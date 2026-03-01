"""High-level transport helpers for deepagents CLI HTTP ACP mode."""
# ruff: noqa: D107,DOC201,ANN201

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deepagents_cli.client.http_client import ServiceHttpClient
from deepagents_cli.core.contracts import (
    DecisionBatch,
    SessionCreateRequest,
    SessionEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass(frozen=True)
class SessionHandle:
    """Client-side runtime session handle."""

    session_id: str
    thread_id: str


class ServiceTransport:
    """Thin orchestration wrapper around JSON-RPC `ServiceHttpClient`."""

    def __init__(self, client: ServiceHttpClient) -> None:
        self._client = client
        self._request_ids = itertools.count(1)
        self._initialized = False

    @property
    def base_url(self) -> str:
        """Service base URL."""
        return self._client.base_url

    def _next_request_id(self) -> int:
        return int(next(self._request_ids))

    async def initialize(self) -> None:
        """Initialize ACP session once per client transport."""
        if self._initialized:
            return
        response = await self._client.call(
            method="initialize",
            params={"clientInfo": {"name": "deepagents-cli"}},
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        self._initialized = True

    async def new_session(self, request: SessionCreateRequest) -> SessionHandle:
        """Create runtime session and return handle."""
        await self.initialize()
        response = await self._client.call(
            method="session/new",
            params={
                "assistantId": request.assistant_id,
                "modelName": request.model_name,
                "modelParams": request.model_params,
                "sandboxType": request.sandbox_type,
                "sandboxId": request.sandbox_id,
                "sandboxSetup": request.sandbox_setup,
                "autoApprove": request.auto_approve,
                "enableShell": request.enable_shell,
                "threadId": request.thread_id,
                "cwd": request.cwd,
            },
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        result = _require_result(response)
        return SessionHandle(
            session_id=str(result["sessionId"]),
            thread_id=str(result["threadId"]),
        )

    async def prompt_stream(self, session_id: str, message: str):
        """Yield runtime events from streaming `session/prompt` call."""
        await self.initialize()
        request_id = self._next_request_id()
        async for event in self._stream_updates(
            method="session/prompt",
            params={"sessionId": session_id, "prompt": message, "stream": True},
            request_id=request_id,
            session_id=session_id,
            include_permission=True,
        ):
            yield event

    async def load_session_stream(
        self,
        runtime_session_id: str,
        thread_id: str,
        *,
        cwd: str | None = None,
    ):
        """Yield replay events from streaming `session/load` call."""
        await self.initialize()
        request_id = self._next_request_id()
        async for event in self._stream_updates(
            method="session/load",
            params={
                "sessionId": thread_id,
                "cwd": cwd,
                "mcpServers": [],
                "runtimeSessionId": runtime_session_id,
                "stream": True,
            },
            request_id=request_id,
            session_id=runtime_session_id,
            include_permission=False,
        ):
            yield event

    async def _stream_updates(
        self,
        *,
        method: str,
        params: dict[str, Any],
        request_id: int,
        session_id: str,
        include_permission: bool,
    ) -> AsyncIterator[SessionEvent]:
        """Parse streaming JSON-RPC messages into `SessionEvent`s`.

        Yields:
            Parsed `SessionEvent` values from the JSON-RPC stream.
        """
        last_event_id = 0
        async for rpc_message in self._client.call_stream(
            method=method,
            params=params,
            request_id=request_id,
        ):
            if rpc_message.get("method") == "session/update":
                update_params = rpc_message.get("params", {})
                if not isinstance(update_params, dict):
                    continue
                event_data = update_params.get("event", {})
                if not isinstance(event_data, dict):
                    continue
                event = SessionEvent.model_validate(event_data)
                last_event_id = max(last_event_id, event.event_id)
                yield event
                continue

            if (
                include_permission
                and rpc_message.get("method") == "session/request_permission"
            ):
                permission_params = rpc_message.get("params", {})
                if not isinstance(permission_params, dict):
                    continue
                tool_call = permission_params.get("toolCall", {})
                if not isinstance(tool_call, dict):
                    tool_call = {}
                action_requests = [tool_call] if tool_call else []
                meta = permission_params.get("_meta", {})
                if not isinstance(meta, dict):
                    meta = {}
                interrupt_id = str(meta.get("interruptId", ""))
                permission_request_id = str(rpc_message.get("id", ""))
                yield SessionEvent(
                    event_id=last_event_id + 1,
                    event_type="approval_required",
                    session_id=session_id,
                    payload={
                        "action_requests": action_requests,
                        "interrupt_id": interrupt_id,
                        "permission_request_id": permission_request_id,
                    },
                )
                continue

            if "error" in rpc_message:
                error = rpc_message.get("error", {})
                msg = "Unknown error"
                if isinstance(error, dict):
                    msg = str(error.get("message", msg))
                yield SessionEvent(
                    event_id=last_event_id + 1,
                    event_type="error",
                    session_id=session_id,
                    payload={"message": msg},
                )
                break

            if "result" in rpc_message:
                yield SessionEvent(
                    event_id=last_event_id + 1,
                    event_type="done",
                    session_id=session_id,
                    payload={"final": True},
                )
                break

    async def compact_session(self, session_id: str) -> dict[str, Any]:
        """Run session compaction through service extension method."""
        await self.initialize()
        response = await self._client.call(
            method="_deepagents/session_compact",
            params={"sessionId": session_id},
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        return _require_result(response)

    async def submit_decision(
        self,
        session_id: str,
        interrupt_id: str,
        decisions: list[dict[str, Any]],
    ) -> None:
        """Submit permission decision response for one request id."""
        _ = session_id
        await self.initialize()
        batch = DecisionBatch.model_validate({"decisions": decisions})
        response = await self._client.respond(
            request_id=interrupt_id,
            result=_decision_batch_to_permission_outcome(batch),
        )
        _raise_if_error(response)

    async def list_threads(
        self,
        *,
        agent_name: str | None,
        limit: int,
        include_message_count: bool,
    ) -> list[dict[str, Any]]:
        """List threads via service API."""
        await self.initialize()
        response = await self._client.call(
            method="session/list",
            params={
                "cwd": None,
                "cursor": None,
            },
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        result = _require_result(response)
        sessions = result.get("sessions", [])
        if not isinstance(sessions, list):
            return []
        threads: list[dict[str, Any]] = []
        for item in sessions:
            if not isinstance(item, dict):
                continue
            meta = item.get("_meta", {})
            if not isinstance(meta, dict):
                meta = {}
            mapped = {
                "thread_id": item.get("sessionId"),
                "updated_at": item.get("updatedAt"),
                "agent_name": meta.get("agentName"),
            }
            if include_message_count:
                mapped["message_count"] = meta.get("messageCount")
            threads.append(mapped)

        if agent_name is not None:
            threads = [t for t in threads if t.get("agent_name") == agent_name]
        return threads[:limit]

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete one thread via service API."""
        await self.initialize()
        response = await self._client.call(
            method="session/delete",
            params={"sessionId": thread_id},
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        result = _require_result(response)
        return bool(result.get("ok", False))

    async def list_agents(self) -> list[dict[str, Any]]:
        """List agents from service-side storage."""
        await self.initialize()
        response = await self._client.call(
            method="_deepagents/agents_list",
            params={},
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        result = _require_result(response)
        agents = result.get("agents", [])
        return [agent for agent in agents if isinstance(agent, dict)]

    async def reset_agent(
        self,
        *,
        agent_name: str,
        source_agent: str | None = None,
    ) -> dict[str, Any]:
        """Reset one service-side agent."""
        await self.initialize()
        response = await self._client.call(
            method="_deepagents/agents_reset",
            params={"agentName": agent_name, "sourceAgent": source_agent},
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        return _require_result(response)

    async def list_skills(self, *, agent: str, project: bool) -> list[dict[str, Any]]:
        """List skills from service-side skill directories."""
        await self.initialize()
        response = await self._client.call(
            method="_deepagents/skills_list",
            params={"agent": agent, "project": project},
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        result = _require_result(response)
        skills = result.get("skills", [])
        return [skill for skill in skills if isinstance(skill, dict)]

    async def create_skill(
        self,
        *,
        name: str,
        agent: str,
        project: bool,
    ) -> dict[str, Any]:
        """Create one skill on service filesystem."""
        await self.initialize()
        response = await self._client.call(
            method="_deepagents/skills_create",
            params={"name": name, "agent": agent, "project": project},
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        return _require_result(response)

    async def skill_info(
        self,
        *,
        name: str,
        agent: str,
        project: bool,
    ) -> dict[str, Any]:
        """Fetch one skill details from service filesystem."""
        await self.initialize()
        response = await self._client.call(
            method="_deepagents/skills_info",
            params={"name": name, "agent": agent, "project": project},
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        return _require_result(response)

    async def delete_skill(
        self,
        *,
        name: str,
        agent: str,
        project: bool,
        force: bool,
    ) -> dict[str, Any]:
        """Delete one skill directory on service filesystem."""
        await self.initialize()
        response = await self._client.call(
            method="_deepagents/skills_delete",
            params={
                "name": name,
                "agent": agent,
                "project": project,
                "force": force,
            },
            request_id=self._next_request_id(),
        )
        _raise_if_error(response)
        return _require_result(response)


def _require_result(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result")
    if not isinstance(result, dict):
        msg = "Missing JSON-RPC result payload"
        raise TypeError(msg)
    return result


def _raise_if_error(response: dict[str, Any]) -> None:
    error = response.get("error")
    if not isinstance(error, dict):
        return
    msg = str(error.get("message", "Unknown JSON-RPC error"))
    raise RuntimeError(msg)


def make_service_transport(base_url: str) -> ServiceTransport:
    """Factory for service transport."""
    return ServiceTransport(ServiceHttpClient(base_url))


def _decision_batch_to_permission_outcome(batch: DecisionBatch) -> dict[str, Any]:
    """Map internal decision batch to ACP permission outcome result."""
    if not batch.decisions:
        return {"outcome": "cancelled"}

    first = batch.decisions[0]
    if first.type == "approve":
        if first.message == "allow-always":
            return {"outcome": "selected", "optionId": "allow-always"}
        return {"outcome": "selected", "optionId": "allow-once"}
    if first.type == "reject":
        return {"outcome": "selected", "optionId": "reject-once"}
    return {"outcome": "cancelled"}
