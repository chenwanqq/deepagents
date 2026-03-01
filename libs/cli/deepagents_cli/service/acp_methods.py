"""ACP-style JSON-RPC method handlers for the CLI service."""
# ruff: noqa: DOC201,DOC501,TRY003,TRY301,EM102,RET504,TC003,E501,I001

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import ValidationError

from deepagents_cli._version import __version__
from deepagents_cli.config import create_model, settings
from deepagents_cli.core.admin import (
    create_skill_core,
    delete_skill_core,
    list_agents_core,
    list_skills_core,
    reset_agent_core,
    skill_info_core,
)
from deepagents_cli.core.contracts import (
    AgentsResetParams,
    DecisionBatch,
    JsonRpcId,
    JsonRpcNotification,
    PermissionOption,
    SessionCancelParams,
    SessionCreateRequest,
    SessionDeleteRequestParams,
    SessionDeleteResult,
    SessionInfo,
    SessionLoadParams,
    SessionListRequestParams,
    SessionListResult,
    SessionNewParams,
    SessionPromptParams,
    SessionCompactParams,
    SessionSetDescriptionParams,
    SkillsCreateParams,
    SkillsDeleteParams,
    SkillsInfoParams,
    SkillsListParams,
)
from deepagents_cli.core.session_store import session_manager
from deepagents_cli.core.threads import delete_thread_core, list_threads_core
from deepagents_cli.service.permission_broker import permission_broker
from deepagents_cli.service.jsonrpc import (
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INTERNAL_ERROR,
    error_response,
    ok_response,
)


def initialize_result() -> dict[str, Any]:
    """Build `initialize` response result."""
    return {
        "protocolVersion": "acp-draft-http-jsonrpc",
        "serverInfo": {
            "name": "deepagents-cli-service",
            "version": __version__,
        },
        "agentCapabilities": {
            "streaming": True,
            "loadSession": True,
            "methods": [
                "initialize",
                "session/new",
                "session/load",
                "session/prompt",
                "session/cancel",
                "session/set_description",
                "session/request_permission",
                "session/list",
                "session/delete",
                "_deepagents/session_compact",
                "_deepagents/agents_list",
                "_deepagents/agents_reset",
                "_deepagents/skills_list",
                "_deepagents/skills_create",
                "_deepagents/skills_info",
                "_deepagents/skills_delete",
            ],
            "sessionCapabilities": {"list": {}, "delete": {}},
        },
    }


async def call_method(
    *,
    method: str,
    params: dict[str, Any] | None,
    request_id: JsonRpcId,
) -> dict[str, Any]:
    """Handle non-streaming JSON-RPC methods and return response payload."""
    try:
        if method == "initialize":
            return ok_response(request_id, initialize_result()).model_dump(
                exclude_none=True
            )
        if method == "session/new":
            payload = SessionNewParams.model_validate(params or {})
            session = await session_manager.create_session(
                SessionCreateRequest(
                    assistant_id=payload.assistant_id,
                    model_name=payload.model_name,
                    model_params=payload.model_params,
                    sandbox_type=payload.sandbox_type,
                    sandbox_id=payload.sandbox_id,
                    sandbox_setup=payload.sandbox_setup,
                    auto_approve=payload.auto_approve,
                    enable_shell=payload.enable_shell,
                    thread_id=payload.thread_id,
                    cwd=payload.cwd,
                )
            )
            return ok_response(
                request_id,
                {"sessionId": session.session_id, "threadId": session.thread_id},
            ).model_dump(exclude_none=True)
        if method == "session/cancel":
            payload = SessionCancelParams.model_validate(params or {})
            cancelled = await session_manager.cancel_run(payload.session_id)
            permission_broker.cancel_pending_for_session(payload.session_id)
            return ok_response(request_id, {"cancelled": cancelled}).model_dump(
                exclude_none=True
            )
        if method == "session/set_description":
            payload = SessionSetDescriptionParams.model_validate(params or {})
            session_manager.set_description(payload.session_id, payload.description)
            return ok_response(request_id, {"ok": True}).model_dump(exclude_none=True)
        if method == "session/list":
            payload = SessionListRequestParams.model_validate(params or {})
            if payload.cursor:
                msg = "cursor not supported yet in this implementation"
                raise KeyError(msg)
            threads = await list_threads_core(
                None,
                limit=200,
                include_message_count=True,
                cwd=payload.cwd,
            )
            sessions = [_to_session_info(thread) for thread in threads]
            result = SessionListResult(sessions=sessions, next_cursor=None)
            return ok_response(
                request_id,
                result.model_dump(by_alias=True, exclude_none=True),
            ).model_dump(exclude_none=True)
        if method == "session/delete":
            payload = SessionDeleteRequestParams.model_validate(params or {})
            await delete_thread_core(payload.session_id)
            result = SessionDeleteResult(ok=True)
            return ok_response(request_id, result.model_dump(exclude_none=True)).model_dump(
                exclude_none=True
            )
        if method == "session/load":
            payload = SessionLoadParams.model_validate(params or {})
            runtime_session_id = payload.runtime_session_id
            if not runtime_session_id:
                msg = "runtimeSessionId is required"
                raise KeyError(msg)
            session = session_manager.get(runtime_session_id)
            if session is None:
                raise KeyError(f"session '{runtime_session_id}' not found")
            session.thread_id = payload.session_id
            return ok_response(
                request_id,
                {"sessionId": payload.session_id, "ok": True},
            ).model_dump(exclude_none=True)
        if method == "_deepagents/session_compact":
            payload = SessionCompactParams.model_validate(params or {})
            result = await compact_session(payload.session_id)
            return ok_response(request_id, result).model_dump(exclude_none=True)
        if method == "_deepagents/agents_list":
            return ok_response(request_id, {"agents": list_agents_core()}).model_dump(
                exclude_none=True
            )
        if method == "_deepagents/agents_reset":
            payload = AgentsResetParams.model_validate(params or {})
            result = reset_agent_core(payload.agent_name, payload.source_agent)
            return ok_response(request_id, result).model_dump(exclude_none=True)
        if method == "_deepagents/skills_list":
            payload = SkillsListParams.model_validate(params or {})
            result = list_skills_core(payload.agent, project=payload.project)
            return ok_response(request_id, {"skills": result}).model_dump(
                exclude_none=True
            )
        if method == "_deepagents/skills_create":
            payload = SkillsCreateParams.model_validate(params or {})
            result = create_skill_core(payload.name, payload.agent, project=payload.project)
            return ok_response(request_id, result).model_dump(exclude_none=True)
        if method == "_deepagents/skills_info":
            payload = SkillsInfoParams.model_validate(params or {})
            result = skill_info_core(payload.name, payload.agent, project=payload.project)
            return ok_response(request_id, result).model_dump(exclude_none=True)
        if method == "_deepagents/skills_delete":
            payload = SkillsDeleteParams.model_validate(params or {})
            result = delete_skill_core(
                payload.name,
                payload.agent,
                project=payload.project,
                force=payload.force,
            )
            return ok_response(request_id, result).model_dump(exclude_none=True)
        if method == "session/prompt":
            payload = SessionPromptParams.model_validate(params or {})
            result = await run_prompt_non_stream(
                request_id=request_id,
                session_id=payload.session_id,
                prompt=payload.prompt,
            )
            return result
    except ValidationError as e:
        return error_response(
            request_id,
            code=JSONRPC_INVALID_PARAMS,
            message="Invalid params",
            data={"errors": e.errors()},
        ).model_dump(exclude_none=True)
    except KeyError as e:
        return error_response(
            request_id,
            code=JSONRPC_INVALID_PARAMS,
            message=str(e),
        ).model_dump(exclude_none=True)
    except RuntimeError as e:
        return error_response(
            request_id,
            code=JSONRPC_INTERNAL_ERROR,
            message=str(e),
        ).model_dump(exclude_none=True)
    except Exception as e:  # noqa: BLE001
        return error_response(
            request_id,
            code=JSONRPC_INTERNAL_ERROR,
            message=str(e),
        ).model_dump(exclude_none=True)

    return error_response(
        request_id,
        code=-32601,
        message=f"Method not found: {method}",
    ).model_dump(exclude_none=True)


async def stream_prompt_messages(
    *,
    request_id: JsonRpcId,
    params: dict[str, Any] | None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield JSON-RPC messages for a streaming `session/prompt` call."""
    try:
        payload = SessionPromptParams.model_validate(params or {})
        session = session_manager.get(payload.session_id)
        if session is None:
            raise KeyError(f"session '{payload.session_id}' not found")
        after_event_id = session.event_counter
        await session_manager.submit_message(payload.session_id, payload.prompt)
        async for event in session_manager.events(
            payload.session_id,
            after_event_id=after_event_id,
        ):
            if event.event_type == "approval_required":
                interrupt_id = str(event.payload.get("interrupt_id", ""))
                action_requests_raw = event.payload.get("action_requests", [])
                action_requests = (
                    action_requests_raw if isinstance(action_requests_raw, list) else []
                )
                tool_call = (
                    action_requests[0]
                    if action_requests and isinstance(action_requests[0], dict)
                    else {"name": "unknown", "args": {}}
                )
                options = [
                    PermissionOption(
                        id="allow-once",
                        kind="allow_once",
                        name="Allow once",
                    ),
                    PermissionOption(
                        id="allow-always",
                        kind="allow_always",
                        name="Allow always",
                    ),
                    PermissionOption(
                        id="reject-once",
                        kind="reject_once",
                        name="Reject once",
                    ),
                ]
                pending = permission_broker.register_request(
                    session_id=payload.session_id,
                    interrupt_id=interrupt_id,
                    action_requests=[
                        req for req in action_requests if isinstance(req, dict)
                    ],
                )
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": pending.request_id,
                    "method": "session/request_permission",
                    "params": {
                        "sessionId": payload.session_id,
                        "toolCall": tool_call,
                        "options": [
                            option.model_dump(by_alias=True, exclude_none=True)
                            for option in options
                        ],
                        "_meta": {
                            "interruptId": interrupt_id,
                            "actionRequests": action_requests,
                        },
                    },
                }
                yield rpc_request
                outcome = await permission_broker.wait_for_outcome(pending.request_id)
                permission_broker.pop(pending.request_id)
                decision_batch = _map_permission_outcome_to_decisions(
                    outcome=outcome,
                    action_request_count=len(pending.action_requests),
                )
                await session_manager.submit_decision(
                    payload.session_id,
                    interrupt_id,
                    decision_batch,
                )
                continue

            if event.event_type == "error":
                yield error_response(
                    request_id,
                    code=JSONRPC_INTERNAL_ERROR,
                    message=str(event.payload.get("message", "unknown error")),
                ).model_dump(exclude_none=True)
                return

            if event.event_type == "done":
                yield ok_response(
                    request_id,
                    {"sessionId": payload.session_id, "final": True},
                ).model_dump(exclude_none=True)
                return

            notification = JsonRpcNotification(
                method="session/update",
                params={
                    "sessionId": payload.session_id,
                    "event": event.model_dump(exclude_none=True),
                },
            )
            yield notification.model_dump(exclude_none=True)

    except ValidationError as e:
        yield error_response(
            request_id,
            code=JSONRPC_INVALID_PARAMS,
            message="Invalid params",
            data={"errors": e.errors()},
        ).model_dump(exclude_none=True)
    except KeyError as e:
        yield error_response(
            request_id,
            code=JSONRPC_INVALID_PARAMS,
            message=str(e),
        ).model_dump(exclude_none=True)
    except RuntimeError as e:
        yield error_response(
            request_id,
            code=JSONRPC_INTERNAL_ERROR,
            message=str(e),
        ).model_dump(exclude_none=True)
    except Exception as e:  # noqa: BLE001
        yield error_response(
            request_id,
            code=JSONRPC_INTERNAL_ERROR,
            message=str(e),
        ).model_dump(exclude_none=True)


async def stream_load_messages(
    *,
    request_id: JsonRpcId,
    params: dict[str, Any] | None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield JSON-RPC messages for streaming `session/load` history replay."""
    try:
        payload = SessionLoadParams.model_validate(params or {})
        runtime_session_id = payload.runtime_session_id
        if not runtime_session_id:
            msg = "runtimeSessionId is required"
            raise KeyError(msg)
        session = session_manager.get(runtime_session_id)
        if session is None:
            raise KeyError(f"session '{runtime_session_id}' not found")

        session.thread_id = payload.session_id
        config = {"configurable": {"thread_id": payload.session_id}}
        state = await session.agent.aget_state(config)
        values = state.values if state and state.values else {}
        messages = values.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        event_id = 0
        pending_tool_calls: dict[str, dict[str, Any]] = {}
        for message in messages:
            if isinstance(message, HumanMessage):
                text = message.content if isinstance(message.content, str) else str(message.content)
                if text:
                    event_id += 1
                    yield _session_update(
                        payload.session_id,
                        event_id,
                        "text_delta",
                        {"text": text, "role": "user"},
                    )
                continue

            if isinstance(message, AIMessage):
                tool_calls = getattr(message, "tool_calls", None)
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        tool_call_id = str(tool_call.get("id") or "")
                        tool_name = str(tool_call.get("name") or "unknown")
                        tool_args = tool_call.get("args", {})
                        payload_args = tool_args if isinstance(tool_args, dict) else {}
                        if tool_call_id:
                            pending_tool_calls[tool_call_id] = {
                                "name": tool_name,
                                "args": payload_args,
                            }
                        event_id += 1
                        yield _session_update(
                            payload.session_id,
                            event_id,
                            "tool_call_started",
                            {
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "args": payload_args,
                                "role": "assistant",
                            },
                        )

                text_blocks: list[str] = []
                if isinstance(message.content, str):
                    text_blocks.append(message.content)
                elif isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text:
                                text_blocks.append(text)
                for text in text_blocks:
                    event_id += 1
                    yield _session_update(
                        payload.session_id,
                        event_id,
                        "text_delta",
                        {"text": text, "role": "assistant"},
                    )
                continue

            if isinstance(message, ToolMessage):
                tool_call_id = str(getattr(message, "tool_call_id", "") or "")
                status = str(getattr(message, "status", "success"))
                content = message.content if isinstance(message.content, str) else str(message.content)
                started = pending_tool_calls.get(tool_call_id, {})
                event_id += 1
                yield _session_update(
                    payload.session_id,
                    event_id,
                    "tool_call_finished",
                    {
                        "tool_call_id": tool_call_id,
                        "name": started.get("name", str(getattr(message, "name", "unknown"))),
                        "status": status,
                        "output": content,
                        "role": "tool",
                    },
                )

        yield ok_response(
            request_id,
            {"sessionId": payload.session_id, "ok": True},
        ).model_dump(exclude_none=True)
    except ValidationError as e:
        yield error_response(
            request_id,
            code=JSONRPC_INVALID_PARAMS,
            message="Invalid params",
            data={"errors": e.errors()},
        ).model_dump(exclude_none=True)
    except KeyError as e:
        yield error_response(
            request_id,
            code=JSONRPC_INVALID_PARAMS,
            message=str(e),
        ).model_dump(exclude_none=True)
    except Exception as e:  # noqa: BLE001
        yield error_response(
            request_id,
            code=JSONRPC_INTERNAL_ERROR,
            message=str(e),
        ).model_dump(exclude_none=True)


async def run_prompt_non_stream(
    *,
    request_id: JsonRpcId,
    session_id: str,
    prompt: str,
) -> dict[str, Any]:
    """Execute one prompt and return final JSON-RPC response payload."""
    session = session_manager.get(session_id)
    if session is None:
        raise KeyError(f"session '{session_id}' not found")

    after_event_id = session.event_counter
    await session_manager.submit_message(session_id, prompt)

    text_chunks: list[str] = []
    async for event in session_manager.events(session_id, after_event_id=after_event_id):
        if event.event_type == "text_delta":
            text = str(event.payload.get("text", ""))
            if text:
                text_chunks.append(text)
            continue
        if event.event_type == "error":
            return error_response(
                request_id,
                code=JSONRPC_INTERNAL_ERROR,
                message=str(event.payload.get("message", "unknown error")),
            ).model_dump(exclude_none=True)
        if event.event_type == "done":
            break

    return ok_response(
        request_id,
        {
            "sessionId": session_id,
            "final": True,
            "output": "".join(text_chunks),
        },
    ).model_dump(exclude_none=True)


def _to_session_info(thread: dict[str, object]) -> SessionInfo:
    """Map internal thread dict to ACP SessionInfo."""
    meta: dict[str, Any] = {}
    if thread.get("agent_name") is not None:
        meta["agentName"] = thread.get("agent_name")
    if thread.get("message_count") is not None:
        meta["messageCount"] = thread.get("message_count")

    return SessionInfo(
        sessionId=str(thread.get("thread_id", "")),
        cwd=str(Path.cwd()),
        updatedAt=thread.get("updated_at"),
        _meta=meta or None,
    )


async def compact_session(session_id: str) -> dict[str, Any]:
    """Compact one runtime session and return aggregate stats."""
    session = session_manager.get(session_id)
    if session is None:
        raise KeyError(f"session '{session_id}' not found")

    model_spec = f"{settings.model_provider}:{settings.model_name}"
    model = create_model(model_spec).model
    from deepagents.middleware.summarization import (
        SummarizationEvent,
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    defaults = compute_summarization_defaults(model)
    middleware = SummarizationMiddleware(
        model=model,
        backend=session.backend,
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
    )
    config = {"configurable": {"thread_id": session.thread_id}}
    state = await session.agent.aget_state(config)
    values = state.values if state and state.values else {}
    messages = values.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    event = values.get("_summarization_event")
    effective = middleware._apply_event_to_messages(messages, event)
    cutoff = middleware._determine_cutoff_index(effective)
    if cutoff == 0:
        return {
            "ok": True,
            "compactedMessages": 0,
            "tokensBefore": 0,
            "tokensAfter": 0,
        }

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)
    summary = await middleware._acreate_summary(to_summarize)
    tokens_before = count_tokens_approximately(effective)
    tokens_after = count_tokens_approximately(to_keep)
    state_cutoff = middleware._compute_state_cutoff(event, cutoff)
    new_event = SummarizationEvent(
        summary_message=summary,
        file_path=event.get("file_path") if isinstance(event, dict) else None,
        cutoff_index=cutoff,
        state_cutoff=state_cutoff,
    )
    await session.agent.aupdate_state(config, {"_summarization_event": new_event})
    return {
        "ok": True,
        "compactedMessages": len(to_summarize),
        "tokensBefore": tokens_before,
        "tokensAfter": tokens_after,
    }


def _session_update(
    session_id: str,
    event_id: int,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build one `session/update` notification payload."""
    notification = JsonRpcNotification(
        method="session/update",
        params={
            "sessionId": session_id,
            "event": {
                "event_id": event_id,
                "event_type": event_type,
                "session_id": session_id,
                "payload": payload,
            },
        },
    )
    return notification.model_dump(exclude_none=True)


def _map_permission_outcome_to_decisions(
    *,
    outcome: dict[str, Any],
    action_request_count: int,
) -> DecisionBatch:
    """Map ACP permission outcome to runtime decision batch."""
    outcome_value = str(outcome.get("outcome", "cancelled"))
    option_id = str(outcome.get("optionId") or "")

    if outcome_value == "cancelled":
        return DecisionBatch(
            decisions=[
                {"type": "reject", "message": "Permission cancelled"}
                for _ in range(action_request_count)
            ]
        )

    if outcome_value == "selected":
        if option_id in {"allow-once", "allow-always"}:
            return DecisionBatch(
                decisions=[{"type": "approve"} for _ in range(action_request_count)]
            )
        if option_id == "reject-once":
            return DecisionBatch(
                decisions=[
                    {"type": "reject", "message": "Permission rejected"}
                    for _ in range(action_request_count)
                ]
            )

    return DecisionBatch(
        decisions=[
            {"type": "reject", "message": "Unknown permission outcome"}
            for _ in range(action_request_count)
        ]
    )
