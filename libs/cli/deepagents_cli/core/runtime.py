"""Core runtime for executing CLI agent sessions independent of UI."""
# ruff: noqa: D107,DOC201,DOC402,DOC501,E501,I001,PLR6301,PLC2801,TRY301

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.human_in_the_loop import HITLRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter, ValidationError

from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import create_model, settings
from deepagents_cli.core.contracts import DecisionBatch, SessionCreateRequest, SessionEvent
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.sessions import generate_thread_id, get_checkpointer
from deepagents_cli.tools import fetch_url, http_request, web_search

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from deepagents.backends import CompositeBackend
    from deepagents.backends.sandbox import SandboxBackendProtocol
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.pregel import Pregel

logger = logging.getLogger(__name__)

_HITL_REQUEST_ADAPTER = TypeAdapter(HITLRequest)
_STREAM_CHUNK_LENGTH = 3
_MESSAGE_DATA_LENGTH = 2


@dataclass
class RuntimeSession:
    """Single runtime session with event queue and interrupt state."""

    session_id: str
    assistant_id: str
    thread_id: str
    agent: Pregel
    backend: CompositeBackend
    checkpointer: BaseCheckpointSaver
    checkpointer_cm: Any
    sandbox_cm: contextlib.AbstractContextManager[SandboxBackendProtocol] | None
    event_queue: asyncio.Queue[SessionEvent] = field(default_factory=asyncio.Queue)
    event_counter: int = 0
    pending_decisions: dict[str, asyncio.Future[DecisionBatch]] = field(default_factory=dict)
    running_task: asyncio.Task[None] | None = None
    closed: bool = False
    description: str | None = None

    def next_event_id(self) -> int:
        """Increment and return the next event id."""
        self.event_counter += 1
        return self.event_counter

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Push one event onto the session queue."""
        await self.event_queue.put(
            SessionEvent(
                event_id=self.next_event_id(),
                event_type=cast("Any", event_type),
                session_id=self.session_id,
                payload=payload,
            )
        )


class CoreSessionManager:
    """Manages lifecycle and execution of runtime sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, RuntimeSession] = {}
        self._create_lock = asyncio.Lock()

    def get(self, session_id: str) -> RuntimeSession | None:
        """Get an existing session."""
        return self._sessions.get(session_id)

    async def create_session(self, request: SessionCreateRequest) -> RuntimeSession:
        """Create a runtime session with initialized agent graph."""
        async with self._create_lock:
            return await self._create_session_locked(request)

    async def _create_session_locked(self, request: SessionCreateRequest) -> RuntimeSession:
        """Create a runtime session while holding the global create lock."""
        model_result = create_model(
            request.model_name, extra_kwargs=request.model_params
        )
        model = model_result.model
        model_result.apply_to_settings()

        sandbox_backend = None
        sandbox_cm = None
        if request.sandbox_type != "none":
            from deepagents_cli.integrations.sandbox_factory import create_sandbox

            sandbox_cm = create_sandbox(
                request.sandbox_type,
                sandbox_id=request.sandbox_id,
                setup_script_path=request.sandbox_setup,
            )
            sandbox_backend = sandbox_cm.__enter__()

        checkpointer_cm = get_checkpointer()
        checkpointer = await checkpointer_cm.__aenter__()

        tools: list[Any] = [http_request, fetch_url]
        if settings.has_tavily:
            tools.append(web_search)

        thread_id = request.thread_id or generate_thread_id()
        with _temporary_cwd(request.cwd):
            agent, backend = create_cli_agent(
                model=model,
                assistant_id=request.assistant_id,
                tools=tools,
                sandbox=sandbox_backend,
                sandbox_type=(
                    request.sandbox_type if request.sandbox_type != "none" else None
                ),
                auto_approve=request.auto_approve,
                enable_shell=request.enable_shell,
                checkpointer=checkpointer,
            )

        session_id = uuid.uuid4().hex
        session = RuntimeSession(
            session_id=session_id,
            assistant_id=request.assistant_id,
            thread_id=thread_id,
            agent=agent,
            backend=backend,
            checkpointer=checkpointer,
            checkpointer_cm=checkpointer_cm,
            sandbox_cm=sandbox_cm,
        )
        self._sessions[session_id] = session
        return session

    async def close_session(self, session_id: str) -> None:
        """Close and remove a runtime session."""
        session = self._sessions.pop(session_id, None)
        if session is None or session.closed:
            return
        session.closed = True
        if session.running_task and not session.running_task.done():
            session.running_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.running_task

        await session.checkpointer_cm.__aexit__(None, None, None)
        if session.sandbox_cm is not None:
            with contextlib.suppress(Exception):
                session.sandbox_cm.__exit__(None, None, None)

    async def submit_message(self, session_id: str, message: str) -> None:
        """Start a run for the given message."""
        session = self._require_session(session_id)
        if session.running_task and not session.running_task.done():
            msg = f"session '{session_id}' already has an active run"
            raise RuntimeError(msg)

        session.running_task = asyncio.create_task(self._run_session(session, message))

    async def submit_decision(
        self,
        session_id: str,
        interrupt_id: str,
        decision_batch: DecisionBatch,
    ) -> None:
        """Submit decisions for one pending interrupt."""
        session = self._require_session(session_id)
        future = session.pending_decisions.get(interrupt_id)
        if future is None:
            msg = f"interrupt '{interrupt_id}' not found"
            raise KeyError(msg)
        if future.done():
            msg = f"interrupt '{interrupt_id}' already resolved"
            raise RuntimeError(msg)
        future.set_result(decision_batch)

    async def cancel_run(self, session_id: str) -> bool:
        """Cancel an active run for a session.

        Returns:
            `True` when a run was cancelled, `False` when there is no active run.
        """
        session = self._require_session(session_id)
        task = session.running_task
        if task is None or task.done():
            return False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return True

    def set_description(self, session_id: str, description: str) -> None:
        """Persist a human-readable description for the active session."""
        session = self._require_session(session_id)
        session.description = description

    async def events(
        self,
        session_id: str,
        *,
        after_event_id: int = 0,
    ) -> AsyncIterator[SessionEvent]:
        """Stream events for a session from queue and skip older ids."""
        session = self._require_session(session_id)
        while True:
            event = await session.event_queue.get()
            if event.event_id <= after_event_id:
                continue
            yield event
            if event.event_type == "done" and event.payload.get("final", False):
                break

    def _require_session(self, session_id: str) -> RuntimeSession:
        session = self.get(session_id)
        if session is None:
            msg = f"session '{session_id}' not found"
            raise KeyError(msg)
        return session

    async def _stream_agent(
        self,
        session: RuntimeSession,
        stream_input: dict[str, Any] | Command,
        config: RunnableConfig,
        file_op_tracker: FileOpTracker,
    ) -> tuple[dict[str, HITLRequest], int]:
        pending_interrupts: dict[str, HITLRequest] = {}
        tool_call_buffers: dict[int | str, dict[str, Any]] = {}
        displayed_tool_ids: set[str] = set()
        total_tokens = 0

        async for chunk in session.agent.astream(
            stream_input,
            stream_mode=["messages", "updates"],
            subgraphs=True,
            config=config,
            durability="exit",
        ):
            if not isinstance(chunk, tuple) or len(chunk) != _STREAM_CHUNK_LENGTH:
                continue
            namespace, stream_mode, data = chunk
            if namespace:
                continue

            if stream_mode == "updates" and isinstance(data, dict) and "__interrupt__" in data:
                interrupts = cast("dict[str, list[Interrupt]]", data).get("__interrupt__", [])
                for interrupt in interrupts:
                    try:
                        hitl_request = _HITL_REQUEST_ADAPTER.validate_python(interrupt.value)
                    except ValidationError:
                        await session.emit(
                            "error",
                            {
                                "message": (
                                    "Received malformed approval request from runtime"
                                )
                            },
                        )
                        continue
                    pending_interrupts[interrupt.id] = hitl_request
                    await session.emit(
                        "approval_required",
                        {
                            "interrupt_id": interrupt.id,
                            "action_requests": hitl_request["action_requests"],
                        },
                    )
                continue

            if stream_mode != "messages":
                continue
            if not isinstance(data, tuple) or len(data) != _MESSAGE_DATA_LENGTH:
                continue

            message_obj, metadata = data
            if isinstance(metadata, dict) and metadata.get("lc_source") == "summarization":
                continue

            if isinstance(message_obj, AIMessage):
                usage_metadata = getattr(message_obj, "usage_metadata", None)
                if usage_metadata and isinstance(usage_metadata, dict):
                    total_tokens = int(usage_metadata.get("total_tokens") or total_tokens)

                content_blocks = getattr(message_obj, "content_blocks", [])
                if not isinstance(content_blocks, list):
                    continue

                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text", "")
                        if text:
                            await session.emit("text_delta", {"text": text})
                    elif block_type in {"tool_call", "tool_call_chunk"}:
                        chunk_id = block.get("id")
                        chunk_name = block.get("name")
                        chunk_index = block.get("index")
                        chunk_args = block.get("args", {})

                        if chunk_index is not None:
                            key: int | str = chunk_index
                        elif chunk_id is not None:
                            key = chunk_id
                        else:
                            key = f"unknown-{len(tool_call_buffers)}"

                        buffer = tool_call_buffers.setdefault(
                            key,
                            {"id": None, "name": None, "args": {}},
                        )
                        if chunk_id:
                            buffer["id"] = chunk_id
                        if chunk_name:
                            buffer["name"] = chunk_name
                        if isinstance(chunk_args, dict) and chunk_args:
                            buffer["args"] = chunk_args

                        buffer_id = buffer.get("id")
                        buffer_name = buffer.get("name")
                        if buffer_id and buffer_name and buffer_id not in displayed_tool_ids:
                            displayed_tool_ids.add(buffer_id)
                            await session.emit(
                                "tool_call_started",
                                {
                                    "tool_call_id": buffer_id,
                                    "name": buffer_name,
                                    "args": buffer.get("args", {}),
                                },
                            )

            elif isinstance(message_obj, ToolMessage):
                record = file_op_tracker.complete_with_message(message_obj)
                diff = record.diff if record and record.diff else None
                tool_call_id = getattr(message_obj, "tool_call_id", None)
                name = getattr(message_obj, "name", "")
                status = getattr(message_obj, "status", "success")
                content = message_obj.content
                output = content if isinstance(content, str) else str(content)
                await session.emit(
                    "tool_call_finished",
                    {
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "status": status,
                        "output": output,
                        "diff": diff,
                    },
                )

        if total_tokens > 0:
            await session.emit("token_usage", {"total_tokens": total_tokens})

        return pending_interrupts, total_tokens

    async def _run_session(self, session: RuntimeSession, message: str) -> None:
        """Execute one user message through the agent with HITL support."""
        config: RunnableConfig = {
            "configurable": {"thread_id": session.thread_id},
            "metadata": {
                "assistant_id": session.assistant_id,
                "agent_name": session.assistant_id,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        }

        file_op_tracker = FileOpTracker(assistant_id=session.assistant_id, backend=session.backend)
        stream_input: dict[str, Any] | Command = {
            "messages": [{"role": "user", "content": message}]
        }

        try:
            while True:
                pending_interrupts, _ = await self._stream_agent(
                    session,
                    stream_input,
                    config,
                    file_op_tracker,
                )
                if not pending_interrupts:
                    break

                resume_payload: dict[str, dict[str, list[dict[str, Any]]]] = {}
                for interrupt_id, hitl_request in pending_interrupts.items():
                    decision_future: asyncio.Future[DecisionBatch] = asyncio.Future()
                    session.pending_decisions[interrupt_id] = decision_future
                    decision_batch = await decision_future
                    decisions = [d.model_dump(exclude_none=True) for d in decision_batch.decisions]

                    if len(decisions) != len(hitl_request["action_requests"]):
                        msg = "decision count does not match action request count"
                        raise ValueError(msg)

                    resume_payload[interrupt_id] = {"decisions": decisions}
                    session.pending_decisions.pop(interrupt_id, None)

                stream_input = Command(resume=resume_payload)

            await session.emit("done", {"final": True})
        except asyncio.CancelledError:
            await session.emit("error", {"message": "Run cancelled"})
            await session.emit("done", {"final": True})
            raise
        except Exception as e:
            logger.exception("Runtime session %s failed", session.session_id)
            await session.emit("error", {"message": str(e)})
            await session.emit("done", {"final": True})


@contextlib.contextmanager
def _temporary_cwd(path: str | None) -> contextlib.AbstractContextManager[None]:
    """Temporarily switch process working directory."""
    if not path:
        yield
        return
    previous = Path.cwd()
    target = Path(path)
    if not target.exists() or not target.is_dir():
        msg = f"cwd '{path}' does not exist or is not a directory"
        raise RuntimeError(msg)
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(previous)
