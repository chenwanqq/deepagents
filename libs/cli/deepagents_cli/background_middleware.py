"""Background task middleware and HITL bridge for deepagents-cli."""
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import asyncio
import logging
import subprocess  # noqa: S404  # Controlled execution with allow-list and pattern checks
import time
import uuid
from typing import TYPE_CHECKING, Any, NotRequired, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from deepagents_cli.background_tasks import (
    BackgroundResumeDecision,
    BackgroundTaskEvent,
    BackgroundTaskMeta,
    PendingBackgroundInterrupt,
    TaskiqRuntime,
)
from deepagents_cli.config import (
    contains_dangerous_patterns,
    is_shell_command_allowed,
    settings,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.types import Command

logger = logging.getLogger(__name__)

BACKGROUND_HITL_TIMEOUT_SECONDS = 10 * 60
BACKGROUND_HITL_POLL_SECONDS = 0.5
DEFAULT_BACKGROUND_SHELL_TIMEOUT_SECONDS = 60
DEFAULT_BACKGROUND_WAIT_SECONDS = 1.0
MAX_BACKGROUND_WAIT_SECONDS = 30.0
MAX_PREVIEW_LENGTH = 400
BACKGROUND_GUIDANCE_PREFIX = "[system]"
GUIDANCE_DEDUP_TAIL_MESSAGES = 3


class BackgroundTasksAgentState(AgentState):
    """State schema for background task runtime metadata."""

    taskiq_mode: NotRequired[str]
    active_background_task_ids: NotRequired[list[str]]
    background_task_meta: NotRequired[dict[str, BackgroundTaskMeta]]
    background_hitl_queue: NotRequired[list[PendingBackgroundInterrupt]]
    background_hitl_resumes: NotRequired[dict[str, BackgroundResumeDecision]]


class BackgroundTasksMiddleware(AgentMiddleware):
    """Intercept `submit_background_task` and execute in the background."""

    state_schema = BackgroundTasksAgentState

    def __init__(
        self,
        *,
        taskiq_runtime: TaskiqRuntime,
        taskiq_mode: str = "inmemory",
    ) -> None:
        """Initialize the middleware.

        Args:
            taskiq_runtime: Runtime used to dispatch and track background jobs.
            taskiq_mode: Active taskiq mode label.
        """
        self._taskiq_runtime = taskiq_runtime
        self._taskiq_mode = taskiq_mode
        self.tools = [
            self._create_submit_background_tool(),
            self._create_get_background_task_status_tool(),
            self._create_wait_background_tasks_tool(),
            self._create_kill_background_task_tool(),
        ]

    @staticmethod
    def _create_submit_background_tool() -> StructuredTool:
        """Create the `submit_background_task` tool exposed by this middleware."""

        def submit_background_task(
            job_kind: str,
            input: dict[str, Any],  # noqa: A002  # Tool contract uses `input`
        ) -> dict[str, Any]:
            """Submit a background job for async execution in CLI runtime.

            Args:
                job_kind: Must be `shell_command`.
                input: Job payload object.

                    - For `shell_command`: `command` and optional `timeout`.

            Returns:
                Placeholder payload. Middleware intercepts and performs real work.
            """
            return {
                "ok": False,
                "error": "submit_background_task must be intercepted by middleware.",
                "job_kind": job_kind,
                "input": input,
            }

        async def asubmit_background_task(
            job_kind: str,
            input: dict[str, Any],  # noqa: A002  # Tool contract uses `input`
        ) -> dict[str, Any]:
            """Async variant for background task submission schema.

            Args:
                job_kind: Must be `shell_command`.
                input: Job payload object.

            Returns:
                Placeholder payload. Middleware intercepts and performs real work.
            """
            await asyncio.sleep(0)
            return submit_background_task(job_kind=job_kind, input=input)

        return StructuredTool.from_function(
            name="submit_background_task",
            func=submit_background_task,
            coroutine=asubmit_background_task,
            description=(
                "Submit a background job and return a task ID immediately. "
                "Supported jobs: "
                "shell_command(command, timeout)."
            ),
        )

    @staticmethod
    def _create_get_background_task_status_tool() -> StructuredTool:
        """Create status tool for querying background queue and task metadata."""

        def get_background_task_status(
            task_id: str | None = None,
            include_queue: bool = True,
        ) -> dict[str, Any]:
            """Get background task status and recent result metadata.

            Args:
                task_id: Optional task ID filter.
                include_queue: Include pending HITL queue details.

            Returns:
                Placeholder payload. Middleware intercepts and performs real work.
            """
            return {
                "ok": False,
                "error": (
                    "get_background_task_status must be intercepted by middleware."
                ),
                "task_id": task_id,
                "include_queue": include_queue,
            }

        async def aget_background_task_status(
            task_id: str | None = None,
            include_queue: bool = True,
        ) -> dict[str, Any]:
            """Async variant for status tool schema."""
            await asyncio.sleep(0)
            return get_background_task_status(
                task_id=task_id, include_queue=include_queue
            )

        return StructuredTool.from_function(
            name="get_background_task_status",
            func=get_background_task_status,
            coroutine=aget_background_task_status,
            description=(
                "Get status for background tasks. "
                "Optionally filter by task_id."
            ),
        )

    @staticmethod
    def _create_wait_background_tasks_tool() -> StructuredTool:
        """Create wait/sleep tool for polling background progress."""

        def wait_background_tasks(
            seconds: float = DEFAULT_BACKGROUND_WAIT_SECONDS,
        ) -> dict[str, Any]:
            """Sleep briefly to wait for background tasks to progress.

            Args:
                seconds: Sleep duration in seconds.

            Returns:
                Placeholder payload. Middleware intercepts and performs real work.
            """
            return {
                "ok": False,
                "error": "wait_background_tasks must be intercepted by middleware.",
                "seconds": seconds,
            }

        async def await_background_tasks(
            seconds: float = DEFAULT_BACKGROUND_WAIT_SECONDS,
        ) -> dict[str, Any]:
            """Async variant for wait tool schema."""
            await asyncio.sleep(0)
            return wait_background_tasks(seconds=seconds)

        return StructuredTool.from_function(
            name="wait_background_tasks",
            func=wait_background_tasks,
            coroutine=await_background_tasks,
            description=(
                "Sleep for a short period and return a summary "
                "of current background task states."
            ),
        )

    @staticmethod
    def _create_kill_background_task_tool() -> StructuredTool:
        """Create tool for canceling a specific background task."""

        def kill_background_task(task_id: str) -> dict[str, Any]:
            """Stop a background task by task ID.

            Args:
                task_id: Background task identifier.

            Returns:
                Placeholder payload. Middleware intercepts and performs real work.
            """
            return {
                "ok": False,
                "error": "kill_background_task must be intercepted by middleware.",
                "task_id": task_id,
            }

        async def akill_background_task(task_id: str) -> dict[str, Any]:
            """Async variant for kill tool schema."""
            await asyncio.sleep(0)
            return kill_background_task(task_id=task_id)

        return StructuredTool.from_function(
            name="kill_background_task",
            func=kill_background_task,
            coroutine=akill_background_task,
            description="Cancel a specific background task by task_id.",
        )

    def wrap_tool_call(  # noqa: PLR6301  # Interface override requires instance method
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Handle sync tool execution.

        The Textual CLI uses async execution paths; this sync branch is a
        defensive fallback.
        """
        tool_name = request.tool_call["name"]
        if tool_name not in {
            "submit_background_task",
            "get_background_task_status",
            "wait_background_tasks",
            "kill_background_task",
        }:
            return handler(request)
        return ToolMessage(
            content=(
                f"{tool_name} requires async agent execution in the CLI."
            ),
            name=tool_name,
            tool_call_id=request.tool_call.get("id", ""),
            status="error",
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Intercept background task submissions."""
        tool_name = request.tool_call["name"]
        if tool_name == "submit_background_task":
            return await self._handle_submit_background_task(request)
        if tool_name == "get_background_task_status":
            return await self._handle_get_background_task_status(request)
        if tool_name == "wait_background_tasks":
            return await self._handle_wait_background_tasks(request)
        if tool_name == "kill_background_task":
            return await self._handle_kill_background_task(request)
        return await handler(request)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject guidance in sync path using current request state only."""
        modified_request = self._get_modified_request_from_active_count(
            request,
            active_count=len(cast("dict[str, Any]", request.state or {}).get(
                "active_background_task_ids",
                [],
            )),
        )
        return handler(modified_request or request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject guidance when background tasks are still running (async)."""
        modified_request = self._get_modified_request(request)
        return await handler(modified_request or request)

    async def _handle_submit_background_task(
        self,
        request: ToolCallRequest,
    ) -> ToolMessage:
        """Handle `submit_background_task` tool call."""
        args = request.tool_call.get("args", {})
        if not isinstance(args, dict):
            return ToolMessage(
                content="Invalid submit_background_task args.",
                name="submit_background_task",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

        parse_error = self._validate_submit_args(args)
        if parse_error:
            return ToolMessage(
                content=parse_error,
                name="submit_background_task",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

        thread_id = self._get_thread_id(request.runtime.config)
        if thread_id is None:
            return ToolMessage(
                content="Could not resolve thread_id for background task submission.",
                name="submit_background_task",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

        task_id = self._taskiq_runtime.create_task_id()
        job_kind = cast("str", args["job_kind"])
        payload = cast("dict[str, Any]", args["input"])
        payload_preview = self._build_payload_preview(job_kind, payload)
        now = time.time()

        task_meta: BackgroundTaskMeta = {
            "task_id": task_id,
            "job_kind": cast("Any", job_kind),
            "status": "running",
            "created_ts": now,
            "updated_ts": now,
            "payload_preview": payload_preview,
        }
        await self._taskiq_runtime.register_task(thread_id, task_id, task_meta)

        active_ids, _ = await self._taskiq_runtime.list_tasks(thread_id)
        await self._taskiq_runtime.update_state(
            thread_id,
            {
                "taskiq_mode": self._taskiq_mode,
                "active_background_task_ids": active_ids,
            },
        )

        handle = await self._taskiq_runtime.submit(
            task_id,
            lambda: self._run_background_job(
                parent_thread_id=thread_id,
                task_id=task_id,
                job_kind=job_kind,
                payload=payload,
                payload_preview=payload_preview,
            ),
        )
        await self._taskiq_runtime.attach_task_handle(task_id, handle)

        return ToolMessage(
            content=f"Background task submitted: {task_id} ({job_kind})",
            name="submit_background_task",
            tool_call_id=request.tool_call.get("id", ""),
            status="success",
        )

    async def _handle_get_background_task_status(
        self,
        request: ToolCallRequest,
    ) -> ToolMessage:
        """Handle `get_background_task_status` tool call."""
        thread_id = self._get_thread_id(request.runtime.config)
        if thread_id is None:
            return ToolMessage(
                content="Could not resolve thread_id for status lookup.",
                name="get_background_task_status",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

        args = request.tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        task_id = args.get("task_id")
        include_queue = args.get("include_queue", True)
        if task_id is not None and not isinstance(task_id, str):
            return ToolMessage(
                content="task_id must be string or null.",
                name="get_background_task_status",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
        if not isinstance(include_queue, bool):
            include_queue = True

        state = await self._taskiq_runtime.get_state_values(thread_id)
        active_ids, meta = await self._taskiq_runtime.list_tasks(
            thread_id, task_id=task_id
        )
        queue = list(state.get("background_hitl_queue", []))
        if task_id:
            filtered_queue = [
                item
                for item in queue
                if isinstance(item, dict) and item.get("task_id") == task_id
            ]
        else:
            filtered_queue = queue

        payload: dict[str, Any] = {
            "taskiq_mode": self._taskiq_mode,
            "active_background_task_ids": active_ids,
            "background_task_meta": meta,
        }
        if include_queue:
            payload["background_hitl_queue"] = filtered_queue

        return ToolMessage(
            content=str(payload),
            name="get_background_task_status",
            tool_call_id=request.tool_call.get("id", ""),
            status="success",
        )

    async def _handle_wait_background_tasks(
        self,
        request: ToolCallRequest,
    ) -> ToolMessage:
        """Handle `wait_background_tasks` tool call."""
        thread_id = self._get_thread_id(request.runtime.config)
        if thread_id is None:
            return ToolMessage(
                content="Could not resolve thread_id for wait operation.",
                name="wait_background_tasks",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

        args = request.tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        raw_seconds = args.get("seconds", DEFAULT_BACKGROUND_WAIT_SECONDS)
        seconds = (
            float(raw_seconds)
            if isinstance(raw_seconds, int | float)
            else DEFAULT_BACKGROUND_WAIT_SECONDS
        )
        seconds = min(MAX_BACKGROUND_WAIT_SECONDS, max(0.0, seconds))
        await asyncio.sleep(seconds)

        state = await self._taskiq_runtime.get_state_values(thread_id)
        active_ids, meta = await self._taskiq_runtime.list_tasks(thread_id)
        payload = {
            "waited_seconds": seconds,
            "active_background_task_ids": active_ids,
            "background_task_meta": meta,
            "background_hitl_queue_length": len(
                list(state.get("background_hitl_queue", []))
            ),
        }
        return ToolMessage(
            content=str(payload),
            name="wait_background_tasks",
            tool_call_id=request.tool_call.get("id", ""),
            status="success",
        )

    async def _handle_kill_background_task(
        self,
        request: ToolCallRequest,
    ) -> ToolMessage:
        """Handle `kill_background_task` tool call."""
        thread_id = self._get_thread_id(request.runtime.config)
        if thread_id is None:
            return ToolMessage(
                content="Could not resolve thread_id for kill operation.",
                name="kill_background_task",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
        args = request.tool_call.get("args", {})
        if not isinstance(args, dict):
            return ToolMessage(
                content="Invalid kill_background_task args.",
                name="kill_background_task",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
        task_id = args.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            return ToolMessage(
                content="task_id must be a non-empty string.",
                name="kill_background_task",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

        owner_thread_id = await self._taskiq_runtime.get_task_thread_id(task_id)
        if owner_thread_id != thread_id:
            return ToolMessage(
                content=f"Unknown background task: {task_id}",
                name="kill_background_task",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )

        canceled = await self._taskiq_runtime.cancel_task(task_id)
        state = await self._taskiq_runtime.get_state_values(thread_id)
        queue = [
            item
            for item in list(state.get("background_hitl_queue", []))
            if item.get("task_id") != task_id
        ]
        resumes = dict(state.get("background_hitl_resumes", {}))
        resumes.pop(task_id, None)
        now = time.time()
        _ = await self._taskiq_runtime.mark_task_terminal(
            task_id,
            status="failure",
            payload={
                "updated_ts": now,
                "error": "Background task killed by user.",
                "result_preview": "no_result",
                "full_result": "",
                "no_result": True,
            },
        )
        active_ids, _ = await self._taskiq_runtime.list_tasks(thread_id)
        await self._taskiq_runtime.update_state(
            thread_id,
            {
                "active_background_task_ids": active_ids,
                "background_hitl_queue": queue,
                "background_hitl_resumes": resumes,
            },
        )
        self._taskiq_runtime.push_event(
            thread_id,
            BackgroundTaskEvent(
                task_id=task_id,
                status="failure",
                message=f"[SYSTEM] Background task {task_id} was killed by user.",
            ),
        )
        return ToolMessage(
            content=(
                f"Kill requested for {task_id}."
                + (" Running task canceled." if canceled else " Task was not running.")
            ),
            name="kill_background_task",
            tool_call_id=request.tool_call.get("id", ""),
            status="success",
        )

    async def _run_background_job(
        self,
        *,
        parent_thread_id: str,
        task_id: str,
        job_kind: str,
        payload: dict[str, Any],
        payload_preview: str,
    ) -> None:
        """Execute a single background job."""
        try:
            if job_kind != "shell_command":
                result = await self._run_unsupported_job_kind(job_kind)
            else:
                result = await self._run_shell_command_job(
                    parent_thread_id=parent_thread_id,
                    task_id=task_id,
                    payload=payload,
                    payload_preview=payload_preview,
                )

            await self._mark_task_success(
                parent_thread_id=parent_thread_id,
                task_id=task_id,
                full_result=result,
                no_result=False,
                diag_text=None,
            )
        except Exception as exc:  # Fail-closed background tasks
            logger.exception("Background task %s failed", task_id)
            await self._mark_task_failure(
                parent_thread_id=parent_thread_id,
                task_id=task_id,
                error_text=str(exc),
            )

    @staticmethod
    def _build_background_guidance(active_count: int) -> str:
        """Build user-message guidance when background tasks are active."""
        noun = "task" if active_count == 1 else "tasks"
        return (
            f"{BACKGROUND_GUIDANCE_PREFIX} {active_count} background {noun} still "
            "running. Prefer waiting for background completion first. "
            "Only if it has been waiting for a long time and appears stuck, "
            "inspect with get_background_task_status and consider "
            "kill_background_task(task_id)."
        )

    @staticmethod
    def _message_text(message: Any) -> str:  # noqa: ANN401
        """Extract text content from request messages."""
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        return ""

    @staticmethod
    def _get_modified_request(request: ModelRequest) -> ModelRequest | None:
        """Append background guidance as a user message when tasks are active."""
        state = cast("dict[str, Any]", request.state or {})
        active_ids = state.get("active_background_task_ids", [])
        if not isinstance(active_ids, list) or not active_ids:
            return None
        return BackgroundTasksMiddleware._get_modified_request_from_active_count(
            request,
            active_count=len(active_ids),
        )

    @staticmethod
    def _get_modified_request_from_active_count(
        request: ModelRequest,
        *,
        active_count: int,
    ) -> ModelRequest | None:
        """Append background guidance as a user message when tasks are active."""
        if active_count <= 0:
            return None
        guidance = BackgroundTasksMiddleware._build_background_guidance(active_count)
        if len(request.messages) > GUIDANCE_DEDUP_TAIL_MESSAGES:
            tail_messages = request.messages[-GUIDANCE_DEDUP_TAIL_MESSAGES:]
        else:
            tail_messages = request.messages
        if any(
            BackgroundTasksMiddleware._message_text(msg) == guidance
            for msg in tail_messages
        ):
            return None
        return request.override(
            messages=[*request.messages, HumanMessage(content=guidance)]
        )

    async def _run_shell_command_job(
        self,
        *,
        parent_thread_id: str,
        task_id: str,
        payload: dict[str, Any],
        payload_preview: str,
    ) -> str:
        """Run validated shell command in background."""
        command = payload.get("command")
        if not isinstance(command, str) or not command.strip():
            msg = "shell_command input.command must be a non-empty string."
            raise ValueError(msg)
        if contains_dangerous_patterns(command):
            msg = "Rejected shell command: dangerous shell pattern detected."
            raise ValueError(msg)
        if not is_shell_command_allowed(command, settings.shell_allow_list):
            interrupt_id = f"bg-shell-{uuid.uuid4().hex[:10]}"
            deadline = time.time() + BACKGROUND_HITL_TIMEOUT_SECONDS
            await self._enqueue_background_interrupt(
                parent_thread_id=parent_thread_id,
                task_id=task_id,
                interrupt_id=interrupt_id,
                action_requests=[
                    {
                        "name": "execute",
                        "args": {"command": command},
                        "allowed_decisions": ["approve", "reject"],
                        "description": (
                            "Shell command is not in allow-list and requires "
                            "approval."
                        ),
                    }
                ],
                payload_preview=payload_preview,
                deadline_ts=deadline,
            )
            decisions = await self._wait_for_resume_or_timeout(
                parent_thread_id=parent_thread_id,
                task_id=task_id,
                interrupt_id=interrupt_id,
                deadline_ts=deadline,
            )
            if decisions is None:
                msg = (
                    "Background HITL approval timed out and was "
                    "automatically rejected."
                )
                raise TimeoutError(msg)
            if not self._is_approved(decisions):
                msg = "Rejected shell command: approval was denied."
                raise PermissionError(msg)

        timeout = payload.get("timeout")
        timeout_s = (
            int(timeout)
            if isinstance(timeout, int) and timeout > 0
            else DEFAULT_BACKGROUND_SHELL_TIMEOUT_SECONDS
        )
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            msg = f"shell_command timed out after {timeout_s}s"
            raise TimeoutError(msg) from exc

        stdout = (stdout_bytes or b"").decode(errors="replace").strip()
        stderr = (stderr_bytes or b"").decode(errors="replace").strip()
        if proc.returncode not in {0, None}:
            msg = (
                f"shell_command failed (exit={proc.returncode}). "
                f"stderr: {stderr or '(empty)'}"
            )
            raise subprocess.CalledProcessError(proc.returncode, command, stderr=stderr)

        return stdout or stderr or "(no output)"

    @staticmethod
    def _is_approved(decisions: list[dict[str, Any]]) -> bool:
        """Return `True` when all decisions are approve."""
        if not decisions:
            return False
        for decision in decisions:
            if not isinstance(decision, dict):
                return False
            if decision.get("type") != "approve":
                return False
        return True

    async def _enqueue_background_interrupt(
        self,
        *,
        parent_thread_id: str,
        task_id: str,
        interrupt_id: str,
        action_requests: list[dict[str, Any]],
        payload_preview: str,
        deadline_ts: float,
    ) -> None:
        """Append a pending background HITL item into checkpointed state."""
        state = await self._taskiq_runtime.get_state_values(parent_thread_id)
        queue = list(state.get("background_hitl_queue", []))
        task_meta = await self._taskiq_runtime.get_task_meta(task_id)
        now = time.time()
        existing_kind = (
            task_meta.get("job_kind", "shell_command")
            if isinstance(task_meta, dict)
            else "shell_command"
        )
        queue.append(
            {
                "interrupt_id": interrupt_id,
                "task_id": task_id,
                "job_kind": cast("Any", existing_kind),
                "action_requests": action_requests,
                "deadline_ts": deadline_ts,
                "payload_preview": payload_preview,
            }
        )
        await self._taskiq_runtime.update_task_meta(
            task_id,
            {
                "status": "waiting_hitl",
                "updated_ts": now,
            },
        )
        await self._taskiq_runtime.update_state(
            parent_thread_id,
            {
                "background_hitl_queue": queue,
            },
        )

    async def _wait_for_resume_or_timeout(
        self,
        *,
        parent_thread_id: str,
        task_id: str,
        interrupt_id: str,
        deadline_ts: float,
    ) -> list[dict[str, Any]] | None:
        """Wait for main-thread approval decision or timeout."""
        while time.time() < deadline_ts:
            state = await self._taskiq_runtime.get_state_values(parent_thread_id)
            resumes = dict(state.get("background_hitl_resumes", {}))
            decision = resumes.get(task_id)
            if decision and decision.get("interrupt_id") == interrupt_id:
                await self._consume_resume_and_queue_item(
                    parent_thread_id=parent_thread_id,
                    task_id=task_id,
                    interrupt_id=interrupt_id,
                    resumes=resumes,
                    queue=list(state.get("background_hitl_queue", [])),
                )
                decisions = decision.get("decisions", [])
                if isinstance(decisions, list):
                    return [d for d in decisions if isinstance(d, dict)]
                return []
            await asyncio.sleep(BACKGROUND_HITL_POLL_SECONDS)

        await self._auto_reject_timeout(
            parent_thread_id=parent_thread_id,
            task_id=task_id,
            interrupt_id=interrupt_id,
        )
        return None

    async def _consume_resume_and_queue_item(
        self,
        *,
        parent_thread_id: str,
        task_id: str,
        interrupt_id: str,
        resumes: dict[str, BackgroundResumeDecision],
        queue: list[PendingBackgroundInterrupt],
    ) -> None:
        """Consume approved background HITL item and mark task as running again."""
        resumes.pop(task_id, None)
        remaining_queue = [
            item
            for item in queue
            if not (
                item.get("task_id") == task_id
                and item.get("interrupt_id") == interrupt_id
            )
        ]
        await self._taskiq_runtime.update_task_meta(
            task_id,
            {
                "status": "running",
                "updated_ts": time.time(),
            },
        )
        await self._taskiq_runtime.update_state(
            parent_thread_id,
            {
                "background_hitl_resumes": resumes,
                "background_hitl_queue": remaining_queue,
            },
        )

    async def _auto_reject_timeout(
        self,
        *,
        parent_thread_id: str,
        task_id: str,
        interrupt_id: str,
    ) -> None:
        """Write timeout auto-reject signal and emit user event."""
        state = await self._taskiq_runtime.get_state_values(parent_thread_id)
        queue = list(state.get("background_hitl_queue", []))
        queue = [
            item
            for item in queue
            if not (
                item.get("task_id") == task_id
                and item.get("interrupt_id") == interrupt_id
            )
        ]
        resumes = dict(state.get("background_hitl_resumes", {}))
        resumes[task_id] = {
            "interrupt_id": interrupt_id,
            "decisions": [{"type": "reject"}],
        }
        now = time.time()
        await self._taskiq_runtime.update_task_meta(
            task_id,
            {
                "status": "failure",
                "updated_ts": now,
                "error": "Background HITL approval timed out.",
            },
        )

        await self._taskiq_runtime.update_state(
            parent_thread_id,
            {
                "background_hitl_queue": queue,
                "background_hitl_resumes": resumes,
            },
        )
        self._taskiq_runtime.push_event(
            parent_thread_id,
            BackgroundTaskEvent(
                task_id=task_id,
                status="failure",
                message=(
                    f"[SYSTEM] Background task {task_id} timed out waiting for "
                    "approval and was automatically rejected."
                ),
            ),
        )

    async def _mark_task_success(
        self,
        *,
        parent_thread_id: str,
        task_id: str,
        full_result: str,
        no_result: bool,
        diag_text: str | None = None,
    ) -> None:
        """Mark background task as successful and emit event."""
        preview = "no_result" if no_result else self._truncate(full_result)
        now = time.time()
        _ = await self._taskiq_runtime.mark_task_terminal(
            task_id,
            status="success",
            payload={
                "updated_ts": now,
                "result_preview": preview,
                "full_result": full_result,
                "no_result": no_result,
            },
        )
        active_ids, _ = await self._taskiq_runtime.list_tasks(parent_thread_id)
        await self._taskiq_runtime.update_state(
            parent_thread_id,
            {"active_background_task_ids": active_ids},
        )
        self._taskiq_runtime.push_event(
            parent_thread_id,
            BackgroundTaskEvent(
                task_id=task_id,
                status="success",
                message=(
                    f"[SYSTEM] Background task {task_id} completed.\n"
                    f"Result: {preview}"
                    + (
                        f"\nDiag: {diag_text}"
                        if no_result and isinstance(diag_text, str) and diag_text
                        else ""
                    )
                ),
            ),
        )

    async def _mark_task_failure(
        self,
        *,
        parent_thread_id: str,
        task_id: str,
        error_text: str,
    ) -> None:
        """Mark background task as failed and emit event."""
        state = await self._taskiq_runtime.get_state_values(parent_thread_id)
        queue = [
            item
            for item in list(state.get("background_hitl_queue", []))
            if item.get("task_id") != task_id
        ]
        resumes = dict(state.get("background_hitl_resumes", {}))
        resumes.pop(task_id, None)
        now = time.time()
        _ = await self._taskiq_runtime.mark_task_terminal(
            task_id,
            status="failure",
            payload={
                "updated_ts": now,
                "error": self._truncate(error_text),
                "result_preview": "no_result",
                "full_result": "",
                "no_result": True,
            },
        )
        active_ids, _ = await self._taskiq_runtime.list_tasks(parent_thread_id)
        await self._taskiq_runtime.update_state(
            parent_thread_id,
            {
                "active_background_task_ids": active_ids,
                "background_hitl_queue": queue,
                "background_hitl_resumes": resumes,
            },
        )
        self._taskiq_runtime.push_event(
            parent_thread_id,
            BackgroundTaskEvent(
                task_id=task_id,
                status="failure",
                message=(
                    f"[SYSTEM] Background task {task_id} failed: "
                    f"{self._truncate(error_text)}"
                ),
            ),
        )

    @staticmethod
    def _get_thread_id(config: RunnableConfig | dict[str, Any]) -> str | None:
        """Extract thread ID from runnable config."""
        configurable = config.get("configurable", {})
        if not isinstance(configurable, dict):
            return None
        thread_id = configurable.get("thread_id")
        return thread_id if isinstance(thread_id, str) and thread_id else None

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate long status/result text for UI display."""
        if len(text) <= MAX_PREVIEW_LENGTH:
            return text
        return f"{text[:MAX_PREVIEW_LENGTH]}..."

    def _build_payload_preview(self, _job_kind: str, payload: dict[str, Any]) -> str:
        """Build short payload preview for state/UI."""
        command = payload.get("command")
        if isinstance(command, str):
            return self._truncate(f"command={command}")
        return "command=<invalid>"

    @staticmethod
    def _validate_submit_args(args: dict[str, Any]) -> str | None:
        """Validate submit tool payload."""
        job_kind = args.get("job_kind")
        payload = args.get("input")
        if job_kind != "shell_command":
            return "job_kind must be 'shell_command'."
        if not isinstance(payload, dict):
            return "input must be an object."

        if not isinstance(payload.get("command"), str):
            return "shell_command requires input.command (str)."
        timeout = payload.get("timeout")
        if timeout is not None and not isinstance(timeout, int):
            return "shell_command input.timeout must be int or null."
        return None

    @staticmethod
    async def _run_unsupported_job_kind(job_kind: str) -> str:
        """Raise a deterministic error for unsupported job kinds."""
        msg = f"Unsupported background job kind: {job_kind}"
        raise ValueError(msg)


__all__ = [
    "BACKGROUND_HITL_TIMEOUT_SECONDS",
    "BackgroundTasksAgentState",
    "BackgroundTasksMiddleware",
]
