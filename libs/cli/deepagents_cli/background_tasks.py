"""Background task runtime and middleware helpers for the CLI."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, ModelResponse
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

from deepagents_cli.config import is_shell_command_allowed, settings

TaskiqMode = Literal["inmemory"]

BACKGROUND_HITL_TIMEOUT_SECONDS = 10 * 60


class BackgroundTaskMeta(TypedDict):
    """Checkpointed metadata for a submitted background task."""

    task_id: str
    job_kind: Literal["agent_subtask", "shell_command"]
    payload: dict[str, Any]
    status: Literal["running", "waiting_hitl", "completed", "failed"]
    error: NotRequired[str]
    result: NotRequired[str]


class PendingBackgroundInterrupt(TypedDict):
    """A queued HITL request raised by a background task."""

    interrupt_id: str
    task_id: str
    action_requests: list[dict[str, Any]]
    deadline: float


class BackgroundResumeDecision(TypedDict):
    """Resume payload written by the foreground approval flow."""

    interrupt_id: str
    decisions: list[dict[str, Any]]


class BackgroundTasksState(AgentState):
    """State schema for background task orchestration."""

    taskiq_mode: NotRequired[Literal["inmemory"]]
    active_background_task_ids: NotRequired[list[str]]
    background_task_meta: NotRequired[dict[str, BackgroundTaskMeta]]
    background_hitl_queue: NotRequired[list[PendingBackgroundInterrupt]]
    background_hitl_resumes: NotRequired[dict[str, BackgroundResumeDecision]]


class BackgroundTaskUpdate(TypedDict):
    """In-memory runtime event consumed by middleware."""

    task_id: str
    status: Literal["completed", "failed"]
    result: NotRequired[str]
    error: NotRequired[str]


@dataclass
class TaskiqRuntime:
    """Minimal in-memory background task runtime for CLI sessions."""

    mode: TaskiqMode = "inmemory"
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    updates: asyncio.Queue[BackgroundTaskUpdate] = field(default_factory=asyncio.Queue)

    def register(self, coro: Any) -> None:
        """Schedule a coroutine and track it for cleanup."""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def publish_update(self, event: BackgroundTaskUpdate) -> None:
        """Publish a background task status transition."""
        self.updates.put_nowait(event)

    def drain_updates(self) -> list[BackgroundTaskUpdate]:
        """Drain all currently available updates without waiting."""
        events: list[BackgroundTaskUpdate] = []
        while True:
            try:
                events.append(self.updates.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events


async def create_taskiq_runtime(mode: TaskiqMode = "inmemory") -> TaskiqRuntime:
    """Create a background task runtime instance."""
    return TaskiqRuntime(mode=mode)


async def close_taskiq_runtime(runtime: TaskiqRuntime | None) -> None:
    """Cancel and drain all runtime tasks."""
    if runtime is None:
        return
    for task in list(runtime.tasks):
        task.cancel()
    if runtime.tasks:
        await asyncio.gather(*runtime.tasks, return_exceptions=True)


async def _run_shell_command(task_id: str, payload: dict[str, Any], runtime: TaskiqRuntime) -> None:
    """Execute a shell command and publish terminal task update."""
    command = str(payload.get("command", "")).strip()
    timeout_raw = payload.get("timeout")
    timeout = int(timeout_raw) if timeout_raw is not None else None

    if not command:
        runtime.publish_update(
            {
                "task_id": task_id,
                "status": "failed",
                "error": "Background shell command is empty",
            }
        )
        return

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if timeout is None:
            stdout_bytes, stderr_bytes = await process.communicate()
        else:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()  # type: ignore[used-before-def]
        runtime.publish_update(
            {
                "task_id": task_id,
                "status": "failed",
                "error": f"Shell command timed out after {timeout} seconds",
            }
        )
        return
    except OSError as exc:
        msg = f"Background shell command failed to start: {exc!s}"
        runtime.publish_update({"task_id": task_id, "status": "failed", "error": msg})
        return

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    if process.returncode == 0:
        output = stdout or "(no output)"
        runtime.publish_update(
            {
                "task_id": task_id,
                "status": "completed",
                "result": output,
            }
        )
        return

    error_text = stderr or stdout or f"Command exited with code {process.returncode}"
    runtime.publish_update(
        {
            "task_id": task_id,
            "status": "failed",
            "error": error_text,
        }
    )


def build_submit_background_task_tool(
    runtime: TaskiqRuntime,
) -> Any:
    """Create `submit_background_task` tool bound to the current runtime."""

    def submit_background_task(  # noqa: PLR0913
        job_kind: Literal["agent_subtask", "shell_command"],
        payload_input: dict[str, Any],
        tool_call_id: Annotated[str, "InjectedToolCallId"],
    ) -> Command:
        task_id = f"bg-{uuid.uuid4().hex[:10]}"
        payload = dict(payload_input)

        meta: BackgroundTaskMeta = {
            "task_id": task_id,
            "job_kind": job_kind,
            "payload": payload,
            "status": "running",
        }

        if job_kind == "shell_command":
            command = str(payload.get("command", ""))
            if not is_shell_command_allowed(command, settings.shell_allow_list):
                meta["status"] = "failed"
                meta["error"] = (
                    "Shell command rejected by allow-list/"
                    "dangerous-pattern policy"
                )
                return Command(
                    update={
                        "background_task_meta": {task_id: meta},
                        "messages": [
                            ToolMessage(
                                content=(
                                    "Background task submission failed: shell command "
                                    "violates allow-list or dangerous-pattern checks."
                                ),
                                tool_call_id=tool_call_id,
                            )
                        ],
                    }
                )

            runtime.register(_run_shell_command(task_id, payload, runtime))
        else:
            meta["status"] = "failed"
            meta["error"] = (
                "Background agent_subtask execution is not implemented yet in "
                "in-memory mode"
            )
            return Command(
                update={
                    "background_task_meta": {task_id: meta},
                    "messages": [
                        ToolMessage(
                            content=(
                                "Background agent_subtask is not yet executable in "
                                "this build and was rejected fail-closed."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        return Command(
            update={
                "taskiq_mode": runtime.mode,
                "active_background_task_ids": [task_id],
                "background_task_meta": {task_id: meta},
                "messages": [
                    ToolMessage(
                        content=f"Submitted background task `{task_id}` ({job_kind}).",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return submit_background_task


class BackgroundTasksMiddleware(AgentMiddleware[BackgroundTasksState, Any, Any]):
    """Middleware that enforces timeout cleanup and publishes task updates."""

    state_schema = BackgroundTasksState

    def __init__(self, runtime: TaskiqRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _expire_waiting_interrupts(
        state: BackgroundTasksState,
    ) -> dict[str, Any] | None:
        queue = list(state.get("background_hitl_queue", []))
        if not queue:
            return None

        now = time.time()
        remaining: list[PendingBackgroundInterrupt] = []
        meta = dict(state.get("background_task_meta", {}))
        expired_task_ids: list[str] = []
        for item in queue:
            if now > item["deadline"]:
                task_id = item["task_id"]
                task_meta = meta.get(task_id)
                if task_meta:
                    task_meta["status"] = "failed"
                    task_meta["error"] = (
                        "Background HITL timed out and was "
                        "auto-rejected"
                    )
                    meta[task_id] = task_meta
                expired_task_ids.append(task_id)
                continue
            remaining.append(item)

        if not expired_task_ids:
            return None

        return {
            "background_hitl_queue": remaining,
            "background_task_meta": meta,
            "messages": [
                SystemMessage(
                    content=(
                        "Background approvals timed out and were automatically rejected "
                        f"for tasks: {', '.join(expired_task_ids)}"
                    )
                )
            ],
        }

    def _apply_runtime_updates(self, state: BackgroundTasksState) -> dict[str, Any] | None:
        events = self.runtime.drain_updates()
        if not events:
            return None

        meta = dict(state.get("background_task_meta", {}))
        active = list(state.get("active_background_task_ids", []))
        messages: list[SystemMessage] = []

        for event in events:
            task_id = event["task_id"]
            task_meta = meta.get(task_id)
            if not task_meta:
                continue

            task_meta["status"] = event["status"]
            if "result" in event:
                task_meta["result"] = event["result"]
            if "error" in event:
                task_meta["error"] = event["error"]
            meta[task_id] = task_meta
            if task_id in active:
                active.remove(task_id)

            if event["status"] == "completed":
                result_preview = event.get("result", "")[:400]
                messages.append(
                    SystemMessage(
                        content=(
                            f"Background task {task_id} completed successfully.\n"
                            f"Result:\n{result_preview}"
                        )
                    )
                )
            else:
                messages.append(
                    SystemMessage(
                        content=(
                            f"Background task {task_id} failed.\n"
                            f"Error: {event.get('error', 'unknown error')}"
                        )
                    )
                )

        update: dict[str, Any] = {
            "background_task_meta": meta,
            "active_background_task_ids": active,
        }
        if messages:
            update["messages"] = messages
        return update

    def _collect_updates(self, state: BackgroundTasksState) -> dict[str, Any] | None:
        timeout_update = self._expire_waiting_interrupts(state)
        runtime_update = self._apply_runtime_updates(state)
        if timeout_update is None:
            return runtime_update
        if runtime_update is None:
            return timeout_update

        merged = dict(timeout_update)
        merged.update({
            "background_task_meta": runtime_update.get(
                "background_task_meta", timeout_update.get("background_task_meta", {})
            ),
            "active_background_task_ids": runtime_update.get(
                "active_background_task_ids",
                state.get("active_background_task_ids", []),
            ),
        })
        timeout_messages = timeout_update.get("messages", [])
        runtime_messages = runtime_update.get("messages", [])
        if timeout_messages or runtime_messages:
            merged["messages"] = [*timeout_messages, *runtime_messages]
        return merged

    def before_model(
        self,
        state: BackgroundTasksState,
        runtime: Any,  # noqa: ANN401, ARG002
    ) -> dict[str, Any] | None:
        return self._collect_updates(state)

    async def abefore_model(
        self,
        state: BackgroundTasksState,
        runtime: Any,  # noqa: ANN401, ARG002
    ) -> dict[str, Any] | None:
        return self._collect_updates(state)

    def after_model(
        self,
        state: BackgroundTasksState,
        runtime: Any,  # noqa: ANN401, ARG002
        response: ModelResponse,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return None

    async def aafter_model(
        self,
        state: BackgroundTasksState,
        runtime: Any,  # noqa: ANN401, ARG002
        response: ModelResponse,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        return None


def clear_inmemory_background_state(
    is_resumed: bool, taskiq_mode: TaskiqMode
) -> dict[str, Any] | None:
    """Return state reset updates for resumed in-memory sessions."""
    if not is_resumed or taskiq_mode != "inmemory":
        return None
    return {
        "active_background_task_ids": [],
        "background_hitl_queue": [],
        "background_hitl_resumes": {},
        "messages": [
            SystemMessage(
                content=(
                    "In-memory background tasks and pending approvals cannot be resumed; "
                    "stale runtime state was cleared."
                )
            )
        ],
    }
