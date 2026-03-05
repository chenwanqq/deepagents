"""Background task runtime and shared state types for deepagents-cli."""
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from taskiq import AsyncTaskiqTask, InMemoryBroker

logger = logging.getLogger(__name__)

TaskiqMode = Literal["inmemory"]
BackgroundTaskStatus = Literal["running", "waiting_hitl", "success", "failure", "retry"]
BackgroundJobKind = Literal["shell_command"]

BACKGROUND_TASK_HISTORY_LIMIT = 200
_TERMINAL_STATUSES: tuple[BackgroundTaskStatus, ...] = ("success", "failure")
_ACTIVE_STATUSES: tuple[BackgroundTaskStatus, ...] = (
    "running",
    "waiting_hitl",
    "retry",
)


class BackgroundResumeDecision(TypedDict):
    """Decision payload written by the main thread to resume background HITL."""

    interrupt_id: str
    decisions: list[dict[str, Any]]


class PendingBackgroundInterrupt(TypedDict):
    """Pending HITL request emitted by a background task."""

    interrupt_id: str
    task_id: str
    job_kind: BackgroundJobKind
    action_requests: list[dict[str, Any]]
    deadline_ts: float
    payload_preview: str


class BackgroundTaskMeta(TypedDict):
    """Per-task runtime metadata mirrored into checkpointed state."""

    task_id: str
    job_kind: BackgroundJobKind
    status: BackgroundTaskStatus
    created_ts: float
    updated_ts: float
    payload_preview: str
    error: NotRequired[str]
    result_preview: NotRequired[str]
    full_result: NotRequired[str]
    no_result: NotRequired[bool]


class BackgroundTasksState(TypedDict):
    """State fields used by background task middleware."""

    taskiq_mode: NotRequired[TaskiqMode]
    active_background_task_ids: NotRequired[list[str]]
    background_task_meta: NotRequired[dict[str, BackgroundTaskMeta]]
    background_hitl_queue: NotRequired[list[PendingBackgroundInterrupt]]
    background_hitl_resumes: NotRequired[dict[str, BackgroundResumeDecision]]


@dataclass(slots=True)
class BackgroundTaskEvent:
    """User-visible background task event for the Textual poller."""

    task_id: str
    status: BackgroundTaskStatus
    message: str


class TaskiqRuntime:
    """Taskiq-backed runtime for first-version in-memory background tasks."""

    def __init__(self, mode: TaskiqMode = "inmemory") -> None:
        """Initialize runtime internals for in-memory background dispatch."""
        self.mode = mode
        self.broker = InMemoryBroker()
        self._agent: Any | None = None
        self._job_factories: dict[str, Callable[[], Awaitable[Any]]] = {}
        self._running_jobs: dict[str, asyncio.Future[Any]] = {}
        self._thread_events: dict[str, deque[BackgroundTaskEvent]] = defaultdict(deque)

        # Background task truth-source registry.
        self._task_index_by_thread: dict[str, deque[str]] = defaultdict(deque)
        self._task_meta_by_id: dict[str, BackgroundTaskMeta] = {}
        self._task_thread_by_id: dict[str, str] = {}
        self._task_handles_by_id: dict[
            str,
            AsyncTaskiqTask[Any] | asyncio.Future[Any],
        ] = {}
        self._lock = asyncio.Lock()

        self._started = False

        @self.broker.task(task_name="deepagents_cli.background.dispatch")
        async def dispatch_background_task(task_id: str) -> Any:  # noqa: ANN401
            return await self._dispatch_task(task_id)

        self._dispatch_background_task = dispatch_background_task

    def bind_agent(self, agent: Any) -> None:  # noqa: ANN401  # LangGraph type is dynamic
        """Bind the compiled agent graph used for checkpoint state reads/writes."""
        self._agent = agent

    def get_agent(self) -> Any | None:  # noqa: ANN401  # LangGraph type is dynamic
        """Return the bound agent graph, if any."""
        return self._agent

    async def startup(self) -> None:
        """Start the in-memory broker."""
        if self._started:
            return
        await self.broker.startup()
        self._started = True

    async def shutdown(self) -> None:
        """Stop the runtime and cancel in-flight jobs."""
        for job in list(self._running_jobs.values()):
            job.cancel()
        if self._running_jobs:
            await asyncio.gather(*self._running_jobs.values(), return_exceptions=True)
        self._running_jobs.clear()
        self._job_factories.clear()
        async with self._lock:
            self._task_handles_by_id.clear()
        if self._started:
            await self.broker.shutdown()
            self._started = False

    async def submit(
        self,
        task_id: str,
        job_factory: Callable[[], Awaitable[Any]],
    ) -> AsyncTaskiqTask[Any]:
        """Submit a background job by ID and return taskiq handle."""
        self._job_factories[task_id] = job_factory
        handle = await self._dispatch_background_task.kiq(task_id)
        await self.attach_task_handle(task_id, handle)
        return handle

    @staticmethod
    def create_task_id() -> str:
        """Create a short, user-facing task ID."""
        return f"bg-{uuid.uuid4().hex[:10]}"

    async def register_task(
        self,
        thread_id: str,
        task_id: str,
        meta_init: BackgroundTaskMeta,
    ) -> None:
        """Register a task in runtime registry under a thread."""
        async with self._lock:
            self._task_meta_by_id[task_id] = self._clone_meta(meta_init)
            self._task_thread_by_id[task_id] = thread_id
            index = self._task_index_by_thread[thread_id]
            if task_id not in index:
                index.append(task_id)

    async def attach_task_handle(
        self,
        task_id: str,
        handle: AsyncTaskiqTask[Any] | asyncio.Future[Any],
    ) -> None:
        """Attach a task handle for cancellation/readiness checks."""
        async with self._lock:
            self._task_handles_by_id[task_id] = handle

    async def update_task_meta(self, task_id: str, patch: dict[str, Any]) -> bool:
        """Patch task metadata atomically.

        Returns:
            `True` if task exists and was updated, otherwise `False`.
        """
        async with self._lock:
            existing = self._task_meta_by_id.get(task_id)
            if existing is None:
                return False
            updated_any = cast("dict[str, Any]", {**self._clone_meta(existing)})
            updated_any.update(patch)
            self._task_meta_by_id[task_id] = cast("BackgroundTaskMeta", updated_any)
            return True

    async def get_task_meta(self, task_id: str) -> BackgroundTaskMeta | None:
        """Get a copy of task metadata by ID."""
        async with self._lock:
            meta = self._task_meta_by_id.get(task_id)
            return self._clone_meta(meta) if meta else None

    async def get_task_thread_id(self, task_id: str) -> str | None:
        """Get owner thread id for a task."""
        async with self._lock:
            return self._task_thread_by_id.get(task_id)

    async def list_tasks(
        self,
        thread_id: str,
        *,
        task_id: str | None = None,
    ) -> tuple[list[str], dict[str, BackgroundTaskMeta]]:
        """List active task IDs and all visible task metadata for a thread."""
        async with self._lock:
            thread_ids = list(self._task_index_by_thread.get(thread_id, deque()))
            if task_id is not None:
                if self._task_thread_by_id.get(task_id) != thread_id:
                    return [], {}
                candidate_ids = [task_id]
            else:
                candidate_ids = thread_ids

            meta: dict[str, BackgroundTaskMeta] = {}
            active_ids: list[str] = []
            for tid in candidate_ids:
                task_meta = self._task_meta_by_id.get(tid)
                if task_meta is None:
                    continue
                cloned = self._clone_meta(task_meta)
                meta[tid] = cloned
                if cloned.get("status") in _ACTIVE_STATUSES:
                    active_ids.append(tid)
            return active_ids, meta

    async def mark_task_terminal(
        self,
        task_id: str,
        *,
        status: Literal["success", "failure"],
        payload: dict[str, Any],
    ) -> bool:
        """Mark terminal state and apply retention policy.

        Returns:
            `True` when task exists and transition was applied.
        """
        thread_id: str | None = None
        async with self._lock:
            existing = self._task_meta_by_id.get(task_id)
            if existing is None:
                return False
            if existing.get("status") in _TERMINAL_STATUSES:
                return True
            updated_any = cast("dict[str, Any]", {**self._clone_meta(existing)})
            updated_any["status"] = status
            updated_any.update(payload)
            self._task_meta_by_id[task_id] = cast("BackgroundTaskMeta", updated_any)
            thread_id = self._task_thread_by_id.get(task_id)

        if thread_id:
            await self.prune_thread_history(thread_id)
        return True

    async def prune_thread_history(
        self,
        thread_id: str,
        *,
        keep_last: int = BACKGROUND_TASK_HISTORY_LIMIT,
    ) -> None:
        """Prune oldest terminal tasks while keeping active tasks intact."""
        keep_last = max(keep_last, 0)
        async with self._lock:
            index = self._task_index_by_thread.get(thread_id)
            if not index:
                return

            terminal_ids: list[str] = []
            for tid in index:
                meta = self._task_meta_by_id.get(tid)
                status = meta.get("status") if meta else None
                if status in _TERMINAL_STATUSES:
                    terminal_ids.append(tid)
            overflow = len(terminal_ids) - keep_last
            if overflow <= 0:
                return

            removable = set(terminal_ids[:overflow])
            kept = deque([tid for tid in index if tid not in removable])
            self._task_index_by_thread[thread_id] = kept
            for tid in removable:
                self._task_meta_by_id.pop(tid, None)
                self._task_thread_by_id.pop(tid, None)
                self._task_handles_by_id.pop(tid, None)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an in-flight task.

        Returns:
            `True` when a task was running and cancellation was requested.
        """
        async with self._lock:
            handle = self._task_handles_by_id.get(task_id)
            meta = self._task_meta_by_id.get(task_id)
            if isinstance(meta, dict) and meta.get("status") in _TERMINAL_STATUSES:
                return False

        canceled = False
        if isinstance(handle, asyncio.Future):
            if not handle.done():
                handle.cancel()
                canceled = True
        elif isinstance(handle, AsyncTaskiqTask):
            try:
                ready = await handle.is_ready()
                canceled = not ready
            except Exception:  # noqa: BLE001
                canceled = False

        if canceled:
            await self.update_task_meta(task_id, {"status": "failure"})
        return canceled

    def push_event(self, thread_id: str, event: BackgroundTaskEvent) -> None:
        """Append a background event for UI consumption."""
        self._thread_events[thread_id].append(event)

    def pop_events(self, thread_id: str) -> list[BackgroundTaskEvent]:
        """Drain all pending events for a thread."""
        queue = self._thread_events.get(thread_id)
        if not queue:
            return []
        events = list(queue)
        queue.clear()
        return events

    async def get_state_values(self, thread_id: str) -> dict[str, Any]:
        """Read checkpointed state values for a thread."""
        if self._agent is None:
            msg = "TaskiqRuntime agent binding is missing."
            raise RuntimeError(msg)
        config = {"configurable": {"thread_id": thread_id}}
        state = await self._agent.aget_state(config)
        if not state or not state.values:
            return {}
        values = state.values
        return values if isinstance(values, dict) else dict(values)

    async def update_state(self, thread_id: str, update: dict[str, Any]) -> None:
        """Write partial checkpointed state for a thread."""
        if self._agent is None:
            msg = "TaskiqRuntime agent binding is missing."
            raise RuntimeError(msg)
        config = {"configurable": {"thread_id": thread_id}}
        await self._agent.aupdate_state(config, update)

    async def _dispatch_task(self, task_id: str) -> Any:  # noqa: ANN401
        """Execute a previously-registered job factory."""
        factory = self._job_factories.get(task_id)
        if factory is None:
            logger.warning("Background task %s missing job factory", task_id)
            return None
        task = asyncio.ensure_future(factory())
        self._running_jobs[task_id] = task
        await self.attach_task_handle(task_id, task)
        try:
            return await task
        finally:
            self._running_jobs.pop(task_id, None)
            self._job_factories.pop(task_id, None)

    @staticmethod
    def _clone_meta(meta: BackgroundTaskMeta) -> BackgroundTaskMeta:
        """Clone task metadata typed dict."""
        cloned: BackgroundTaskMeta = {
            "task_id": meta["task_id"],
            "job_kind": meta["job_kind"],
            "status": meta["status"],
            "created_ts": meta["created_ts"],
            "updated_ts": meta["updated_ts"],
            "payload_preview": meta["payload_preview"],
        }
        if "error" in meta:
            cloned["error"] = meta["error"]
        if "result_preview" in meta:
            cloned["result_preview"] = meta["result_preview"]
        if "full_result" in meta:
            cloned["full_result"] = meta["full_result"]
        if "no_result" in meta:
            cloned["no_result"] = meta["no_result"]
        return cloned


def create_taskiq_runtime(mode: TaskiqMode = "inmemory") -> TaskiqRuntime:
    """Create a Taskiq runtime for the requested mode."""
    return TaskiqRuntime(mode=mode)


async def close_taskiq_runtime(runtime: TaskiqRuntime | None) -> None:
    """Close runtime resources safely."""
    if runtime is None:
        return
    await runtime.shutdown()


__all__ = [
    "BACKGROUND_TASK_HISTORY_LIMIT",
    "BackgroundJobKind",
    "BackgroundResumeDecision",
    "BackgroundTaskEvent",
    "BackgroundTaskMeta",
    "BackgroundTaskStatus",
    "BackgroundTasksState",
    "PendingBackgroundInterrupt",
    "TaskiqMode",
    "TaskiqRuntime",
    "close_taskiq_runtime",
    "create_taskiq_runtime",
]
