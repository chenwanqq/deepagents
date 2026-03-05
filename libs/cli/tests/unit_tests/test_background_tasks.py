"""Unit tests for background task runtime."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from deepagents_cli.background_tasks import (
    BACKGROUND_TASK_HISTORY_LIMIT,
    BackgroundTaskEvent,
    close_taskiq_runtime,
    create_taskiq_runtime,
)


class _FakeAgent:
    """Minimal async agent stub for state read/write tests."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        thread_id = config["configurable"]["thread_id"]
        return SimpleNamespace(values=self._state.get(thread_id, {}))

    async def aupdate_state(
        self,
        config: dict[str, Any],
        update: dict[str, Any],
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        merged = dict(self._state.get(thread_id, {}))
        merged.update(update)
        self._state[thread_id] = merged


async def test_taskiq_runtime_submit_executes_job() -> None:
    """Submitted jobs run via the taskiq in-memory broker."""
    runtime = create_taskiq_runtime()
    await runtime.startup()

    marker = {"value": 0}

    async def _job() -> int:
        await asyncio.sleep(0)
        marker["value"] = 1
        return 1

    task_id = runtime.create_task_id()
    await runtime.register_task(
        "t-1",
        task_id,
        {
            "task_id": task_id,
            "job_kind": "shell_command",
            "status": "running",
            "created_ts": 1.0,
            "updated_ts": 1.0,
            "payload_preview": "x",
        },
    )
    handle = await runtime.submit(task_id, lambda: _job())
    result = await handle.wait_result(timeout=2)
    assert marker["value"] == 1
    assert result.return_value == 1

    await close_taskiq_runtime(runtime)


def test_taskiq_runtime_event_queue() -> None:
    """Runtime should buffer and drain events per thread."""
    runtime = create_taskiq_runtime()
    runtime.push_event(
        "thread-1",
        BackgroundTaskEvent(
            task_id="bg-1",
            status="success",
            message="done",
        ),
    )
    drained = runtime.pop_events("thread-1")
    assert len(drained) == 1
    assert drained[0].task_id == "bg-1"
    assert runtime.pop_events("thread-1") == []


async def test_taskiq_runtime_state_helpers() -> None:
    """State helper methods should delegate through bound agent."""
    runtime = create_taskiq_runtime()
    fake_agent = _FakeAgent()
    runtime.bind_agent(fake_agent)

    await runtime.update_state("t-1", {"k": "v"})
    values = await runtime.get_state_values("t-1")
    assert values["k"] == "v"


async def test_runtime_register_concurrent_no_lost_updates() -> None:
    """Concurrent registration should keep complete index and metadata."""
    runtime = create_taskiq_runtime()

    async def _register(i: int) -> None:
        task_id = f"bg-{i}"
        await runtime.register_task(
            "thread-many",
            task_id,
            {
                "task_id": task_id,
                "job_kind": "shell_command",
                "status": "running",
                "created_ts": float(i),
                "updated_ts": float(i),
                "payload_preview": f"command=echo {i}",
            },
        )

    await asyncio.gather(*[_register(i) for i in range(20)])
    active, meta = await runtime.list_tasks("thread-many")
    assert len(meta) == 20
    assert len(active) == 20


async def test_runtime_mark_terminal_and_prune() -> None:
    """Terminal transition should preserve only latest N completed tasks."""
    runtime = create_taskiq_runtime()
    thread_id = "thread-prune"
    total = BACKGROUND_TASK_HISTORY_LIMIT + 5

    for i in range(total):
        task_id = f"bg-{i}"
        await runtime.register_task(
            thread_id,
            task_id,
            {
                "task_id": task_id,
                "job_kind": "shell_command",
                "status": "running",
                "created_ts": float(i),
                "updated_ts": float(i),
                "payload_preview": "x",
            },
        )
        await runtime.mark_task_terminal(
            task_id,
            status="success",
            payload={
                "updated_ts": float(i) + 0.1,
                "result_preview": "ok",
                "full_result": "ok",
                "no_result": False,
            },
        )

    _active, meta = await runtime.list_tasks(thread_id)
    assert len(meta) == BACKGROUND_TASK_HISTORY_LIMIT
    assert "bg-0" not in meta
    assert f"bg-{total - 1}" in meta


async def test_runtime_cancel_task_state_transitions() -> None:
    """Cancel should return false for terminal and true for live future."""
    runtime = create_taskiq_runtime()
    await runtime.register_task(
        "thread-cancel",
        "bg-live",
        {
            "task_id": "bg-live",
            "job_kind": "shell_command",
            "status": "running",
            "created_ts": 1.0,
            "updated_ts": 1.0,
            "payload_preview": "x",
        },
    )
    fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    await runtime.attach_task_handle("bg-live", fut)

    canceled = await runtime.cancel_task("bg-live")
    assert canceled is True
    meta = await runtime.get_task_meta("bg-live")
    assert isinstance(meta, dict)
    assert meta["status"] == "failure"

    await runtime.register_task(
        "thread-cancel",
        "bg-done",
        {
            "task_id": "bg-done",
            "job_kind": "shell_command",
            "status": "success",
            "created_ts": 1.0,
            "updated_ts": 1.0,
            "payload_preview": "x",
        },
    )
    not_canceled = await runtime.cancel_task("bg-done")
    assert not_canceled is False
