"""Unit tests for background HITL bridge middleware."""

from __future__ import annotations

import ast
import asyncio
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from deepagents_cli.background_middleware import BackgroundTasksMiddleware
from deepagents_cli.background_tasks import create_taskiq_runtime
from deepagents_cli.config import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.agents.middleware.types import (
        ModelRequest,
        ModelResponse,
        ToolCallRequest,
    )


class _FakeModelRequest:
    """Minimal request object for model-call middleware tests."""

    def __init__(self, messages: list[Any], state: dict[str, Any]) -> None:
        self.messages = messages
        self.state = state
        self.runtime = SimpleNamespace(config={"configurable": {"thread_id": "t-1"}})

    def override(self, *, messages: list[Any] | None = None) -> _FakeModelRequest:
        return _FakeModelRequest(messages or self.messages, dict(self.state))


class _FakeAgent:
    """Minimal async agent stub with thread-scoped state."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        thread_id = config["configurable"]["thread_id"]
        return SimpleNamespace(values=self._state.get(thread_id, {}), interrupts=[])

    async def aupdate_state(
        self, config: dict[str, Any], update: dict[str, Any]
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        merged = dict(self._state.get(thread_id, {}))
        merged.update(update)
        self._state[thread_id] = merged


def _build_request(
    args: dict[str, Any],
    *,
    thread_id: str,
    tool_name: str = "submit_background_task",
) -> ToolCallRequest:
    runtime = SimpleNamespace(config={"configurable": {"thread_id": thread_id}})
    return cast(
        "ToolCallRequest",
        SimpleNamespace(
            tool_call={"name": tool_name, "id": "call-1", "args": args},
            runtime=runtime,
            state={},
        ),
    )


async def _noop_handler(_request: ToolCallRequest) -> Any:  # noqa: ANN401
    await asyncio.sleep(0)
    msg = "handler should not be called for submit_background_task"
    raise AssertionError(msg)


def _parse_tool_payload(content: object) -> dict[str, Any]:
    parsed = ast.literal_eval(str(content))
    return parsed if isinstance(parsed, dict) else {}


async def test_submit_shell_background_task_success(monkeypatch) -> None:
    """Submitting a safe shell command should enqueue and complete."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)

    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

    monkeypatch.setattr(settings, "shell_allow_list", ["echo"])
    request = _build_request(
        {
            "job_kind": "shell_command",
            "input": {"command": "echo hello"},
        },
        thread_id="thread-ok",
    )

    response = await middleware.awrap_tool_call(request, _noop_handler)
    assert "Background task submitted" in str(response.content)

    await asyncio.sleep(0.2)
    _active, meta = await runtime.list_tasks("thread-ok")
    task_id = next(iter(meta))
    assert meta[task_id]["status"] == "success"
    assert runtime.pop_events("thread-ok")

    await runtime.shutdown()


async def test_submit_background_task_rejects_non_shell_job_kind() -> None:
    """Any non-shell job kind should be rejected."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)
    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

    response = await middleware.awrap_tool_call(
        _build_request(
            {
                "job_kind": "unsupported_job",
                "input": {"command": "echo hello"},
            },
            thread_id="thread-agent-reject",
        ),
        _noop_handler,
    )
    assert response.status == "error"
    assert "job_kind must be 'shell_command'" in str(response.content)

    await runtime.shutdown()


async def test_submit_shell_background_task_validation_failure(monkeypatch) -> None:
    """Dangerous shell command should fail closed in background worker."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)

    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

    monkeypatch.setattr(settings, "shell_allow_list", ["echo"])
    request = _build_request(
        {
            "job_kind": "shell_command",
            "input": {"command": "echo $HOME"},
        },
        thread_id="thread-fail",
    )
    await middleware.awrap_tool_call(request, _noop_handler)
    await asyncio.sleep(0.2)

    _active, meta = await runtime.list_tasks("thread-fail")
    task_id = next(iter(meta))
    assert meta[task_id]["status"] == "failure"
    assert "dangerous shell pattern" in meta[task_id]["error"]

    await runtime.shutdown()


async def test_submit_shell_background_task_requires_hitl_then_approve(
    monkeypatch,
) -> None:
    """Unallowed shell command should queue HITL and run after approval."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    try:
        agent = _FakeAgent()
        runtime.bind_agent(agent)
        middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

        monkeypatch.setattr(settings, "shell_allow_list", ["echo"])
        thread_id = "thread-shell-hitl"
        request = _build_request(
            {
                "job_kind": "shell_command",
                "input": {"command": "ls"},
            },
            thread_id=thread_id,
        )
        response = await middleware.awrap_tool_call(request, _noop_handler)
        assert "Background task submitted" in str(response.content)

        item: dict[str, Any] | None = None
        task_id = ""
        for _ in range(20):
            state = (
                await agent.aget_state({"configurable": {"thread_id": thread_id}})
            ).values
            queue = state.get("background_hitl_queue", [])
            if isinstance(queue, list) and queue and isinstance(queue[0], dict):
                item = queue[0]
                task_id = str(item["task_id"])
                break
            await asyncio.sleep(0.05)

        assert item is not None
        interrupt_id = str(item["interrupt_id"])
        task_meta = await runtime.get_task_meta(task_id)
        assert isinstance(task_meta, dict)
        assert task_meta["status"] == "waiting_hitl"

        await agent.aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            {
                "background_hitl_resumes": {
                    task_id: {
                        "interrupt_id": interrupt_id,
                        "decisions": [{"type": "approve"}],
                    }
                }
            },
        )

        final_meta: dict[str, Any] = {}
        for _ in range(40):
            meta_map = (await runtime.list_tasks(thread_id))[1]
            maybe_meta = meta_map.get(task_id)
            final_meta = cast("dict[str, Any]", maybe_meta or {})
            if final_meta.get("status") == "success":
                break
            await asyncio.sleep(0.05)
        assert final_meta["status"] == "success"
        assert final_meta["no_result"] is False
    finally:
        await runtime.shutdown()


async def test_wait_for_resume_timeout_auto_reject() -> None:
    """Pending background HITL should auto-reject after deadline."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)
    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

    thread_id = "thread-timeout"
    await runtime.register_task(
        thread_id,
        "bg-1",
        {
            "task_id": "bg-1",
            "job_kind": "shell_command",
            "status": "waiting_hitl",
            "created_ts": time.time() - 2,
            "updated_ts": time.time() - 1,
            "payload_preview": "x",
        },
    )
    await agent.aupdate_state(
        {"configurable": {"thread_id": thread_id}},
        {
            "background_hitl_queue": [
                {
                    "interrupt_id": "int-1",
                    "task_id": "bg-1",
                    "job_kind": "shell_command",
                    "action_requests": [{"name": "execute", "args": {"command": "ls"}}],
                    "deadline_ts": time.time() - 1,
                    "payload_preview": "x",
                }
            ],
            "background_hitl_resumes": {},
        },
    )

    result = await middleware._wait_for_resume_or_timeout(
        parent_thread_id=thread_id,
        task_id="bg-1",
        interrupt_id="int-1",
        deadline_ts=time.time() - 0.1,
    )
    assert result is None
    state = (
        await agent.aget_state({"configurable": {"thread_id": thread_id}})
    ).values
    assert state["background_hitl_queue"] == []
    assert state["background_hitl_resumes"]["bg-1"]["decisions"][0]["type"] == "reject"
    meta = await runtime.get_task_meta("bg-1")
    assert isinstance(meta, dict)
    assert meta["status"] == "failure"

    await runtime.shutdown()


async def test_get_background_task_status_tool() -> None:
    """Status tool should return metadata from runtime and queue from state."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)
    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

    await runtime.register_task(
        "thread-status",
        "bg-100",
        {
            "task_id": "bg-100",
            "job_kind": "shell_command",
            "status": "success",
            "created_ts": 1.0,
            "updated_ts": 2.0,
            "payload_preview": "command=echo hi",
            "result_preview": "hi",
            "full_result": "hi",
            "no_result": False,
        },
    )
    await agent.aupdate_state(
        {"configurable": {"thread_id": "thread-status"}},
        {"background_hitl_queue": []},
    )

    response = await middleware.awrap_tool_call(
        _build_request(
            {"task_id": "bg-100", "include_queue": True},
            thread_id="thread-status",
            tool_name="get_background_task_status",
        ),
        _noop_handler,
    )
    assert response.status == "success"
    payload = _parse_tool_payload(response.content)
    assert payload["active_background_task_ids"] == []
    assert payload["background_task_meta"]["bg-100"]["result_preview"] == "hi"

    await runtime.shutdown()


async def test_wait_background_tasks_tool() -> None:
    """Wait tool should sleep and return current runtime summary."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)
    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

    await runtime.register_task(
        "thread-wait",
        "bg-200",
        {
            "task_id": "bg-200",
            "job_kind": "shell_command",
            "status": "running",
            "created_ts": 1.0,
            "updated_ts": 2.0,
            "payload_preview": "x",
        },
    )
    await agent.aupdate_state(
        {"configurable": {"thread_id": "thread-wait"}},
        {"background_hitl_queue": []},
    )

    response = await middleware.awrap_tool_call(
        _build_request(
            {"seconds": 0.01},
            thread_id="thread-wait",
            tool_name="wait_background_tasks",
        ),
        _noop_handler,
    )
    assert response.status == "success"
    payload = _parse_tool_payload(response.content)
    assert abs(float(payload["waited_seconds"]) - 0.01) < 1e-9
    assert payload["active_background_task_ids"] == ["bg-200"]

    await runtime.shutdown()


async def test_mark_task_success_stores_full_result_and_no_result_flag() -> None:
    """Success metadata should store full result and explicit no-result marker."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)
    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

    await runtime.register_task(
        "thread-success-meta",
        "bg-300",
        {
            "task_id": "bg-300",
            "job_kind": "shell_command",
            "status": "running",
            "created_ts": 1.0,
            "updated_ts": 2.0,
            "payload_preview": "x",
        },
    )
    await middleware._mark_task_success(
        parent_thread_id="thread-success-meta",
        task_id="bg-300",
        full_result="",
        no_result=True,
    )
    meta = await runtime.get_task_meta("bg-300")
    assert isinstance(meta, dict)
    assert meta["status"] == "success"
    assert meta["result_preview"] == "no_result"
    assert meta["full_result"] == ""
    assert meta["no_result"] is True

    await runtime.shutdown()


async def test_kill_background_task_tool_marks_failure_and_drains_state() -> None:
    """Kill tool should remove task from queue/resume and mark it failed."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)
    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)

    await runtime.register_task(
        "thread-kill",
        "bg-kill-1",
        {
            "task_id": "bg-kill-1",
            "job_kind": "shell_command",
            "status": "running",
            "created_ts": 1.0,
            "updated_ts": 1.0,
            "payload_preview": "command=sleep 30",
        },
    )
    await agent.aupdate_state(
        {"configurable": {"thread_id": "thread-kill"}},
        {
            "background_hitl_queue": [
                {"task_id": "bg-kill-1", "interrupt_id": "int-1"},
                {"task_id": "bg-other", "interrupt_id": "int-2"},
            ],
            "background_hitl_resumes": {
                "bg-kill-1": {
                    "interrupt_id": "int-1",
                    "decisions": [{"type": "approve"}],
                }
            },
        },
    )

    response = await middleware.awrap_tool_call(
        _build_request(
            {"task_id": "bg-kill-1"},
            thread_id="thread-kill",
            tool_name="kill_background_task",
        ),
        _noop_handler,
    )
    assert response.status == "success"
    state = (
        await agent.aget_state({"configurable": {"thread_id": "thread-kill"}})
    ).values
    assert state["background_hitl_queue"] == [
        {"task_id": "bg-other", "interrupt_id": "int-2"}
    ]
    assert "bg-kill-1" not in state["background_hitl_resumes"]
    meta = await runtime.get_task_meta("bg-kill-1")
    assert isinstance(meta, dict)
    assert meta["status"] == "failure"

    await runtime.shutdown()


async def test_concurrent_submit_reports_all_tasks(monkeypatch) -> None:
    """Concurrent submit should not lose metadata for any task."""
    runtime = create_taskiq_runtime()
    await runtime.startup()
    agent = _FakeAgent()
    runtime.bind_agent(agent)
    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)
    monkeypatch.setattr(settings, "shell_allow_list", ["echo"])

    thread_id = "thread-many"

    async def _submit(i: int) -> None:
        req = _build_request(
            {
                "job_kind": "shell_command",
                "input": {"command": f"echo t{i}"},
            },
            thread_id=thread_id,
        )
        _ = await middleware.awrap_tool_call(req, _noop_handler)

    await asyncio.gather(*[_submit(i) for i in range(8)])

    for _ in range(40):
        _active, meta = await runtime.list_tasks(thread_id)
        if len(meta) == 8 and all(
            m.get("status") in {"success", "failure"} for m in meta.values()
        ):
            break
        await asyncio.sleep(0.05)

    _active, meta = await runtime.list_tasks(thread_id)
    assert len(meta) == 8

    response = await middleware.awrap_tool_call(
        _build_request(
            {"task_id": None, "include_queue": True},
            thread_id=thread_id,
            tool_name="get_background_task_status",
        ),
        _noop_handler,
    )
    payload = _parse_tool_payload(response.content)
    assert len(payload["background_task_meta"]) == 8

    await runtime.shutdown()


def test_wrap_model_call_injects_background_guidance_as_user_message() -> None:
    """Model middleware should append `[system]` guidance in messages."""
    runtime = create_taskiq_runtime()
    middleware = BackgroundTasksMiddleware(taskiq_runtime=runtime)
    request = _FakeModelRequest(
        messages=[{"role": "user", "content": "hello"}],
        state={"active_background_task_ids": ["bg-1"]},
    )

    captured: dict[str, ModelRequest] = {}

    def _handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return cast("ModelResponse", "ok")

    result = middleware.wrap_model_call(
        cast("ModelRequest", request),
        cast("Callable[[ModelRequest], ModelResponse]", _handler),
    )
    assert result == "ok"
    new_messages = captured["request"].messages
    assert len(new_messages) == 2
    assert "[system]" in str(new_messages[-1].content)
