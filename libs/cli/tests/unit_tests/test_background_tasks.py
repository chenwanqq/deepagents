"""Unit tests for CLI background task helpers."""

import asyncio
import time

from langchain_core.messages import SystemMessage

from deepagents_cli.background_tasks import (
    BACKGROUND_HITL_TIMEOUT_SECONDS,
    BackgroundTasksMiddleware,
    TaskiqRuntime,
    build_submit_background_task_tool,
    clear_inmemory_background_state,
)


def test_submit_background_task_rejects_unsafe_shell() -> None:
    """Shell commands that fail allow-list checks should fail closed."""
    runtime = TaskiqRuntime()
    tool = build_submit_background_task_tool(runtime)

    command = tool(
        job_kind="shell_command",
        payload_input={"command": "rm -rf /", "timeout": 5},
        tool_call_id="call-1",
    )

    update = command.update
    assert update is not None
    assert "background_task_meta" in update
    [meta] = list(update["background_task_meta"].values())
    assert meta["status"] == "failed"


def test_submit_background_task_rejects_agent_subtask_for_now() -> None:
    """Agent subtask is fail-closed until executor bridge is implemented."""
    runtime = TaskiqRuntime()
    tool = build_submit_background_task_tool(runtime)

    command = tool(
        job_kind="agent_subtask",
        payload_input={"description": "x", "subagent_type": "coder"},
        tool_call_id="call-2",
    )

    update = command.update
    assert update is not None
    [meta] = list(update["background_task_meta"].values())
    assert meta["status"] == "failed"


def test_clear_inmemory_background_state_on_resume() -> None:
    """In-memory resume should clear unrecoverable background state."""
    update = clear_inmemory_background_state(is_resumed=True, taskiq_mode="inmemory")
    assert update is not None
    assert update["active_background_task_ids"] == []
    assert update["background_hitl_queue"] == []
    assert update["background_hitl_resumes"] == {}


def test_background_middleware_expires_hitl_queue() -> None:
    """Expired background HITL items should auto-reject and emit a system message."""
    middleware = BackgroundTasksMiddleware(TaskiqRuntime())
    state = {
        "background_hitl_queue": [
            {
                "interrupt_id": "i1",
                "task_id": "t1",
                "action_requests": [{"name": "shell"}],
                "deadline": time.time() - 1,
            }
        ],
        "background_task_meta": {
            "t1": {
                "task_id": "t1",
                "job_kind": "shell_command",
                "payload": {"command": "ls"},
                "status": "waiting_hitl",
            }
        },
    }

    update = middleware.before_model(state, runtime=None)
    assert update is not None
    assert update["background_hitl_queue"] == []
    assert update["background_task_meta"]["t1"]["status"] == "failed"
    assert BACKGROUND_HITL_TIMEOUT_SECONDS == 600
    [msg] = update["messages"]
    assert isinstance(msg, SystemMessage)


def test_background_middleware_applies_runtime_updates() -> None:
    """Runtime completion events should be reflected in checkpoint state."""
    runtime = TaskiqRuntime()
    middleware = BackgroundTasksMiddleware(runtime)
    runtime.publish_update(
        {
            "task_id": "t1",
            "status": "completed",
            "result": "done",
        }
    )

    state = {
        "active_background_task_ids": ["t1", "t2"],
        "background_task_meta": {
            "t1": {
                "task_id": "t1",
                "job_kind": "shell_command",
                "payload": {"command": "ls"},
                "status": "running",
            }
        },
    }

    update = middleware.before_model(state, runtime=None)
    assert update is not None
    assert update["background_task_meta"]["t1"]["status"] == "completed"
    assert update["active_background_task_ids"] == ["t2"]
    assert update["messages"]


def test_submit_shell_command_registers_background_worker() -> None:
    """Safe shell command should spawn a runtime worker."""
    runtime = TaskiqRuntime()
    tool = build_submit_background_task_tool(runtime)

    command = tool(
        job_kind="shell_command",
        payload_input={"command": "ls"},
        tool_call_id="call-3",
    )

    update = command.update
    assert update is not None
    assert update["active_background_task_ids"]
    assert runtime.tasks

    async def _cleanup() -> None:
        await asyncio.gather(*runtime.tasks, return_exceptions=True)

    asyncio.run(_cleanup())
