"""Unit tests for HTTP JSON-RPC service transport."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepagents_cli.client.service_transport import ServiceTransport


class _StaticAsyncIterator:
    """Simple async iterator wrapper for deterministic stream tests."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _StaticAsyncIterator:
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class TestPermissionFlow:
    """Tests ACP permission request/response behavior."""

    @pytest.mark.asyncio
    async def test_prompt_stream_emits_permission_event(self) -> None:
        """`session/request_permission` should map to `approval_required` event."""
        client = MagicMock()
        client.call_stream = MagicMock(
            return_value=_StaticAsyncIterator(
                [
                    {
                        "jsonrpc": "2.0",
                        "id": "perm-1",
                        "method": "session/request_permission",
                        "params": {
                            "toolCall": {"name": "bash", "args": {"command": "ls"}},
                            "_meta": {"interruptId": "interrupt-1"},
                        },
                    },
                    {"jsonrpc": "2.0", "id": 1, "result": {"final": True}},
                ]
            )
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        events = [event async for event in transport.prompt_stream("sid-1", "hello")]

        assert len(events) == 2
        approval = events[0]
        assert approval.event_type == "approval_required"
        assert approval.payload["permission_request_id"] == "perm-1"
        assert approval.payload["interrupt_id"] == "interrupt-1"
        assert approval.payload["action_requests"] == [
            {"name": "bash", "args": {"command": "ls"}}
        ]
        assert events[1].event_type == "done"

    @pytest.mark.asyncio
    async def test_submit_decision_posts_jsonrpc_response(self) -> None:
        """Decision submission should send JSON-RPC response payload."""
        client = MagicMock()
        client.respond = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": None, "result": {"ok": True}}
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        await transport.submit_decision(
            "sid-1",
            "perm-1",
            [{"type": "approve", "message": "allow-always"}],
        )

        client.respond.assert_awaited_once_with(
            request_id="perm-1",
            result={"outcome": "selected", "optionId": "allow-always"},
        )

    @pytest.mark.asyncio
    async def test_list_threads_calls_session_list_and_maps_result(self) -> None:
        """List threads should use `session/list` and map sessions to threads."""
        client = MagicMock()
        client.call = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "sessions": [
                        {
                            "sessionId": "tid-1",
                            "cwd": "/repo",
                            "updatedAt": "2026-01-01T00:00:00Z",
                            "_meta": {"agentName": "agent-a", "messageCount": 4},
                        },
                        {
                            "sessionId": "tid-2",
                            "cwd": "/repo",
                            "updatedAt": None,
                            "_meta": {"agentName": "agent-b", "messageCount": 2},
                        },
                    ]
                },
            }
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        threads = await transport.list_threads(
            agent_name="agent-a",
            limit=20,
            include_message_count=True,
        )

        client.call.assert_awaited_once()
        kwargs = client.call.await_args.kwargs
        assert kwargs["method"] == "session/list"
        assert kwargs["params"] == {"cwd": None, "cursor": None}
        assert threads == [
            {
                "thread_id": "tid-1",
                "updated_at": "2026-01-01T00:00:00Z",
                "agent_name": "agent-a",
                "message_count": 4,
            }
        ]

    @pytest.mark.asyncio
    async def test_list_threads_non_list_sessions_returns_empty(self) -> None:
        """Invalid `sessions` payload should return empty thread list."""
        client = MagicMock()
        client.call = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": 1, "result": {"sessions": "bad"}}
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        threads = await transport.list_threads(
            agent_name=None,
            limit=20,
            include_message_count=False,
        )

        assert threads == []

    @pytest.mark.asyncio
    async def test_delete_thread_calls_session_delete(self) -> None:
        """Delete thread should call `session/delete` with `sessionId`."""
        client = MagicMock()
        client.call = AsyncMock(
            return_value={"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        deleted = await transport.delete_thread("tid-1")

        assert deleted is True
        client.call.assert_awaited_once()
        kwargs = client.call.await_args.kwargs
        assert kwargs["method"] == "session/delete"
        assert kwargs["params"] == {"sessionId": "tid-1"}

    @pytest.mark.asyncio
    async def test_delete_thread_missing_ok_returns_false(self) -> None:
        """Delete thread should return false when result.ok is absent."""
        client = MagicMock()
        client.call = AsyncMock(return_value={"jsonrpc": "2.0", "id": 1, "result": {}})
        transport = ServiceTransport(client)
        transport._initialized = True

        deleted = await transport.delete_thread("tid-2")

        assert deleted is False

    @pytest.mark.asyncio
    async def test_load_session_stream_parses_updates_and_done(self) -> None:
        """Load stream should parse `session/update` and final result."""
        client = MagicMock()
        client.call_stream = MagicMock(
            return_value=_StaticAsyncIterator(
                [
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "sessionId": "tid-1",
                            "event": {
                                "event_id": 1,
                                "event_type": "text_delta",
                                "session_id": "sid-1",
                                "payload": {"text": "hello", "role": "assistant"},
                            },
                        },
                    },
                    {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}},
                ]
            )
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        events = [
            event
            async for event in transport.load_session_stream(
                "sid-1",
                "tid-1",
                cwd="/repo",
            )
        ]

        assert len(events) == 2
        assert events[0].event_type == "text_delta"
        assert events[1].event_type == "done"

    @pytest.mark.asyncio
    async def test_compact_session_calls_extension_method(self) -> None:
        """Compact should call `_deepagents/session_compact`."""
        client = MagicMock()
        client.call = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "id": 4,
                "result": {"ok": True, "compactedMessages": 2},
            }
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        result = await transport.compact_session("sid-9")

        assert result["ok"] is True
        kwargs = client.call.await_args.kwargs
        assert kwargs["method"] == "_deepagents/session_compact"
        assert kwargs["params"] == {"sessionId": "sid-9"}

    @pytest.mark.asyncio
    async def test_list_agents_calls_extension_method(self) -> None:
        """List agents should call `_deepagents/agents_list`."""
        client = MagicMock()
        client.call = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "id": 5,
                "result": {"agents": [{"name": "agent"}]},
            }
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        agents = await transport.list_agents()

        assert agents == [{"name": "agent"}]
        kwargs = client.call.await_args.kwargs
        assert kwargs["method"] == "_deepagents/agents_list"

    @pytest.mark.asyncio
    async def test_skills_info_calls_extension_method(self) -> None:
        """Skill info should call `_deepagents/skills_info`."""
        client = MagicMock()
        client.call = AsyncMock(
            return_value={
                "jsonrpc": "2.0",
                "id": 6,
                "result": {"skill": {"name": "demo"}, "content": "# demo"},
            }
        )
        transport = ServiceTransport(client)
        transport._initialized = True

        result = await transport.skill_info(name="demo", agent="agent", project=False)

        assert result["skill"]["name"] == "demo"
        kwargs = client.call.await_args.kwargs
        assert kwargs["method"] == "_deepagents/skills_info"
