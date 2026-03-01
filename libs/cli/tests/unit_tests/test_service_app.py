"""Unit tests for `/acp` JSON-RPC endpoint routing."""

from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from deepagents_cli.service.app import app


class TestAcpEndpoint:
    """Tests for request/response payload handling on `/acp`."""

    def test_accepts_jsonrpc_response_payload(self) -> None:
        """Response payloads should be accepted and acknowledged."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": "perm-1",
            "result": {"outcome": "selected", "optionId": "allow-once"},
        }

        with patch(
            "deepagents_cli.service.app.permission_broker.resolve_response",
            return_value=True,
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["result"] == {"ok": True}

    def test_response_payload_without_pending_request_returns_404(self) -> None:
        """Unknown response IDs should return a JSON-RPC error response."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": "missing-id",
            "result": {"outcome": "cancelled"},
        }

        with patch(
            "deepagents_cli.service.app.permission_broker.resolve_response",
            return_value=False,
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 404
        body = response.json()
        assert (
            body["error"]["message"] == "No pending permission request for response id"
        )

    def test_prompt_stream_returns_ndjson(self) -> None:
        """Streaming prompt should return NDJSON content and JSON lines."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": {"sessionId": "sid", "prompt": "hello", "stream": True},
        }
        messages = [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "sid", "event": {"event_type": "text_delta"}},
            },
            {"jsonrpc": "2.0", "id": 1, "result": {"final": True}},
        ]

        async def _stream_messages(**_kwargs: object):  # noqa: ANN202, RUF029
            for msg in messages:
                yield msg

        with patch(
            "deepagents_cli.service.app.stream_prompt_messages",
            side_effect=_stream_messages,
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        lines = [line for line in response.text.splitlines() if line]
        assert len(lines) == 2
        assert json.loads(lines[0])["method"] == "session/update"
        assert json.loads(lines[1])["result"]["final"] is True

    def test_prompt_non_stream_returns_json(self) -> None:
        """Non-stream prompt should return regular JSON-RPC response."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {"sessionId": "sid", "prompt": "hello", "stream": False},
        }

        with patch(
            "deepagents_cli.service.app.call_method",
            return_value={"jsonrpc": "2.0", "id": 2, "result": {"final": True}},
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

    def test_initialize_includes_session_list_capability(self) -> None:
        """Initialize should declare session list capability."""
        client = TestClient(app)
        payload = {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {}}

        response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert "session/list" in body["result"]["agentCapabilities"]["methods"]
        assert "session/delete" in body["result"]["agentCapabilities"]["methods"]
        assert (
            body["result"]["agentCapabilities"]["sessionCapabilities"]["list"] == {}
        )
        assert (
            body["result"]["agentCapabilities"]["sessionCapabilities"]["delete"] == {}
        )
        assert body["result"]["agentCapabilities"]["loadSession"] is True
        assert "session/load" in body["result"]["agentCapabilities"]["methods"]

    def test_session_load_stream_returns_ndjson(self) -> None:
        """`session/load` should return NDJSON stream when `stream=true`."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 30,
            "method": "session/load",
            "params": {
                "sessionId": "tid-1",
                "runtimeSessionId": "sid-1",
                "stream": True,
            },
        }

        async def _stream_messages(**_kwargs: object):  # noqa: ANN202, RUF029
            yield {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": "tid-1", "event": {"event_type": "text_delta"}},
            }
            yield {"jsonrpc": "2.0", "id": 30, "result": {"ok": True}}

        with patch(
            "deepagents_cli.service.app.stream_load_messages",
            side_effect=_stream_messages,
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        lines = [line for line in response.text.splitlines() if line]
        assert len(lines) == 2

    def test_session_list_returns_sessions(self) -> None:
        """`session/list` should return ACP sessions payload."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "session/list",
            "params": {"cwd": "/repo"},
        }

        async def _list_threads_core(*_args: object, **_kwargs: object):  # noqa: ANN202, RUF029
            return [
                {
                    "thread_id": "tid-1",
                    "agent_name": "agent",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "message_count": 3,
                }
            ]

        with patch(
            "deepagents_cli.service.acp_methods.list_threads_core",
            side_effect=_list_threads_core,
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        sessions = body["result"]["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["sessionId"] == "tid-1"
        assert "cwd" in sessions[0]
        assert sessions[0]["updatedAt"] == "2026-01-01T00:00:00Z"
        assert sessions[0]["_meta"]["agentName"] == "agent"
        assert sessions[0]["_meta"]["messageCount"] == 3

    def test_session_list_with_cursor_returns_invalid_params(self) -> None:
        """`session/list` cursor should fail with invalid params."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "session/list",
            "params": {"cursor": "abc"},
        }

        response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["error"]["code"] == -32602
        assert "cursor not supported" in body["error"]["message"]

    def test_legacy_session_list_method_not_found(self) -> None:
        """Legacy `_deepagents/session_list` should not exist."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "_deepagents/session_list",
            "params": {},
        }

        response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["error"]["code"] == -32601

    def test_session_delete_returns_ok_true(self) -> None:
        """`session/delete` should return ok true when target exists."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "session/delete",
            "params": {"sessionId": "tid-1"},
        }

        with patch(
            "deepagents_cli.service.acp_methods.delete_thread_core",
            return_value=True,
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["ok"] is True

    def test_session_delete_not_found_still_returns_ok_true(self) -> None:
        """`session/delete` should be idempotent and succeed when missing."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "session/delete",
            "params": {"sessionId": "missing"},
        }

        with patch(
            "deepagents_cli.service.acp_methods.delete_thread_core",
            return_value=False,
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["ok"] is True

    def test_legacy_session_delete_method_not_found(self) -> None:
        """Legacy `_deepagents/session_delete` should not exist."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "_deepagents/session_delete",
            "params": {"threadId": "tid-1"},
        }

        response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["error"]["code"] == -32601

    def test_agents_list_extension_returns_agents(self) -> None:
        """`_deepagents/agents_list` should return service-side agents."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 40,
            "method": "_deepagents/agents_list",
            "params": {},
        }

        with patch(
            "deepagents_cli.service.acp_methods.list_agents_core",
            return_value=[{"name": "agent", "path": "/tmp/agent"}],
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["agents"][0]["name"] == "agent"

    def test_skills_list_extension_returns_skills(self) -> None:
        """`_deepagents/skills_list` should return service-side skills."""
        client = TestClient(app)
        payload = {
            "jsonrpc": "2.0",
            "id": 41,
            "method": "_deepagents/skills_list",
            "params": {"agent": "agent", "project": False},
        }

        with patch(
            "deepagents_cli.service.acp_methods.list_skills_core",
            return_value=[{"name": "demo", "source": "user"}],
        ):
            response = client.post("/acp", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["skills"][0]["name"] == "demo"
