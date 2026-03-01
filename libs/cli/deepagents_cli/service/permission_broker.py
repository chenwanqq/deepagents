"""Broker for ACP permission requests and responses."""
# ruff: noqa: DOC201,DOC501

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from deepagents_cli.core.contracts import JsonRpcResponseIn


@dataclass
class PendingPermission:
    """Pending permission request context."""

    request_id: str
    session_id: str
    interrupt_id: str
    action_requests: list[dict[str, Any]]
    future: asyncio.Future[dict[str, Any]]


class PermissionBroker:
    """Tracks pending permission prompts and resolves client responses."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingPermission] = {}

    def register_request(
        self,
        *,
        session_id: str,
        interrupt_id: str,
        action_requests: list[dict[str, Any]],
    ) -> PendingPermission:
        """Create a pending permission request context."""
        request_id = uuid.uuid4().hex
        pending = PendingPermission(
            request_id=request_id,
            session_id=session_id,
            interrupt_id=interrupt_id,
            action_requests=action_requests,
            future=asyncio.Future(),
        )
        self._pending[request_id] = pending
        return pending

    async def wait_for_outcome(self, request_id: str) -> dict[str, Any]:
        """Wait for a response outcome for one permission request."""
        pending = self._pending.get(request_id)
        if pending is None:
            msg = f"permission request '{request_id}' not found"
            raise KeyError(msg)
        outcome = await pending.future
        return outcome

    def resolve_response(self, payload: JsonRpcResponseIn) -> bool:
        """Resolve one pending request from JSON-RPC response payload."""
        request_id = str(payload.id)
        pending = self._pending.get(request_id)
        if pending is None:
            return False

        result = payload.result or {}
        if payload.error is not None:
            outcome = {"outcome": "cancelled"}
        else:
            outcome = {
                "outcome": str(result.get("outcome", "cancelled")),
                "optionId": result.get("optionId"),
            }

        if not pending.future.done():
            pending.future.set_result(outcome)
        return True

    def pop(self, request_id: str) -> PendingPermission | None:
        """Remove and return pending request by id."""
        return self._pending.pop(request_id, None)

    def cancel_pending_for_session(self, session_id: str) -> None:
        """Cancel all pending permission requests for one session."""
        to_cancel = [
            req_id
            for req_id, pending in self._pending.items()
            if pending.session_id == session_id
        ]
        for req_id in to_cancel:
            pending = self._pending.pop(req_id, None)
            if pending and not pending.future.done():
                pending.future.set_result({"outcome": "cancelled"})


permission_broker = PermissionBroker()
