"""JSON-RPC 2.0 helpers for ACP HTTP transport."""
# ruff: noqa: DOC201

from __future__ import annotations

from typing import Any

from deepagents_cli.core.contracts import JsonRpcError, JsonRpcId, JsonRpcResponse

JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


def ok_response(request_id: JsonRpcId, result: dict[str, Any]) -> JsonRpcResponse:
    """Build a successful JSON-RPC response."""
    return JsonRpcResponse(id=request_id, result=result)


def error_response(
    request_id: JsonRpcId,
    *,
    code: int,
    message: str,
    data: dict[str, Any] | None = None,
) -> JsonRpcResponse:
    """Build a JSON-RPC error response."""
    return JsonRpcResponse(
        id=request_id,
        error=JsonRpcError(code=code, message=message, data=data),
    )
