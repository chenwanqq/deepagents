"""FastAPI service exposing ACP-style JSON-RPC over HTTP."""
# ruff: noqa: DOC201,TC003

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from deepagents_cli._version import __version__
from deepagents_cli.core.contracts import (
    HealthResponse,
    JsonRpcRequest,
    JsonRpcResponseIn,
)
from deepagents_cli.service.acp_methods import (
    call_method,
    stream_load_messages,
    stream_prompt_messages,
)
from deepagents_cli.service.jsonrpc import (
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_REQUEST,
    error_response,
    ok_response,
)
from deepagents_cli.service.ndjson_rpc_stream import encode_ndjson_json
from deepagents_cli.service.permission_broker import permission_broker

logger = logging.getLogger(__name__)

app = FastAPI(title="deepagents-cli-service", version=__version__)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Service health endpoint."""
    return HealthResponse(version=__version__)


@app.post("/acp", response_model=None)
async def handle_acp(request: Request) -> JSONResponse | StreamingResponse:
    """Handle JSON-RPC requests and stream `session/prompt` over NDJSON."""
    try:
        raw_payload = cast("dict[str, Any]", await request.json())
    except Exception:  # noqa: BLE001
        response = error_response(
            None,
            code=JSONRPC_INVALID_REQUEST,
            message="Invalid JSON payload",
        )
        return JSONResponse(response.model_dump(exclude_none=True), status_code=400)

    parsed_response: JsonRpcResponseIn | None = None
    try:
        parsed = JsonRpcRequest.model_validate(raw_payload)
    except ValidationError:
        try:
            parsed_response = JsonRpcResponseIn.model_validate(raw_payload)
        except ValidationError as e:
            response = error_response(
                None,
                code=JSONRPC_INVALID_REQUEST,
                message="Invalid JSON-RPC message",
                data={"errors": e.errors()},
            )
            return JSONResponse(response.model_dump(exclude_none=True), status_code=400)

        resolved = permission_broker.resolve_response(parsed_response)
        if not resolved:
            response = error_response(
                parsed_response.id,
                code=JSONRPC_INVALID_REQUEST,
                message="No pending permission request for response id",
            )
            return JSONResponse(response.model_dump(exclude_none=True), status_code=404)
        ack = ok_response(None, {"ok": True})
        return JSONResponse(ack.model_dump(exclude_none=True))
    stream_methods = {"session/prompt", "session/load"}
    is_stream_method = parsed.method in stream_methods
    stream_flag = bool((parsed.params or {}).get("stream", True))

    if is_stream_method and stream_flag:

        async def _event_generator() -> AsyncIterator[str]:
            try:
                stream_fn = (
                    stream_prompt_messages
                    if parsed.method == "session/prompt"
                    else stream_load_messages
                )
                async for message in stream_fn(
                    request_id=parsed.id,
                    params=parsed.params,
                ):
                    yield encode_ndjson_json(message)
            except Exception as e:
                logger.exception("Unexpected error in NDJSON stream")
                yield encode_ndjson_json(
                    error_response(
                        parsed.id,
                        code=JSONRPC_INTERNAL_ERROR,
                        message=str(e),
                    ).model_dump(exclude_none=True)
                )

        return StreamingResponse(
            _event_generator(),
            media_type="application/x-ndjson",
        )

    result = await call_method(
        method=parsed.method,
        params=parsed.params,
        request_id=parsed.id,
    )
    return JSONResponse(result)
