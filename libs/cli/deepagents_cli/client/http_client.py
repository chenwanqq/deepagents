"""Async JSON-RPC over streamable HTTP NDJSON client for CLI service."""
# ruff: noqa: D107,DOC201,DOC402,DOC501,ANN401,TC003,S113

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from deepagents_cli.client.ndjson_stream import parse_ndjson_line

logger = logging.getLogger(__name__)


class ServiceHttpClient:
    """HTTP client wrapper for the deepagents local service."""

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """Service base URL."""
        return self._base_url

    async def call(
        self,
        *,
        method: str,
        params: dict[str, Any] | None,
        request_id: str | int,
    ) -> dict[str, Any]:
        """Perform one non-streaming JSON-RPC request."""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/acp", json=payload)
            resp.raise_for_status()
            return cast_dict(resp.json())

    async def call_stream(
        self,
        *,
        method: str,
        params: dict[str, Any] | None,
        request_id: str | int,
    ) -> AsyncIterator[dict[str, Any]]:
        """Perform one streaming JSON-RPC request over NDJSON."""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        async with httpx.AsyncClient(timeout=None) as client, client.stream(
            "POST",
            f"{self._base_url}/acp",
            json=payload,
            headers={"accept": "application/x-ndjson"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                message = parse_ndjson_line(line)
                if message is None:
                    if line:
                        logger.debug("Skipping invalid NDJSON line: %s", line)
                    continue
                yield cast_dict(message)

    async def respond(
        self,
        *,
        request_id: str | int,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one JSON-RPC response payload to the service."""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
            "error": error,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base_url}/acp", json=payload)
            resp.raise_for_status()
            return cast_dict(resp.json())


def cast_dict(value: Any) -> dict[str, Any]:
    """Cast JSON-like value to dict, raising if incompatible."""
    if not isinstance(value, dict):
        msg = "Expected JSON object"
        raise TypeError(msg)
    return value
