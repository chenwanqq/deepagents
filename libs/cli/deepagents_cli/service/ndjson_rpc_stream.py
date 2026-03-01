"""NDJSON framing helpers for JSON-RPC message streaming."""
# ruff: noqa: DOC201

from __future__ import annotations

import json
from typing import Any


def encode_ndjson_json(payload: dict[str, Any]) -> str:
    """Encode one JSON payload into one NDJSON line."""
    return f"{json.dumps(payload, ensure_ascii=True)}\n"
