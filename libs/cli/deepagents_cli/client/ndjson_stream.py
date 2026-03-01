"""NDJSON parsing helpers for the deepagents CLI HTTP transport."""
# ruff: noqa: DOC201

from __future__ import annotations

import json
from typing import Any


def parse_ndjson_line(line: str) -> dict[str, Any] | None:
    """Parse one NDJSON line into a JSON object."""
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
