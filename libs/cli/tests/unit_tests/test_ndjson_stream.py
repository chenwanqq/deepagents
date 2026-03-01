"""Unit tests for NDJSON stream parsing helpers."""

from deepagents_cli.client.ndjson_stream import parse_ndjson_line


def test_parse_ndjson_line_valid_object() -> None:
    payload = parse_ndjson_line('{"jsonrpc":"2.0","id":1,"result":{"ok":true}}')
    assert payload is not None
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == 1


def test_parse_ndjson_line_empty_returns_none() -> None:
    assert parse_ndjson_line("") is None


def test_parse_ndjson_line_invalid_json_returns_none() -> None:
    assert parse_ndjson_line("{bad-json}") is None


def test_parse_ndjson_line_non_object_returns_none() -> None:
    assert parse_ndjson_line('["not","object"]') is None
