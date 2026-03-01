"""Thin wrappers around thread persistence APIs for service layer."""
# ruff: noqa: DOC201

from __future__ import annotations

from pathlib import Path

from deepagents_cli.sessions import delete_thread, list_threads


async def list_threads_core(
    agent_name: str | None,
    *,
    limit: int,
    include_message_count: bool,
    cwd: str | None = None,
) -> list[dict[str, object]]:
    """List conversation threads from checkpoint storage."""
    if cwd is not None and cwd != str(Path.cwd()):
        return []
    threads = await list_threads(
        agent_name=agent_name,
        limit=limit,
        include_message_count=include_message_count,
    )
    return [dict(thread) for thread in threads]


async def delete_thread_core(thread_id: str) -> bool:
    """Delete one thread by id."""
    return await delete_thread(thread_id)
