"""Core admin operations for HTTP service extensions."""
# ruff: noqa: DOC201,DOC501,TRY003,EM101,EM102,E501,I001

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from deepagents_cli.agent import DEFAULT_AGENT_NAME
from deepagents_cli.config import get_default_coding_instructions, settings
from deepagents_cli.skills.load import list_skills


MAX_SKILL_NAME_LENGTH = 64


def list_agents_core() -> list[dict[str, Any]]:
    """List agents from service-side filesystem."""
    agents_dir = settings.user_deepagents_dir
    if not agents_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for agent_path in sorted(agents_dir.iterdir()):
        if not agent_path.is_dir():
            continue
        agent_md = agent_path / "AGENTS.md"
        items.append(
            {
                "name": agent_path.name,
                "path": str(agent_path),
                "isDefault": agent_path.name == DEFAULT_AGENT_NAME,
                "hasAgentsMd": agent_md.exists(),
            }
        )
    return items


def reset_agent_core(agent_name: str, source_agent: str | None) -> dict[str, Any]:
    """Reset one agent directory and return summary."""
    _ensure_valid_name(agent_name, label="agent")
    agents_dir = settings.user_deepagents_dir
    agent_dir = agents_dir / agent_name

    if source_agent:
        _ensure_valid_name(source_agent, label="source agent")
        source_md = agents_dir / source_agent / "AGENTS.md"
        if not source_md.exists():
            msg = f"source agent '{source_agent}' not found or has no AGENTS.md"
            raise KeyError(msg)
        content = source_md.read_text(encoding="utf-8")
    else:
        content = get_default_coding_instructions()

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "AGENTS.md").write_text(content, encoding="utf-8")
    return {"ok": True, "agent": agent_name, "path": str(agent_dir)}


def list_skills_core(agent: str, *, project: bool) -> list[dict[str, Any]]:
    """List skills on the service filesystem."""
    _ensure_valid_name(agent, label="agent")
    user_skills_dir = settings.get_user_skills_dir(agent)
    project_skills_dir = settings.get_project_skills_dir()
    user_agent_skills_dir = settings.get_user_agent_skills_dir()
    project_agent_skills_dir = settings.get_project_agent_skills_dir()

    if project:
        if not project_skills_dir:
            return []
        skills = list_skills(
            user_skills_dir=None,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=None,
            project_agent_skills_dir=project_agent_skills_dir,
        )
    else:
        skills = list_skills(
            built_in_skills_dir=settings.get_built_in_skills_dir(),
            user_skills_dir=user_skills_dir,
            project_skills_dir=project_skills_dir,
            user_agent_skills_dir=user_agent_skills_dir,
            project_agent_skills_dir=project_agent_skills_dir,
        )
    return [dict(skill) for skill in skills]


def create_skill_core(name: str, agent: str, *, project: bool) -> dict[str, Any]:
    """Create one skill and return location."""
    _ensure_valid_name(agent, label="agent")
    _ensure_valid_name(name, label="skill")
    if project:
        if not settings.project_root:
            raise KeyError("not in a project directory")
        skills_dir = settings.ensure_project_skills_dir()
        if skills_dir is None:
            raise RuntimeError("could not create project skills directory")
    else:
        skills_dir = settings.ensure_user_skills_dir(agent)
    skill_dir = skills_dir / name
    if skill_dir.exists():
        msg = f"skill '{name}' already exists"
        raise KeyError(msg)
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(_generate_template(name), encoding="utf-8")
    return {"ok": True, "name": name, "path": str(skill_dir)}


def skill_info_core(name: str, agent: str, *, project: bool) -> dict[str, Any]:
    """Get one skill details and contents."""
    skills = list_skills_core(agent, project=project)
    match = next((skill for skill in skills if skill.get("name") == name), None)
    if not isinstance(match, dict):
        raise KeyError(f"skill '{name}' not found")
    skill_path = Path(str(match.get("path", "")))
    content = skill_path.read_text(encoding="utf-8")
    supporting = [item.name for item in skill_path.parent.iterdir() if item.name != "SKILL.md"]
    return {"skill": match, "content": content, "supportingFiles": supporting}


def delete_skill_core(
    name: str,
    agent: str,
    *,
    project: bool,
    force: bool,
) -> dict[str, Any]:
    """Delete one skill directory."""
    _ = force
    _ensure_valid_name(agent, label="agent")
    _ensure_valid_name(name, label="skill")
    skills = list_skills_core(agent, project=project)
    match = next((skill for skill in skills if skill.get("name") == name), None)
    if not isinstance(match, dict):
        raise KeyError(f"skill '{name}' not found")
    source = str(match.get("source", ""))
    if source == "built-in":
        raise RuntimeError("cannot delete built-in skill")

    skill_path = Path(str(match.get("path", "")))
    skill_dir = skill_path.parent
    base_dir = settings.get_project_skills_dir() if source == "project" else settings.get_user_skills_dir(agent)
    if base_dir is None:
        raise RuntimeError("cannot determine base skills directory")
    resolved_skill = skill_dir.resolve()
    resolved_base = base_dir.resolve()
    if not resolved_skill.is_relative_to(resolved_base):
        raise RuntimeError("refusing to delete outside skill directory")
    if skill_dir.is_symlink():
        raise RuntimeError("refusing to delete symlink skill directory")

    shutil.rmtree(skill_dir)
    return {"ok": True, "name": name}


def _ensure_valid_name(value: str, *, label: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{label} name cannot be empty")
    if len(value) > MAX_SKILL_NAME_LENGTH:
        raise ValueError(f"{label} name cannot exceed 64 characters")
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"{label} name cannot contain path components")
    if value.startswith("-") or value.endswith("-") or "--" in value:
        raise ValueError(
            f"{label} name must be lowercase alphanumeric with single hyphens only"
        )
    for char in value:
        if char == "-":
            continue
        if (char.isalpha() and char.islower()) or char.isdigit() or char == "_":
            continue
        raise ValueError(
            f"{label} name must be lowercase alphanumeric with single hyphens only"
        )


def _generate_template(skill_name: str) -> str:
    """Generate a default SKILL.md template."""
    title = skill_name.title().replace("-", " ")
    return f"""---
name: {skill_name}
description: "TODO: Explain what this skill does and when to use it."
---

# {title}

## Overview

TODO: Describe when this skill should be selected.

## Instructions

1. TODO: Add concrete step-by-step instructions.
2. TODO: Add validation and edge-case handling guidance.
3. TODO: Add expected output format.
"""
