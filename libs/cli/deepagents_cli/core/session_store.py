"""Global session store for service process runtime sessions."""

from __future__ import annotations

from deepagents_cli.core.runtime import CoreSessionManager

session_manager = CoreSessionManager()
"""Process-local runtime session manager instance."""
