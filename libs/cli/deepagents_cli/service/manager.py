"""Local lifecycle manager for the deepagents CLI FastAPI service."""
# ruff: noqa: DOC201,DOC501,PLR2004,S404,SIM105

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

_SERVICE_HOST = "127.0.0.1"
_SERVICE_DIR = Path.home() / ".deepagents" / "service"
_RUNTIME_PATH = _SERVICE_DIR / "runtime.json"
_HEALTH_TIMEOUT_SECONDS = 12.0
_POLL_INTERVAL_SECONDS = 0.2
_TERMINATE_GRACE_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class LocalServiceLease:
    """Lifecycle lease for one service binding."""

    base_url: str
    managed: bool
    pid: int | None = None
    port: int | None = None
    spawn_token: str | None = None


def _is_pid_alive(pid: int) -> bool:
    """Return whether a PID appears to be alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _choose_port() -> int:
    """Choose a free localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_SERVICE_HOST, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _healthz_ok(base_url: str) -> bool:
    """Probe service health endpoint."""
    try:
        with urlopen(f"{base_url}/healthz", timeout=1.5) as resp:  # noqa: S310
            return resp.status == 200
    except (URLError, TimeoutError, OSError):
        return False


def _read_runtime() -> dict[str, Any] | None:
    """Read runtime metadata if present."""
    if not _RUNTIME_PATH.exists():
        return None
    try:
        return json.loads(_RUNTIME_PATH.read_text())
    except (OSError, ValueError):
        return None


def _write_runtime(runtime: dict[str, Any]) -> None:
    """Persist runtime metadata."""
    _SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    _RUNTIME_PATH.write_text(json.dumps(runtime))


def _remove_runtime_file() -> None:
    """Delete stale runtime metadata."""
    try:
        _RUNTIME_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def _spawn_service(*, host: str, port: int) -> subprocess.Popen[Any]:
    """Spawn local service process."""
    cmd = [
        sys.executable,
        "-m",
        "deepagents_cli.service.server_main",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _build_base_url(port: int) -> str:
    """Build loopback service URL from port."""
    return f"http://{_SERVICE_HOST}:{port}"


def _runtime_matches_lease(
    runtime: dict[str, Any] | None,
    lease: LocalServiceLease,
) -> bool:
    """Check whether runtime file still points to the leased service."""
    if runtime is None:
        return False
    runtime_pid = int(runtime.get("pid", 0) or 0)
    runtime_port = int(runtime.get("port", 0) or 0)
    runtime_token = runtime.get("spawn_token")
    return (
        runtime_pid == (lease.pid or 0)
        and runtime_port == (lease.port or 0)
        and runtime_token == lease.spawn_token
    )


def acquire_service_lease(service_url: str | None = None) -> LocalServiceLease:
    """Acquire a service lease for this CLI invocation."""
    if service_url:
        return LocalServiceLease(base_url=service_url, managed=False)

    runtime = _read_runtime()
    if runtime:
        pid = int(runtime.get("pid", 0))
        port = int(runtime.get("port", 0))
        base_url = _build_base_url(port)
        if _is_pid_alive(pid) and _healthz_ok(base_url):
            return LocalServiceLease(
                base_url=base_url,
                managed=False,
                pid=pid,
                port=port,
                spawn_token=runtime.get("spawn_token"),
            )
        _remove_runtime_file()

    port = _choose_port()
    process = _spawn_service(host=_SERVICE_HOST, port=port)
    base_url = _build_base_url(port)
    spawn_token = uuid.uuid4().hex

    deadline = time.time() + _HEALTH_TIMEOUT_SECONDS
    while time.time() < deadline:
        if process.poll() is not None:
            break
        if _healthz_ok(base_url):
            _write_runtime(
                {
                    "pid": process.pid,
                    "port": port,
                    "spawn_token": spawn_token,
                    "owner_pid": os.getpid(),
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
            return LocalServiceLease(
                base_url=base_url,
                managed=True,
                pid=process.pid,
                port=port,
                spawn_token=spawn_token,
            )
        time.sleep(_POLL_INTERVAL_SECONDS)

    with contextlib.suppress(OSError):
        process.terminate()

    msg = "Failed to start local deepagents service"
    raise RuntimeError(msg)


def release_service_lease(lease: LocalServiceLease) -> None:
    """Release the service lease and stop managed processes."""
    if not lease.managed or lease.pid is None:
        return

    runtime = _read_runtime()
    if not _runtime_matches_lease(runtime, lease):
        return

    if _is_pid_alive(lease.pid):
        with contextlib.suppress(OSError):
            os.kill(lease.pid, signal.SIGTERM)

        deadline = time.time() + _TERMINATE_GRACE_SECONDS
        while time.time() < deadline:
            if not _is_pid_alive(lease.pid):
                break
            time.sleep(_POLL_INTERVAL_SECONDS)

        if _is_pid_alive(lease.pid):
            with contextlib.suppress(OSError):
                os.kill(lease.pid, signal.SIGKILL)

    runtime_after = _read_runtime()
    if _runtime_matches_lease(runtime_after, lease):
        _remove_runtime_file()


def ensure_local_service_url() -> str:
    """Ensure a local service is running and return its base URL."""
    return acquire_service_lease().base_url
