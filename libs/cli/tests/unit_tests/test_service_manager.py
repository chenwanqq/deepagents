"""Unit tests for local service lease lifecycle manager."""

import signal
from unittest.mock import MagicMock, patch

from deepagents_cli.service.manager import (
    LocalServiceLease,
    acquire_service_lease,
    release_service_lease,
)


def test_acquire_with_explicit_service_url_is_unmanaged() -> None:
    """Explicit service URL should not be lifecycle-managed."""
    lease = acquire_service_lease("http://127.0.0.1:9999")
    assert lease.base_url == "http://127.0.0.1:9999"
    assert lease.managed is False


def test_acquire_reuses_existing_runtime_without_management() -> None:
    """Healthy runtime file should be reused as unmanaged lease."""
    with (
        patch(
            "deepagents_cli.service.manager._read_runtime",
            return_value={"pid": 123, "port": 8888, "spawn_token": "old"},
        ),
        patch("deepagents_cli.service.manager._is_pid_alive", return_value=True),
        patch("deepagents_cli.service.manager._healthz_ok", return_value=True),
        patch("deepagents_cli.service.manager._spawn_service") as mock_spawn,
    ):
        lease = acquire_service_lease(None)

    assert lease.base_url == "http://127.0.0.1:8888"
    assert lease.managed is False
    assert lease.pid == 123
    assert lease.port == 8888
    mock_spawn.assert_not_called()


def test_acquire_starts_new_service_and_marks_managed() -> None:
    """Missing runtime should spawn and return managed lease."""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.pid = 456

    with (
        patch("deepagents_cli.service.manager._read_runtime", return_value=None),
        patch("deepagents_cli.service.manager._choose_port", return_value=7777),
        patch(
            "deepagents_cli.service.manager._spawn_service",
            return_value=mock_process,
        ),
        patch("deepagents_cli.service.manager._healthz_ok", return_value=True),
        patch("deepagents_cli.service.manager._write_runtime") as mock_write,
    ):
        lease = acquire_service_lease(None)

    assert lease.base_url == "http://127.0.0.1:7777"
    assert lease.managed is True
    assert lease.pid == 456
    assert lease.port == 7777
    assert isinstance(lease.spawn_token, str)
    assert lease.spawn_token
    mock_write.assert_called_once()
    write_payload = mock_write.call_args.args[0]
    assert write_payload["pid"] == 456
    assert write_payload["port"] == 7777
    assert write_payload["spawn_token"] == lease.spawn_token


def test_release_terminates_matching_managed_lease() -> None:
    """Managed lease with matching runtime should be terminated and cleaned up."""
    lease = LocalServiceLease(
        base_url="http://127.0.0.1:7777",
        managed=True,
        pid=456,
        port=7777,
        spawn_token="token-a",
    )

    with (
        patch(
            "deepagents_cli.service.manager._read_runtime",
            side_effect=[
                {"pid": 456, "port": 7777, "spawn_token": "token-a"},
                {"pid": 456, "port": 7777, "spawn_token": "token-a"},
            ],
        ),
        patch(
            "deepagents_cli.service.manager._is_pid_alive",
            side_effect=[True, False, False],
        ),
        patch("deepagents_cli.service.manager.os.kill") as mock_kill,
        patch("deepagents_cli.service.manager._remove_runtime_file") as mock_remove,
    ):
        release_service_lease(lease)

    mock_kill.assert_called_once_with(456, signal.SIGTERM)
    mock_remove.assert_called_once()


def test_release_does_not_kill_when_runtime_mismatch() -> None:
    """Mismatched runtime should not be terminated."""
    lease = LocalServiceLease(
        base_url="http://127.0.0.1:7777",
        managed=True,
        pid=456,
        port=7777,
        spawn_token="token-a",
    )

    with (
        patch(
            "deepagents_cli.service.manager._read_runtime",
            return_value={"pid": 456, "port": 7777, "spawn_token": "token-b"},
        ),
        patch("deepagents_cli.service.manager.os.kill") as mock_kill,
        patch("deepagents_cli.service.manager._remove_runtime_file") as mock_remove,
    ):
        release_service_lease(lease)

    mock_kill.assert_not_called()
    mock_remove.assert_not_called()
