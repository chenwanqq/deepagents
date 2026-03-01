"""Unit tests for status bar widget."""

from typing import ClassVar

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from deepagents_cli.widgets.status import StatusBar


class StatusHostApp(App[None]):
    """Minimal host app to mount the status bar widget."""

    CSS_PATH: ClassVar[str | None] = None

    def compose(self) -> ComposeResult:
        """Mount a status bar for widget tests."""
        yield StatusBar(cwd="/tmp", id="status")


class TestStatusBarTransport:
    """Tests for transport display in status bar."""

    @pytest.mark.asyncio
    async def test_set_transport_display_updates_widget(self) -> None:
        """`set_transport_display` should update transport text."""
        app = StatusHostApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            status = app.query_one("#status", StatusBar)
            status.set_transport_display("HTTP:7777")
            await pilot.pause()

            display = status.query_one("#transport-display", Static)
            assert "HTTP:7777" in str(display.content)

    @pytest.mark.asyncio
    async def test_status_bar_compose_includes_transport_widget(self) -> None:
        """Transport display should exist alongside other status widgets."""
        app = StatusHostApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            status = app.query_one("#status", StatusBar)

            assert status.query_one("#transport-display", Static) is not None
            assert status.query_one("#tokens-display", Static) is not None
            assert status.query_one("#model-display", Static) is not None
            assert status.query_one("#auto-approve-indicator", Static) is not None
