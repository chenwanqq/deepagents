"""Server entrypoint for the deepagents CLI FastAPI service."""

from __future__ import annotations

import argparse

from deepagents_cli.service.app import app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deepagents CLI local service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    """Start uvicorn with the CLI service app."""
    args = _parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
