"""Entry point: run the combined GraphQL + Dash server."""

from __future__ import annotations

import logging
import os

import uvicorn
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def _find_project_root() -> Path:
    """Find project root by walking up from this file until finding data/."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "data" / "master" / "master_panel.parquet").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return Path.cwd()


def main():
    base_dir = _find_project_root()
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    port = int(os.environ.get("PORT", 8050))

    from src.api.app import create_api

    app = create_api(base_dir, debug=debug)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
