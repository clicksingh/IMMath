"""Entry point: run the combined GraphQL + Dash server."""

from __future__ import annotations

import logging
import os

import uvicorn
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def main():
    base_dir = Path(__file__).resolve().parents[2]
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    port = int(os.environ.get("PORT", 8050))

    from src.api.app import create_api

    app = create_api(base_dir, debug=debug)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
