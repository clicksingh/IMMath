"""Entry point: run the combined GraphQL + Dash server."""

from __future__ import annotations

import logging
import uvicorn
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def main():
    base_dir = Path(__file__).resolve().parents[2]
    from src.api.app import create_api

    app = create_api(base_dir, debug=True)
    uvicorn.run(app, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
