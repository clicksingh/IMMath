"""FastAPI application factory — mounts GraphQL API and Dash dashboard.

GraphQL API:     POST /graphql
GraphiQL:        GET  /graphql  (debug mode only)
Health check:    GET  /health
Dash dashboard:  GET  /dash/*  and GET / (redirects to /dash)
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.wsgi import WSGIMiddleware
from strawberry.fastapi import GraphQLRouter

from .errors import graphql_error_formatter
from .schema import create_schema

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8050

# Prefix where the Dash dashboard is served
DASH_PREFIX = "/dash"


def create_api(base_dir: Path, debug: bool = False) -> FastAPI:
    """Create the FastAPI application with GraphQL and Dash mounted.

    Args:
        base_dir: Project root directory (contains data/ and outputs/).
        debug: Enable GraphiQL playground and introspection.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="IMMath ACI Research API",
        version="1.0.0",
        debug=debug,
    )

    # CORS — allow browser access from localhost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Store base_dir in app state for middleware access
    @app.middleware("http")
    async def inject_base_dir(request: Request, call_next):
        request.state.base_dir = str(base_dir)
        response = await call_next(request)
        return response

    # Health check
    @app.get("/health")
    def health():
        return {"status": "ok", "version": "1.0.0"}

    # GraphQL
    schema = create_schema(debug=debug)
    graphql_app = GraphQLRouter(
        schema,
        graphql_ide="graphiql" if debug else None,
    )
    app.include_router(graphql_app, prefix="/graphql")

    # Dash dashboard — mounted as WSGI sub-app at /dash
    try:
        from src.viz.dashboard import create_app as create_dash_app

        dash_app = create_dash_app(base_dir)
        app.mount(DASH_PREFIX, WSGIMiddleware(dash_app.server))

        # Redirect root to /dash for browser convenience
        @app.get("/")
        def root_redirect():
            return RedirectResponse(url=DASH_PREFIX + "/")

        logger.info("Dash dashboard mounted at %s", DASH_PREFIX)
    except Exception:
        logger.warning("Could not mount Dash dashboard — data files may be missing")

    return app
