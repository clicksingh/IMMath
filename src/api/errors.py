"""Structured GraphQL error formatting.

Provides consistent error responses with extensions.code for:
- RATE_LIMITED
- QUERY_TOO_DEEP
- COMPLEXITY_EXCEEDED
- INVALID_API_KEY
- General execution errors
"""

from __future__ import annotations

from graphql import GraphQLError


def graphql_error_formatter(error: Exception, debug: bool = False) -> dict:
    """Format a GraphQL error with structured extensions.

    Returns a dict matching the GraphQL error spec with added extensions.code.
    """
    if isinstance(error, GraphQLError):
        extensions = dict(error.extensions) if error.extensions else {}
        if "code" not in extensions:
            extensions["code"] = "EXECUTION_ERROR"
        return {
            "message": error.message,
            "locations": [
                {"line": loc.line, "column": loc.column}
                for loc in (error.locations or [])
            ],
            "path": error.path,
            "extensions": extensions,
        }

    return {
        "message": str(error),
        "extensions": {"code": "INTERNAL_ERROR"},
    }
