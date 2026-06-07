"""Query depth limiting — Strawberry SchemaExtension.

Walks the parsed GraphQL AST before execution and rejects queries
deeper than the tier's max_depth threshold.
"""

from __future__ import annotations

import logging
from typing import Any

from graphql import GraphQLError
from graphql.language import visitor
from strawberry.extensions import SchemaExtension

from ..middleware.api_key import TIER_LIMITS

logger = logging.getLogger(__name__)

DEFAULT_MAX_DEPTH = 10


class DepthLimitExtension(SchemaExtension):
    """Reject GraphQL queries that exceed depth limits."""

    def on_request_start(self) -> None:
        execution_context = self.execution_context
        query = execution_context.graphql_document
        if query is None:
            return

        # Get tier from request context
        tier = "anonymous"
        try:
            request = execution_context.context.get("request")
            if request and hasattr(request, "state") and hasattr(request.state, "tier"):
                tier = request.state.tier
        except (AttributeError, TypeError):
            pass

        max_depth = TIER_LIMITS.get(tier, TIER_LIMITS["anonymous"])["max_depth"]

        # Calculate max depth via AST visitor
        class DepthVisitor(visitor.Visitor):
            def __init__(self):
                self.current_depth = 0
                self.max_depth_seen = 0

            def enter_field(self, node, *args, **kwargs):
                self.current_depth += 1
                if self.current_depth > self.max_depth_seen:
                    self.max_depth_seen = self.current_depth

            def leave_field(self, node, *args, **kwargs):
                self.current_depth -= 1

        depth_visitor = DepthVisitor()
        visitor.visit(query, depth_visitor)

        if depth_visitor.max_depth_seen > max_depth:
            raise GraphQLError(
                f"Query depth {depth_visitor.max_depth_seen} exceeds "
                f"maximum {max_depth} for tier '{tier}'",
                extensions={
                    "code": "QUERY_TOO_DEEP",
                    "depth": depth_visitor.max_depth_seen,
                    "max_depth": max_depth,
                    "tier": tier,
                },
            )
