"""Introspection control — disables __schema and __type queries in production."""

from __future__ import annotations

import logging

from graphql import GraphQLError
from graphql.language import visitor
from strawberry.extensions import SchemaExtension

logger = logging.getLogger(__name__)


class IntrospectionGuard(SchemaExtension):
    """Block introspection queries when not in debug mode."""

    def on_request_start(self) -> None:
        execution_context = self.execution_context
        query = execution_context.graphql_document
        if query is None:
            return

        class IntrospectionVisitor(visitor.Visitor):
            def __init__(self):
                self.found = False

            def enter_field(self, node, *args, **kwargs):
                name = node.name.value
                if name.startswith("__"):
                    self.found = True
                    return visitor.SKIP

        introspection_visitor = IntrospectionVisitor()
        visitor.visit(query, introspection_visitor)

        if introspection_visitor.found:
            raise GraphQLError(
                "GraphQL introspection is not allowed in production",
                extensions={"code": "INTROSPECTION_DISABLED"},
            )
