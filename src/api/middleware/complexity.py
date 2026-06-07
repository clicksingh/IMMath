"""Query complexity budget — Strawberry SchemaExtension.

Estimates query cost before execution:
- Scalar field: 1 point
- Connection field: 10 + first argument value
Rejects queries exceeding the tier's complexity budget.
"""

from __future__ import annotations

import logging

from graphql import GraphQLError
from graphql.language import visitor
from strawberry.extensions import SchemaExtension

from ..middleware.api_key import TIER_LIMITS

logger = logging.getLogger(__name__)


class ComplexityExtension(SchemaExtension):
    """Estimate and enforce query complexity budgets."""

    def on_request_start(self) -> None:
        execution_context = self.execution_context
        query = execution_context.graphql_document
        if query is None:
            return

        tier = "anonymous"
        try:
            request = execution_context.context.get("request")
            if request and hasattr(request, "state") and hasattr(request.state, "tier"):
                tier = request.state.tier
        except (AttributeError, TypeError):
            pass

        budget = TIER_LIMITS.get(tier, TIER_LIMITS["anonymous"])["complexity_budget"]

        # Walk AST to calculate complexity
        class ComplexityVisitor(visitor.Visitor):
            def __init__(self):
                self.cost = 0

            def enter_field(self, node, *args, **kwargs):
                field_name = node.name.value
                # Skip introspection fields
                if field_name.startswith("__"):
                    return

                # Check if this is a connection type (has 'first' arg)
                is_connection = False
                first_val = 25  # default
                for arg in node.arguments or []:
                    if arg.name.value == "first":
                        is_connection = True
                        try:
                            first_val = int(arg.value.value)
                        except (AttributeError, ValueError):
                            pass

                # Connection fields have 'Connection' in typical names
                if is_connection or "Connection" in field_name or field_name in (
                    "masterPanel", "counterfactual", "welfareLoss",
                    "decomposition", "cohortNiv",
                ):
                    self.cost += 10 + first_val
                else:
                    self.cost += 1

        complexity_visitor = ComplexityVisitor()
        visitor.visit(query, complexity_visitor)

        if complexity_visitor.cost > budget:
            raise GraphQLError(
                f"Query complexity {complexity_visitor.cost} exceeds "
                f"budget {budget} for tier '{tier}'",
                extensions={
                    "code": "COMPLEXITY_EXCEEDED",
                    "complexity": complexity_visitor.cost,
                    "budget": budget,
                    "tier": tier,
                },
            )
