"""Strawberry GraphQL schema — 6 query methods for ACI research data."""

from __future__ import annotations

import strawberry

from .services import data_loader
from .types.counterfactual import CounterfactualConnection, paginate_counterfactual
from .types.cohort_niv import CohortNIVConnection, paginate_cohort_niv
from .types.decomposition import DecompositionConnection, paginate_decomposition
from .types.lambda_result import LambdaResult
from .types.master_panel import MasterPanelConnection, paginate_master_panel
from .types.welfare import WelfareLossConnection, paginate_welfare


def _base_dir_from_info(info: strawberry.types.Info) -> str | None:
    """Extract base_dir from FastAPI request state if available."""
    try:
        return info.context.get("request", {}).get("state", {}).get("base_dir")
    except (AttributeError, TypeError):
        return None


@strawberry.type
class Query:
    @strawberry.field
    def health(self) -> str:
        return "ok"

    @strawberry.field
    def master_panel(
        self,
        info: strawberry.types.Info,
        province: str | None = None,
        cohort_type: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        first: int = 25,
        after: str | None = None,
    ) -> MasterPanelConnection:
        df = data_loader.get_master_panel(_base_dir_from_info(info))
        if province:
            df = df[df["province"] == province]
        if cohort_type:
            df = df[df["cohort_type"] == cohort_type]
        if year_min is not None:
            df = df[df["year"] >= year_min]
        if year_max is not None:
            df = df[df["year"] <= year_max]
        return paginate_master_panel(df, first=first, after=after)

    @strawberry.field
    def counterfactual(
        self,
        info: strawberry.types.Info,
        province: str | None = None,
        cohort_type: str | None = None,
        scenario: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        first: int = 25,
        after: str | None = None,
    ) -> CounterfactualConnection:
        df = data_loader.get_counterfactual(_base_dir_from_info(info))
        if province:
            df = df[df["province"] == province]
        if cohort_type:
            df = df[df["cohort_type"] == cohort_type]
        if scenario:
            df = df[df["scenario"] == scenario]
        if year_min is not None:
            df = df[df["year"] >= year_min]
        if year_max is not None:
            df = df[df["year"] <= year_max]
        return paginate_counterfactual(df, first=first, after=after)

    @strawberry.field
    def welfare_loss(
        self,
        info: strawberry.types.Info,
        province: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        first: int = 25,
        after: str | None = None,
    ) -> WelfareLossConnection:
        df = data_loader.get_welfare_loss(_base_dir_from_info(info))
        if province:
            df = df[df["province"] == province]
        if year_min is not None:
            df = df[df["year"] >= year_min]
        if year_max is not None:
            df = df[df["year"] <= year_max]
        return paginate_welfare(df, first=first, after=after)

    @strawberry.field
    def decomposition(
        self,
        info: strawberry.types.Info,
        dimension: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        first: int = 25,
        after: str | None = None,
    ) -> DecompositionConnection:
        df = data_loader.get_decomposition(_base_dir_from_info(info))
        if dimension:
            df = df[df["dimension"] == dimension]
        if year_min is not None:
            df = df[df["year"] >= year_min]
        if year_max is not None:
            df = df[df["year"] <= year_max]
        return paginate_decomposition(df, first=first, after=after)

    @strawberry.field
    def cohort_niv(
        self,
        info: strawberry.types.Info,
        cohort_type: str | None = None,
        first: int = 25,
        after: str | None = None,
    ) -> CohortNIVConnection:
        df = data_loader.get_cohort_niv(_base_dir_from_info(info))
        if cohort_type:
            df = df[df["cohort_type"] == cohort_type]
        return paginate_cohort_niv(df, first=first, after=after)

    @strawberry.field
    def lambda_results(
        self,
        info: strawberry.types.Info,
    ) -> list[LambdaResult]:
        df = data_loader.get_lambda_results(_base_dir_from_info(info))
        return [LambdaResult.from_row(row) for _, row in df.iterrows()]


def create_schema(debug: bool = False) -> strawberry.Schema:
    """Create the GraphQL schema with optional protection extensions."""
    extensions = []
    extensions.append(
        __import__("src.api.middleware.depth_limit", fromlist=["DepthLimitExtension"]).DepthLimitExtension
    )
    extensions.append(
        __import__("src.api.middleware.complexity", fromlist=["ComplexityExtension"]).ComplexityExtension
    )
    if not debug:
        extensions.append(
            __import__("src.api.middleware.introspection", fromlist=["IntrospectionGuard"]).IntrospectionGuard
        )

    return strawberry.Schema(
        query=Query,
        extensions=extensions,
    )


# Default schema for import convenience (debug mode)
schema = create_schema(debug=True)
