"""GraphQL types for Counterfactual Series data."""

from __future__ import annotations

import pandas as pd
import strawberry

from .pagination import Connection, Edge, paginate_df


def _safe_float(row: pd.Series, col: str) -> float | None:
    v = row.get(col)
    if pd.isna(v):
        return None
    return float(v)


@strawberry.type
class Counterfactual:
    year: int
    province: str
    cohort_type: str
    actual_intake: float | None
    optimal_intake: float | None
    aci_value: float | None
    niv_per_unit: float | None
    scenario: str

    @classmethod
    def from_row(cls, row: pd.Series) -> Counterfactual:
        return cls(
            year=int(row["year"]),
            province=str(row["province"]),
            cohort_type=str(row["cohort_type"]),
            actual_intake=_safe_float(row, "actual_intake"),
            optimal_intake=_safe_float(row, "optimal_intake"),
            aci_value=_safe_float(row, "aci_value"),
            niv_per_unit=_safe_float(row, "niv_per_unit"),
            scenario=str(row["scenario"]),
        )


@strawberry.type
class CounterfactualEdge(Edge[Counterfactual]):
    pass


@strawberry.type
class CounterfactualConnection(Connection[Counterfactual]):
    pass


def paginate_counterfactual(
    df: pd.DataFrame,
    first: int = 25,
    after: str | None = None,
) -> CounterfactualConnection:
    return paginate_df(df, Counterfactual.from_row, first=first, after=after)
