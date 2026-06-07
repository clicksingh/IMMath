"""GraphQL types for Cohort NIV (Net Immigration Value) data — 8 rows."""

from __future__ import annotations

from typing import Any

import pandas as pd
import strawberry

from .pagination import Connection, Edge, PageInfo, paginate_df


@strawberry.type
class CohortNIV:
    cohort_type: str
    display_name: str
    pv_benefits: float
    pv_tax: float
    pv_transition: float
    pv_costs: float
    pv_housing: float
    pv_health: float
    pv_education: float
    pv_integrity: float
    pv_settlement: float
    niv: float
    transition_probability: float
    time_horizon: int
    discount_rate: float

    @classmethod
    def from_row(cls, row: pd.Series) -> CohortNIV:
        return cls(
            cohort_type=str(row["cohort_type"]),
            display_name=str(row["display_name"]),
            pv_benefits=float(row["pv_benefits"]),
            pv_tax=float(row["pv_tax"]),
            pv_transition=float(row["pv_transition"]),
            pv_costs=float(row["pv_costs"]),
            pv_housing=float(row["pv_housing"]),
            pv_health=float(row["pv_health"]),
            pv_education=float(row["pv_education"]),
            pv_integrity=float(row["pv_integrity"]),
            pv_settlement=float(row["pv_settlement"]),
            niv=float(row["niv"]),
            transition_probability=float(row["transition_probability"]),
            time_horizon=int(row["time_horizon"]),
            discount_rate=float(row["discount_rate"]),
        )


@strawberry.type
class CohortNIVEdge(Edge[CohortNIV]):
    pass


@strawberry.type
class CohortNIVConnection(Connection[CohortNIV]):
    pass


def paginate_cohort_niv(
    df: pd.DataFrame,
    first: int = 25,
    after: str | None = None,
) -> CohortNIVConnection:
    return paginate_df(df, CohortNIV.from_row, first=first, after=after)
