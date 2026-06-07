"""GraphQL types for Welfare Loss decomposition data."""

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
class WelfareLoss:
    year: int
    province: str
    actual_welfare: float | None
    optimal_welfare: float | None
    welfare_loss: float | None
    actual_total_intake: float | None
    optimal_total_intake: float | None
    intake_gap: float | None

    @classmethod
    def from_row(cls, row: pd.Series) -> WelfareLoss:
        return cls(
            year=int(row["year"]),
            province=str(row["province"]),
            actual_welfare=_safe_float(row, "actual_welfare"),
            optimal_welfare=_safe_float(row, "optimal_welfare"),
            welfare_loss=_safe_float(row, "welfare_loss"),
            actual_total_intake=_safe_float(row, "actual_total_intake"),
            optimal_total_intake=_safe_float(row, "optimal_total_intake"),
            intake_gap=_safe_float(row, "intake_gap"),
        )


@strawberry.type
class WelfareLossEdge(Edge[WelfareLoss]):
    pass


@strawberry.type
class WelfareLossConnection(Connection[WelfareLoss]):
    pass


def paginate_welfare(
    df: pd.DataFrame,
    first: int = 25,
    after: str | None = None,
) -> WelfareLossConnection:
    return paginate_df(df, WelfareLoss.from_row, first=first, after=after)
