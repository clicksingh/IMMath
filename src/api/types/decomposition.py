"""GraphQL types for Dimensional Decomposition data."""

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
class DimensionalDecomposition:
    year: int
    dimension: str
    avg_score: float | None
    intake_gap: float | None
    contribution: float | None
    contribution_pct: float | None

    @classmethod
    def from_row(cls, row: pd.Series) -> DimensionalDecomposition:
        return cls(
            year=int(row["year"]),
            dimension=str(row["dimension"]),
            avg_score=_safe_float(row, "avg_score"),
            intake_gap=_safe_float(row, "intake_gap"),
            contribution=_safe_float(row, "contribution"),
            contribution_pct=_safe_float(row, "contribution_pct"),
        )


@strawberry.type
class DecompositionEdge(Edge[DimensionalDecomposition]):
    pass


@strawberry.type
class DecompositionConnection(Connection[DimensionalDecomposition]):
    pass


def paginate_decomposition(
    df: pd.DataFrame,
    first: int = 25,
    after: str | None = None,
) -> DecompositionConnection:
    return paginate_df(df, DimensionalDecomposition.from_row, first=first, after=after)
