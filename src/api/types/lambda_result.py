"""GraphQL types for Lambda regression results (8 rows, no pagination)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import strawberry


@strawberry.type
class LambdaResult:
    variable: str
    coefficient: float | None
    std_error: float | None
    t_statistic: float | None
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None
    significant_10pct: bool | None
    notes: str | None

    @classmethod
    def from_row(cls, row: pd.Series) -> LambdaResult:
        def _float(v: Any) -> float | None:
            if pd.isna(v):
                return None
            return float(v)

        def _str(v: Any) -> str | None:
            if pd.isna(v):
                return None
            return str(v)

        return cls(
            variable=str(row.get("variable", "")),
            coefficient=_float(row.get("coefficient")),
            std_error=_float(row.get("std_error")),
            t_statistic=_float(row.get("t_statistic")),
            p_value=_float(row.get("p_value")),
            ci_lower=_float(row.get("ci_lower")),
            ci_upper=_float(row.get("ci_upper")),
            significant_10pct=bool(row["significant_10pct"]) if pd.notna(row.get("significant_10pct")) else None,
            notes=_str(row.get("notes")),
        )
