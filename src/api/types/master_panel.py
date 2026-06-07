"""GraphQL types for Master Panel data — the core analytical base.

Groups 27 columns into nested types for cleaner queries and meaningful
depth limiting.
"""

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
class MasterPanelHousing:
    starts_annual: float | None
    starts_per_capita: float | None
    starts_per_capita_growth: float | None
    vacancy_rate: float | None
    avg_rent_2br: float | None


@strawberry.type
class MasterPanelLabour:
    unemployment_rate: float | None
    median_wage_hourly: float | None
    job_vacancy_rate: float | None
    job_quality_index: float | None


@strawberry.type
class MasterPanelCapacities:
    health_capacity: float | None
    school_capacity: float | None


@strawberry.type
class NormalizedDims:
    vacancy_rate_norm: float | None
    starts_growth_norm: float | None
    health_capacity_norm: float | None
    school_capacity_norm: float | None
    job_quality_norm: float | None
    fiscal_balance_norm: float | None


@strawberry.type
class ACIScores:
    housing_heavy: float | None
    equal: float | None
    fiscal_heavy: float | None


@strawberry.type
class MasterPanel:
    year: int
    quarter: int
    province: str
    cohort_type: str
    intake_count: int | None
    housing: MasterPanelHousing
    labour: MasterPanelLabour
    capacities: MasterPanelCapacities
    normalized: NormalizedDims
    aci: ACIScores
    aci_nan_flag: str | None

    @classmethod
    def from_row(cls, row: pd.Series) -> MasterPanel:
        intake = row.get("intake_count")
        return cls(
            year=int(row["year"]),
            quarter=int(row["quarter"]),
            province=str(row["province"]),
            cohort_type=str(row["cohort_type"]),
            intake_count=int(intake) if pd.notna(intake) else None,
            housing=MasterPanelHousing(
                starts_annual=_safe_float(row, "starts_annual"),
                starts_per_capita=_safe_float(row, "starts_per_capita"),
                starts_per_capita_growth=_safe_float(row, "starts_per_capita_growth"),
                vacancy_rate=_safe_float(row, "vacancy_rate"),
                avg_rent_2br=_safe_float(row, "avg_rent_2br"),
            ),
            labour=MasterPanelLabour(
                unemployment_rate=_safe_float(row, "unemployment_rate"),
                median_wage_hourly=_safe_float(row, "median_wage_hourly"),
                job_vacancy_rate=_safe_float(row, "job_vacancy_rate"),
                job_quality_index=_safe_float(row, "job_quality_index"),
            ),
            capacities=MasterPanelCapacities(
                health_capacity=_safe_float(row, "health_capacity"),
                school_capacity=_safe_float(row, "school_capacity"),
            ),
            normalized=NormalizedDims(
                vacancy_rate_norm=_safe_float(row, "vacancy_rate_norm"),
                starts_growth_norm=_safe_float(row, "starts_growth_norm"),
                health_capacity_norm=_safe_float(row, "health_capacity_norm"),
                school_capacity_norm=_safe_float(row, "school_capacity_norm"),
                job_quality_norm=_safe_float(row, "job_quality_norm"),
                fiscal_balance_norm=_safe_float(row, "fiscal_balance_norm"),
            ),
            aci=ACIScores(
                housing_heavy=_safe_float(row, "aci_housing_heavy"),
                equal=_safe_float(row, "aci_equal"),
                fiscal_heavy=_safe_float(row, "aci_fiscal_heavy"),
            ),
            aci_nan_flag=str(row["aci_nan_flag"]) if pd.notna(row.get("aci_nan_flag")) else None,
        )


@strawberry.type
class MasterPanelEdge(Edge[MasterPanel]):
    pass


@strawberry.type
class MasterPanelConnection(Connection[MasterPanel]):
    pass


def paginate_master_panel(
    df: pd.DataFrame,
    first: int = 25,
    after: str | None = None,
) -> MasterPanelConnection:
    return paginate_df(df, MasterPanel.from_row, first=first, after=after)
