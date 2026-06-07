"""Data Joiner — Merges all sources into master analytical base.

Produces master_panel.parquet indexed by (year, quarter, province, cohort_type)
with all dimensions from the data pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import ircc_loader, cmhc_loader, statcan_loader, cihi_loader, school_loader

logger = logging.getLogger(__name__)

PROVINCES = [
    "ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "NT", "YT", "NU"
]


def normalize_to_01(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1].

    Handles NaN values by preserving them.
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_aci(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Absorption Capacity Index.

    ACI_r,t = w1*(vacancy_rate) + w2*(starts_per_capita_growth)
              + w3*(health_capacity) + w4*(school_capacity)
              + w5*(job_quality) + w6*(municipal_fiscal_balance)

    All components normalized to [0,1] before weighting.

    Returns:
        DataFrame with ACI columns for three weight scenarios.
    """
    df = df.copy()

    # Normalize components to [0, 1]
    components = {
        "vacancy_rate_norm": normalize_to_01(df["vacancy_rate"]),
        "starts_growth_norm": normalize_to_01(df["starts_per_capita_growth"].fillna(0)),
        "health_capacity_norm": normalize_to_01(df["health_capacity"]),
        "school_capacity_norm": normalize_to_01(df["school_capacity"]),
        "job_quality_norm": normalize_to_01(df["job_quality_index"]),
        "fiscal_balance_norm": normalize_to_01(df["municipal_fiscal_balance_pc"]),
    }

    for name, series in components.items():
        df[name] = series

    # Three weight scenarios
    scenarios = {
        "aci_housing_heavy": {
            "vacancy_rate_norm": 0.25,
            "starts_growth_norm": 0.25,
            "health_capacity_norm": 0.10,
            "school_capacity_norm": 0.10,
            "job_quality_norm": 0.15,
            "fiscal_balance_norm": 0.15,
        },
        "aci_equal": {
            "vacancy_rate_norm": 0.167,
            "starts_growth_norm": 0.167,
            "health_capacity_norm": 0.167,
            "school_capacity_norm": 0.167,
            "job_quality_norm": 0.166,
            "fiscal_balance_norm": 0.166,
        },
        "aci_fiscal_heavy": {
            "vacancy_rate_norm": 0.10,
            "starts_growth_norm": 0.10,
            "health_capacity_norm": 0.15,
            "school_capacity_norm": 0.15,
            "job_quality_norm": 0.25,
            "fiscal_balance_norm": 0.25,
        },
    }

    for scenario_name, weights in scenarios.items():
        df[scenario_name] = sum(
            df[col] * weight for col, weight in weights.items()
        )

    return df


def build_master_panel(
    base_dir: Path,
    raw_dir: Path | None = None,
    cleaned_dir: Path | None = None,
    manual_dir: Path | None = None,
    master_dir: Path | None = None,
) -> pd.DataFrame:
    """Build the master analytical panel from all data sources.

    Pipeline: load all sources → clean → join → compute ACI → save.

    Args:
        base_dir: Project root directory.
        raw_dir: Override for raw data directory.
        cleaned_dir: Override for cleaned data directory.
        manual_dir: Override for manual data directory.
        master_dir: Override for master data directory.

    Returns:
        Master panel DataFrame.
    """
    base_dir = Path(base_dir)
    raw_dir = Path(raw_dir) if raw_dir else base_dir / "data" / "raw"
    cleaned_dir = Path(cleaned_dir) if cleaned_dir else base_dir / "data" / "cleaned"
    manual_dir = Path(manual_dir) if manual_dir else base_dir / "manual_data"
    master_dir = Path(master_dir) if master_dir else base_dir / "data" / "master"

    master_dir.mkdir(parents=True, exist_ok=True)

    # Run each loader
    logger.info("=== Running data pipeline ===")
    ircc_df = ircc_loader.run(raw_dir, cleaned_dir)
    cmhc_df = cmhc_loader.run(raw_dir, cleaned_dir)
    statcan_df = statcan_loader.run(raw_dir, cleaned_dir)
    cihi_df = cihi_loader.run(raw_dir, cleaned_dir, manual_dir)
    school_df = school_loader.run(manual_dir, cleaned_dir)

    # Aggregate IRCC to annual by province (sum across quarters and cohort types)
    ircc_annual = (
        ircc_df
        .groupby(["year", "province"])["intake_count"]
        .sum()
        .reset_index()
        .rename(columns={"intake_count": "total_intake_annual"})
    )

    # Also keep quarterly + cohort breakdown
    ircc_detail = ircc_df.copy()

    # Start with IRCC detail as base
    master = ircc_detail.copy()

    # Aggregate housing/labour/fiscal to annual provincial level
    # Merge CMHC (housing)
    cmhc_annual = cmhc_df[["year", "province", "starts_annual", "starts_per_capita",
                           "starts_per_capita_growth", "vacancy_rate", "avg_rent_2br"]].copy()

    # Merge StatCan (labour + fiscal)
    statcan_annual = statcan_df[["year", "province", "unemployment_rate", "median_wage_hourly",
                                  "job_vacancy_rate", "job_quality_index",
                                  "municipal_fiscal_balance_pc"]].copy()

    # Merge CIHI (health)
    cihi_annual = cihi_df[["year", "province", "health_capacity"]].copy() if not cihi_df.empty else pd.DataFrame()

    # Merge school
    school_annual = school_df[["year", "province", "school_capacity"]].copy() if not school_df.empty else pd.DataFrame()

    # Aggregate intake to annual for merging with annual-only sources
    intake_annual = (
        master
        .groupby(["year", "province", "cohort_type"])["intake_count"]
        .sum()
        .reset_index()
        .rename(columns={"intake_count": "intake_annual"})
    )

    # Build annual panel first
    annual_panel = intake_annual.copy()

    # Merge housing
    annual_panel = annual_panel.merge(cmhc_annual, on=["year", "province"], how="left")

    # Merge labour/fiscal
    annual_panel = annual_panel.merge(statcan_annual, on=["year", "province"], how="left")

    # Merge health
    if not cihi_annual.empty:
        annual_panel = annual_panel.merge(cihi_annual, on=["year", "province"], how="left")
    else:
        annual_panel["health_capacity"] = np.nan

    # Merge school
    if not school_annual.empty:
        annual_panel = annual_panel.merge(school_annual, on=["year", "province"], how="left")
    else:
        annual_panel["school_capacity"] = np.nan

    # Compute ACI
    annual_panel = compute_aci(annual_panel)

    # Flag dimensions with significant NaN
    aci_cols = ["vacancy_rate", "starts_per_capita_growth", "health_capacity",
                "school_capacity", "job_quality_index", "municipal_fiscal_balance_pc"]
    nan_per_row = annual_panel[aci_cols].isna().sum(axis=1)
    annual_panel["aci_nan_flag"] = nan_per_row.apply(
        lambda x: "complete" if x == 0 else f"missing_{x}_of_{len(aci_cols)}"
    )

    # Now expand back to quarterly for IRCC detail
    # Keep annual panel for ACI and annual-only metrics
    # Map quarterly detail onto annual ACI
    master = master.merge(
        annual_panel.drop(columns=["intake_annual"]),
        on=["year", "province", "cohort_type"],
        how="left"
    )

    # Save
    output_path = master_dir / "master_panel.parquet"
    master.to_parquet(output_path, index=False)
    logger.info("Saved master panel to %s (%d rows)", output_path, len(master))

    # Also save annual panel
    annual_path = master_dir / "annual_panel.parquet"
    annual_panel.to_parquet(annual_path, index=False)
    logger.info("Saved annual panel to %s (%d rows)", annual_path, len(annual_panel))

    return master


def run(base_dir: Path) -> pd.DataFrame:
    """Entry point for the full data pipeline."""
    return build_master_panel(base_dir)
