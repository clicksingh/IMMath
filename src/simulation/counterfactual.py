"""Counterfactual Intake Generator.

Produces the counterfactual intake series: what intake *should* have
looked like had ACI constraints been binding, vs what actually happened.
"""

import logging
from pathlib import Path

import pandas as pd

from .aci_optimizer import run_optimization

logger = logging.getLogger(__name__)


def generate_counterfactual(
    annual_panel: pd.DataFrame,
    npv_df: pd.DataFrame,
    aci_scenarios: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate counterfactual intake for all ACI scenarios.

    Args:
        annual_panel: Annual panel with ACI columns.
        npv_df: Cohort NIV DataFrame.
        aci_scenarios: List of ACI column names to run.

    Returns:
        Dictionary mapping scenario name to counterfactual DataFrame.
    """
    if aci_scenarios is None:
        aci_scenarios = ["aci_housing_heavy", "aci_equal", "aci_fiscal_heavy"]

    results = {}

    for scenario in aci_scenarios:
        logger.info("Running counterfactual for scenario: %s", scenario)
        cf = run_optimization(annual_panel, npv_df, aci_scenario=scenario)
        if not cf.empty:
            cf["scenario"] = scenario
            results[scenario] = cf
            logger.info(
                "  %s: %d records, total optimal = %d, total actual = %d",
                scenario, len(cf),
                cf["optimal_intake"].sum(),
                cf["actual_intake"].sum(),
            )

    return results


def compute_counterfactual_summary(
    counterfactuals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize counterfactual vs actual intake by year.

    Returns:
        DataFrame with yearly comparison across scenarios.
    """
    summaries = []

    for scenario, cf in counterfactuals.items():
        yearly = cf.groupby("year").agg({
            "actual_intake": "sum",
            "optimal_intake": "sum",
        }).reset_index()

        yearly["scenario"] = scenario
        yearly["delta"] = yearly["optimal_intake"] - yearly["actual_intake"]
        yearly["delta_pct"] = (
            (yearly["optimal_intake"] - yearly["actual_intake"])
            / yearly["actual_intake"].replace(0, 1) * 100
        )

        summaries.append(yearly)

    return pd.concat(summaries, ignore_index=True)


def export_counterfactual_series(
    counterfactuals: dict[str, pd.DataFrame],
    output_dir: Path,
) -> Path:
    """Export counterfactual series as CSV.

    Args:
        counterfactuals: Dictionary of scenario → DataFrame.
        output_dir: Output directory.

    Returns:
        Path to exported CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all scenarios
    all_cf = pd.concat(counterfactuals.values(), ignore_index=True)

    # Sort and export
    all_cf = all_cf.sort_values(["scenario", "year", "province", "cohort_type"])
    output_path = output_dir / "counterfactual_series.csv"
    all_cf.to_csv(output_path, index=False)

    logger.info("Exported counterfactual series to %s (%d rows)", output_path, len(all_cf))

    # Also export summary
    summary = compute_counterfactual_summary(counterfactuals)
    summary_path = output_dir / "counterfactual_summary.csv"
    summary.to_csv(summary_path, index=False)

    return output_path


def run(
    annual_panel_path: Path,
    npv_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Run the full counterfactual pipeline.

    Args:
        annual_panel_path: Path to annual_panel.parquet.
        npv_df: Cohort NIV DataFrame.
        output_dir: Output directory for CSVs.

    Returns:
        Dictionary of scenario → counterfactual DataFrame.
    """
    annual_panel = pd.read_parquet(annual_panel_path)

    counterfactuals = generate_counterfactual(annual_panel, npv_df)
    export_counterfactual_series(counterfactuals, output_dir)

    return counterfactuals
