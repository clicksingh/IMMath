"""Welfare Loss Calculator.

Compares actual intake vs counterfactual to compute welfare loss.
Decomposes loss by dimension for the orthogonal residual argument.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_welfare_loss(
    counterfactual_df: pd.DataFrame,
    welfare_weights: dict | None = None,
) -> pd.DataFrame:
    """Compute welfare loss from actual vs optimal allocation.

    welfare_loss_t = w^T . V(x_actual) - w^T . V(x_counterfactual)

    Args:
        counterfactual_df: Counterfactual comparison DataFrame.
        welfare_weights: Weight dictionary for dimensions.

    Returns:
        DataFrame with welfare loss by year and dimension.
    """
    if welfare_weights is None:
        welfare_weights = {"equal": 1.0}

    records = []

    for year in sorted(counterfactual_df["year"].unique()):
        year_data = counterfactual_df[counterfactual_df["year"] == year]

        # Total welfare = sum of NIV * intake
        actual_welfare = (year_data["niv_per_unit"] * year_data["actual_intake"]).sum()
        optimal_welfare = (year_data["niv_per_unit"] * year_data["optimal_intake"]).sum()

        # Welfare loss (positive means actual was worse than optimal)
        total_loss = optimal_welfare - actual_welfare

        # Decompose by province
        for province in year_data["province"].unique():
            prov_data = year_data[year_data["province"] == province]

            prov_actual_w = (prov_data["niv_per_unit"] * prov_data["actual_intake"]).sum()
            prov_optimal_w = (prov_data["niv_per_unit"] * prov_data["optimal_intake"]).sum()

            records.append({
                "year": year,
                "province": province,
                "actual_welfare": round(prov_actual_w, 2),
                "optimal_welfare": round(prov_optimal_w, 2),
                "welfare_loss": round(prov_optimal_w - prov_actual_w, 2),
                "actual_total_intake": prov_data["actual_intake"].sum(),
                "optimal_total_intake": prov_data["optimal_intake"].sum(),
                "intake_gap": prov_data["optimal_intake"].sum() - prov_data["actual_intake"].sum(),
            })

    df = pd.DataFrame(records)
    return df


def decompose_by_dimension(
    counterfactual_df: pd.DataFrame,
    master_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose welfare loss by ACI dimension.

    Identifies which dimensions account for the largest gap between
    actual and counterfactual allocation.

    Args:
        counterfactual_df: Counterfactual comparison data.
        master_panel: Master panel with ACI component columns.

    Returns:
        DataFrame with dimensional decomposition.
    """
    # Get normalized ACI components from master panel
    component_cols = [
        "vacancy_rate_norm", "starts_growth_norm", "health_capacity_norm",
        "school_capacity_norm", "job_quality_norm", "fiscal_balance_norm",
    ]

    dim_names = [
        "housing_vacancy", "housing_starts", "health_capacity",
        "school_capacity", "job_quality", "fiscal_balance",
    ]

    records = []

    for year in sorted(counterfactual_df["year"].unique()):
        year_cf = counterfactual_df[counterfactual_df["year"] == year]
        year_panel = master_panel[master_panel["year"] == year]

        for dim_col, dim_name in zip(component_cols, dim_names):
            # Average dimension score across provinces
            if dim_col in year_panel.columns:
                dim_score = year_panel[dim_col].mean()

                # Intake gap (how much actual deviated from optimal)
                intake_gap = year_cf["optimal_intake"].sum() - year_cf["actual_intake"].sum()

                # Dimensional contribution to loss
                # Proportional to how far the dimension is from optimal
                dim_contribution = (1.0 - dim_score) * abs(intake_gap) if not np.isnan(dim_score) else 0

                records.append({
                    "year": year,
                    "dimension": dim_name,
                    "avg_score": round(dim_score, 4) if not np.isnan(dim_score) else None,
                    "intake_gap": intake_gap,
                    "contribution": round(dim_contribution, 2),
                })

    df = pd.DataFrame(records)

    # Normalize contributions to percentages
    if not df.empty:
        total_by_year = df.groupby("year")["contribution"].transform("sum")
        df["contribution_pct"] = (
            df["contribution"] / total_by_year.replace(0, 1) * 100
        ).round(2)

    return df


def export_welfare_loss(
    welfare_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Export welfare loss results.

    Returns:
        Tuple of (welfare_loss_path, decomposition_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_path = output_dir / "welfare_loss_decomposition.csv"
    welfare_df.to_csv(loss_path, index=False)
    logger.info("Exported welfare loss to %s", loss_path)

    decomp_path = output_dir / "dimensional_decomposition.csv"
    decomposition_df.to_csv(decomp_path, index=False)
    logger.info("Exported dimensional decomposition to %s", decomp_path)

    return loss_path, decomp_path


def run(
    counterfactuals: dict[str, pd.DataFrame],
    master_panel: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full welfare loss pipeline.

    Returns:
        Tuple of (welfare_loss_df, decomposition_df).
    """
    # Use the equal-weight scenario as primary
    primary = counterfactuals.get("aci_equal")
    if primary is None:
        logger.error("No aci_equal counterfactual found")
        return pd.DataFrame(), pd.DataFrame()

    welfare_df = compute_welfare_loss(primary)
    decomposition_df = decompose_by_dimension(primary, master_panel)

    export_welfare_loss(welfare_df, decomposition_df, output_dir)

    return welfare_df, decomposition_df
