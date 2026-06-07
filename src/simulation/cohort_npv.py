"""Cohort Net Present Value (NPV) Calculator.

Computes the Net Immigration Value (NIV) for each cohort type using
vector benefit/cost model. Parameters loaded from cohort_params.yaml.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_cohort_params(config_path: Path) -> dict:
    """Load cohort parameters from YAML config.

    Args:
        config_path: Path to cohort_params.yaml.

    Returns:
        Dictionary of cohort parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Cohort params file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def compute_cohort_npv(cohort_params: dict, discount_rate_override: float | None = None) -> pd.DataFrame:
    """Compute NIV for each cohort type.

    NIV_i = PV(B_i) - PV(C_i)

    Where B_i is the benefit vector (earnings, tax contribution, transition value)
    and C_i is the cost vector (housing, health, education, integrity).

    Args:
        cohort_params: Loaded YAML config dictionary.
        discount_rate_override: Optional override for discount rate.

    Returns:
        DataFrame with cohort NIV breakdown by component.
    """
    records = []

    for cohort_id, params in cohort_params.get("cohort_types", {}).items():
        r = discount_rate_override if discount_rate_override is not None else params.get("discount_rate", 0.03)
        T = params.get("time_horizon_years", 25)

        # Benefits
        annual_earnings = params.get("annual_earnings", params.get("annual_earnings_post_grad", 0))
        tax_contribution = params.get("annual_tax_contribution", 0)
        transition_prob = params.get("transition_probability", 1.0)

        # For students, earnings start after graduation (year 3)
        is_student = "student" in cohort_id
        earnings_start_year = 3 if is_student else 1

        # Present value of earnings stream
        pv_earnings = 0.0
        for t in range(earnings_start_year, T + 1):
            pv_earnings += annual_earnings / ((1 + r) ** t)

        # Transition value (future PR stream)
        # Value of the transition pathway itself
        pv_transition = transition_prob * pv_earnings * params.get("rho_future", 0.3)

        # PV of tax contributions
        pv_tax = 0.0
        for t in range(1, T + 1):
            pv_tax += tax_contribution / ((1 + r) ** t)

        # Total benefits
        pv_benefits = pv_tax + pv_transition

        # Costs
        housing_annual = params.get("housing_consumption_annual", 0)
        health_annual = params.get("health_service_consumption_annual", 0)
        education_annual = params.get("education_infrastructure_cost", 0)
        integrity = params.get("integrity_processing_cost", 0)
        settlement = params.get("settlement_services_annual", 0)

        # PV of costs
        pv_housing = sum(housing_annual / ((1 + r) ** t) for t in range(1, T + 1))
        pv_health = sum(health_annual / ((1 + r) ** t) for t in range(1, T + 1))
        pv_education = sum(education_annual / ((1 + r) ** t) for t in range(1, T + 1))
        pv_integrity = integrity  # One-time cost
        pv_settlement = sum(settlement / ((1 + r) ** t) for t in range(1, T + 1))

        pv_costs = pv_housing + pv_health + pv_education + pv_integrity + pv_settlement

        # NIV
        niv = pv_benefits - pv_costs

        records.append({
            "cohort_type": cohort_id,
            "display_name": params.get("display_name", cohort_id),
            "pv_benefits": round(pv_benefits, 2),
            "pv_tax": round(pv_tax, 2),
            "pv_transition": round(pv_transition, 2),
            "pv_costs": round(pv_costs, 2),
            "pv_housing": round(pv_housing, 2),
            "pv_health": round(pv_health, 2),
            "pv_education": round(pv_education, 2),
            "pv_integrity": round(pv_integrity, 2),
            "pv_settlement": round(pv_settlement, 2),
            "niv": round(niv, 2),
            "transition_probability": transition_prob,
            "time_horizon": T,
            "discount_rate": r,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("niv", ascending=False).reset_index(drop=True)
    return df


def compute_regional_npv_adjustment(
    npv_df: pd.DataFrame,
    master_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Adjust cohort NIV by regional absorption conditions.

    Reduces NIV for cohorts in provinces with low ACI (indicating
    strained infrastructure).

    Args:
        npv_df: Cohort NIV DataFrame.
        master_panel: Master panel with ACI columns.

    Returns:
        DataFrame with region-adjusted NIV by (province, cohort_type).
    """
    # Use the equal-weight ACI for default adjustment
    annual = master_panel.groupby(["year", "province", "cohort_type"]).agg({
        "intake_count": "sum",
        "aci_equal": "first",
    }).reset_index()

    # Merge NIV
    merged = annual.merge(
        npv_df[["cohort_type", "niv"]],
        on="cohort_type",
        how="left",
    )

    # Regional adjustment: scale NIV by ACI
    # Low ACI = less absorption capacity = reduced effective NIV
    # ACI in [0, 1], so multiply directly
    merged["niv_adjusted"] = merged["niv"] * merged["aci_equal"]

    return merged


def run(config_path: Path) -> pd.DataFrame:
    """Run the cohort NPV pipeline.

    Args:
        config_path: Path to cohort_params.yaml.

    Returns:
        DataFrame with NIV for each cohort type.
    """
    params = load_cohort_params(config_path)
    npv_df = compute_cohort_npv(params)

    logger.info("Computed NIV for %d cohort types", len(npv_df))
    for _, row in npv_df.iterrows():
        logger.info("  %s: NIV = $%.0f", row["cohort_type"], row["niv"])

    return npv_df
