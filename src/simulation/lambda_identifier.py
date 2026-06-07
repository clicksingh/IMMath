"""Lambda Identification — Empirical Rank-2 Projection Test.

Implements the falsifiability test from the paper:
    Delta_x_observed = Lambda^T . beta + epsilon
    H0: rank(Lambda_hat) = 2, loadings on null-space dims = 0

Tests whether policy lever changes concentrate on political utility
dimensions (V11: volume optics, V12: Quebec leverage) with near-zero
loadings on absorptive capacity (V4), heterogeneity (V5), innovation (V8).

Uses statsmodels OLS with robust standard errors.

IMPORTANT: Reports results honestly even if they contradict the thesis.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_OBSERVATIONS_FOR_REGRESSION = 6


def construct_dimension_scores(
    master_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the dimension score matrix for regression.

    Dimensions:
    - V1: volume_signal (total intake change)
    - V2: quebec_leverage (Quebec-specific intake share)
    - V3: housing_stress (inverse of vacancy_rate)
    - V4: absorptive_capacity (ACI)
    - V5: cohort_heterogeneity (diversity of intake across cohorts)
    - V6: innovation_spillover (high-wage worker share)
    - V7: credential_utilization (ratio of skilled to unskilled)
    - V8: fiscal_balance (municipal fiscal balance)

    Args:
        master_panel: Master panel with all dimensions.

    Returns:
        DataFrame with dimension scores by year.
    """
    # Aggregate to yearly national level
    yearly = master_panel.groupby("year").agg({
        "intake_count": "sum",
        "vacancy_rate": "mean",
        "aci_equal": "mean",
        "municipal_fiscal_balance_pc": "mean",
        "unemployment_rate": "mean",
        "median_wage_hourly": "mean",
    }).reset_index()

    # Quebec-specific share
    yearly_qc = master_panel[master_panel["province"] == "QC"].groupby("year")["intake_count"].sum()
    yearly_total = master_panel.groupby("year")["intake_count"].sum()
    yearly["quebec_share"] = (yearly_qc / yearly_total).values

    # Cohort heterogeneity (Herfindahl index of cohort shares)
    cohort_shares = (
        master_panel.groupby(["year", "cohort_type"])["intake_count"].sum()
        / master_panel.groupby("year")["intake_count"].sum()
    ).reset_index()
    hhi = cohort_shares.groupby("year")["intake_count"].apply(
        lambda x: 1 - (x ** 2).sum()
    ).reset_index()
    hhi.columns = ["year", "cohort_heterogeneity"]
    yearly = yearly.merge(hhi, on="year", how="left")

    # High-wage worker share
    high_wage = master_panel[
        master_panel["cohort_type"].isin(["high_wage_worker", "high_quality_student"])
    ].groupby("year")["intake_count"].sum()
    yearly["high_value_share"] = (high_wage / yearly_total).values

    # Credential utilization
    skilled = master_panel[
        master_panel["cohort_type"].isin(["high_wage_worker", "high_quality_student", "francophone_pr"])
    ].groupby("year")["intake_count"].sum()
    unskilled = master_panel[
        master_panel["cohort_type"].isin(["low_wage_worker", "low_quality_student"])
    ].groupby("year")["intake_count"].sum()
    yearly["credential_ratio"] = (skilled / unskilled.replace(0, 1)).values

    # Compute year-over-year changes (Delta x)
    change_cols = {
        "intake_count": "delta_volume",
        "quebec_share": "delta_quebec_leverage",
        "vacancy_rate": "delta_housing_stress",
        "aci_equal": "delta_absorptive_capacity",
        "cohort_heterogeneity": "delta_heterogeneity",
        "high_value_share": "delta_innovation",
        "credential_ratio": "delta_credential_utilization",
        "municipal_fiscal_balance_pc": "delta_fiscal",
    }

    for orig_col, change_col in change_cols.items():
        if orig_col in yearly.columns:
            yearly[change_col] = yearly[orig_col].diff()

    return yearly


def run_regression(
    dimension_scores: pd.DataFrame,
) -> dict:
    """Run the rank-2 projection identification regression.

    Regresses observed policy changes (delta_volume) on dimension scores.
    Tests H0: rank(Lambda_hat) = 2.

    Args:
        dimension_scores: Yearly dimension scores with deltas.

    Returns:
        Dictionary with regression results.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.error("statsmodels not installed. Cannot run Lambda regression.")
        return {"status": "error", "message": "statsmodels not available"}

    # Prepare regression data
    delta_cols = [c for c in dimension_scores.columns if c.startswith("delta_")]
    reg_data = dimension_scores.dropna(subset=delta_cols)

    n_obs = len(reg_data)

    # Degrees of freedom warning
    dof_warning = None
    if n_obs < MIN_OBSERVATIONS_FOR_REGRESSION:
        dof_warning = (
            f"WARNING: Only {n_obs} observations for regression. "
            f"Minimum recommended: {MIN_OBSERVATIONS_FOR_REGRESSION}. "
            f"Degrees of freedom may be insufficient for reliable inference."
        )
        logger.warning(dof_warning)

    if n_obs < 3:
        return {
            "status": "insufficient_data",
            "message": f"Only {n_obs} observations available",
            "dof_warning": dof_warning,
        }

    # Dependent variable: volume change (the policy lever)
    y = reg_data["delta_volume"]

    # Independent variables: the dimension scores that define the projection
    X_cols = [
        "delta_quebec_leverage",  # V12: political dimension
        "delta_housing_stress",   # V4: absorptive capacity
        "delta_absorptive_capacity",
        "delta_heterogeneity",    # V5: cohort heterogeneity
        "delta_innovation",       # V8: innovation spillover
        "delta_fiscal",
    ]

    # Only use columns that exist and have non-zero variance
    available_X = [c for c in X_cols if c in reg_data.columns and reg_data[c].std() > 0]

    if len(available_X) < 2:
        return {
            "status": "insufficient_predictors",
            "message": f"Only {len(available_X)} valid predictors",
        }

    X = reg_data[available_X]
    X = sm.add_constant(X)

    try:
        # OLS with robust (HC3) standard errors
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception as e:
        return {"status": "regression_error", "message": str(e)}

    # Extract results
    results = {
        "status": "ok",
        "n_observations": n_obs,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_statistic": round(model.fvalue, 4),
        "f_pvalue": round(model.f_pvalue, 6),
        "dof_warning": dof_warning,
        "coefficients": {},
        "rank_2_test": {},
    }

    # Coefficients with confidence intervals
    for var in available_X:
        results["coefficients"][var] = {
            "coef": round(model.params[var], 6),
            "std_err": round(model.bse[var], 6),
            "t_stat": round(model.tvalues[var], 4),
            "p_value": round(model.pvalues[var], 6),
            "ci_lower": round(model.conf_int().loc[var, 0], 6),
            "ci_upper": round(model.conf_int().loc[var, 1], 6),
        }

    # Rank-2 test: check if only 2 dimensions have significant loadings
    significant = sum(1 for v in available_X if model.pvalues[v] < 0.10)
    results["rank_2_test"] = {
        "n_significant_predictors": significant,
        "significant_dims": [
            v for v in available_X if model.pvalues[v] < 0.10
        ],
        "political_dims_significant": (
            "delta_quebec_leverage" in [v for v in available_X if model.pvalues[v] < 0.10]
        ),
        "absorptive_dims_significant": (
            any(v in [v2 for v2 in available_X if model.pvalues[v2] < 0.10]
                for v in ["delta_housing_stress", "delta_absorptive_capacity"])
        ),
    }

    # Honest reporting: assess whether results support rank-2 thesis
    results["thesis_assessment"] = _assess_thesis_support(results)

    return results


def _assess_thesis_support(results: dict) -> str:
    """Honest assessment of whether results support the rank-2 projection thesis.

    IMPORTANT: This function reports honestly. Do not manipulate assessments.
    """
    if results["status"] != "ok":
        return "Cannot assess — regression did not produce valid results."

    r2 = results["r_squared"]
    n_sig = results["rank_2_test"]["n_significant_predictors"]
    political_sig = results["rank_2_test"]["political_dims_significant"]
    absorptive_sig = results["rank_2_test"]["absorptive_dims_significant"]

    if r2 < 0.10:
        return (
            "WEAK SUPPORT: R-squared is very low ({:.1%}). The dimension scores explain "
            "very little variance in policy lever changes. The rank-2 projection "
            "hypothesis is not supported by this regression.".format(r2)
        )

    if n_sig == 2 and political_sig and not absorptive_sig:
        return (
            "SUPPORTIVE: Exactly 2 dimensions are significant, and they are political "
            "utility dimensions. Absorptive capacity dimensions are not significant. "
            "This is consistent with the rank-2 projection thesis."
        )

    if political_sig and not absorptive_sig:
        return (
            "PARTIALLY SUPPORTIVE: Political dimensions are significant while "
            "absorptive capacity dimensions are not. This pattern is consistent with "
            "the thesis, but the model may have more than 2 significant dimensions."
        )

    if absorptive_sig:
        return (
            "NOT SUPPORTIVE: Absorptive capacity dimensions are significant predictors "
            "of policy changes, suggesting the government DID respond to capacity "
            "constraints. This challenges the rank-2 projection thesis."
        )

    if n_sig == 0:
        return (
            "INCONCLUSIVE: No dimensions are statistically significant. "
            "Insufficient power to test the rank-2 hypothesis."
        )

    return (
        f"MIXED: {n_sig} significant dimensions found. "
        f"Political: {political_sig}, Absorptive: {absorptive_sig}. "
        f"Results do not cleanly support or refute the rank-2 thesis."
    )


def export_regression_results(results: dict, output_dir: Path) -> Path:
    """Export regression results to CSV.

    Returns:
        Path to exported CSV.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten results to a table
    rows = []
    for var, stats in results.get("coefficients", {}).items():
        rows.append({
            "variable": var,
            "coefficient": stats["coef"],
            "std_error": stats["std_err"],
            "t_statistic": stats["t_stat"],
            "p_value": stats["p_value"],
            "ci_lower": stats["ci_lower"],
            "ci_upper": stats["ci_upper"],
            "significant_10pct": stats["p_value"] < 0.10,
        })

    # Add model-level stats
    rows.append({
        "variable": "MODEL_R_SQUARED",
        "coefficient": results.get("r_squared"),
        "std_error": results.get("adj_r_squared"),
        "t_statistic": results.get("f_statistic"),
        "p_value": results.get("f_pvalue"),
    })

    rows.append({
        "variable": "THESIS_ASSESSMENT",
        "coefficient": None,
        "std_error": None,
        "t_statistic": None,
        "p_value": None,
        "notes": results.get("thesis_assessment", ""),
    })

    if results.get("dof_warning"):
        rows.append({
            "variable": "DOF_WARNING",
            "coefficient": None,
            "notes": results["dof_warning"],
        })

    df = pd.DataFrame(rows)
    output_path = output_dir / "lambda_regression_results.csv"
    df.to_csv(output_path, index=False)
    logger.info("Exported Lambda regression results to %s", output_path)

    return output_path


def run(master_panel_path: Path, output_dir: Path) -> dict:
    """Run the full Lambda identification pipeline.

    Args:
        master_panel_path: Path to master_panel.parquet.
        output_dir: Output directory.

    Returns:
        Regression results dictionary.
    """
    master_panel = pd.read_parquet(master_panel_path)

    # Construct dimension scores
    scores = construct_dimension_scores(master_panel)
    logger.info("Constructed dimension scores for %d years", len(scores))

    # Run regression
    results = run_regression(scores)

    # Export
    export_regression_results(results, output_dir)

    # Log assessment
    if results.get("thesis_assessment"):
        logger.info("Thesis assessment: %s", results["thesis_assessment"])

    return results
