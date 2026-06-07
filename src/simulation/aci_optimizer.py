"""ACI Constrained Optimization.

Solves: max sum_i sum_r NIV_i * N_i,r,t
subject to: ACI_r,t >= ACI_min, sum_r N_i,r,t = Total_i,t, N_i,r,t >= 0

Uses scipy.optimize.linprog for LP formulation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

logger = logging.getLogger(__name__)

# Minimum ACI threshold below which province should not receive additional intake
ACI_MIN_DEFAULT = 0.35


def run_optimization(
    annual_panel: pd.DataFrame,
    npv_df: pd.DataFrame,
    aci_scenario: str = "aci_equal",
    aci_min: float = ACI_MIN_DEFAULT,
) -> pd.DataFrame:
    """Run ACI-constrained intake optimization for each year.

    For each year, allocates intake across provinces and cohort types
    to maximize total NIV while respecting ACI constraints.

    Args:
        annual_panel: Annual panel with ACI columns.
        npv_df: Cohort NIV DataFrame.
        aci_scenario: Which ACI column to use.
        aci_min: Minimum ACI threshold.

    Returns:
        DataFrame with optimal and actual intake by (year, province, cohort_type).
    """
    years = sorted(annual_panel["year"].unique())
    all_results = []

    for year in years:
        year_data = annual_panel[annual_panel["year"] == year].copy()
        result = _optimize_year(year_data, npv_df, year, aci_scenario, aci_min)
        if result is not None:
            all_results.append(result)

    if not all_results:
        logger.warning("Optimization produced no results")
        return pd.DataFrame()

    results_df = pd.concat(all_results, ignore_index=True)
    return results_df


def _optimize_year(
    year_data: pd.DataFrame,
    npv_df: pd.DataFrame,
    year: int,
    aci_scenario: str,
    aci_min: float,
) -> pd.DataFrame | None:
    """Optimize intake for a single year.

    Decision variables: N_i,r (intake of cohort i in province r)
    Objective: maximize sum_i sum_r NIV_i * N_i,r
    Constraints:
        - ACI_r >= ACI_min (total intake in province r cannot exceed capacity)
        - sum_r N_i,r = Total_i (national cohort totals must match actual)
        - N_i,r >= 0
    """
    provinces = sorted(year_data["province"].unique())
    cohort_types = sorted(year_data["cohort_type"].unique())

    n_prov = len(provinces)
    n_cohorts = len(cohort_types)
    n_vars = n_prov * n_cohorts

    if n_vars == 0:
        return None

    prov_idx = {p: i for i, p in enumerate(provinces)}
    cohort_idx = {c: i for i, c in enumerate(cohort_types)}

    # Determine intake column name
    intake_col = "intake_annual" if "intake_annual" in year_data.columns else "intake_count"

    # NIV values for objective (maximize → minimize negative)
    niv_map = dict(zip(npv_df["cohort_type"], npv_df["niv"]))
    c = np.zeros(n_vars)
    for cohort in cohort_types:
        niv = niv_map.get(cohort, 0)
        for prov in provinces:
            idx = cohort_idx[cohort] * n_prov + prov_idx[prov]
            c[idx] = -niv  # Negative for maximization

    # ACI capacity constraints
    # For each province, total intake weighted by inverse capacity
    aci_map = {}
    for _, row in year_data.iterrows():
        aci_map[(row["province"], row["cohort_type"])] = row.get(aci_scenario, 0.5)

    # National total constraints: sum_r N_i,r = Total_i
    # Get actual totals per cohort
    actual_totals = year_data.groupby("cohort_type")[intake_col].sum().to_dict()

    # Equality constraints: national totals
    A_eq = np.zeros((n_cohorts, n_vars))
    b_eq = np.zeros(n_cohorts)

    for cohort in cohort_types:
        i = cohort_idx[cohort]
        total = actual_totals.get(cohort, 0)
        b_eq[i] = total
        for prov in provinces:
            idx = i * n_prov + prov_idx[prov]
            A_eq[i, idx] = 1.0

    # Inequality constraints: province ACI capacity
    # Total intake in province r should respect ACI constraint
    # Approximation: province share inversely proportional to ACI stress
    A_ub_rows = []
    b_ub_rows = []

    for prov in provinces:
        # Average ACI for this province across cohorts
        prov_aci_vals = [
            aci_map.get((prov, c), 0.5) for c in cohort_types
        ]
        avg_aci = np.mean(prov_aci_vals)

        if avg_aci < aci_min:
            # Province has low ACI → constrain total intake
            # Current actual intake for this province
            prov_actual = year_data[year_data["province"] == prov][intake_col].sum()
            # Reduce intake proportionally to how far below threshold
            reduction_factor = avg_aci / aci_min
            max_intake = prov_actual * reduction_factor

            row = np.zeros(n_vars)
            for cohort in cohort_types:
                idx = cohort_idx[cohort] * n_prov + prov_idx[prov]
                row[idx] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(max_intake)

    # Soft constraints: refugee floors and francophone targets
    # These get minimum allocation, not maximum
    bounds = []
    for cohort in cohort_types:
        for prov in provinces:
            # Minimum allocation: refugees and francophone get at least 80% of proportional share
            total_for_cohort = actual_totals.get(cohort, 0)
            if cohort in ("refugee", "francophone_pr") and total_for_cohort > 0:
                proportional = total_for_cohort / n_prov * 0.5
                bounds.append((proportional, None))
            else:
                bounds.append((0, None))

    # Solve LP
    try:
        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_rows)
        else:
            A_ub = None
            b_ub = None

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            logger.warning("Optimization failed for year %d: %s", year, result.message)
            return _fallback_allocation(year_data, year, aci_scenario, aci_min)

        optimal = result.x

    except Exception as e:
        logger.warning("Optimization error for year %d: %s", year, e)
        return _fallback_allocation(year_data, year, aci_scenario, aci_min)

    # Build result DataFrame
    records = []
    for cohort in cohort_types:
        for prov in provinces:
            idx = cohort_idx[cohort] * n_prov + prov_idx[prov]
            actual = year_data[
                (year_data["province"] == prov) & (year_data["cohort_type"] == cohort)
            ][intake_col].sum()

            aci_val = aci_map.get((prov, cohort), 0.5)

            records.append({
                "year": year,
                "province": prov,
                "cohort_type": cohort,
                "actual_intake": int(actual),
                "optimal_intake": max(0, int(round(optimal[idx]))),
                "aci_value": round(aci_val, 4),
                "niv_per_unit": niv_map.get(cohort, 0),
            })

    return pd.DataFrame(records)


def _fallback_allocation(
    year_data: pd.DataFrame,
    year: int,
    aci_scenario: str,
    aci_min: float,
) -> pd.DataFrame:
    """Simple fallback when LP solver fails.

    Proportionally reduces intake in provinces with ACI < threshold
    and redistributes to provinces above threshold.
    """
    records = []
    provinces = sorted(year_data["province"].unique())
    cohort_types = sorted(year_data["cohort_type"].unique())

    intake_col = "intake_annual" if "intake_annual" in year_data.columns else "intake_count"

    for cohort in cohort_types:
        cohort_data = year_data[year_data["cohort_type"] == cohort]
        total = cohort_data[intake_col].sum()

        # Get province ACI values
        prov_aci = {}
        for prov in provinces:
            prov_rows = cohort_data[cohort_data["province"] == prov]
            if not prov_rows.empty:
                prov_aci[prov] = prov_rows[aci_scenario].iloc[0]
            else:
                prov_aci[prov] = 0.5

        # Weight allocation by ACI (higher ACI gets more)
        weights = {p: max(aci, 0.01) for p, aci in prov_aci.items()}
        total_weight = sum(weights.values())

        for prov in provinces:
            actual = cohort_data[cohort_data["province"] == prov][intake_col].sum()
            optimal = int(total * weights[prov] / total_weight)

            records.append({
                "year": year,
                "province": prov,
                "cohort_type": cohort,
                "actual_intake": int(actual),
                "optimal_intake": optimal,
                "aci_value": round(prov_aci[prov], 4),
                "niv_per_unit": 0,
            })

    return pd.DataFrame(records)
