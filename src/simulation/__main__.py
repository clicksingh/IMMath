"""Simulation Runner — Orchestrates the full simulation pipeline.

Runs: cohort NPV → ACI optimization → counterfactual → welfare loss → Lambda ID
"""

import logging
from pathlib import Path

import pandas as pd

from .cohort_npv import run as run_npv
from .cohort_npv import compute_cohort_npv
from .counterfactual import run as run_counterfactual
from .welfare_loss import run as run_welfare_loss
from .lambda_identifier import run as run_lambda

logger = logging.getLogger(__name__)


def run_simulation(base_dir: Path) -> dict:
    """Run the full simulation pipeline.

    Args:
        base_dir: Project root directory.

    Returns:
        Dictionary with all simulation outputs.
    """
    base_dir = Path(base_dir)
    config_path = base_dir / "config" / "cohort_params.yaml"
    master_path = base_dir / "data" / "master" / "master_panel.parquet"
    annual_path = base_dir / "data" / "master" / "annual_panel.parquet"
    output_dir = base_dir / "outputs" / "data"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Cohort NPV
    logger.info("=== Step 1: Computing cohort NPV ===")
    npv_df = run_npv(config_path)
    npv_df.to_csv(output_dir / "cohort_niv.csv", index=False)

    # Step 2: Counterfactual
    logger.info("=== Step 2: Generating counterfactual intake ===")
    counterfactuals = run_counterfactual(annual_path, npv_df, output_dir)

    # Step 3: Welfare Loss
    logger.info("=== Step 3: Computing welfare loss ===")
    master_panel = pd.read_parquet(master_path)
    welfare_df, decomp_df = run_welfare_loss(counterfactuals, master_panel, output_dir)

    # Step 4: Lambda Identification
    logger.info("=== Step 4: Running Lambda identification ===")
    lambda_results = run_lambda(master_path, output_dir)

    # Generate summary stats
    _generate_summary_stats(npv_df, counterfactuals, welfare_df, lambda_results, output_dir)

    return {
        "npv": npv_df,
        "counterfactuals": counterfactuals,
        "welfare": welfare_df,
        "decomposition": decomp_df,
        "lambda": lambda_results,
    }


def _generate_summary_stats(
    npv_df: pd.DataFrame,
    counterfactuals: dict,
    welfare_df: pd.DataFrame,
    lambda_results: dict,
    output_dir: Path,
) -> None:
    """Generate summary_stats.txt with key numbers for paper citations."""
    lines = [
        "=" * 60,
        "ACI Research Tool — Summary Statistics",
        "=" * 60,
        "",
        "COHORT NIV RANKINGS",
        "-" * 40,
    ]

    for _, row in npv_df.iterrows():
        lines.append(f"  {row['display_name']:40s}  NIV: ${row['niv']:>12,.0f}")

    lines.extend(["", "COUNTERFACTUAL VS ACTUAL (aci_equal scenario)", "-" * 40])

    if "aci_equal" in counterfactuals:
        cf = counterfactuals["aci_equal"]
        yearly = cf.groupby("year").agg({
            "actual_intake": "sum",
            "optimal_intake": "sum",
        }).reset_index()

        for _, row in yearly.iterrows():
            delta = row["optimal_intake"] - row["actual_intake"]
            pct = delta / row["actual_intake"] * 100 if row["actual_intake"] > 0 else 0
            lines.append(
                f"  {row['year']}: Actual={row['actual_intake']:>8,}  "
                f"Optimal={row['optimal_intake']:>8,}  "
                f"Delta={delta:>+8,} ({pct:>+.1f}%)"
            )

    lines.extend(["", "WELFARE LOSS", "-" * 40])
    if not welfare_df.empty:
        total_loss = welfare_df.groupby("year")["welfare_loss"].sum()
        for year, loss in total_loss.items():
            lines.append(f"  {year}: ${loss:>15,.0f}")

    lines.extend(["", "LAMBDA IDENTIFICATION", "-" * 40])
    lines.append(f"  Status: {lambda_results.get('status', 'unknown')}")
    lines.append(f"  R-squared: {lambda_results.get('r_squared', 'N/A')}")
    lines.append(f"  N observations: {lambda_results.get('n_observations', 'N/A')}")
    if lambda_results.get("thesis_assessment"):
        lines.append(f"  Assessment: {lambda_results['thesis_assessment']}")
    if lambda_results.get("dof_warning"):
        lines.append(f"  WARNING: {lambda_results['dof_warning']}")

    lines.extend(["", "=" * 60])

    output_path = output_dir / ".." / "reports" / "summary_stats.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Saved summary stats to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_simulation(Path("."))
    print("Simulation complete.")
