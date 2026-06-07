"""Static Charts — All 6 Plotly charts exported as HTML + PNG.

1. ACI vs Actual Intake
2. Counterfactual vs Actual Intake
3. Welfare Loss Decomposition
4. Rank-2 Projection Visualization
5. ACI Sensitivity Analysis
6. Cohort NIV Comparison
"""

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .chart_config import (
    COLORS, COHORT_COLORS, PROVINCE_COLORS, DIMENSION_COLORS,
    LAYOUT_DEFAULTS, EXPORT_SETTINGS, get_cohort_display_name,
    get_province_display_name,
)

logger = logging.getLogger(__name__)


def _apply_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply standard layout to a figure."""
    layout = {**LAYOUT_DEFAULTS, "title": title}
    fig.update_layout(**layout)
    return fig


def _export_chart(fig: go.Figure, name: str, output_dir: Path) -> tuple[Path, Path]:
    """Export chart as HTML and PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"{name}.html"
    png_path = output_dir / f"{name}.png"

    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), **EXPORT_SETTINGS)
        logger.info("Exported %s (HTML + PNG)", name)
    except Exception as e:
        logger.warning("PNG export failed for %s: %s (HTML saved)", name, e)

    return html_path, png_path


# --- Chart 1: ACI vs Actual Intake ---

def chart_aci_vs_intake(
    master_panel: pd.DataFrame,
    provinces: list[str] | None = None,
) -> go.Figure:
    """Line chart: ACI on left axis, actual intake on right axis, per province."""
    if provinces is None:
        provinces = ["ON", "QC", "BC", "AB"]

    # Aggregate to annual by province
    annual = (
        master_panel
        .groupby(["year", "province"])
        .agg({"intake_count": "sum", "aci_equal": "mean"})
        .reset_index()
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[get_province_display_name(p) for p in provinces],
    )

    for idx, prov in enumerate(provinces):
        row = idx // 2 + 1
        col = idx % 2 + 1
        prov_data = annual[annual["province"] == prov].sort_values("year")

        # ACI line
        fig.add_trace(
            go.Scatter(
                x=prov_data["year"], y=prov_data["aci_equal"],
                name=f"{prov} ACI", mode="lines+markers",
                line=dict(color=PROVINCE_COLORS.get(prov, COLORS["primary"]), width=2),
                marker=dict(size=6),
                showlegend=(idx == 0),
                legendgroup="aci",
            ),
            row=row, col=col,
        )

        # Intake on secondary axis (scaled)
        fig.add_trace(
            go.Bar(
                x=prov_data["year"], y=prov_data["intake_count"],
                name=f"{prov} Intake",
                marker_color=PROVINCE_COLORS.get(prov, COLORS["secondary"]),
                opacity=0.4,
                showlegend=(idx == 0),
                legendgroup="intake",
            ),
            row=row, col=col,
        )

    fig.update_layout(title_text="ACI vs Actual Intake (2018-2026)")
    _apply_layout(fig, "ACI vs Actual Intake by Province (2018-2026)")
    fig.update_yaxes(title_text="ACI Score / Intake", row=1, col=1)

    return fig


# --- Chart 2: Counterfactual vs Actual ---

def chart_counterfactual_vs_actual(
    counterfactual_df: pd.DataFrame,
) -> go.Figure:
    """Stacked area chart: actual vs counterfactual by cohort type and year."""
    yearly = counterfactual_df.groupby(["year", "cohort_type"]).agg({
        "actual_intake": "sum",
        "optimal_intake": "sum",
    }).reset_index()

    cohorts = sorted(yearly["cohort_type"].unique())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Actual Intake", "ACI-Constrained (Optimal)"],
    )

    for cohort in cohorts:
        cohort_data = yearly[yearly["cohort_type"] == cohort].sort_values("year")
        color = COHORT_COLORS.get(cohort, COLORS["grey"])
        name = get_cohort_display_name(cohort)

        fig.add_trace(
            go.Scatter(
                x=cohort_data["year"], y=cohort_data["actual_intake"],
                name=name, stackgroup="actual",
                line=dict(color=color),
                legendgroup=cohort,
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=cohort_data["year"], y=cohort_data["optimal_intake"],
                name=name, stackgroup="optimal",
                line=dict(color=color),
                legendgroup=cohort,
                showlegend=False,
            ),
            row=1, col=2,
        )

    _apply_layout(fig, "Counterfactual vs Actual Intake by Cohort Type")
    fig.update_yaxes(title_text="Intake Count")

    return fig


# --- Chart 3: Welfare Loss Decomposition ---

def chart_welfare_loss_decomposition(
    decomposition_df: pd.DataFrame,
) -> go.Figure:
    """Waterfall chart: welfare loss by dimension."""
    # Average contribution by dimension across years
    dim_avg = (
        decomposition_df
        .groupby("dimension")["contribution_pct"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = go.Figure()

    colors = [DIMENSION_COLORS.get(d, COLORS["grey"]) for d in dim_avg["dimension"]]

    fig.add_trace(go.Bar(
        x=dim_avg["dimension"],
        y=dim_avg["contribution_pct"],
        marker_color=colors,
        text=dim_avg["contribution_pct"].round(1).astype(str) + "%",
        textposition="outside",
    ))

    _apply_layout(fig, "Welfare Loss Decomposition by Dimension")
    fig.update_yaxes(title_text="Average Contribution (%)")
    fig.update_xaxes(title_text="Dimension")

    return fig


# --- Chart 4: Rank-2 Projection ---

def chart_rank2_projection(
    master_panel: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
) -> go.Figure:
    """2D scatter: electoral manifold (P=volume, Q=Quebec leverage)."""
    # Aggregate yearly
    yearly = master_panel.groupby("year").agg({
        "intake_count": "sum",
        "vacancy_rate": "mean",
        "aci_equal": "mean",
    }).reset_index()

    # Quebec share
    qc_share = (
        master_panel[master_panel["province"] == "QC"]
        .groupby("year")["intake_count"].sum()
        / master_panel.groupby("year")["intake_count"].sum()
    ).reset_index()
    qc_share.columns = ["year", "qc_share"]

    yearly = yearly.merge(qc_share, on="year", how="left")

    fig = go.Figure()

    # Plot years as points on P-Q plane
    fig.add_trace(go.Scatter(
        x=yearly["intake_count"],
        y=yearly["qc_share"],
        mode="markers+text",
        text=yearly["year"].astype(str),
        textposition="top center",
        marker=dict(
            size=14,
            color=yearly["aci_equal"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="ACI Score"),
        ),
        name="Policy Year",
    ))

    _apply_layout(fig, "Rank-2 Projection: Electoral Manifold (Volume vs Quebec Leverage)")
    fig.update_xaxes(title_text="P: Total Intake (Volume Signal)")
    fig.update_yaxes(title_text="Q: Quebec Intake Share (Coalition Leverage)")

    return fig


# --- Chart 5: ACI Sensitivity ---

def chart_aci_sensitivity(
    counterfactuals: dict[str, pd.DataFrame],
) -> go.Figure:
    """Three-panel chart: counterfactual under each weight scenario."""
    scenario_names = {
        "aci_housing_heavy": "Housing-Heavy Weights",
        "aci_equal": "Equal Weights",
        "aci_fiscal_heavy": "Fiscal-Heavy Weights",
    }

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=list(scenario_names.values()),
    )

    for col_idx, (scenario, cf) in enumerate(counterfactuals.items()):
        yearly = cf.groupby("year").agg({
            "actual_intake": "sum",
            "optimal_intake": "sum",
        }).reset_index().sort_values("year")

        fig.add_trace(
            go.Scatter(
                x=yearly["year"], y=yearly["actual_intake"],
                mode="lines+markers", name="Actual" if col_idx == 0 else "",
                line=dict(color=COLORS["accent"], dash="dash", width=2),
                showlegend=(col_idx == 0),
                legendgroup="actual",
            ),
            row=1, col=col_idx + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=yearly["year"], y=yearly["optimal_intake"],
                mode="lines+markers", name="Optimal" if col_idx == 0 else "",
                line=dict(color=COLORS["success"], width=2),
                showlegend=(col_idx == 0),
                legendgroup="optimal",
            ),
            row=1, col=col_idx + 1,
        )

    _apply_layout(fig, "ACI Sensitivity Analysis: Counterfactual Under Three Weight Scenarios")
    fig.update_yaxes(title_text="Total Intake")

    return fig


# --- Chart 6: Cohort NIV Comparison ---

def chart_cohort_niv(npv_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart ranking cohort types by NIV."""
    df = npv_df.sort_values("niv", ascending=True)

    colors = [
        COLORS["success"] if niv > 0 else COLORS["accent"]
        for niv in df["niv"]
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["display_name"],
        x=df["niv"],
        orientation="h",
        marker_color=colors,
        text=df["niv"].apply(lambda x: f"${x:,.0f}"),
        textposition="outside",
    ))

    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["dark"])

    _apply_layout(fig, "Cohort Net Immigration Value (NIV) Comparison")
    fig.update_xaxes(title_text="Net Immigration Value (CAD)")
    fig.update_yaxes(title_text="")

    return fig


# --- Generate All Charts ---

def generate_all_charts(
    base_dir: Path,
    output_dir: Path | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Generate all 6 static charts.

    Returns:
        Dictionary mapping chart name to (html_path, png_path).
    """
    base_dir = Path(base_dir)
    if output_dir is None:
        output_dir = base_dir / "outputs" / "charts"

    # Load data
    master_panel = pd.read_parquet(base_dir / "data" / "master" / "master_panel.parquet")
    counterfactual_df = pd.read_csv(base_dir / "outputs" / "data" / "counterfactual_series.csv")
    welfare_df = pd.read_csv(base_dir / "outputs" / "data" / "welfare_loss_decomposition.csv")
    decomp_df = pd.read_csv(base_dir / "outputs" / "data" / "dimensional_decomposition.csv")
    npv_df = pd.read_csv(base_dir / "outputs" / "data" / "cohort_niv.csv")

    # Load all counterfactual scenarios
    counterfactuals = {}
    for scenario in ["aci_housing_heavy", "aci_equal", "aci_fiscal_heavy"]:
        scenario_data = counterfactual_df[counterfactual_df["scenario"] == scenario]
        if not scenario_data.empty:
            counterfactuals[scenario] = scenario_data

    results = {}

    # Chart 1
    fig1 = chart_aci_vs_intake(master_panel)
    results["01_aci_vs_intake"] = _export_chart(fig1, "01_aci_vs_intake", output_dir)

    # Chart 2
    if "aci_equal" in counterfactuals:
        fig2 = chart_counterfactual_vs_actual(counterfactuals["aci_equal"])
        results["02_counterfactual_vs_actual"] = _export_chart(fig2, "02_counterfactual_vs_actual", output_dir)

    # Chart 3
    if not decomp_df.empty:
        fig3 = chart_welfare_loss_decomposition(decomp_df)
        results["03_welfare_loss_decomposition"] = _export_chart(fig3, "03_welfare_loss_decomposition", output_dir)

    # Chart 4
    fig4 = chart_rank2_projection(master_panel, counterfactual_df)
    results["04_rank2_projection"] = _export_chart(fig4, "04_rank2_projection", output_dir)

    # Chart 5
    if counterfactuals:
        fig5 = chart_aci_sensitivity(counterfactuals)
        results["05_aci_sensitivity"] = _export_chart(fig5, "05_aci_sensitivity", output_dir)

    # Chart 6
    fig6 = chart_cohort_niv(npv_df)
    results["06_cohort_niv"] = _export_chart(fig6, "06_cohort_niv", output_dir)

    logger.info("Generated %d charts", len(results))
    return results
