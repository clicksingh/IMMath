"""Dash Dashboard — Interactive dashboard for ACI Research Tool.

Runs on localhost:8050 with province selector, year slider,
weight toggle, ACI gauge, counterfactual comparison, and welfare loss table.
"""

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

from .chart_config import (
    COLORS, COHORT_COLORS, PROVINCE_COLORS,
    get_cohort_display_name, get_province_display_name,
)

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8050


def create_app(base_dir: Path, url_base_pathname: str | None = None) -> Dash:
    """Create and configure the Dash application.

    Args:
        base_dir: Project root directory.
        url_base_pathname: Base path when mounted behind a proxy (e.g. "/dash/").

    Returns:
        Configured Dash app.
    """
    # Load data
    master_panel = pd.read_parquet(base_dir / "data" / "master" / "master_panel.parquet")
    counterfactual_df = pd.read_csv(base_dir / "outputs" / "data" / "counterfactual_series.csv")
    welfare_df = pd.read_csv(base_dir / "outputs" / "data" / "welfare_loss_decomposition.csv")
    npv_df = pd.read_csv(base_dir / "outputs" / "data" / "cohort_niv.csv")

    provinces = sorted(master_panel["province"].unique())
    years = sorted(master_panel["year"].unique())

    # Create app
    dash_kwargs = {
        "external_stylesheets": [dbc.themes.BOOTSTRAP],
        "title": "ACI Research Tool",
    }
    if url_base_pathname:
        dash_kwargs["url_base_pathname"] = url_base_pathname

    app = Dash(__name__, **dash_kwargs)

    app.layout = dbc.Container([
        html.H1(
            "Canadian Immigration ACI Research Tool",
            style={"color": COLORS["primary"], "textAlign": "center", "marginBottom": 30},
        ),

        # Controls row
        dbc.Row([
            dbc.Col([
                html.Label("Province"),
                dcc.Dropdown(
                    id="province-selector",
                    options=[{"label": get_province_display_name(p), "value": p} for p in provinces],
                    value="ON",
                    clearable=False,
                ),
            ], width=3),
            dbc.Col([
                html.Label("Year Range"),
                dcc.RangeSlider(
                    id="year-slider",
                    min=min(years),
                    max=max(years),
                    step=1,
                    value=[min(years), max(years)],
                    marks={str(y): str(y) for y in years},
                ),
            ], width=5),
            dbc.Col([
                html.Label("Weight Scenario"),
                dcc.RadioItems(
                    id="weight-toggle",
                    options=[
                        {"label": "Housing-Heavy", "value": "aci_housing_heavy"},
                        {"label": "Equal", "value": "aci_equal"},
                        {"label": "Fiscal-Heavy", "value": "aci_fiscal_heavy"},
                    ],
                    value="aci_equal",
                    inline=True,
                ),
            ], width=4),
        ], style={"marginBottom": 30}),

        # ACI Gauge
        dbc.Row([
            dbc.Col([
                html.H4("ACI Gauge", style={"textAlign": "center"}),
                dcc.Graph(id="aci-gauge"),
            ], width=4),
            dbc.Col([
                html.H4("Intake: Actual vs Counterfactual", style={"textAlign": "center"}),
                dcc.Graph(id="intake-comparison"),
            ], width=8),
        ]),

        # Welfare Loss Table
        dbc.Row([
            dbc.Col([
                html.H4("Welfare Loss Breakdown", style={"textAlign": "center", "marginTop": 30}),
                html.Div(id="welfare-table"),
            ], width=12),
        ]),
    ], fluid=True)

    @app.callback(
        Output("aci-gauge", "figure"),
        [Input("province-selector", "value"),
         Input("year-slider", "value"),
         Input("weight-toggle", "value")],
    )
    def update_aci_gauge(province, year_range, scenario):
        filtered = master_panel[
            (master_panel["province"] == province) &
            (master_panel["year"] >= year_range[0]) &
            (master_panel["year"] <= year_range[1])
        ]
        aci_val = filtered[scenario].mean()

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=aci_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"ACI — {get_province_display_name(province)}"},
            delta={"reference": 0.35, "decreasing": {"color": COLORS["accent"]}},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": COLORS["primary"]},
                "steps": [
                    {"range": [0, 0.35], "color": "#FADBD8"},
                    {"range": [0.35, 0.65], "color": "#FEF9E7"},
                    {"range": [0.65, 1.0], "color": "#D5F5E3"},
                ],
                "threshold": {"line": {"color": COLORS["accent"], "width": 4}, "value": 0.35},
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @app.callback(
        Output("intake-comparison", "figure"),
        [Input("province-selector", "value"),
         Input("year-slider", "value"),
         Input("weight-toggle", "value")],
    )
    def update_intake_comparison(province, year_range, scenario):
        cf_scenario = scenario.replace("aci_", "aci_") if scenario in ("aci_housing_heavy", "aci_equal", "aci_fiscal_heavy") else "aci_equal"

        cf_filtered = counterfactual_df[
            (counterfactual_df["province"] == province) &
            (counterfactual_df["year"] >= year_range[0]) &
            (counterfactual_df["year"] <= year_range[1]) &
            (counterfactual_df["scenario"] == cf_scenario)
        ]

        yearly = cf_filtered.groupby("year").agg({
            "actual_intake": "sum",
            "optimal_intake": "sum",
        }).reset_index().sort_values("year")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly["year"], y=yearly["actual_intake"],
            name="Actual", marker_color=COLORS["accent"], opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            x=yearly["year"], y=yearly["optimal_intake"],
            name="Optimal (ACI-Constrained)", marker_color=COLORS["success"], opacity=0.7,
        ))

        fig.update_layout(
            title="Counterfactual vs Actual Intake",
            barmode="group",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    @app.callback(
        Output("welfare-table", "children"),
        [Input("province-selector", "value"),
         Input("year-slider", "value")],
    )
    def update_welfare_table(province, year_range):
        filtered = welfare_df[
            (welfare_df["province"] == province) &
            (welfare_df["year"] >= year_range[0]) &
            (welfare_df["year"] <= year_range[1])
        ]

        if filtered.empty:
            return html.P("No welfare loss data for selected filters.")

        table = dbc.Table.from_dataframe(
            filtered[["year", "actual_welfare", "optimal_welfare", "welfare_loss", "actual_total_intake", "optimal_total_intake"]],
            striped=True, bordered=True, hover=True, size="sm",
        )
        return table

    return app


def run_dashboard(base_dir: Path, port: int = DEFAULT_PORT) -> None:
    """Run the Dash dashboard.

    Args:
        base_dir: Project root directory.
        port: Port to run on (default 8050).
    """
    app = create_app(base_dir)
    logger.info("Starting dashboard on http://localhost:%d", port)
    app.run(port=port, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard(Path("."))
