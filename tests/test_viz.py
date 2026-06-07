"""Tests for the visualization layer."""

from pathlib import Path

import pandas as pd
import pytest

from src.viz.chart_config import get_cohort_display_name, get_province_display_name
from src.viz.static_charts import (
    chart_aci_vs_intake,
    chart_counterfactual_vs_actual,
    chart_welfare_loss_decomposition,
    chart_rank2_projection,
    chart_aci_sensitivity,
    chart_cohort_niv,
)
import plotly.graph_objects as go


@pytest.fixture
def master_panel():
    return pd.read_parquet("data/master/master_panel.parquet")


@pytest.fixture
def counterfactual_df():
    return pd.read_csv("outputs/data/counterfactual_series.csv")


@pytest.fixture
def npv_df():
    return pd.read_csv("outputs/data/cohort_niv.csv")


@pytest.fixture
def decomp_df():
    return pd.read_csv("outputs/data/dimensional_decomposition.csv")


class TestChartConfig:
    def test_cohort_display_names(self):
        assert get_cohort_display_name("high_wage_worker") == "High-Wage Worker"
        assert get_cohort_display_name("refugee") == "Refugee"
        assert get_cohort_display_name("unknown") == "unknown"

    def test_province_display_names(self):
        assert get_province_display_name("ON") == "Ontario"
        assert get_province_display_name("QC") == "Quebec"


class TestStaticCharts:
    def test_chart_aci_vs_intake(self, master_panel):
        fig = chart_aci_vs_intake(master_panel)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_chart_counterfactual_vs_actual(self, counterfactual_df):
        cf_equal = counterfactual_df[counterfactual_df["scenario"] == "aci_equal"]
        fig = chart_counterfactual_vs_actual(cf_equal)
        assert isinstance(fig, go.Figure)

    def test_chart_welfare_loss_decomposition(self, decomp_df):
        fig = chart_welfare_loss_decomposition(decomp_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_chart_rank2_projection(self, master_panel, counterfactual_df):
        fig = chart_rank2_projection(master_panel, counterfactual_df)
        assert isinstance(fig, go.Figure)

    def test_chart_aci_sensitivity(self, counterfactual_df):
        counterfactuals = {
            scenario: counterfactual_df[counterfactual_df["scenario"] == scenario]
            for scenario in counterfactual_df["scenario"].unique()
        }
        fig = chart_aci_sensitivity(counterfactuals)
        assert isinstance(fig, go.Figure)

    def test_chart_cohort_niv(self, npv_df):
        fig = chart_cohort_niv(npv_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_all_chart_outputs_exist(self):
        charts_dir = Path("outputs/charts")
        expected = [
            "01_aci_vs_intake", "02_counterfactual_vs_actual",
            "03_welfare_loss_decomposition", "04_rank2_projection",
            "05_aci_sensitivity", "06_cohort_niv",
        ]
        for name in expected:
            assert (charts_dir / f"{name}.html").exists(), f"Missing {name}.html"
            assert (charts_dir / f"{name}.png").exists(), f"Missing {name}.png"


class TestChartGeneration:
    def test_generate_all_charts(self, tmp_path):
        from src.viz.static_charts import generate_all_charts

        base_dir = Path(".")
        results = generate_all_charts(base_dir, output_dir=tmp_path)
        assert len(results) >= 5
        for name, (html_path, png_path) in results.items():
            assert html_path.exists(), f"Missing HTML for {name}"
            assert html_path.suffix == ".html"
