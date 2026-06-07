"""Tests for the simulation engine modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.simulation.cohort_npv import load_cohort_params, compute_cohort_npv
from src.simulation.aci_optimizer import run_optimization, _fallback_allocation
from src.simulation.counterfactual import generate_counterfactual, compute_counterfactual_summary
from src.simulation.welfare_loss import compute_welfare_loss, decompose_by_dimension
from src.simulation.lambda_identifier import construct_dimension_scores, _assess_thesis_support


@pytest.fixture
def sample_cohort_params():
    """Minimal cohort params for testing."""
    return {
        "cohort_types": {
            "high_wage_worker": {
                "display_name": "High-Wage Worker",
                "annual_earnings": 95000,
                "transition_probability": 0.65,
                "annual_tax_contribution": 30400,
                "housing_consumption_annual": 22000,
                "health_service_consumption_annual": 3200,
                "education_infrastructure_cost": 0,
                "integrity_processing_cost": 600,
                "discount_rate": 0.03,
                "time_horizon_years": 25,
            },
            "low_wage_worker": {
                "display_name": "Low-Wage Worker",
                "annual_earnings": 32000,
                "transition_probability": 0.15,
                "annual_tax_contribution": 5760,
                "housing_consumption_annual": 14500,
                "health_service_consumption_annual": 3000,
                "education_infrastructure_cost": 0,
                "integrity_processing_cost": 1200,
                "discount_rate": 0.03,
                "time_horizon_years": 25,
            },
            "refugee": {
                "display_name": "Refugee",
                "annual_earnings": 22000,
                "transition_probability": 1.0,
                "annual_tax_contribution": 3300,
                "housing_consumption_annual": 16000,
                "health_service_consumption_annual": 4200,
                "education_infrastructure_cost": 3500,
                "integrity_processing_cost": 2800,
                "settlement_services_annual": 8500,
                "discount_rate": 0.03,
                "time_horizon_years": 30,
            },
        },
        "welfare_weights": {"equal": 1.0},
    }


@pytest.fixture
def sample_annual_panel():
    """Minimal annual panel for testing optimization."""
    records = []
    provinces = ["ON", "QC", "BC"]
    cohorts = ["high_wage_worker", "low_wage_worker", "refugee"]

    for year in [2020, 2021]:
        for prov in provinces:
            for cohort in cohorts:
                records.append({
                    "year": year,
                    "province": prov,
                    "cohort_type": cohort,
                    "intake_count": np.random.randint(1000, 50000),
                    "aci_equal": np.random.uniform(0.2, 0.8),
                    "aci_housing_heavy": np.random.uniform(0.2, 0.8),
                    "aci_fiscal_heavy": np.random.uniform(0.2, 0.8),
                })

    return pd.DataFrame(records)


class TestCohortNPV:
    def test_load_cohort_params(self):
        config_path = Path("config/cohort_params.yaml")
        params = load_cohort_params(config_path)
        assert "cohort_types" in params
        assert len(params["cohort_types"]) == 8

    def test_compute_npv(self, sample_cohort_params):
        df = compute_cohort_npv(sample_cohort_params)
        assert len(df) == 3
        assert "niv" in df.columns
        # High-wage worker should have higher NIV than low-wage
        hw_niv = df[df["cohort_type"] == "high_wage_worker"]["niv"].iloc[0]
        lw_niv = df[df["cohort_type"] == "low_wage_worker"]["niv"].iloc[0]
        assert hw_niv > lw_niv

    def test_npv_components_present(self, sample_cohort_params):
        df = compute_cohort_npv(sample_cohort_params)
        expected_cols = ["pv_benefits", "pv_costs", "pv_tax", "pv_housing",
                         "pv_health", "niv", "transition_probability"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_npv_discount_rate_override(self, sample_cohort_params):
        df_default = compute_cohort_npv(sample_cohort_params)
        df_high_r = compute_cohort_npv(sample_cohort_params, discount_rate_override=0.10)
        # Higher discount rate should shrink present values toward zero
        # (absolute NIV decreases for both positive and negative cohorts)
        assert df_high_r["niv"].abs().sum() < df_default["niv"].abs().sum()


class TestACIOptimizer:
    def test_optimization_runs(self, sample_annual_panel, sample_cohort_params):
        npv_df = compute_cohort_npv(sample_cohort_params)
        result = run_optimization(sample_annual_panel, npv_df, aci_scenario="aci_equal")
        assert not result.empty
        assert "optimal_intake" in result.columns
        assert "actual_intake" in result.columns

    def test_optimization_preserves_totals(self, sample_annual_panel, sample_cohort_params):
        npv_df = compute_cohort_npv(sample_cohort_params)
        result = run_optimization(sample_annual_panel, npv_df)
        # National totals should roughly match
        for year in result["year"].unique():
            yr = result[result["year"] == year]
            actual_total = yr["actual_intake"].sum()
            optimal_total = yr["optimal_intake"].sum()
            # Allow 5% tolerance
            assert abs(actual_total - optimal_total) / max(actual_total, 1) < 0.05

    def test_fallback_allocation(self, sample_annual_panel):
        year_data = sample_annual_panel[sample_annual_panel["year"] == 2020]
        result = _fallback_allocation(year_data, 2020, "aci_equal", 0.35)
        assert not result.empty
        assert result["optimal_intake"].sum() > 0


class TestCounterfactual:
    def test_generate_counterfactual(self, sample_annual_panel, sample_cohort_params):
        npv_df = compute_cohort_npv(sample_cohort_params)
        results = generate_counterfactual(sample_annual_panel, npv_df)
        assert "aci_equal" in results
        assert not results["aci_equal"].empty

    def test_counterfactual_summary(self, sample_annual_panel, sample_cohort_params):
        npv_df = compute_cohort_npv(sample_cohort_params)
        counterfactuals = generate_counterfactual(sample_annual_panel, npv_df)
        summary = compute_counterfactual_summary(counterfactuals)
        assert not summary.empty
        assert "delta" in summary.columns


class TestWelfareLoss:
    def test_compute_welfare_loss(self, sample_annual_panel, sample_cohort_params):
        npv_df = compute_cohort_npv(sample_cohort_params)
        counterfactuals = generate_counterfactual(sample_annual_panel, npv_df)
        welfare = compute_welfare_loss(counterfactuals["aci_equal"])
        assert not welfare.empty
        assert "welfare_loss" in welfare.columns

    def test_decompose_by_dimension(self, sample_annual_panel, sample_cohort_params, tmp_path):
        npv_df = compute_cohort_npv(sample_cohort_params)
        counterfactuals = generate_counterfactual(sample_annual_panel, npv_df)

        # Create a minimal master panel for decomposition
        master = sample_annual_panel.copy()
        for col in ["vacancy_rate_norm", "starts_growth_norm", "health_capacity_norm",
                     "school_capacity_norm", "job_quality_norm", "fiscal_balance_norm"]:
            master[col] = np.random.uniform(0.1, 0.9, len(master))

        decomp = decompose_by_dimension(counterfactuals["aci_equal"], master)
        assert not decomp.empty
        assert "dimension" in decomp.columns
        assert "contribution_pct" in decomp.columns


class TestLambdaIdentifier:
    def test_construct_dimension_scores(self, tmp_path):
        master_path = Path("data/master/master_panel.parquet")
        if not master_path.exists():
            pytest.skip("Master panel not yet built")

        master = pd.read_parquet(master_path)
        scores = construct_dimension_scores(master)
        assert not scores.empty
        assert "delta_volume" in scores.columns

    def test_thesis_assessment_honest(self):
        results_supportive = {
            "status": "ok",
            "r_squared": 0.65,
            "rank_2_test": {
                "n_significant_predictors": 2,
                "political_dims_significant": True,
                "absorptive_dims_significant": False,
            },
        }
        assessment = _assess_thesis_support(results_supportive)
        assert "SUPPORTIVE" in assessment

        results_contradictory = {
            "status": "ok",
            "r_squared": 0.5,
            "rank_2_test": {
                "n_significant_predictors": 3,
                "political_dims_significant": True,
                "absorptive_dims_significant": True,
            },
        }
        assessment = _assess_thesis_support(results_contradictory)
        assert "NOT SUPPORTIVE" in assessment

    def test_thesis_assessment_all_branches(self):
        from src.simulation.lambda_identifier import _assess_thesis_support, run_regression

        # Error status
        assert "Cannot assess" in _assess_thesis_support({"status": "error"})

        # Weak support (low R²)
        weak = {"status": "ok", "r_squared": 0.05, "rank_2_test": {"n_significant_predictors": 0, "political_dims_significant": False, "absorptive_dims_significant": False}}
        assert "WEAK" in _assess_thesis_support(weak)

        # Partially supportive
        partial = {"status": "ok", "r_squared": 0.5, "rank_2_test": {"n_significant_predictors": 1, "political_dims_significant": True, "absorptive_dims_significant": False}}
        assert "PARTIALLY SUPPORTIVE" in _assess_thesis_support(partial)

        # Inconclusive
        inconcl = {"status": "ok", "r_squared": 0.3, "rank_2_test": {"n_significant_predictors": 0, "political_dims_significant": False, "absorptive_dims_significant": False}}
        assert "INCONCLUSIVE" in _assess_thesis_support(inconcl)

        # Mixed
        mixed = {"status": "ok", "r_squared": 0.4, "rank_2_test": {"n_significant_predictors": 1, "political_dims_significant": False, "absorptive_dims_significant": False}}
        assert "MIXED" in _assess_thesis_support(mixed)

    def test_run_regression_with_sufficient_data(self):
        from src.simulation.lambda_identifier import run_regression

        master_path = Path("data/master/master_panel.parquet")
        if not master_path.exists():
            pytest.skip("Master panel not yet built")

        master = pd.read_parquet(master_path)
        scores = construct_dimension_scores(master)
        results = run_regression(scores)
        assert results["status"] in ("ok", "insufficient_data", "insufficient_predictors")
        if results["status"] == "ok":
            assert "r_squared" in results
            assert "thesis_assessment" in results

    def test_run_regression_insufficient_data(self):
        from src.simulation.lambda_identifier import run_regression

        # Only 2 rows — should return insufficient_data
        tiny = pd.DataFrame({
            "year": [2020, 2021],
            "delta_volume": [100, -50],
            "delta_quebec_leverage": [0.01, -0.01],
            "delta_housing_stress": [0.02, -0.01],
            "delta_absorptive_capacity": [0.01, 0.01],
            "delta_heterogeneity": [0.0, 0.0],
            "delta_innovation": [0.0, 0.0],
            "delta_fiscal": [0.01, -0.01],
        })
        results = run_regression(tiny)
        assert results["status"] == "insufficient_data"

    def test_export_regression_results(self, tmp_path):
        from src.simulation.lambda_identifier import export_regression_results

        results = {
            "status": "ok",
            "r_squared": 0.5,
            "adj_r_squared": 0.4,
            "f_statistic": 3.0,
            "f_pvalue": 0.05,
            "thesis_assessment": "MIXED: test",
            "coefficients": {
                "delta_quebec_leverage": {"coef": 0.5, "std_err": 0.2, "t_stat": 2.5, "p_value": 0.03, "ci_lower": 0.1, "ci_upper": 0.9},
            },
        }
        path = export_regression_results(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) >= 2  # coefficient row + model stats

    def test_export_with_dof_warning(self, tmp_path):
        from src.simulation.lambda_identifier import export_regression_results

        results = {
            "status": "ok",
            "r_squared": 0.3,
            "coefficients": {},
            "thesis_assessment": "INCONCLUSIVE",
            "dof_warning": "WARNING: Only 5 observations",
        }
        path = export_regression_results(results, tmp_path)
        assert path.exists()


class TestCounterfactualExport:
    def test_export_counterfactual_series(self, sample_annual_panel, sample_cohort_params, tmp_path):
        from src.simulation.counterfactual import export_counterfactual_series

        npv_df = compute_cohort_npv(sample_cohort_params)
        counterfactuals = generate_counterfactual(sample_annual_panel, npv_df)
        path = export_counterfactual_series(counterfactuals, tmp_path)
        assert path.exists()
        assert path.name == "counterfactual_series.csv"

        # Summary should also exist
        summary_path = tmp_path / "counterfactual_summary.csv"
        assert summary_path.exists()


class TestWelfareLossExport:
    def test_export_welfare_loss(self, sample_annual_panel, sample_cohort_params, tmp_path):
        from src.simulation.welfare_loss import export_welfare_loss

        npv_df = compute_cohort_npv(sample_cohort_params)
        counterfactuals = generate_counterfactual(sample_annual_panel, npv_df)
        welfare_df = compute_welfare_loss(counterfactuals["aci_equal"])

        master = sample_annual_panel.copy()
        for col in ["vacancy_rate_norm", "starts_growth_norm", "health_capacity_norm",
                     "school_capacity_norm", "job_quality_norm", "fiscal_balance_norm"]:
            master[col] = np.random.uniform(0.1, 0.9, len(master))

        decomp_df = decompose_by_dimension(counterfactuals["aci_equal"], master)
        loss_path, decomp_path = export_welfare_loss(welfare_df, decomp_df, tmp_path)
        assert loss_path.exists()
        assert decomp_path.exists()

    def test_welfare_loss_run_pipeline(self, sample_annual_panel, sample_cohort_params, tmp_path):
        from src.simulation.welfare_loss import run as welfare_run

        npv_df = compute_cohort_npv(sample_cohort_params)
        counterfactuals = generate_counterfactual(sample_annual_panel, npv_df)

        master = sample_annual_panel.copy()
        for col in ["vacancy_rate_norm", "starts_growth_norm", "health_capacity_norm",
                     "school_capacity_norm", "job_quality_norm", "fiscal_balance_norm"]:
            master[col] = np.random.uniform(0.1, 0.9, len(master))

        w_df, d_df = welfare_run(counterfactuals, master, tmp_path)
        assert not w_df.empty
        assert not d_df.empty


class TestCohortNPVExtended:
    def test_regional_npv_adjustment(self, sample_cohort_params):
        from src.simulation.cohort_npv import compute_cohort_npv, compute_regional_npv_adjustment

        npv_df = compute_cohort_npv(sample_cohort_params)
        master_path = Path("data/master/master_panel.parquet")
        if not master_path.exists():
            pytest.skip("Master panel not yet built")

        master = pd.read_parquet(master_path)
        adjusted = compute_regional_npv_adjustment(npv_df, master)
        assert not adjusted.empty
        assert "niv_adjusted" in adjusted.columns
