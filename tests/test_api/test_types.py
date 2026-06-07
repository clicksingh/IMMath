"""Tests for GraphQL type from_row() mapping and NaN handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.types.cohort_niv import CohortNIV
from src.api.types.counterfactual import Counterfactual
from src.api.types.decomposition import DimensionalDecomposition
from src.api.types.lambda_result import LambdaResult
from src.api.types.master_panel import MasterPanel
from src.api.types.welfare import WelfareLoss


class TestCohortNIVType:
    def test_from_row(self):
        row = pd.Series({
            "cohort_type": "high_wage_worker",
            "display_name": "Test Worker",
            "pv_benefits": 500000.0,
            "pv_tax": 200000.0,
            "pv_transition": 100000.0,
            "pv_costs": 150000.0,
            "pv_housing": 50000.0,
            "pv_health": 30000.0,
            "pv_education": 20000.0,
            "pv_integrity": 10000.0,
            "pv_settlement": 15000.0,
            "niv": 425000.0,
            "transition_probability": 0.8,
            "time_horizon": 20,
            "discount_rate": 0.03,
        })
        result = CohortNIV.from_row(row)
        assert result.cohort_type == "high_wage_worker"
        assert result.niv == 425000.0
        assert result.time_horizon == 20


class TestLambdaResultType:
    def test_from_row_with_nan(self):
        row = pd.Series({
            "variable": "test_var",
            "coefficient": np.nan,
            "std_error": 0.5,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "significant_10pct": np.nan,
            "notes": None,
        })
        result = LambdaResult.from_row(row)
        assert result.variable == "test_var"
        assert result.coefficient is None
        assert result.std_error == 0.5
        assert result.significant_10pct is None
        assert result.notes is None


class TestWelfareLossType:
    def test_from_row(self):
        row = pd.Series({
            "year": 2023,
            "province": "ON",
            "actual_welfare": 1000000.0,
            "optimal_welfare": 1100000.0,
            "welfare_loss": -100000.0,
            "actual_total_intake": 500000,
            "optimal_total_intake": 510000,
            "intake_gap": 10000,
        })
        result = WelfareLoss.from_row(row)
        assert result.year == 2023
        assert result.province == "ON"
        assert result.welfare_loss == -100000.0


class TestDecompositionType:
    def test_from_row_with_none(self):
        row = pd.Series({
            "year": 2023,
            "dimension": "housing_vacancy",
            "avg_score": np.nan,
            "intake_gap": 5000.0,
            "contribution": -200.0,
            "contribution_pct": 0.15,
        })
        result = DimensionalDecomposition.from_row(row)
        assert result.dimension == "housing_vacancy"
        assert result.avg_score is None
        assert result.contribution_pct == 0.15


class TestCounterfactualType:
    def test_from_row(self):
        row = pd.Series({
            "year": 2022,
            "province": "BC",
            "cohort_type": "high_quality_student",
            "actual_intake": 100000,
            "optimal_intake": 95000,
            "aci_value": 0.45,
            "niv_per_unit": 99711.0,
            "scenario": "aci_equal",
        })
        result = Counterfactual.from_row(row)
        assert result.year == 2022
        assert result.scenario == "aci_equal"
        assert result.actual_intake == 100000


class TestMasterPanelType:
    def test_from_row(self):
        row = pd.Series({
            "year": 2023, "quarter": 2, "province": "AB",
            "cohort_type": "francophone_pr", "intake_count": 5000,
            "starts_annual": 30000.0, "starts_per_capita": 6.38,
            "starts_per_capita_growth": -0.02, "vacancy_rate": 3.5,
            "avg_rent_2br": 1200.0,
            "unemployment_rate": 5.8, "median_wage_hourly": 28.0,
            "job_vacancy_rate": 4.0, "job_quality_index": 4.83,
            "municipal_fiscal_balance_pc": 350.0,
            "health_capacity": 0.65, "school_capacity": 0.72,
            "vacancy_rate_norm": 0.35, "starts_growth_norm": 0.4,
            "health_capacity_norm": 0.65, "school_capacity_norm": 0.72,
            "job_quality_norm": 0.5, "fiscal_balance_norm": 0.6,
            "aci_housing_heavy": 0.38, "aci_equal": 0.42, "aci_fiscal_heavy": 0.45,
            "aci_nan_flag": "complete",
        })
        result = MasterPanel.from_row(row)
        assert result.year == 2023
        assert result.housing.starts_annual == 30000.0
        assert result.labour.unemployment_rate == 5.8
        assert result.aci.equal == 0.42
        assert result.aci_nan_flag == "complete"

    def test_from_row_with_nan(self):
        row = pd.Series({
            "year": 2020, "quarter": 1, "province": "NT",
            "cohort_type": "refugee", "intake_count": np.nan,
            "starts_annual": np.nan, "starts_per_capita": np.nan,
            "starts_per_capita_growth": np.nan, "vacancy_rate": np.nan,
            "avg_rent_2br": np.nan,
            "unemployment_rate": np.nan, "median_wage_hourly": np.nan,
            "job_vacancy_rate": np.nan, "job_quality_index": np.nan,
            "municipal_fiscal_balance_pc": np.nan,
            "health_capacity": np.nan, "school_capacity": np.nan,
            "vacancy_rate_norm": np.nan, "starts_growth_norm": np.nan,
            "health_capacity_norm": np.nan, "school_capacity_norm": np.nan,
            "job_quality_norm": np.nan, "fiscal_balance_norm": np.nan,
            "aci_housing_heavy": np.nan, "aci_equal": np.nan, "aci_fiscal_heavy": np.nan,
            "aci_nan_flag": "missing_3_of_6",
        })
        result = MasterPanel.from_row(row)
        assert result.intake_count is None
        assert result.housing.starts_annual is None
        assert result.aci.equal is None
