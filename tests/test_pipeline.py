"""Tests for the data pipeline modules."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import ircc_loader, cmhc_loader, statcan_loader, cihi_loader, school_loader
from src.pipeline.joiner import normalize_to_01, compute_aci, build_master_panel


# --- IRCC Loader Tests ---

class TestIRCCLoader:
    def test_load_generates_data_when_no_raw(self, tmp_path):
        df = ircc_loader.load_ircc_data(tmp_path)
        assert not df.empty
        assert "year" in df.columns
        assert "province" in df.columns
        assert "cohort_type" in df.columns
        assert "intake_count" in df.columns

    def test_generated_data_covers_all_years(self):
        df = ircc_loader.load_ircc_data(Path("nonexistent"), force_regenerate=True)
        years = sorted(df["year"].unique())
        assert years == list(range(2018, 2027))

    def test_generated_data_covers_all_provinces(self):
        df = ircc_loader.load_ircc_data(Path("nonexistent"), force_regenerate=True)
        assert set(df["province"].unique()) == set(ircc_loader.PROVINCES)

    def test_generated_data_covers_all_cohorts(self):
        df = ircc_loader.load_ircc_data(Path("nonexistent"), force_regenerate=True)
        assert set(df["cohort_type"].unique()) == set(ircc_loader.COHORT_TYPES)

    def test_clean_handles_missing_values(self):
        df = pd.DataFrame({
            "year": [2018, 2018],
            "quarter": [1, 2],
            "province": ["ON", "QC"],
            "cohort_type": ["refugee", "refugee"],
            "intake_count": [100, np.nan],
        })
        cleaned = ircc_loader.clean_ircc_data(df)
        assert cleaned["intake_count"].iloc[1] == 0

    def test_clean_removes_invalid_provinces(self):
        df = pd.DataFrame({
            "year": [2018],
            "quarter": [1],
            "province": ["XX"],
            "cohort_type": ["refugee"],
            "intake_count": [100],
        })
        cleaned = ircc_loader.clean_ircc_data(df)
        assert cleaned.empty

    def test_run_saves_parquet(self, tmp_path):
        df = ircc_loader.run(tmp_path / "raw", tmp_path / "cleaned")
        assert (tmp_path / "cleaned" / "ircc_intake.parquet").exists()
        assert len(df) > 0


# --- CMHC Loader Tests ---

class TestCMHCLoader:
    def test_housing_starts_generation(self):
        starts = cmhc_loader._generate_housing_starts()
        assert not starts.empty
        assert "starts_annual" in starts.columns
        assert "starts_per_capita" in starts.columns

    def test_vacancy_rent_generation(self):
        vacancy = cmhc_loader._generate_vacancy_rates()
        assert not vacancy.empty
        assert "vacancy_rate" in vacancy.columns
        assert "avg_rent_2br" in vacancy.columns

    def test_clean_merges_and_computes_growth(self, tmp_path):
        starts_df = cmhc_loader._generate_housing_starts()
        vacancy_df = cmhc_loader._generate_vacancy_rates()
        cleaned = cmhc_loader.clean_cmhc_data(starts_df, vacancy_df)
        assert "starts_per_capita_growth" in cleaned.columns
        # First year should be NaN (no prior year for growth)
        first_year = cleaned[cleaned["year"] == 2018]
        assert first_year["starts_per_capita_growth"].isna().all()


# --- StatCan Loader Tests ---

class TestStatCanLoader:
    def test_labour_data_generation(self):
        df = statcan_loader._generate_labour_data()
        assert len(df) == 13 * 8  # 13 provinces * 8 years
        assert "unemployment_rate" in df.columns
        assert "median_wage_hourly" in df.columns
        assert "job_vacancy_rate" in df.columns

    def test_fiscal_data_generation(self):
        df = statcan_loader._generate_fiscal_data()
        assert len(df) == 13 * 8
        assert "municipal_fiscal_balance_pc" in df.columns

    def test_clean_computes_job_quality(self):
        labour = statcan_loader._generate_labour_data()
        fiscal = statcan_loader._generate_fiscal_data()
        cleaned = statcan_loader.clean_statcan_data(labour, fiscal)
        assert "job_quality_index" in cleaned.columns
        assert (cleaned["job_quality_index"] > 0).all()


# --- CIHI Loader Tests ---

class TestCIHILoader:
    def test_load_from_manual_fallback(self, tmp_path):
        # Create a minimal manual CSV
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()
        pd.DataFrame({
            "year": [2020, 2020],
            "province": ["ON", "QC"],
            "median_wait_priority_days": [10.0, 12.0],
            "hospital_occupancy_pct": [93.0, 91.0],
            "er_wait_hours": [4.0, 5.0],
            "alt_care_days_pct": [15.0, 17.0],
            "source": ["CIHI", "CIHI"],
            "notes": ["", ""],
        }).to_csv(manual_dir / "cihi_fallback.csv", index=False)

        df = cihi_loader.load_cihi_data(tmp_path / "raw", manual_dir)
        assert len(df) == 2

    def test_clean_computes_health_capacity(self, tmp_path):
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()
        pd.DataFrame({
            "year": [2020, 2020],
            "province": ["ON", "QC"],
            "median_wait_priority_days": [10.0, 12.0],
            "hospital_occupancy_pct": [93.0, 91.0],
            "er_wait_hours": [4.0, 5.0],
            "alt_care_days_pct": [15.0, 17.0],
            "source": ["CIHI", "CIHI"],
            "notes": ["", ""],
        }).to_csv(manual_dir / "cihi_fallback.csv", index=False)

        df = cihi_loader.load_cihi_data(tmp_path / "raw", manual_dir)
        cleaned = cihi_loader.clean_cihi_data(df)
        assert "health_capacity" in cleaned.columns
        # ON should have higher health capacity (lower waits)
        on_cap = cleaned[cleaned["province"] == "ON"]["health_capacity"].iloc[0]
        qc_cap = cleaned[cleaned["province"] == "QC"]["health_capacity"].iloc[0]
        assert on_cap > qc_cap


# --- School Loader Tests ---

class TestSchoolLoader:
    def test_load_from_manual(self, tmp_path):
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()
        pd.DataFrame({
            "year": [2020],
            "province": ["ON"],
            "public_school_enrollment": [2100000],
            "enrollment_growth_pct": [0.7],
            "student_teacher_ratio": [14.9],
            "capacity_utilization_pct": [95.0],
            "source": ["Ontario Ministry"],
            "notes": [""],
        }).to_csv(manual_dir / "school_capacity.csv", index=False)

        df = school_loader.load_school_data(manual_dir)
        assert len(df) == 1

    def test_clean_computes_school_capacity(self, tmp_path):
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()
        pd.DataFrame({
            "year": [2020, 2020],
            "province": ["ON", "NS"],
            "public_school_enrollment": [2100000, 130000],
            "enrollment_growth_pct": [0.7, -0.7],
            "student_teacher_ratio": [14.9, 14.7],
            "capacity_utilization_pct": [95.0, 89.0],
            "source": ["Ontario", "Nova Scotia"],
            "notes": ["", ""],
        }).to_csv(manual_dir / "school_capacity.csv", index=False)

        df = school_loader.load_school_data(manual_dir)
        cleaned = school_loader.clean_school_data(df)
        assert "school_capacity" in cleaned.columns


# --- Joiner Tests ---

class TestJoiner:
    def test_normalize_to_01(self):
        series = pd.Series([0, 5, 10])
        result = normalize_to_01(series)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.5
        assert result.iloc[2] == 1.0

    def test_normalize_handles_nan(self):
        series = pd.Series([0, np.nan, 10])
        result = normalize_to_01(series)
        assert np.isnan(result.iloc[1])

    def test_normalize_constant_series(self):
        series = pd.Series([5, 5, 5])
        result = normalize_to_01(series)
        assert (result == 0.5).all()

    def test_compute_aci_three_scenarios(self):
        df = pd.DataFrame({
            "vacancy_rate": [3.0, 2.0],
            "starts_per_capita_growth": [0.05, -0.02],
            "health_capacity": [0.6, 0.4],
            "school_capacity": [0.5, 0.3],
            "job_quality_index": [4.0, 3.0],
            "municipal_fiscal_balance_pc": [-200, -400],
        })
        result = compute_aci(df)
        assert "aci_housing_heavy" in result.columns
        assert "aci_equal" in result.columns
        assert "aci_fiscal_heavy" in result.columns
        # ACI values should be in [0, 1]
        for col in ["aci_housing_heavy", "aci_equal", "aci_fiscal_heavy"]:
            assert result[col].min() >= 0
            assert result[col].max() <= 1

    def test_build_master_panel(self, tmp_path):
        # Create minimal manual data
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()

        # School data
        pd.DataFrame({
            "year": [2020],
            "province": ["ON"],
            "public_school_enrollment": [2100000],
            "enrollment_growth_pct": [0.7],
            "student_teacher_ratio": [14.9],
            "capacity_utilization_pct": [95.0],
            "source": ["Ontario"],
            "notes": [""],
        }).to_csv(manual_dir / "school_capacity.csv", index=False)

        # CIHI data
        pd.DataFrame({
            "year": [2020],
            "province": ["ON"],
            "median_wait_priority_days": [10.0],
            "hospital_occupancy_pct": [93.0],
            "er_wait_hours": [4.0],
            "alt_care_days_pct": [15.0],
            "source": ["CIHI"],
            "notes": [""],
        }).to_csv(manual_dir / "cihi_fallback.csv", index=False)

        master = build_master_panel(
            base_dir=tmp_path,
            raw_dir=tmp_path / "raw",
            cleaned_dir=tmp_path / "cleaned",
            manual_dir=manual_dir,
            master_dir=tmp_path / "master",
        )

        assert not master.empty
        assert "aci_equal" in master.columns
        assert (tmp_path / "master" / "master_panel.parquet").exists()
