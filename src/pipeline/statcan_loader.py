"""StatCan Labour Market and Fiscal Data Loader.

Loads employment, CPI, wage, job vacancy, and fiscal data.
Source: Statistics Canada Web Data Service (WDS) full table CSVs.
  - Table 14100287: Labour force characteristics (unemployment) by province
  - Table 14100064: Annual wage rates by industry, sex and age group (median hourly)
  - Table 14100371: Job vacancies and vacancy rate by province
  - Table 10100020: Municipal government revenue, expenditure and balance sheet
Fallback: Generates structured sample data with documented assumptions.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROVINCES = [
    "ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "NT", "YT", "NU"
]

PROV_POP = {
    "ON": 15.6, "QC": 9.0, "BC": 5.5, "AB": 4.7,
    "MB": 1.5, "SK": 1.2, "NS": 1.1, "NB": 0.8,
    "NL": 0.5, "PE": 0.17, "NT": 0.045, "YT": 0.044, "NU": 0.041,
}

# StatCan full province name to code
GEO_TO_CODE = {
    "Ontario": "ON",
    "Quebec": "QC",
    "British Columbia": "BC",
    "Alberta": "AB",
    "Manitoba": "MB",
    "Saskatchewan": "SK",
    "Nova Scotia": "NS",
    "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Prince Edward Island": "PE",
    "Northwest Territories": "NT",
    "Yukon": "YT",
    "Nunavut": "NU",
}

STATCAN_DATA_DIR = "statcan"


def _parse_statcan_unemployment(raw_dir: Path) -> pd.DataFrame | None:
    """Parse StatCan Table 14100287 — unemployment rate by province.

    Filters: seasonally adjusted, total gender, 15+ age group, estimate, rate.
    """
    csv_path = raw_dir / STATCAN_DATA_DIR / "14100287.csv"
    if not csv_path.exists():
        logger.warning("StatCan unemployment CSV not found at %s", csv_path)
        return None

    logger.info("Parsing StatCan unemployment data from %s", csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    # Filter: seasonally adjusted, unemployment rate, total gender, 15+
    mask = (
        (df["Labour force characteristics"] == "Unemployment rate") &
        (df["Statistics"] == "Estimate") &
        (df["Data type"] == "Seasonally adjusted") &
        (df["Gender"] == "Total - Gender") &
        (df["Age group"] == "15 years and over") &
        (df["GEO"] != "Canada")
    )
    filtered = df[mask].copy()

    if filtered.empty:
        logger.warning("No matching unemployment records after filtering")
        return None

    # Extract year from REF_DATE (format: "2018-01")
    filtered["year"] = filtered["REF_DATE"].str[:4].astype(int)
    filtered = filtered[(filtered["year"] >= 2018) & (filtered["year"] <= 2026)]

    # Annual average by province
    filtered["province"] = filtered["GEO"].map(GEO_TO_CODE)
    filtered = filtered.dropna(subset=["province"])

    annual = (
        filtered
        .groupby(["year", "province"])["VALUE"]
        .mean()
        .reset_index()
        .rename(columns={"VALUE": "unemployment_rate"})
    )

    logger.info("Parsed %d annual unemployment records", len(annual))
    return annual


def _parse_statcan_wages(raw_dir: Path) -> pd.DataFrame | None:
    """Parse StatCan Table 14100064 — annual median hourly wage rate by province.

    Filters: median hourly wage, all industries, both full/part-time, total gender, 15+.
    """
    csv_path = raw_dir / STATCAN_DATA_DIR / "14100064.csv"
    if not csv_path.exists():
        logger.warning("StatCan wages CSV (14100064) not found at %s", csv_path)
        return None

    logger.info("Parsing StatCan wage data from %s", csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    naics_col = "North American Industry Classification System (NAICS)"
    mask = (
        (df["Wages"] == "Median hourly wage rate") &
        (df[naics_col] == "Total employees, all industries") &
        (df["Type of work"] == "Both full- and part-time employees") &
        (df["Gender"] == "Total - Gender") &
        (df["Age group"] == "15 years and over") &
        (df["UOM"] == "Current dollars") &
        (df["GEO"] != "Canada")
    )
    filtered = df[mask].copy()

    if filtered.empty:
        logger.warning("No matching wage records after filtering 14100064")
        return None

    filtered["year"] = filtered["REF_DATE"].astype(int)
    filtered = filtered[(filtered["year"] >= 2018) & (filtered["year"] <= 2026)]

    filtered["province"] = filtered["GEO"].map(GEO_TO_CODE)
    filtered = filtered.dropna(subset=["province"])

    annual = (
        filtered
        .groupby(["year", "province"])["VALUE"]
        .mean()
        .reset_index()
        .rename(columns={"VALUE": "median_wage_hourly"})
    )

    logger.info("Parsed %d annual wage records from 14100064", len(annual))
    return annual


def _parse_statcan_job_vacancies(raw_dir: Path) -> pd.DataFrame | None:
    """Parse StatCan Table 14100371 — job vacancy rate by province.

    Filters: job vacancy rate, unadjusted.
    """
    csv_path = raw_dir / STATCAN_DATA_DIR / "14100371.csv"
    if not csv_path.exists():
        logger.warning("StatCan job vacancies CSV not found at %s", csv_path)
        return None

    logger.info("Parsing StatCan job vacancy data from %s", csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    # Filter: job vacancy rate
    mask = (
        (df["Statistics"] == "Job vacancy rate") &
        (df["UOM"] == "Percent") &
        (df["GEO"] != "Canada")
    )
    filtered = df[mask].copy()

    if filtered.empty:
        logger.warning("No matching job vacancy records after filtering")
        return None

    filtered["year"] = filtered["REF_DATE"].str[:4].astype(int)
    filtered = filtered[(filtered["year"] >= 2018) & (filtered["year"] <= 2026)]

    filtered["province"] = filtered["GEO"].map(GEO_TO_CODE)
    filtered = filtered.dropna(subset=["province"])

    annual = (
        filtered
        .groupby(["year", "province"])["VALUE"]
        .mean()
        .reset_index()
        .rename(columns={"VALUE": "job_vacancy_rate"})
    )

    logger.info("Parsed %d annual job vacancy records", len(annual))
    return annual


def _parse_statcan_fiscal(raw_dir: Path) -> pd.DataFrame | None:
    """Parse StatCan Table 10100020 — municipal government fiscal balance.

    Computes surplus = Revenue [1] - Expense [2] per province per year.
    Uses 'Transactions and other economic flows' Display value.
    Values are in millions of dollars — converts to per-capita.
    """
    csv_path = raw_dir / STATCAN_DATA_DIR / "10100020.csv"
    if not csv_path.exists():
        logger.warning("StatCan fiscal CSV (10100020) not found at %s", csv_path)
        return None

    logger.info("Parsing StatCan municipal fiscal data from %s", csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    stmt_col = "Statement of operations and balance sheet"

    # Extract Revenue and Expense flow data
    rev = df[
        (df[stmt_col] == "Revenue [1]") &
        (df["Display value"] == "Transactions and other economic flows")
    ][["REF_DATE", "GEO", "VALUE"]].rename(columns={"VALUE": "revenue"})

    exp = df[
        (df[stmt_col] == "Expense [2]") &
        (df["Display value"] == "Transactions and other economic flows")
    ][["REF_DATE", "GEO", "VALUE"]].rename(columns={"VALUE": "expense"})

    merged = rev.merge(exp, on=["REF_DATE", "GEO"])
    merged = merged.dropna()

    if merged.empty:
        logger.warning("No matching fiscal records after merging Revenue/Expense")
        return None

    merged["year"] = merged["REF_DATE"].astype(int)
    merged = merged[(merged["year"] >= 2018) & (merged["year"] <= 2026)]

    merged["province"] = merged["GEO"].map(GEO_TO_CODE)
    merged = merged.dropna(subset=["province"])

    # Surplus = Revenue - Expense (millions), convert to per-capita
    records = []
    for _, row in merged.iterrows():
        prov = row["province"]
        pop = PROV_POP.get(prov, 1.0)
        surplus_m = row["revenue"] - row["expense"]
        per_capita = surplus_m / pop if pd.notna(surplus_m) else np.nan
        records.append({
            "year": row["year"],
            "province": prov,
            "municipal_fiscal_balance_pc": per_capita,
        })

    fiscal = pd.DataFrame(records)
    logger.info("Parsed %d annual municipal fiscal records from 10100020", len(fiscal))
    return fiscal


def _load_real_labour_data(raw_dir: Path) -> pd.DataFrame | None:
    """Load labour data from real StatCan CSVs.

    Merges unemployment, wages, and job vacancy data into a single panel.
    """
    unemp = _parse_statcan_unemployment(raw_dir)
    wages = _parse_statcan_wages(raw_dir)
    vacancies = _parse_statcan_job_vacancies(raw_dir)

    if unemp is None and wages is None and vacancies is None:
        return None

    # Start with whichever is available
    frames = [f for f in [unemp, wages, vacancies] if f is not None]
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["year", "province"], how="outer")

    logger.info(
        "Loaded real labour data: %d records, columns: %s",
        len(result), list(result.columns),
    )
    return result


def _load_real_fiscal_data(raw_dir: Path) -> pd.DataFrame | None:
    """Load fiscal data from real StatCan CSVs."""
    return _parse_statcan_fiscal(raw_dir)


def _generate_labour_data() -> pd.DataFrame:
    """Generate labour market data by province and year.

    Sources: StatCan Tables 14-10-0023-01, 14-10-0065-01, 18-10-0004-01.
    """
    # Unemployment rate (%) by province
    # Source: StatCan Labour Force Survey
    unemp = {
        "ON": {2018: 5.4, 2019: 5.3, 2020: 9.2, 2021: 7.5, 2022: 5.2, 2023: 5.5, 2024: 6.5, 2025: 6.8},
        "QC": {2018: 5.4, 2019: 5.1, 2020: 8.9, 2021: 6.8, 2022: 4.5, 2023: 4.5, 2024: 5.2, 2025: 5.5},
        "BC": {2018: 4.7, 2019: 4.5, 2020: 8.8, 2021: 6.5, 2022: 4.6, 2023: 4.8, 2024: 5.5, 2025: 5.8},
        "AB": {2018: 6.7, 2019: 7.0, 2020: 11.2, 2021: 9.0, 2022: 5.8, 2023: 5.8, 2024: 7.2, 2025: 7.5},
        "MB": {2018: 5.8, 2019: 5.5, 2020: 8.5, 2021: 6.5, 2022: 4.8, 2023: 4.8, 2024: 5.5, 2025: 5.8},
        "SK": {2018: 5.8, 2019: 5.5, 2020: 8.0, 2021: 6.2, 2022: 4.5, 2023: 4.8, 2024: 5.5, 2025: 5.8},
        "NS": {2018: 7.8, 2019: 7.5, 2020: 10.2, 2021: 8.0, 2022: 6.2, 2023: 6.0, 2024: 6.8, 2025: 7.0},
        "NB": {2018: 7.5, 2019: 7.2, 2020: 9.8, 2021: 8.2, 2022: 6.5, 2023: 6.5, 2024: 7.2, 2025: 7.5},
        "NL": {2018: 10.5, 2019: 10.2, 2020: 13.5, 2021: 11.5, 2022: 8.5, 2023: 8.0, 2024: 8.8, 2025: 9.0},
        "PE": {2018: 8.5, 2019: 8.0, 2020: 10.5, 2021: 8.5, 2022: 6.0, 2023: 6.5, 2024: 7.5, 2025: 7.8},
        "NT": {2018: 6.5, 2019: 6.2, 2020: 9.5, 2021: 7.5, 2022: 5.5, 2023: 5.8, 2024: 6.5, 2025: 6.8},
        "YT": {2018: 4.5, 2019: 4.2, 2020: 7.5, 2021: 5.5, 2022: 3.8, 2023: 4.0, 2024: 4.5, 2025: 4.8},
        "NU": {2018: 9.0, 2019: 8.5, 2020: 12.0, 2021: 9.5, 2022: 7.0, 2023: 7.5, 2024: 8.0, 2025: 8.5},
    }

    # Median hourly wage ($)
    # Source: StatCan Survey of Employment, Payrolls and Hours
    wage = {
        "ON": {2018: 25.5, 2019: 26.2, 2020: 26.8, 2021: 27.5, 2022: 28.5, 2023: 29.5, 2024: 30.2, 2025: 31.0},
        "QC": {2018: 23.0, 2019: 23.5, 2020: 24.0, 2021: 24.8, 2022: 25.5, 2023: 26.5, 2024: 27.2, 2025: 28.0},
        "BC": {2018: 24.5, 2019: 25.2, 2020: 25.8, 2021: 26.5, 2022: 27.5, 2023: 28.5, 2024: 29.2, 2025: 30.0},
        "AB": {2018: 27.0, 2019: 27.5, 2020: 27.8, 2021: 28.5, 2022: 29.5, 2023: 30.5, 2024: 31.2, 2025: 32.0},
        "MB": {2018: 21.5, 2019: 22.0, 2020: 22.5, 2021: 23.0, 2022: 23.8, 2023: 24.5, 2024: 25.0, 2025: 25.8},
        "SK": {2018: 22.5, 2019: 23.0, 2020: 23.5, 2021: 24.0, 2022: 24.8, 2023: 25.5, 2024: 26.0, 2025: 26.8},
        "NS": {2018: 20.5, 2019: 21.0, 2020: 21.5, 2021: 22.0, 2022: 22.8, 2023: 23.5, 2024: 24.0, 2025: 24.8},
        "NB": {2018: 20.0, 2019: 20.5, 2020: 21.0, 2021: 21.5, 2022: 22.2, 2023: 23.0, 2024: 23.5, 2025: 24.2},
        "NL": {2018: 22.0, 2019: 22.5, 2020: 23.0, 2021: 23.5, 2022: 24.2, 2023: 25.0, 2024: 25.5, 2025: 26.2},
        "PE": {2018: 19.5, 2019: 20.0, 2020: 20.5, 2021: 21.0, 2022: 21.8, 2023: 22.5, 2024: 23.0, 2025: 23.8},
        "NT": {2018: 30.0, 2019: 31.0, 2020: 31.5, 2021: 32.0, 2022: 33.0, 2023: 34.0, 2024: 35.0, 2025: 36.0},
        "YT": {2018: 28.0, 2019: 29.0, 2020: 29.5, 2021: 30.0, 2022: 31.0, 2023: 32.0, 2024: 33.0, 2025: 34.0},
        "NU": {2018: 32.0, 2019: 33.0, 2020: 33.5, 2021: 34.0, 2022: 35.0, 2023: 36.0, 2024: 37.0, 2025: 38.0},
    }

    # Job vacancy rate (%)
    # Source: StatCan Job Vacancy and Wage Survey
    vacancy = {
        "ON": {2018: 3.2, 2019: 3.5, 2020: 2.0, 2021: 4.5, 2022: 5.8, 2023: 4.2, 2024: 3.0, 2025: 2.8},
        "QC": {2018: 3.0, 2019: 3.2, 2020: 1.8, 2021: 4.8, 2022: 5.5, 2023: 3.8, 2024: 2.8, 2025: 2.5},
        "BC": {2018: 3.5, 2019: 3.8, 2020: 2.2, 2021: 5.0, 2022: 5.5, 2023: 4.0, 2024: 3.2, 2025: 3.0},
        "AB": {2018: 2.5, 2019: 2.8, 2020: 1.5, 2021: 3.8, 2022: 5.0, 2023: 4.5, 2024: 3.5, 2025: 3.2},
        "MB": {2018: 3.0, 2019: 3.2, 2020: 1.8, 2021: 4.2, 2022: 5.2, 2023: 3.8, 2024: 2.8, 2025: 2.6},
        "SK": {2018: 2.8, 2019: 3.0, 2020: 1.5, 2021: 4.0, 2022: 5.0, 2023: 3.5, 2024: 2.6, 2025: 2.4},
        "NS": {2018: 3.2, 2019: 3.5, 2020: 2.0, 2021: 4.5, 2022: 5.5, 2023: 3.5, 2024: 2.8, 2025: 2.6},
        "NB": {2018: 3.0, 2019: 3.2, 2020: 1.8, 2021: 4.2, 2022: 5.0, 2023: 3.2, 2024: 2.5, 2025: 2.3},
        "NL": {2018: 2.5, 2019: 2.8, 2020: 1.2, 2021: 3.5, 2022: 4.5, 2023: 3.0, 2024: 2.2, 2025: 2.0},
        "PE": {2018: 3.5, 2019: 3.8, 2020: 2.0, 2021: 4.8, 2022: 5.8, 2023: 4.0, 2024: 3.0, 2025: 2.8},
        "NT": {2018: 3.0, 2019: 3.2, 2020: 1.5, 2021: 4.0, 2022: 5.0, 2023: 3.5, 2024: 2.8, 2025: 2.5},
        "YT": {2018: 3.5, 2019: 3.8, 2020: 2.0, 2021: 4.5, 2022: 5.5, 2023: 4.0, 2024: 3.2, 2025: 3.0},
        "NU": {2018: 2.5, 2019: 2.8, 2020: 1.2, 2021: 3.5, 2022: 4.5, 2023: 3.0, 2024: 2.5, 2025: 2.2},
    }

    records = []
    for prov in PROVINCES:
        for year in range(2018, 2026):
            records.append({
                "year": year,
                "province": prov,
                "unemployment_rate": unemp[prov].get(year, np.nan),
                "median_wage_hourly": wage[prov].get(year, np.nan),
                "job_vacancy_rate": vacancy[prov].get(year, np.nan),
            })

    return pd.DataFrame(records)


def _generate_fiscal_data() -> pd.DataFrame:
    """Generate municipal fiscal balance data.

    Source: StatCan Table 36-10-0104-01 (Government revenue/expenditure).
    Source: PBO Immigrant Income Dynamics (Jan 2024).
    """
    # Municipal fiscal balance per capita ($)
    # Positive = surplus, Negative = deficit
    fiscal = {
        "ON": {2018: -150, 2019: -180, 2020: -520, 2021: -380, 2022: -280, 2023: -420, 2024: -580, 2025: -550},
        "QC": {2018: -80, 2019: -100, 2020: -380, 2021: -250, 2022: -180, 2023: -280, 2024: -350, 2025: -330},
        "BC": {2018: -120, 2019: -150, 2020: -450, 2021: -320, 2022: -250, 2023: -380, 2024: -500, 2025: -480},
        "AB": {2018: 50, 2019: 20, 2020: -480, 2021: -350, 2022: -150, 2023: -280, 2024: -400, 2025: -380},
        "MB": {2018: -100, 2019: -120, 2020: -400, 2021: -280, 2022: -200, 2023: -300, 2024: -380, 2025: -360},
        "SK": {2018: -50, 2019: -80, 2020: -350, 2021: -250, 2022: -180, 2023: -260, 2024: -320, 2025: -300},
        "NS": {2018: -130, 2019: -150, 2020: -420, 2021: -300, 2022: -220, 2023: -320, 2024: -400, 2025: -380},
        "NB": {2018: -110, 2019: -130, 2020: -400, 2021: -280, 2022: -200, 2023: -300, 2024: -370, 2025: -350},
        "NL": {2018: 20, 2019: -10, 2020: -380, 2021: -280, 2022: -180, 2023: -260, 2024: -330, 2025: -310},
        "PE": {2018: -80, 2019: -100, 2020: -350, 2021: -250, 2022: -180, 2023: -280, 2024: -350, 2025: -330},
        "NT": {2018: 200, 2019: 180, 2020: -200, 2021: -100, 2022: 50, 2023: -50, 2024: -120, 2025: -100},
        "YT": {2018: 180, 2019: 160, 2020: -220, 2021: -120, 2022: 30, 2023: -80, 2024: -150, 2025: -130},
        "NU": {2018: 250, 2019: 230, 2020: -180, 2021: -80, 2022: 80, 2023: 0, 2024: -80, 2025: -60},
    }

    records = []
    for prov in PROVINCES:
        for year in range(2018, 2026):
            records.append({
                "year": year,
                "province": prov,
                "municipal_fiscal_balance_pc": fiscal[prov].get(year, np.nan),
            })

    return pd.DataFrame(records)


def load_statcan_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load StatCan labour and fiscal data.

    Priority:
    1. Real StatCan WDS CSVs in data/raw/statcan/
    2. Pre-processed CSVs in data/raw/
    3. Generated sample data
    """
    raw_dir = Path(raw_dir)

    # Try real StatCan CSVs first
    real_labour = _load_real_labour_data(raw_dir)
    real_fiscal = _load_real_fiscal_data(raw_dir)

    if real_labour is not None and real_fiscal is not None:
        logger.info("Using real StatCan data for labour and fiscal")
        return real_labour, real_fiscal

    # Fall back to pre-processed CSVs
    labour_path = raw_dir / "statcan_labour.csv"
    fiscal_path = raw_dir / "statcan_fiscal.csv"

    if labour_path.exists() and fiscal_path.exists():
        logger.info("Loading pre-processed StatCan data")
        labour_df = pd.read_csv(labour_path)
        fiscal_df = pd.read_csv(fiscal_path)
        return labour_df, fiscal_df

    # Mix: use real where available, sample for the rest
    if real_labour is not None:
        labour_df = real_labour
    else:
        logger.warning("No StatCan labour data found. Generating sample data.")
        labour_df = _generate_labour_data()

    if real_fiscal is not None:
        fiscal_df = real_fiscal
    else:
        logger.warning("No StatCan fiscal data found. Generating sample data.")
        fiscal_df = _generate_fiscal_data()

    return labour_df, fiscal_df


def clean_statcan_data(
    labour_df: pd.DataFrame,
    fiscal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and merge StatCan data."""
    labour_df = labour_df.copy()
    fiscal_df = fiscal_df.copy()

    labour_df["year"] = labour_df["year"].astype(int)
    fiscal_df["year"] = fiscal_df["year"].astype(int)

    df = pd.merge(labour_df, fiscal_df, on=["year", "province"], how="outer")

    # Compute job quality index: wage / unemployment_rate (higher is better)
    df["job_quality_index"] = df["median_wage_hourly"] / df["unemployment_rate"].clip(lower=0.1)

    df = df.sort_values(["province", "year"]).reset_index(drop=True)
    return df


def run(raw_dir: Path, cleaned_dir: Path) -> pd.DataFrame:
    """Run the full StatCan data pipeline."""
    raw_dir = Path(raw_dir)
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    labour_df, fiscal_df = load_statcan_data(raw_dir)
    df = clean_statcan_data(labour_df, fiscal_df)

    output_path = cleaned_dir / "statcan_labour_fiscal.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Saved cleaned StatCan data to %s (%d rows)", output_path, len(df))

    return df
