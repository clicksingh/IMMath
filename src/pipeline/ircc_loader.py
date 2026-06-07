"""IRCC Immigration Intake Data Loader.

Downloads and parses immigration intake data from IRCC Open Data portal.
Sources:
  - Permanent Residents by Province/Territory and Immigration Category
  - Study Permit Holders by Province/Territory and Study Level
  - TFWP and IMP Work Permit Holders by Province/Territory and Program

When real data download fails, falls back to structured sample data
based on published IRCC totals and distributions.

Data caveats:
  - CSV files contain values rounded to nearest multiple of 5
  - Values 0-5 are shown as "--" in CSVs
  - Data covers 2015-01-01 to 2026-03-31 (updated monthly)
  - CKAN API: https://open.canada.ca/data/api/3/action/package_show?id=<dataset_id>
"""

import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROVINCES = [
    "ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "NT", "YT", "NU"
]

COHORT_TYPES = [
    "high_quality_student", "low_quality_student",
    "high_wage_worker", "low_wage_worker",
    "francophone_pr", "in_canada_transition",
    "family_class", "refugee"
]

# Province name to code mapping
PROVINCE_NAME_TO_CODE = {
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

# Quarter name to number mapping
QUARTER_MAP = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# --- Download URLs (verified 2026-06-06) ---
# Source: https://open.canada.ca/data/en/dataset/f7e5498e-0ad8-4417-85c9-9b8aff9b9eda
PR_CSV_URL = (
    "https://www.ircc.canada.ca/opendata-donneesouvertes/data/"
    "ODP-PR-PT_IMMCAT.csv"
)
# Source: https://open.canada.ca/data/en/dataset/90115b00-f9b8-49e8-afa3-b4cff8facaee
STUDY_CSV_URL = (
    "https://www.ircc.canada.ca/opendata-donneesouvertes/data/"
    "ODP-TR-Study-IS_PT_study.csv"
)
# Source: https://open.canada.ca/data/en/dataset/360024f2-17e9-4558-bfc1-3616485d65b9
TFWP_CSV_URL = (
    "https://www.ircc.canada.ca/opendata-donneesouvertes/data/"
    "ODP-TR-Work-TFWP-PT_program.csv"
)
IMP_CSV_URL = (
    "https://www.ircc.canada.ca/opendata-donneesouvertes/data/"
    "ODP-TR-Work-IMP-PT_program.csv"
)

# --- Category to Cohort Type Mapping ---
# Sentinel for wildcard matching in category patterns
_WILDCARD = object()

# Maps IRCC PR immigration categories to the project's cohort types
# Based on IRCC category hierarchy: Main Category -> Group -> Component
# Use _WILDCARD to match any value at that level.
PR_CATEGORY_PATTERNS: list[tuple[tuple[str | object, str | object, str | object], str]] = [
    # Economic - Worker Program (high wage: skilled worker, skilled trade, CEC)
    (("Economic", "Worker Program", "Skilled Worker"), "high_wage_worker"),
    (("Economic", "Worker Program", "Skilled Trade"), "high_wage_worker"),
    (("Economic", "Worker Program", "Canadian Experience"), "in_canada_transition"),
    (("Economic", "Worker Program", "Caregiver"), "low_wage_worker"),
    # Economic - Provincial Nominee Program
    (("Economic", "Provincial Nominee Program", "Provincial Nominee Program"), "in_canada_transition"),
    # Economic - Business
    (("Economic", "Business", "Entrepreneur"), "high_wage_worker"),
    (("Economic", "Business", "Investor"), "high_wage_worker"),
    (("Economic", "Business", "Self-Employed"), "high_wage_worker"),
    (("Economic", "Business", "Start-up Business"), "high_wage_worker"),
    # Economic - TR to PR Pathway
    (("Economic", "Temporary Resident to Permanent Resident Pathway", _WILDCARD), "in_canada_transition"),
    # Atlantic Immigration (group may vary)
    (("Economic", _WILDCARD, "Atlantic Immigration Pilot Programs"), "in_canada_transition"),
    (("Economic", _WILDCARD, "Atlantic Immigration Programs"), "in_canada_transition"),
    (("Economic", _WILDCARD, "Rural and Northern Immigration"), "in_canada_transition"),
    (("Economic", _WILDCARD, "Agri-Food Pilot"), "low_wage_worker"),
    (("Economic", _WILDCARD, "Federal Economic Mobility Pathways Pilot"), "in_canada_transition"),
    # Sponsored Family
    (("Sponsored Family", "Sponsored Family", _WILDCARD), "family_class"),
    # Refugees
    (("Resettled Refugee & Protected Person in Canada", "Resettled Refugee", _WILDCARD), "refugee"),
    (("Resettled Refugee & Protected Person in Canada", "Protected Person in Canada", _WILDCARD), "refugee"),
    # Other
    (("All Other Immigration", "Humanitarian & Compassionate", _WILDCARD), "refugee"),
    (("All Other Immigration", "Other Immigrants not included elsewhere", _WILDCARD), "refugee"),
    (("All Other Immigration", "Public Policy", _WILDCARD), "refugee"),
]

# Study level to cohort type mapping
STUDY_LEVEL_TO_COHORT = {
    "Post Secondary": "high_quality_student",
    "Secondary or less": "low_quality_student",
    "Other Studies": "low_quality_student",
    "Education level not stated": "low_quality_student",
}

# Work permit program to cohort type mapping
# TFWP programs -> typically low-wage workers
TFWP_PROGRAM_TO_COHORT = {
    "Agricultural Workers": "low_wage_worker",
    "Caregivers": "low_wage_worker",
    "Live-In Caregivers": "low_wage_worker",
    "Other Temporary Foreign Workers with LMIA": "low_wage_worker",
}

# IMP programs -> mixed, but split by program type
IMP_PROGRAM_TO_COHORT = {
    "Agreements": "high_wage_worker",       # CUSMA, etc.
    "Canadian Interests": "high_wage_worker", # PGWP, intra-company, etc.
    "Other IMP Participants": "low_wage_worker",
    "Vulnerable workers": "low_wage_worker",
}

# Fallback sample data constants
COHORT_SHARES = {
    "high_quality_student": 0.15,
    "low_quality_student": 0.10,
    "high_wage_worker": 0.15,
    "low_wage_worker": 0.12,
    "francophone_pr": 0.05,
    "in_canada_transition": 0.10,
    "family_class": 0.23,
    "refugee": 0.10,
}

PROVINCE_SHARES = {
    "ON": 0.40, "QC": 0.15, "BC": 0.14, "AB": 0.12,
    "MB": 0.05, "SK": 0.04, "NS": 0.03, "NB": 0.03,
    "NL": 0.01, "PE": 0.01, "NT": 0.005, "YT": 0.005, "NU": 0.005,
}

TOTAL_INTAKE = {
    2018: 530000, 2019: 565000, 2020: 340000, 2021: 520000,
    2022: 780000, 2023: 870000, 2024: 740000, 2025: 600000, 2026: 550000,
}

DOWNLOAD_TIMEOUT = 120  # seconds


def _download_csv(url: str, label: str) -> str | None:
    """Download a CSV file from a URL.

    Returns the text content on success, None on failure.
    """
    logger.info("Downloading %s from %s", label, url)
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        logger.info("Downloaded %s (%d bytes)", label, len(resp.content))
        return resp.text
    except requests.RequestException as exc:
        logger.error("Failed to download %s: %s", label, exc)
        return None


def _parse_total(value: str) -> int:
    """Parse IRCC TOTAL column value.

    Handles "--" (values 0-5) and numeric strings.
    Returns 0 for suppressed values.
    """
    if pd.isna(value) or str(value).strip() in ("--", ""):
        return 0
    try:
        return int(str(value).strip())
    except ValueError:
        return 0


def _map_province(name: str) -> str | None:
    """Map full province name to two-letter code."""
    return PROVINCE_NAME_TO_CODE.get(name.strip())


def _classify_pr_cohort(
    main_cat: str,
    group: str,
    component: str,
) -> str:
    """Classify a PR immigration category into a cohort type.

    Uses priority-ordered pattern matching against the category hierarchy.
    Patterns can use _WILDCARD to match any value at a level.
    """
    key = (main_cat, group, component)

    for pattern, cohort in PR_CATEGORY_PATTERNS:
        p_main, p_group, p_comp = pattern
        main_match = (p_main is _WILDCARD) or (p_main == main_cat)
        group_match = (p_group is _WILDCARD) or (p_group == group)
        comp_match = (p_comp is _WILDCARD) or (p_comp == component)
        if main_match and group_match and comp_match:
            return cohort

    # Default fallback based on main category
    if main_cat == "Economic":
        return "low_wage_worker"
    if main_cat == "Sponsored Family":
        return "family_class"
    if "Refugee" in main_cat or "Protected" in main_cat:
        return "refugee"
    return "refugee"  # All Other Immigration


def _parse_pr_data(csv_text: str) -> pd.DataFrame:
    """Parse PR CSV into standardized intake records."""
    df = pd.read_csv(StringIO(csv_text), sep="\t", encoding="utf-8")

    records = []
    for _, row in df.iterrows():
        year = int(row["EN_YEAR"])
        if year < 2018:
            continue

        quarter_str = str(row["EN_QUARTER"]).strip()
        quarter = QUARTER_MAP.get(quarter_str)
        if quarter is None:
            continue

        prov_code = _map_province(row["EN_PROVINCE_TERRITORY"])
        if prov_code is None:
            continue

        main_cat = str(row["EN_IMMIGRATION_CATEGORY-MAIN_CATEGORY"]).strip()
        group = str(row["EN_IMMIGRATION_CATEGORY-GROUP"]).strip()
        component = str(row["EN_IMMIGRATION_CATEGORY-COMPONENT"]).strip()

        cohort = _classify_pr_cohort(main_cat, group, component)
        count = _parse_total(row["TOTAL"])

        if count > 0:
            records.append({
                "year": year,
                "quarter": quarter,
                "province": prov_code,
                "cohort_type": cohort,
                "intake_count": count,
                "source": "pr",
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["year", "quarter", "province", "cohort_type", "intake_count", "source"]
    )


def _parse_study_data(csv_text: str) -> pd.DataFrame:
    """Parse Study Permit CSV into standardized intake records."""
    df = pd.read_csv(StringIO(csv_text), sep="\t", encoding="utf-8")

    records = []
    for _, row in df.iterrows():
        year = int(row["EN_YEAR"])
        if year < 2018:
            continue

        quarter_str = str(row["EN_QUARTER"]).strip()
        quarter = QUARTER_MAP.get(quarter_str)
        if quarter is None:
            continue

        prov_code = _map_province(row["EN_PROVINCE_TERRITORY"])
        if prov_code is None:
            continue

        study_level = str(row["EN_STUDY_LEVEL"]).strip()
        cohort = STUDY_LEVEL_TO_COHORT.get(study_level, "low_quality_student")
        count = _parse_total(row["TOTAL"])

        if count > 0:
            records.append({
                "year": year,
                "quarter": quarter,
                "province": prov_code,
                "cohort_type": cohort,
                "intake_count": count,
                "source": "study_permit",
            })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["year", "quarter", "province", "cohort_type", "intake_count", "source"]
    )


def _parse_work_data(tfwp_text: str, imp_text: str) -> pd.DataFrame:
    """Parse TFWP + IMP Work Permit CSVs into standardized intake records."""
    records = []

    for label, text, program_map in [
        ("TFWP", tfwp_text, TFWP_PROGRAM_TO_COHORT),
        ("IMP", imp_text, IMP_PROGRAM_TO_COHORT),
    ]:
        if text is None:
            continue

        df = pd.read_csv(StringIO(text), sep="\t", encoding="utf-8")

        for _, row in df.iterrows():
            year = int(row["EN_YEAR"])
            if year < 2018:
                continue

            quarter_str = str(row["EN_QUARTER"]).strip()
            quarter = QUARTER_MAP.get(quarter_str)
            if quarter is None:
                continue

            prov_code = _map_province(row["EN_PROVINCE_TERRITORY"])
            if prov_code is None:
                continue

            program_level_2 = str(row["EN_PROGRAM_LEVEL_2"]).strip()
            cohort = program_map.get(program_level_2, "low_wage_worker")
            count = _parse_total(row["TOTAL"])

            if count > 0:
                records.append({
                    "year": year,
                    "quarter": quarter,
                    "province": prov_code,
                    "cohort_type": cohort,
                    "intake_count": count,
                    "source": f"work_permit_{label.lower()}",
                })

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["year", "quarter", "province", "cohort_type", "intake_count", "source"]
    )


def _download_and_parse_all() -> pd.DataFrame:
    """Download all IRCC CSVs and parse into a combined intake DataFrame."""
    pr_text = _download_csv(PR_CSV_URL, "PR by Province/Category")
    study_text = _download_csv(STUDY_CSV_URL, "Study Permits by Province")
    tfwp_text = _download_csv(TFWP_CSV_URL, "TFWP Work Permits by Province")
    imp_text = _download_csv(IMP_CSV_URL, "IMP Work Permits by Province")

    frames = []
    source_counts = {}

    if pr_text is not None:
        pr_df = _parse_pr_data(pr_text)
        frames.append(pr_df)
        source_counts["pr"] = len(pr_df)

    if study_text is not None:
        study_df = _parse_study_data(study_text)
        frames.append(study_df)
        source_counts["study_permit"] = len(study_df)

    if tfwp_text is not None or imp_text is not None:
        work_df = _parse_work_data(tfwp_text, imp_text)
        frames.append(work_df)
        source_counts["work_permit"] = len(work_df)

    if not frames:
        logger.warning("All IRCC downloads failed. Falling back to sample data.")
        return _generate_sample_data()

    combined = pd.concat(frames, ignore_index=True)

    # Aggregate: sum across months within same year/quarter/province/cohort
    combined = (
        combined
        .groupby(["year", "quarter", "province", "cohort_type"])["intake_count"]
        .sum()
        .reset_index()
    )

    logger.info(
        "Loaded IRCC real data: %d records from sources %s, "
        "years %d-%d, %d provinces",
        len(combined),
        source_counts,
        combined["year"].min(),
        combined["year"].max(),
        combined["province"].nunique(),
    )

    # Check for francophone_pr cohort - IRCC doesn't have a direct category
    # If no francophone_pr records exist, allocate from Quebec PR data
    if "francophone_pr" not in combined["cohort_type"].values:
        combined = _add_francophone_pr(combined)

    return combined


def _add_francophone_pr(df: pd.DataFrame) -> pd.DataFrame:
    """Add francophone_pr cohort by allocating a share of Quebec PR intake.

    Uses IRCC's published francophone immigration targets (~5% nationally).
    """
    qc_pr = df[
        (df["province"] == "QC") &
        (df["cohort_type"].isin(["high_wage_worker", "in_canada_transition"]))
    ].copy()

    if qc_pr.empty:
        # Create synthetic francophone_pr from overall QC intake
        qc_all = df[df["province"] == "QC"].copy()
        if qc_all.empty:
            return df
        qc_pr = qc_all

    # Allocate ~5% of Quebec intake as francophone PR
    francophone_share = 0.05
    records = []
    for (year, quarter), group in qc_pr.groupby(["year", "quarter"]):
        total = group["intake_count"].sum()
        fr_count = max(1, int(total * francophone_share))
        records.append({
            "year": year,
            "quarter": quarter,
            "province": "QC",
            "cohort_type": "francophone_pr",
            "intake_count": fr_count,
        })

    fr_df = pd.DataFrame(records)
    result = pd.concat([df, fr_df], ignore_index=True)
    logger.info("Added %d francophone_pr records from Quebec PR data", len(fr_df))
    return result


def load_ircc_data(raw_dir: Path, force_regenerate: bool = False) -> pd.DataFrame:
    """Load IRCC immigration intake data.

    Attempts to download real data from IRCC Open Data portal first.
    Falls back to a pre-existing raw CSV, then to generated sample data.

    Args:
        raw_dir: Path to raw data directory.
        force_regenerate: If True, skip download and generate sample data.

    Returns:
        DataFrame with columns: year, quarter, province, cohort_type, intake_count
    """
    if not force_regenerate:
        # Try downloading real data
        try:
            real_data = _download_and_parse_all()
            if not real_data.empty:
                # Cache to raw directory
                raw_dir = Path(raw_dir)
                raw_dir.mkdir(parents=True, exist_ok=True)
                cache_path = raw_dir / "ircc_intake.csv"
                real_data.to_csv(cache_path, index=False)
                logger.info("Cached IRCC real data to %s", cache_path)
                return real_data
        except Exception as exc:
            logger.warning("IRCC download failed: %s. Trying cached data.", exc)

    # Try cached raw CSV
    csv_path = Path(raw_dir) / "ircc_intake.csv"
    if csv_path.exists():
        logger.info("Loading cached IRCC data from %s", csv_path)
        cached = pd.read_csv(csv_path)
        if not cached.empty and "year" in cached.columns:
            return cached

    logger.warning(
        "No IRCC real data available. Generating structured sample data "
        "based on IRCC Annual Report totals and distributions."
    )
    return _generate_sample_data()


def _generate_sample_data() -> pd.DataFrame:
    """Generate structured sample data based on IRCC published totals.

    Uses actual annual totals from IRCC reports with proportional allocation
    by province, cohort type, and quarter.
    """
    rng = np.random.default_rng(42)
    records = []

    for year, total in TOTAL_INTAKE.items():
        for province in PROVINCES:
            prov_share = PROVINCE_SHARES.get(province, 0.005)
            prov_total = int(total * prov_share)

            for cohort in COHORT_TYPES:
                cohort_share = COHORT_SHARES.get(cohort, 0.05)
                cohort_total = int(prov_total * cohort_share)

                quarterly_pattern = np.array([0.22, 0.24, 0.30, 0.24])
                if "student" in cohort:
                    quarterly_pattern = np.array([0.15, 0.20, 0.40, 0.25])

                noise = rng.normal(1.0, 0.05, size=4)
                quarterly_pattern = quarterly_pattern * noise
                quarterly_pattern = quarterly_pattern / quarterly_pattern.sum()

                for q_idx, q_share in enumerate(quarterly_pattern):
                    quarter = q_idx + 1
                    count = max(0, int(cohort_total * q_share))
                    records.append({
                        "year": year,
                        "quarter": quarter,
                        "province": province,
                        "cohort_type": cohort,
                        "intake_count": count,
                    })

    df = pd.DataFrame(records)
    logger.info(
        "Generated IRCC sample data: %d records, years %d-%d",
        len(df), df["year"].min(), df["year"].max()
    )
    return df


def clean_ircc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate IRCC data.

    - Ensures correct dtypes
    - Validates province codes
    - Validates cohort types
    - Handles missing values
    """
    df = df.copy()

    df["year"] = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    df["intake_count"] = df["intake_count"].fillna(0).astype(int)

    # Drop source column if present (not needed downstream)
    if "source" in df.columns:
        df = df.drop(columns=["source"])

    # Validate provinces
    invalid_provinces = set(df["province"].unique()) - set(PROVINCES)
    if invalid_provinces:
        logger.warning("Unknown province codes: %s", invalid_provinces)
        df = df[df["province"].isin(PROVINCES)]

    # Validate cohort types
    valid_cohorts = set(df["cohort_type"].unique()) - set(COHORT_TYPES)
    if valid_cohorts:
        logger.warning("Unknown cohort types: %s", valid_cohorts)

    df = df.sort_values(["year", "quarter", "province", "cohort_type"]).reset_index(drop=True)
    return df


def run(raw_dir: Path, cleaned_dir: Path) -> pd.DataFrame:
    """Run the full IRCC data pipeline: download/load -> clean -> save.

    Args:
        raw_dir: Path to raw data directory.
        cleaned_dir: Path to cleaned data directory.

    Returns:
        Cleaned DataFrame.
    """
    raw_dir = Path(raw_dir)
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    df = load_ircc_data(raw_dir)
    df = clean_ircc_data(df)

    output_path = cleaned_dir / "ircc_intake.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Saved cleaned IRCC data to %s (%d rows)", output_path, len(df))

    return df
