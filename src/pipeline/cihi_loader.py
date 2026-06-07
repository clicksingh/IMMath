"""CIHI Health Capacity Data Loader.

Loads health system capacity data from CIHI or manual CSV fallback.
Source: CIHI Wait Times for Health Services.
Fallback: manual_data/cihi_fallback.csv.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROVINCES = [
    "ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "NT", "YT", "NU"
]


def load_cihi_data(raw_dir: Path, manual_dir: Path) -> pd.DataFrame:
    """Load CIHI health capacity data.

    Priority:
    1. Raw CSV from CIHI scrape/download
    2. Manual fallback CSV from manual_data/

    Args:
        raw_dir: Path to raw data directory.
        manual_dir: Path to manual_data directory.

    Returns:
        DataFrame with health capacity indicators.
    """
    raw_path = raw_dir / "cihi_health.csv"
    manual_path = manual_dir / "cihi_fallback.csv"

    if raw_path.exists():
        logger.info("Loading CIHI data from %s", raw_path)
        return pd.read_csv(raw_path)

    if manual_path.exists():
        logger.info("Loading CIHI data from manual fallback: %s", manual_path)
        return pd.read_csv(manual_path)

    logger.error(
        "No CIHI data found. Checked: %s, %s. "
        "Please provide health capacity data.",
        raw_path, manual_path
    )
    return pd.DataFrame()


def clean_cihi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate CIHI data.

    Normalizes health capacity indicators and constructs a composite
    health_capacity score.
    """
    if df.empty:
        logger.warning("Empty CIHI DataFrame — returning empty result.")
        return df

    df = df.copy()
    df["year"] = df["year"].astype(int)

    # Validate provinces
    invalid = set(df["province"].unique()) - set(PROVINCES)
    if invalid:
        logger.warning("Unknown province codes in CIHI data: %s", invalid)
        df = df[df["province"].isin(PROVINCES)]

    # Handle duplicate rows for AB in 2024 (data entry issue)
    df = df.drop_duplicates(subset=["year", "province"], keep="first")

    # Normalize health capacity indicators (lower wait time = higher capacity)
    # Invert wait times so higher value = better capacity
    wait_max = df["median_wait_priority_days"].max()
    df["wait_capacity"] = 1.0 - (df["median_wait_priority_days"] / wait_max)

    # Lower occupancy is better (more slack capacity)
    df["occupancy_slack"] = 1.0 - (df["hospital_occupancy_pct"] / 100.0)

    # Lower ER wait is better
    er_max = df["er_wait_hours"].max()
    df["er_capacity"] = 1.0 - (df["er_wait_hours"] / er_max)

    # Composite health capacity (equal weight on sub-components)
    df["health_capacity"] = (
        0.4 * df["wait_capacity"] +
        0.3 * df["occupancy_slack"] +
        0.3 * df["er_capacity"]
    )

    # Flag NaN-heavy dimensions
    health_cols = ["median_wait_priority_days", "hospital_occupancy_pct", "er_wait_hours"]
    nan_count = df[health_cols].isna().sum(axis=1)
    df["health_data_flag"] = nan_count.map(
        lambda x: "complete" if x == 0 else f"missing_{x}_of_{len(health_cols)}"
    )

    df = df.sort_values(["province", "year"]).reset_index(drop=True)
    return df


def run(raw_dir: Path, cleaned_dir: Path, manual_dir: Path | None = None) -> pd.DataFrame:
    """Run the full CIHI data pipeline."""
    raw_dir = Path(raw_dir)
    cleaned_dir = Path(cleaned_dir)
    manual_dir = Path(manual_dir) if manual_dir else Path("manual_data")
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    df = load_cihi_data(raw_dir, manual_dir)
    df = clean_cihi_data(df)

    if not df.empty:
        output_path = cleaned_dir / "cihi_health.parquet"
        df.to_parquet(output_path, index=False)
        logger.info("Saved cleaned CIHI data to %s (%d rows)", output_path, len(df))

    return df
