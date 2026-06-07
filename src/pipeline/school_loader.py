"""School Capacity Data Loader.

Loads school capacity data from manual CSV stub.
Source: Provincial education ministry enrollment data.
File: manual_data/school_capacity.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROVINCES = [
    "ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "NT", "YT", "NU"
]


def load_school_data(manual_dir: Path) -> pd.DataFrame:
    """Load school capacity data from manual CSV.

    Args:
        manual_dir: Path to manual_data directory.

    Returns:
        DataFrame with school capacity indicators.
    """
    manual_path = manual_dir / "school_capacity.csv"

    if not manual_path.exists():
        logger.error(
            "No school capacity data found at %s. "
            "Please populate manual_data/school_capacity.csv.",
            manual_path
        )
        return pd.DataFrame()

    logger.info("Loading school capacity data from %s", manual_path)
    return pd.read_csv(manual_path)


def clean_school_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate school capacity data.

    Constructs a normalized school_capacity score from enrollment
    and utilization metrics.
    """
    if df.empty:
        logger.warning("Empty school capacity DataFrame — returning empty result.")
        return df

    df = df.copy()
    df["year"] = df["year"].astype(int)

    # Validate provinces
    invalid = set(df["province"].unique()) - set(PROVINCES)
    if invalid:
        logger.warning("Unknown province codes: %s", invalid)
        df = df[df["province"].isin(PROVINCES)]

    # Lower capacity utilization means more slack = better capacity
    # Normalize to [0, 1] where 1 = best capacity (most slack)
    if "capacity_utilization_pct" in df.columns:
        util_max = df["capacity_utilization_pct"].max()
        if util_max > 0:
            df["capacity_slack"] = 1.0 - (df["capacity_utilization_pct"] / util_max)
        else:
            df["capacity_slack"] = np.nan

    # Lower student-teacher ratio is better, normalize inversely
    if "student_teacher_ratio" in df.columns:
        str_max = df["student_teacher_ratio"].max()
        if str_max > 0:
            df["teaching_capacity"] = 1.0 - (df["student_teacher_ratio"] / str_max)
        else:
            df["teaching_capacity"] = np.nan

    # Composite school capacity score
    components = []
    if "capacity_slack" in df.columns:
        components.append(0.6 * df["capacity_slack"])
    if "teaching_capacity" in df.columns:
        components.append(0.4 * df["teaching_capacity"])

    if components:
        df["school_capacity"] = sum(components)
    else:
        df["school_capacity"] = np.nan

    # Flag data quality
    key_cols = ["public_school_enrollment", "capacity_utilization_pct", "student_teacher_ratio"]
    nan_count = df[key_cols].isna().sum(axis=1)
    df["school_data_flag"] = nan_count.map(
        lambda x: "complete" if x == 0 else f"missing_{x}_of_{len(key_cols)}"
    )

    df = df.sort_values(["province", "year"]).reset_index(drop=True)
    return df


def run(manual_dir: Path, cleaned_dir: Path) -> pd.DataFrame:
    """Run the full school capacity data pipeline."""
    manual_dir = Path(manual_dir)
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    df = load_school_data(manual_dir)
    df = clean_school_data(df)

    if not df.empty:
        output_path = cleaned_dir / "school_capacity.parquet"
        df.to_parquet(output_path, index=False)
        logger.info("Saved cleaned school data to %s (%d rows)", output_path, len(df))

    return df
