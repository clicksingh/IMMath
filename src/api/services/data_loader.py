"""Central data loading for the GraphQL API.

Reads parquet/CSV files once (lru_cache) and returns copies
to preserve immutability. Reuses the same file paths as the
Dash dashboard and simulation engine.
"""

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Walk up from this file to find the project root (directory containing data/)
def _find_project_root() -> Path:
    """Find project root by walking up from this file until finding data/."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "data" / "master" / "master_panel.parquet").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    # Fallback: use cwd
    return Path.cwd()

_DEFAULT_BASE = _find_project_root()


def _resolve_base(base_dir: Path | None) -> Path:
    return Path(base_dir) if base_dir else _DEFAULT_BASE


@lru_cache(maxsize=1)
def _get_master_panel_cached(base_str: str) -> pd.DataFrame:
    base = Path(base_str)
    path = base / "data" / "master" / "master_panel.parquet"
    logger.info("Loading master panel from %s", path)
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _get_counterfactual_cached(base_str: str) -> pd.DataFrame:
    base = Path(base_str)
    path = base / "outputs" / "data" / "counterfactual_series.csv"
    logger.info("Loading counterfactual from %s", path)
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _get_welfare_loss_cached(base_str: str) -> pd.DataFrame:
    base = Path(base_str)
    path = base / "outputs" / "data" / "welfare_loss_decomposition.csv"
    logger.info("Loading welfare loss from %s", path)
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _get_decomposition_cached(base_str: str) -> pd.DataFrame:
    base = Path(base_str)
    path = base / "outputs" / "data" / "dimensional_decomposition.csv"
    logger.info("Loading dimensional decomposition from %s", path)
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _get_cohort_niv_cached(base_str: str) -> pd.DataFrame:
    base = Path(base_str)
    path = base / "outputs" / "data" / "cohort_niv.csv"
    logger.info("Loading cohort NIV from %s", path)
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _get_lambda_results_cached(base_str: str) -> pd.DataFrame:
    base = Path(base_str)
    path = base / "outputs" / "data" / "lambda_regression_results.csv"
    logger.info("Loading lambda results from %s", path)
    return pd.read_csv(path)


def get_master_panel(base_dir: Path | None = None) -> pd.DataFrame:
    return _get_master_panel_cached(str(_resolve_base(base_dir))).copy()


def get_counterfactual(base_dir: Path | None = None) -> pd.DataFrame:
    return _get_counterfactual_cached(str(_resolve_base(base_dir))).copy()


def get_welfare_loss(base_dir: Path | None = None) -> pd.DataFrame:
    return _get_welfare_loss_cached(str(_resolve_base(base_dir))).copy()


def get_decomposition(base_dir: Path | None = None) -> pd.DataFrame:
    return _get_decomposition_cached(str(_resolve_base(base_dir))).copy()


def get_cohort_niv(base_dir: Path | None = None) -> pd.DataFrame:
    return _get_cohort_niv_cached(str(_resolve_base(base_dir))).copy()


def get_lambda_results(base_dir: Path | None = None) -> pd.DataFrame:
    return _get_lambda_results_cached(str(_resolve_base(base_dir))).copy()
