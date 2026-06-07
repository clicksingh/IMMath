"""Tests for the Data Sources tab module."""

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.viz.data_sources_tab import (
    _format_size,
    _freshness_badge,
    _get_file_stats,
    build_data_sources_tab,
)


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500 B"

    def test_kilobytes(self):
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(5120) == "5.0 KB"

    def test_megabytes(self):
        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"


class TestGetFileStats:
    def test_missing_file(self, tmp_path):
        stats = _get_file_stats(tmp_path / "nonexistent.parquet")
        assert stats["exists"] is False
        assert stats["row_count"] is None
        assert stats["last_modified"] is None

    def test_parquet_file(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        stats = _get_file_stats(path)
        assert stats["exists"] is True
        assert stats["row_count"] == 3
        assert stats["size_bytes"] > 0
        assert stats["last_modified"] is not None

    def test_csv_file(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("col1,col2\n1,2\n3,4\n5,6\n")

        stats = _get_file_stats(path)
        assert stats["exists"] is True
        assert stats["row_count"] == 3  # 4 lines - 1 header
        assert stats["size_bytes"] > 0


class TestFreshnessBadge:
    def test_fresh(self):
        badge = _freshness_badge(time.time())
        assert badge.style["backgroundColor"] == "#27AE60"

    def test_aging(self):
        badge = _freshness_badge(time.time() - 45 * 86400)
        assert badge.style["backgroundColor"] == "#F39C12"

    def test_stale(self):
        badge = _freshness_badge(time.time() - 100 * 86400)
        assert badge.style["backgroundColor"] == "#E74C3C"

    def test_missing(self):
        badge = _freshness_badge(None)
        assert badge.style["backgroundColor"] == "#E74C3C"


class TestBuildDataSourcesTab:
    def test_renders_without_error(self):
        from pathlib import Path
        base_dir = Path(__file__).resolve().parents[1]
        result = build_data_sources_tab(base_dir)
        assert result is not None

    def test_contains_source_names(self):
        base_dir = Path(__file__).resolve().parents[1]
        result = build_data_sources_tab(base_dir)
        # Convert to string to check content
        from dash import html
        # Check that IRCC and StatCan are mentioned
        text = str(result)
        assert "IRCC" in text
        assert "StatCan" in text
        assert "CMHC" in text
        assert "CIHI" in text
        assert "School" in text
