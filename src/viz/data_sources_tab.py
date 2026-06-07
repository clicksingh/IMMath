"""Data Sources tab — provenance, freshness status, and download links.

Renders a static tab (no callbacks) showing all data sources with
their origin, freshness indicators, and download buttons. Reads
metadata from config/data_sources.yaml.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dash import html
import dash_bootstrap_components as dbc

from .chart_config import COLORS

logger = logging.getLogger(__name__)

# Seconds before a file is considered stale
_FRESH_THRESHOLD = 30 * 86400   # 30 days
_STALE_THRESHOLD = 90 * 86400   # 90 days

# Download route prefix (served by FastAPI)
_DOWNLOAD_PREFIX = "/dash/downloads"

_MECHANISM_COLORS = {
    "auto": COLORS["success"],
    "semi-auto": COLORS["secondary"],
    "manual": COLORS["warning"],
}


def _format_size(size_bytes: int) -> str:
    """Return human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _get_file_stats(path: Path) -> dict:
    """Return file existence, size, modification time, and row count."""
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "size_display": "--",
                "last_modified": None, "last_modified_display": "N/A",
                "row_count": None}

    stat = path.stat()
    mtime = stat.st_mtime
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    row_count = None
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
            row_count = pq.read_metadata(path).num_rows
        except Exception:
            row_count = None
    elif suffix == ".csv":
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                row_count = sum(1 for _ in f) - 1  # subtract header
        except Exception:
            row_count = None

    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "size_display": _format_size(stat.st_size),
        "last_modified": mtime,
        "last_modified_display": dt.strftime("%Y-%m-%d"),
        "row_count": row_count,
    }


def _freshness_badge(last_modified: float | None) -> html.Span:
    """Return a colored dot indicating data freshness."""
    if last_modified is None:
        color = COLORS["accent"]
        title = "File not found"
    elif time.time() - last_modified < _FRESH_THRESHOLD:
        color = COLORS["success"]
        title = "Fresh (< 30 days)"
    elif time.time() - last_modified < _STALE_THRESHOLD:
        color = COLORS["warning"]
        title = "Aging (30-90 days)"
    else:
        color = COLORS["accent"]
        title = "Stale (> 90 days)"

    return html.Span(
        "",
        title=title,
        style={
            "display": "inline-block",
            "width": 12,
            "height": 12,
            "borderRadius": "50%",
            "backgroundColor": color,
            "marginRight": 6,
            "verticalAlign": "middle",
        },
    )


def _build_source_card(source: dict, base_dir: Path) -> dbc.Card:
    """Render one data source as a Bootstrap card."""
    mechanism = source.get("update_mechanism", "manual")
    badge_color = _MECHANISM_COLORS.get(mechanism, COLORS["grey"])

    # File stats from cleaned output
    cleaned_path = base_dir / source["cleaned_file"]
    stats = _get_file_stats(cleaned_path)

    # Count raw files that exist
    raw_existing = sum(
        1 for f in source.get("raw_files", [])
        if (base_dir / f).exists()
    )
    raw_total = len(source.get("raw_files", []))

    # Download URL for cleaned file
    category = "cleaned"
    filename = Path(source["cleaned_file"]).name
    download_url = f"{_DOWNLOAD_PREFIX}/{category}/{filename}"

    rows_text = f"{stats['row_count']:,}" if stats["row_count"] is not None else "--"
    raw_text = f"{raw_existing}/{raw_total} files" if raw_total else "--"

    header = dbc.CardHeader([
        html.Strong(source["display_name"]),
            html.Span(
            mechanism.upper(),
            style={
                "marginLeft": 10,
                "padding": "2px 8px",
                "borderRadius": 4,
                "backgroundColor": badge_color,
                "color": "white",
                "fontSize": "0.75em",
                "fontWeight": "bold",
                "float": "right",
            },
        ),
    ])

    body = dbc.CardBody([
        html.P(source["organization"], style={"marginBottom": 4, "fontWeight": 500}),
        html.Small(
            source["description"].strip(),
            style={"color": COLORS["dark"], "display": "block", "marginBottom": 8},
        ),
        html.Div([
            html.Span(f"Coverage: {source['coverage_years']}", style={"marginRight": 16}),
            html.Span(f"Frequency: {source['update_frequency']}"),
        ], style={"fontSize": "0.85em", "color": COLORS["grey"], "marginBottom": 8}),
        html.A(
            source["url"] if source["url"] != "#" else None,
            href=source["url"] if source["url"] != "#" else None,
            target="_blank",
            style={"fontSize": "0.85em"} if source["url"] != "#" else {"display": "none"},
        ),
    ])

    footer = dbc.CardFooter([
        _freshness_badge(stats["last_modified"]),
        html.Span(f"Updated: {stats['last_modified_display']}", style={"marginRight": 16}),
        html.Span(f"Rows: {rows_text}", style={"marginRight": 16}),
        html.Span(f"Size: {stats['size_display']}", style={"marginRight": 16}),
        html.Span(f"Raw: {raw_text}", style={"marginRight": 8}),
        html.A(
            "Download",
            href=download_url,
            download=filename,
            className="btn btn-sm btn-outline-primary",
            style={"float": "right", "fontSize": "0.8em"},
        ) if stats["exists"] else html.Span(
            "No file",
            style={"float": "right", "color": COLORS["accent"], "fontSize": "0.8em"},
        ),
    ], style={"fontSize": "0.85em"})

    return dbc.Card([header, body, footer])


def _build_derived_outputs(outputs: list[dict], base_dir: Path) -> dbc.Table:
    """Render derived outputs as a table with download links."""
    rows = []
    for out in outputs:
        path = base_dir / out["file"]
        stats = _get_file_stats(path)
        rows.append({
            "Name": out["display_name"],
            "Description": out["description"].strip(),
            "Rows": f"{stats['row_count']:,}" if stats["row_count"] is not None else "--",
            "Size": stats["size_display"],
            "Updated": stats["last_modified_display"],
            "Status": "OK" if stats["exists"] else "Missing",
            "": html.A(
                "Download",
                href=f"{_DOWNLOAD_PREFIX}/{out['category']}/{Path(out['file']).name}",
                download=Path(out["file"]).name,
                className="btn btn-sm btn-outline-primary",
            ) if stats["exists"] else html.Span("--", style={"color": COLORS["grey"]}),
        })

    return dbc.Table.from_dataframe(
        __import__("pandas").DataFrame(rows),
        striped=True, bordered=True, hover=True, size="sm",
        style={"fontSize": "0.9em"},
    )


def _count_total_records(sources: list[dict], base_dir: Path) -> int:
    """Sum row counts across all cleaned files."""
    total = 0
    for src in sources:
        stats = _get_file_stats(base_dir / src["cleaned_file"])
        if stats["row_count"] is not None:
            total += stats["row_count"]
    return total


def _count_fresh_sources(sources: list[dict], base_dir: Path) -> int:
    """Count sources with fresh (< 30 days) cleaned files."""
    count = 0
    for src in sources:
        stats = _get_file_stats(base_dir / src["cleaned_file"])
        if stats["last_modified"] and time.time() - stats["last_modified"] < _FRESH_THRESHOLD:
            count += 1
    return count


def build_data_sources_tab(base_dir: Path) -> html.Div:
    """Build the Data Sources tab content.

    Args:
        base_dir: Project root directory.

    Returns:
        Dash html.Div with the complete tab layout.
    """
    config_path = base_dir / "config" / "data_sources.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        logger.error("Failed to load data sources config: %s", exc)
        return html.Div([
            html.H4("Data Sources", style={"color": COLORS["accent"]}),
            html.P(f"Could not load data source configuration: {exc}"),
        ])

    sources = config.get("sources", [])
    outputs = config.get("derived_outputs", [])

    # Summary banner
    total_records = _count_total_records(sources, base_dir)
    fresh_count = _count_fresh_sources(sources, base_dir)

    summary = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{len(sources)}", style={"color": COLORS["primary"], "marginBottom": 0}),
            html.Small("Data Sources"),
        ]), style={"textAlign": "center"}), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{total_records:,}", style={"color": COLORS["primary"], "marginBottom": 0}),
            html.Small("Total Records"),
        ]), style={"textAlign": "center"}), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"{fresh_count}/{len(sources)}", style={"color": COLORS["success"] if fresh_count == len(sources) else COLORS["warning"], "marginBottom": 0}),
            html.Small("Fresh Sources"),
        ]), style={"textAlign": "center"}), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("2024-2025", style={"color": COLORS["primary"], "marginBottom": 0}),
            html.Small("Academic Year"),
        ]), style={"textAlign": "center"}), width=3),
    ], style={"marginBottom": 24})

    # Source cards in a grid
    source_cards = []
    for src in sources:
        source_cards.append(dbc.Col(
            _build_source_card(src, base_dir),
            width=4, style={"marginBottom": 16},
        ))

    source_grid = dbc.Row(source_cards)

    # Derived outputs section
    outputs_section = html.Div([
        html.H4("Derived Outputs", style={"marginTop": 24, "color": COLORS["primary"]}),
        html.P(
            "Files generated by the pipeline from raw and cleaned sources.",
            style={"color": COLORS["grey"]},
        ),
        _build_derived_outputs(outputs, base_dir),
    ])

    # Freshness legend
    legend = html.Div([
        html.Small("Freshness: "),
        _freshness_badge(time.time()),
        html.Small("< 30 days  ", style={"marginRight": 12}),
        _freshness_badge(time.time() - 45 * 86400),
        html.Small("30-90 days  ", style={"marginRight": 12}),
        _freshness_badge(None),
        html.Small("Missing"),
    ], style={"marginTop": 16, "marginBottom": 16})

    return html.Div([
        html.H3("Data Sources & Provenance", style={
            "color": COLORS["primary"], "marginBottom": 8,
        }),
        html.P(
            "This research tool integrates data from Canadian government open data "
            "portals. Automatic sources can be refreshed via the pipeline; manual "
            "sources require curated updates.",
            style={"color": COLORS["grey"], "marginBottom": 20},
        ),
        summary,
        source_grid,
        outputs_section,
        legend,
    ])
