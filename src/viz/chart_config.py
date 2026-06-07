"""Chart Configuration — Shared styling for all visualizations."""

# Color palette (accessible, print-friendly)
COLORS = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#E74C3C",
    "success": "#27AE60",
    "warning": "#F39C12",
    "dark": "#2C3E50",
    "light": "#ECF0F1",
    "grey": "#95A5A6",
}

# Cohort type colors
COHORT_COLORS = {
    "high_quality_student": "#2E86C1",
    "low_quality_student": "#85C1E9",
    "high_wage_worker": "#27AE60",
    "low_wage_worker": "#82E0AA",
    "francophone_pr": "#8E44AD",
    "in_canada_transition": "#F39C12",
    "family_class": "#E74C3C",
    "refugee": "#C0392B",
}

# Province colors
PROVINCE_COLORS = {
    "ON": "#1B4F72",
    "QC": "#8E44AD",
    "BC": "#27AE60",
    "AB": "#E74C3C",
    "MB": "#F39C12",
    "SK": "#2E86C1",
    "NS": "#D35400",
    "NB": "#16A085",
    "NL": "#C0392B",
    "PE": "#7D3C98",
    "NT": "#5D6D7E",
    "YT": "#AAB7B8",
    "NU": "#34495E",
}

# Dimension colors for welfare loss decomposition
DIMENSION_COLORS = {
    "housing_vacancy": "#2E86C1",
    "housing_starts": "#1B4F72",
    "health_capacity": "#E74C3C",
    "school_capacity": "#27AE60",
    "job_quality": "#F39C12",
    "fiscal_balance": "#8E44AD",
}

# Common layout settings
LAYOUT_DEFAULTS = {
    "font": {"family": "Helvetica Neue, Arial, sans-serif", "size": 12, "color": "#2C3E50"},
    "title": {"font": {"size": 16, "color": "#1B4F72"}},
    "paper_bgcolor": "white",
    "plot_bgcolor": "#FAFAFA",
    "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
}

# Axis defaults
AXIS_DEFAULTS = {
    "gridcolor": "#E5E8E8",
    "zerolinecolor": "#BDC3C7",
    "title": {"font": {"size": 13}},
}

# Export settings
EXPORT_SETTINGS = {
    "width": 1200,
    "height": 700,
    "scale": 2,
}


def get_cohort_display_name(cohort_type: str) -> str:
    """Get human-readable display name for cohort type."""
    names = {
        "high_quality_student": "High-Quality Student",
        "low_quality_student": "Low-Quality Student",
        "high_wage_worker": "High-Wage Worker",
        "low_wage_worker": "Low-Wage Worker",
        "francophone_pr": "Francophone PR",
        "in_canada_transition": "In-Canada Transition",
        "family_class": "Family Class",
        "refugee": "Refugee",
    }
    return names.get(cohort_type, cohort_type)


def get_province_display_name(province: str) -> str:
    """Get full province name."""
    names = {
        "ON": "Ontario", "QC": "Quebec", "BC": "British Columbia",
        "AB": "Alberta", "MB": "Manitoba", "SK": "Saskatchewan",
        "NS": "Nova Scotia", "NB": "New Brunswick",
        "NL": "Newfoundland & Labrador", "PE": "Prince Edward Island",
        "NT": "Northwest Territories", "YT": "Yukon", "NU": "Nunavut",
    }
    return names.get(province, province)
