# Canadian Immigration ACI Research Tool

A three-layer Python research tool for analyzing Canada's 2018-2026 immigration restructuring through Absorption Capacity Index (ACI) constrained optimization.

## Architecture

```
Layer 1: Data Pipeline   → master_panel.parquet
Layer 2: Simulation      → counterfactual_series.csv, welfare_loss, NIV, lambda regression
Layer 3: Visualization   → 6 Plotly charts (HTML+PNG) + Dash dashboard
```

### Project Structure

```
src/
  pipeline/          # Data loaders + joiner
    ircc_loader.py   # Immigration intake (IRCC open data)
    cmhc_loader.py   # Housing starts & vacancy rates (CMHC)
    statcan_loader.py # Labour & fiscal data (StatCan)
    cihi_loader.py   # Health system capacity (CIHI)
    school_loader.py  # School capacity (manual)
    joiner.py         # Merge all sources + compute ACI
  simulation/        # Optimization & analysis
    cohort_npv.py     # Cohort Net Immigration Value
    aci_optimizer.py  # LP constrained optimization (scipy)
    counterfactual.py # Counterfactual intake generator
    welfare_loss.py   # Welfare loss calculator + decomposition
    lambda_identifier.py  # Rank-2 projection test (statsmodels OLS)
  viz/               # Visualization
    chart_config.py   # Shared styling constants
    static_charts.py  # 6 Plotly charts
    dashboard.py      # Dash interactive dashboard
config/
  cohort_params.yaml  # 8 cohort types with documented parameters
data/
  master/            # master_panel.parquet, annual_panel.parquet
outputs/
  data/              # CSV outputs
  charts/            # HTML + PNG charts
  reports/           # summary_stats.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11+.

### Dependencies

- pandas, numpy, scipy — data processing & optimization
- statsmodels — Lambda regression with HC3 robust SEs
- plotly, kaleido — interactive charts with PNG export
- dash, dash-bootstrap-components — interactive dashboard
- pyarrow — parquet file support
- pyyaml — YAML config loading
- requests, beautifulsoup4 — data source scraping (fallback)
- pytest, pytest-cov — testing

## Usage

### Run the full pipeline

```bash
# Layer 1: Data Pipeline
python -m src.pipeline

# Layer 2: Simulation Engine
python -m src.simulation

# Layer 3: Generate charts
python -c "from src.viz.static_charts import generate_all_charts; generate_all_charts('.')"

# Layer 3: Launch dashboard
python -m src.viz.dashboard
```

### Run tests

```bash
python -m pytest tests/ -v --cov=src
```

## Data Sources

| Source | Variables | Provider |
|--------|-----------|----------|
| Immigration intake | Arrivals by permit type, province, category | IRCC Open Data |
| Housing starts & vacancy rates | `starts_annual`, `vacancy_rate`, `avg_rent_2br` | CMHC |
| Wait times & bed capacity | Wait time benchmarks, occupancy rates | CIHI |
| School enrollment | Student-teacher ratio, capacity utilization | Provincial education reports |
| Labour & fiscal | Unemployment, wages, vacancy rate, fiscal balance | StatCan |

### Data Download Status

| Source | Status | Method |
|--------|--------|--------|
| IRCC immigration intake | **Live download** | Downloads PR, study permit, TFWP, and IMP CSVs from `ircc.canada.ca/opendata-donneesouvertes/data/` on each run |
| Housing starts & vacancy rates | Sample data | Generated from published CMHC totals (awaiting raw CSV or API integration) |
| Health wait times & bed capacity | Manual fallback | Loads from `manual_data/cihi_fallback.csv` |
| School enrollment | Manual fallback | Loads from `manual_data/school_capacity.csv` |
| Labour & fiscal | Sample data | Generated from published StatCan totals (awaiting raw CSV or API integration) |

The IRCC loader downloads real data from four IRCC Open Data CSV files covering 2015-2026 (filtering to 2018+). Data is cached in `data/raw/ircc_intake.csv` after first download. Category mapping from IRCC immigration categories to project cohort types follows documented IRCC classification rules.

Sources that are not yet downloading live data fall back to generated sample data based on published totals and documented distributions. Each loader checks for raw CSV files first and falls back to generated data with appropriate provincial shares and temporal trends.

## Key Concepts

### Absorption Capacity Index (ACI)

```
ACI_r,t = ω1·(vacancy_rate) + ω2·(starts_per_capita_growth) + ω3·(health_capacity)
        + ω4·(school_capacity) + ω5·(job_quality) + ω6·(fiscal_balance)
```

Three weight scenarios:
- **Housing-Heavy**: ω_housing = 0.35
- **Equal**: all ω = 1/6
- **Fiscal-Heavy**: ω_fiscal = 0.35

### Cohort Net Immigration Value (NIV)

```
NIV_i = PV(tax_contribution + transition_value) - PV(housing + health + education + integrity + settlement)
```

Computed for 8 cohort types: high/low-quality students, high/low-wage workers, francophone PR, in-Canada transition, family class, refugees.

### ACI Constrained Optimization

Linear program via `scipy.optimize.linprog`:
```
max  Σ_i Σ_r NIV_i · N_{i,r,t}
s.t. ACI_{r,t} ≥ ACI_min,  Σ_r N_{i,r,t} = Total_{i,t},  N_{i,r,t} ≥ 0
```

### Lambda Identification

Statsmodels OLS with HC3 robust SEs tests whether policy lever changes concentrate on political utility dimensions (volume optics, Quebec leverage) with near-zero loadings on absorptive capacity. Results are reported honestly even when they contradict the thesis.

## Output Files

| File | Description |
|------|-------------|
| `data/master/master_panel.parquet` | Full analytical base (year × quarter × province × cohort_type) |
| `outputs/data/counterfactual_series.csv` | Optimal vs actual intake under 3 ACI scenarios |
| `outputs/data/welfare_loss_decomposition.csv` | Welfare loss by province/year |
| `outputs/data/dimensional_decomposition.csv` | Loss attribution by ACI dimension |
| `outputs/data/cohort_niv.csv` | NIV rankings for 8 cohort types |
| `outputs/data/lambda_regression_results.csv` | Rank-2 projection regression results |
| `outputs/charts/*.html` | 6 interactive Plotly charts |
| `outputs/charts/*.png` | High-resolution PNG exports |

## Cohort Parameters

All cohort parameters are documented in `config/cohort_params.yaml` with sources:

- Annual earnings from StatCan wage data
- Transition probabilities from IRCC PR admission rates
- Tax contributions estimated from CRA marginal rates
- Housing/health/education costs from CMHC, CIHI, provincial reports
- Discount rates and time horizons based on OECD longitudinal studies

## Assumptions & Limitations

1. **Data availability**: When raw open data is unavailable, structured sample data is generated based on published totals. This is documented in each loader.
2. **ACI normalization**: Min-max normalization to [0,1] range; constant series handled by assigning 0.5.
3. **LP feasibility**: The optimizer includes a fallback allocation when the LP solver fails to find a feasible solution.
4. **Lambda regression**: With 8-9 annual observations, statistical power is limited. Results are reported honestly as INCONCLUSIVE when insufficient.
5. **Refugee/francophone floors**: Soft constraints ensure minimum allocations for humanitarian and linguistic categories.

## Test Coverage

54 tests across 3 test modules with 81% line coverage:
- `tests/test_pipeline.py` — 22 tests for data loaders, joiner, ACI computation
- `tests/test_simulation.py` — 22 tests for NPV, optimizer, counterfactual, welfare, lambda
- `tests/test_viz.py` — 10 tests for chart config, all 6 charts, output verification
