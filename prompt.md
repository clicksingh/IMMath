# Claude Code Prompt: Canadian Immigration ACI Research Tool

## Project Overview

Build a three-layer Python research tool for analyzing Canada's 2018–2026 immigration restructuring. The tool has three components that feed into each other:

1. **Data Pipeline** — pulls, cleans, and joins public Canadian datasets into one analytical base
2. **Simulation Engine** — runs an Absorption Capacity Index (ACI) constrained optimization against actual data and produces a counterfactual intake series (what intake *should* have looked like had ACI constraints been binding)
3. **Visualization Layer** — makes outputs legible for a policy/academic audience

The final output feeds a working paper arguing Canada's immigration failure was a dimensional reduction problem: the state optimized a rank-2 electoral projection (volume optics + Quebec coalition leverage) while the welfare-relevant dimensions (housing capacity, cohort heterogeneity, innovation spillovers, credential utilization) lived in the null space of that projection.

---

## Tech Stack

- Python 3.11+
- `pandas`, `numpy`, `scipy` for data and optimization
- `requests`, `beautifulsoup4` for any scraping needed
- `statsmodels` for regression (for Λ identification in Section 6 of the paper)
- `plotly` for interactive visualizations (not matplotlib — we need exportable HTML charts for the paper)
- `dash` for the dashboard layer if interactive UI is needed
- `jupyter` for iterative analysis notebooks
- Store cleaned data as parquet files locally
- All outputs go to `/outputs/` directory

---

## Layer 1: Data Pipeline

### Goal
Pull all public datasets, clean them, and join into a single analytical base indexed by `(year, quarter, province, cohort_type)`.

### Data Sources to Pull

**Immigration Intake (IRCC Open Data)**
- URL pattern: `https://open.canada.ca/data/en/dataset/` — search for "temporary residents" and "permanent residents" by province and category
- Target variables: monthly arrivals by permit type (study permit, work permit, PR by category), by province
- Years needed: 2018–2026
- Fallback: IRCC publishes annual reports with tables — scrape or manually load CSVs if API is unavailable

**Housing Capacity (CMHC)**
- Housing starts by province and CMA: `https://www.cmhc-schl.gc.ca/en/professionals/housing-markets-data-and-research/housing-data/data-tables/housing-market-data/housing-starts-data-table`
- Rental vacancy rates by CMA: CMHC Rental Market Survey, published annually each fall
- Target variables: `starts_annual`, `vacancy_rate`, `avg_rent_2br` by CMA
- Years needed: 2018–2025

**Health System Capacity (CIHI)**
- Canadian Institute for Health Information publishes wait times and bed capacity
- URL: `https://www.cihi.ca/en/wait-times-for-health-services`
- Target variables: median wait time for priority procedures by province, hospital occupancy rate
- This data is messier — build a manual CSV loader as fallback with documented sources

**Labour Market Quality (StatCan)**
- Table 14-10-0023-01: Employment by industry and province
- Table 18-10-0004-01: CPI by province (for real wage calculation)
- Target variables: unemployment rate, median wage by province, job vacancy rate
- Years: 2018–2026

**School/Municipal Capacity**
- This is the messiest dimension — use provincial education ministry enrollment data as proxy
- Build a stub that loads a manually populated CSV (`school_capacity.csv`) with documented sources per province
- Do not hallucinate numbers here — leave as NaN if not available and flag in output

**Fiscal Capacity (StatCan/PBO)**
- Parliamentary Budget Officer published immigrant income dynamics Jan 2024
- StatCan Table 36-10-0104-01: Government revenue and expenditure by province
- Target variables: municipal fiscal balance per capita by province

### Data Pipeline Architecture

```
/src/
  pipeline/
    ircc_loader.py       # IRCC intake data
    cmhc_loader.py       # Housing starts + vacancy
    statcan_loader.py    # Labour market + fiscal
    cihi_loader.py       # Health capacity (with manual fallback)
    school_loader.py     # School capacity stub
    joiner.py            # Merges all sources into master analytical base
    
/data/
  raw/                   # Downloaded files, never modified
  cleaned/               # Cleaned parquets per source
  master/                # Final joined analytical base
    master_panel.parquet # (year, quarter, province, cohort) indexed
    
/manual_data/
  school_capacity.csv    # Manually populated with sources documented in comments
  cihi_fallback.csv      # Health capacity manual fallback
```

### ACI Index Construction

After joining, construct the regional Absorption Capacity Index per province per year:

```
ACI_r,t = ω1·(vacancy_rate) + ω2·(starts_per_capita_growth) + ω3·(health_capacity) + ω4·(school_capacity) + ω5·(job_quality) + ω6·(municipal_fiscal_balance)
```

Where all components are normalized to [0,1] before weighting.

**Weight vector**: Run three scenarios:
- `ω_housing_heavy`: housing dimensions (ω1, ω2) get 0.5 combined weight
- `ω_equal`: all six dimensions equal weight (~0.167 each)  
- `ω_fiscal_heavy`: fiscal and labour dimensions get 0.5 combined weight

This sensitivity analysis is critical — the paper's conclusions must be robust across weight assumptions.

Store ACI as a column in master panel.

---

## Layer 2: Simulation Engine

### Goal
Answer: *What would ACI-constrained federal intake have looked like in 2018–2024 if the government had treated ACI_r,t ≥ ACI_min as a binding constraint?*

This is the counterfactual benchmark the papers currently lack.

### Architecture

```
/src/
  simulation/
    aci_optimizer.py      # Core constrained optimization
    cohort_npv.py         # Per-cohort net present value calculator
    counterfactual.py     # Runs counterfactual vs actual comparison
    welfare_loss.py       # Computes welfare loss from projection
    lambda_identifier.py  # Empirical identification of rank-2 projection matrix Λ
```

### Cohort NPV Calculator (`cohort_npv.py`)

Implement the student net contribution function from the paper:

```
Net_s = Y_s + ρ_future · p_trans - C_s - Ω_housing_r - Ω_services_r - Ω_integrity_j
```

And the general cohort NIV:

```
NIV_i = PV(B_i) - PV(C_i)
```

Where B_i and C_i are vectors, not scalars. Use the following cohort types:
- `high_quality_student`
- `low_quality_student` 
- `high_wage_worker`
- `low_wage_worker`
- `francophone_pr`
- `in_canada_transition`
- `family_class`
- `refugee`

For each cohort, load parameter assumptions from a YAML config file (`/config/cohort_params.yaml`) so they can be adjusted without touching code. Document sources for each parameter assumption in the YAML comments. Do not hardcode fiscal NPV assumptions — make them explicit and adjustable.

### ACI Optimizer (`aci_optimizer.py`)

For each year t and province r, solve:

```
max Σ_i Σ_r NIV_i · N_i,r,t

subject to:
  ACI_r,t ≥ ACI_min          # absorption constraint
  Σ_r N_i,r,t = Total_i,t    # national totals (from IRCC actual)
  N_i,r,t ≥ 0
  Legal obligations satisfied (refugee floors, francophone targets as soft constraints)
```

Use `scipy.optimize.linprog` or `scipy.optimize.minimize` with constraints. The optimization is linear in N if NIV parameters are fixed — use LP formulation where possible for speed.

Run this for every year 2018–2024 to produce `counterfactual_intake_series`.

### Welfare Loss Calculator (`welfare_loss.py`)

Compare actual intake vs counterfactual:

```
welfare_loss_t = w^T · V(x_actual) - w^T · V(x_counterfactual)
```

Where w is the social welfare weight vector (also sensitivity-tested across three scenarios).

Decompose the loss by dimension — which dimensions account for most of the gap between actual and counterfactual? This directly populates the "orthogonal residual" argument in the paper.

### Λ Identification (`lambda_identifier.py`)

Implement the empirical falsifiability test from Paper 1, Section 6:

```
Δx_observed = Λ^T · β + ε
H0: rank(Λ̂) = 2, loadings on null-space dims = 0
```

Regress observed policy lever changes (actual intake adjustments by category) on estimated dimension scores. Test whether fitted loadings concentrate on political utility dimensions (V11, V12) with near-zero loadings on absorptive capacity (V4), heterogeneity (V5), innovation (V8).

Use `statsmodels` OLS with robust standard errors. Report rank test results. This is the paper's key falsifiable claim — the output here either supports or challenges the projection thesis.

**Important**: If the regression doesn't support rank-2, report that honestly. Don't torture the data.

---

## Layer 3: Visualization

### Goal
Produce publication-ready charts and an interactive dashboard.

### Static Charts (Plotly, export as HTML + PNG)

1. **ACI vs Actual Intake (2018–2024)**: Line chart per province showing ACI_r,t on left axis, actual intake on right axis. Highlights where intake exceeded ACI capacity.

2. **Counterfactual vs Actual Intake**: Stacked area chart showing what the composition *should* have been vs what it was, by cohort type and year.

3. **Welfare Loss Decomposition**: Waterfall chart showing welfare loss by dimension (which dimensions account for the gap). This is the money chart for the paper.

4. **Rank-2 Projection Visualization**: 2D scatter plot of the electoral manifold (P axis = volume signal, Q axis = Quebec leverage) with actual policy levers plotted as points. Illustrates the dimensional reduction visually.

5. **ACI Sensitivity Analysis**: Three panel chart showing counterfactual intake under housing-heavy, equal, and fiscal-heavy weight scenarios. If conclusions are robust, all three panels tell the same story.

6. **Cohort NIV Comparison**: Horizontal bar chart ranking cohort types by NIV under actual vs ACI-constrained allocation.

### Interactive Dashboard (Dash)

Single-page dashboard with:
- Province selector dropdown
- Year range slider (2018–2026)
- Weight scenario toggle (housing / equal / fiscal)
- ACI gauge per province (current vs minimum threshold)
- Counterfactual vs actual intake comparison
- Welfare loss breakdown table

Dashboard should run locally on `localhost:8050`. Not deploying this anywhere.

```
/src/
  viz/
    static_charts.py     # All 6 static Plotly charts
    dashboard.py         # Dash app
    chart_config.py      # Shared styling (colors, fonts, labels)
```

---

## Output Structure

```
/outputs/
  charts/
    01_aci_vs_intake.html
    01_aci_vs_intake.png
    02_counterfactual_vs_actual.html
    ...
  data/
    counterfactual_series.csv      # Main result for paper
    welfare_loss_decomposition.csv
    lambda_regression_results.csv  # Λ identification output
  reports/
    summary_stats.txt              # Key numbers for paper citations
```

---

## What NOT to Do

- Do not fabricate data. If a source is unavailable, log a warning and use NaN. Flag all NaN-heavy dimensions in output.
- Do not hardcode parameter assumptions. Everything adjustable goes in `/config/`.
- Do not over-engineer the ACI weights. They are assumptions. Make them explicit, run sensitivity, move on.
- Do not use matplotlib. Plotly only for visualization.
- Do not build a web scraper that's fragile. If IRCC/CMHC data requires authentication or dynamic loading, build a manual CSV loader with a documented template instead.
- Do not run the Λ regression with fewer than 6 years of data without flagging the degrees-of-freedom problem explicitly.

---

## Deliverables Checklist

- [ ] `master_panel.parquet` with all dimensions, all provinces, 2018–2026
- [ ] `cohort_params.yaml` with all NIV parameters documented with sources
- [ ] `counterfactual_series.csv` — the core benchmark
- [ ] `welfare_loss_decomposition.csv` — the core argument
- [ ] `lambda_regression_results.csv` — the falsifiability test
- [ ] All 6 static charts as HTML + PNG
- [ ] Working Dash dashboard
- [ ] `README.md` explaining how to run everything, what data sources are used, and what assumptions were made

---

## Context for the Paper

The tool is building empirical support for this argument:

> Canada's 2024–2026 immigration restructuring is best understood as the optimization of a rank-2 electoral projection whose null space contains the dimensions of greatest long-run welfare weight. The fiscal and absorptive-capacity damage is the orthogonal residual that projection necessarily discards. The state did not make a computational error — it optimized exactly what its incentive structure allowed it to see.

Every output should be legible to a policy economist who has read the NAS 2017 immigration report and is familiar with Canadian public finance. The audience is not the general public — it's academics, think tanks, and senior policy staff.

The paper this feeds will be submitted as a working paper (likely SSRN) and potentially to a Canadian public policy journal. Standards matter.
