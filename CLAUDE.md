# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.8+, virtual environment at `.venv/`. Activate with `source .venv/bin/activate`.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

Entry-point scripts live at the project root and are run directly. Library modules inside `src/` are run as package modules (required for relative imports to resolve):

```bash
python eda_narrative.py            # EDA narrative + visualizations
python -m src.risk_model           # Train divorce risk model + diagnostics
python -m src.payoff_model         # EV calculations across example profiles
python -m src.validation           # 5-fold cross-validation
python demo.py                     # End-to-end walkthrough
```

Run tests:
```bash
pytest tests/
```

## Data

Raw data is an IPUMS ACS CSV at a hardcoded local path (`RAW_PATH` in `ev_of_marriage_by_covariate.py`). On first load it is cleaned and cached as Parquet at `CACHE_PATH`. Subsequent calls to `load_ipums_ev_data()` use the cache by default (`use_cache=True`). To force a re-process, pass `use_cache=False`.

The data file paths are machine-specific — override via environment variables `IPUMS_RAW_PATH` and `IPUMS_CACHE_PATH` (see `src/config.py`).

## Project Layout

```
src/                        # Library package — import as src.module
  config.py                 # Path configuration (env var overrides)
  utils.py                  # Shared utilities (save_visual)
  ev_of_marriage_by_covariate.py
  analysis_relationships.py
  risk_model.py
  payoff_model.py
  validation.py
demo.py                     # Entry-point scripts (run directly)
eda_narrative.py
tests/
  test_core.py
models/                     # Saved model artifacts (.pkl, .csv)
visuals/                    # Generated PNG outputs
```

## Architecture

The pipeline is linear: data loading → EDA → risk model → payoff model → validation.

**`src/ev_of_marriage_by_covariate.py`** — Data ETL. Single public entrypoint: `load_ipums_ev_data()`. Reads raw IPUMS CSV, filters to household population, applies label maps (SEX, MARST, EDUC, STATEFIP), adds derived flags (`is_married_now`, `is_divorced`, etc.) and `age_at_marriage` / `marriage_duration`. All downstream modules import from here.

**`src/analysis_relationships.py`** — Exploratory analysis. Reads the cleaned DataFrame and writes PNGs to `visuals/`.

**`src/utils.py`** — `save_visual()`: shared helper imported by both `analysis_relationships` and `risk_model`.

**`src/risk_model.py`** — Divorce risk modeling. Key functions:
- `build_logit_dataset(df, age_min, age_max)` → `(X, y)`: filters to ever-married in age range, one-hot encodes education, returns feature matrix + binary divorce target.
- `fit_divorce_logit(X, y)` → result dict: L2 logistic regression with StandardScaler on numeric cols only (`AGE`, `is_female`, `log_incwage`, `YRMARR`); returns model, scaler, metrics, and splits.
- `compare_models(X, y)` → DataFrame: benchmarks logistic regression vs. random forest.
- `predict_divorce_prob(X, model, scaler, num_cols)` → probabilities (batch, DataFrame input).
- `save_model` / `load_model`: pickle to/from `models/`.
- `fit_divorce_cox(df)`: optional Cox PH model (requires `lifelines`).

**`src/payoff_model.py`** — Economic EV calculations. Key functions:
- `estimate_income_effects(df, method="regression")`: regression-controlled income effect of marital status.
- `build_feature_row(profile, feature_cols)`: converts a profile dict to a 1-row DataFrame matching the training design matrix.
- `predict_divorce_prob_from_profile(profile, model, scaler, feature_cols)`: profile dict → scalar probability.
- `compute_ev_of_marriage(profile, p_divorce, married_uplift, divorced_penalty)`: discounted cash flow EV comparison, returns `delta_EV_marry_minus_single`.
- `sensitivity_analysis(profile, p_divorce, married_uplift, divorced_penalty)` → DataFrame: varies horizon, discount rate, divorce cost one-at-a-time.
- `train_default_risk_model(cache_path)`: trains and optionally caches to `models/divorce_risk_model.pkl`.

**`src/validation.py`** — `cross_validate_model(X, y)` and `run_validation()`: 5-fold stratified CV reporting ROC AUC and Brier score.

## Profile Dict Schema

Used by `payoff_model.py` functions:
```python
{
    "age": int,           # 18–100
    "sex": "Male" | "Female",
    "educ_label": str,    # must match values in get_label_maps() educ_map
    "incwage": float,     # annual wage income, non-negative
    "yrmarr": int,        # year of marriage, 1950–present
}
```

Valid `educ_label` values: `"N/A or no schooling"`, `"Gr 1–4"`, `"Gr 5–8"`, `"Gr 9"`, `"Gr 10"`, `"Gr 11"`, `"Gr 12 (no college)"`, `"1 yr college"`, `"2 yrs college"`, `"3 yrs college"`, `"4 yrs college"`, `"5+ yrs college"`.
