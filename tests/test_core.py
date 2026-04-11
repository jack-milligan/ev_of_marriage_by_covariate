"""
tests/test_core.py

Unit tests for core functions. Run from the project root:
    pytest tests/

No real data is required — all tests use synthetic DataFrames.
"""

import numpy as np
import pandas as pd
import pytest

from src.payoff_model import (
    validate_profile,
    compute_ev_of_marriage,
    build_feature_row,
    predict_divorce_prob_from_profile,
    sensitivity_analysis,
)
from src.risk_model import build_logit_dataset


# =========================================================
# Helpers
# =========================================================

def make_valid_profile(**overrides):
    base = {
        "age": 32,
        "sex": "Male",
        "educ_label": "4 yrs college",
        "incwage": 80_000,
        "yrmarr": 2020,
    }
    base.update(overrides)
    return base


def make_feature_cols():
    """Minimal feature column list matching the training design matrix shape."""
    return [
        "AGE", "is_female", "log_incwage", "YRMARR",
        "EDUC_4 yrs college", "EDUC_5+ yrs college", "EDUC_Gr 12 (no college)",
    ]


def make_synthetic_ipums_df(n=300, seed=42):
    """Synthetic DataFrame with the columns expected by build_logit_dataset."""
    rng = np.random.default_rng(seed)
    educ_labels = ["Gr 12 (no college)", "4 yrs college", "5+ yrs college", "1 yr college"]
    marst_labels = ["Married, spouse present", "Divorced", "Never married/single"]

    marst = rng.choice(marst_labels, size=n, p=[0.6, 0.2, 0.2])
    return pd.DataFrame({
        "AGE": rng.integers(25, 46, size=n),
        "is_female": rng.integers(0, 2, size=n),
        "INCWAGE": rng.integers(20_000, 120_001, size=n).astype(float),
        "YRMARR": rng.integers(2000, 2023, size=n).astype(float),
        "EDUC_label": rng.choice(educ_labels, size=n),
        "MARST_label": marst,
        "is_ever_married": np.where(
            np.isin(marst, ["Married, spouse present", "Divorced"]), 1, 0
        ),
    })


# =========================================================
# validate_profile
# =========================================================

def test_validate_profile_valid():
    validate_profile(make_valid_profile())  # should not raise


def test_validate_profile_missing_key():
    profile = make_valid_profile()
    del profile["age"]
    with pytest.raises(ValueError, match="Missing required profile keys"):
        validate_profile(profile)


def test_validate_profile_bad_age():
    with pytest.raises(ValueError, match="Invalid age"):
        validate_profile(make_valid_profile(age=150))


def test_validate_profile_bad_sex():
    with pytest.raises(ValueError, match="Invalid sex"):
        validate_profile(make_valid_profile(sex="X"))


def test_validate_profile_bad_educ():
    with pytest.raises(ValueError, match="Invalid educ_label"):
        validate_profile(make_valid_profile(educ_label="PhD"))


def test_validate_profile_negative_income():
    with pytest.raises(ValueError, match="Invalid incwage"):
        validate_profile(make_valid_profile(incwage=-500))


# =========================================================
# compute_ev_of_marriage
# =========================================================

def test_compute_ev_no_divorce_no_cost():
    """With p_divorce=0 and zero discount rate, delta_EV = uplift * income * years."""
    profile = {"incwage": 100_000}
    ev = compute_ev_of_marriage(
        profile, p_divorce=0.0, married_uplift=0.10, divorced_penalty=-0.05,
        horizon_years=10, discount_rate=0.0, divorce_fixed_cost=0,
    )
    expected_delta = 100_000 * 0.10 * 10
    assert abs(ev["delta_EV_marry_minus_single"] - expected_delta) < 1


def test_compute_ev_certain_divorce():
    """With p_divorce=1, EV(marry) = PV(divorced_income) - divorce_cost."""
    profile = {"incwage": 100_000}
    ev = compute_ev_of_marriage(
        profile, p_divorce=1.0, married_uplift=0.10, divorced_penalty=0.0,
        horizon_years=10, discount_rate=0.0, divorce_fixed_cost=20_000,
    )
    assert ev["delta_EV_marry_minus_single"] == pytest.approx(-20_000)


def test_compute_ev_invalid_p_divorce():
    with pytest.raises(ValueError):
        compute_ev_of_marriage({"incwage": 50_000}, p_divorce=1.5,
                               married_uplift=0.1, divorced_penalty=-0.05)


def test_compute_ev_positive_ev_low_risk():
    """Large uplift + low divorce probability should yield positive delta EV."""
    profile = {"incwage": 80_000}
    ev = compute_ev_of_marriage(
        profile, p_divorce=0.05, married_uplift=0.25, divorced_penalty=-0.10,
    )
    assert ev["delta_EV_marry_minus_single"] > 0


# =========================================================
# build_feature_row
# =========================================================

def test_build_feature_row_columns_match():
    feature_cols = make_feature_cols()
    row = build_feature_row(make_valid_profile(), feature_cols)
    assert list(row.columns) == feature_cols


def test_build_feature_row_is_female_male():
    row = build_feature_row(make_valid_profile(sex="Male"), make_feature_cols())
    assert row["is_female"].iloc[0] == 0.0


def test_build_feature_row_is_female_female():
    row = build_feature_row(make_valid_profile(sex="Female"), make_feature_cols())
    assert row["is_female"].iloc[0] == 1.0


def test_build_feature_row_education_dummy():
    feature_cols = make_feature_cols()
    row = build_feature_row(make_valid_profile(educ_label="4 yrs college"), feature_cols)
    assert row["EDUC_4 yrs college"].iloc[0] == 1.0
    assert row["EDUC_5+ yrs college"].iloc[0] == 0.0


# =========================================================
# build_logit_dataset
# =========================================================

def test_build_logit_dataset_basic():
    df = make_synthetic_ipums_df(n=300)
    X, y = build_logit_dataset(df, age_min=25, age_max=45)
    assert len(X) > 0
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_build_logit_dataset_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        build_logit_dataset(pd.DataFrame(), age_min=25, age_max=45)


def test_build_logit_dataset_age_filter():
    df = make_synthetic_ipums_df(n=500)
    X, y = build_logit_dataset(df, age_min=30, age_max=35)
    # All rows in X should come from the age-filtered subset
    assert len(X) <= len(df[(df["AGE"] >= 30) & (df["AGE"] <= 35)])


# =========================================================
# sensitivity_analysis
# =========================================================

def test_sensitivity_analysis_returns_dataframe():
    from unittest.mock import MagicMock
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.85, 0.15]])
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 4))

    results = sensitivity_analysis(
        make_valid_profile(), p_divorce=0.10,
        married_uplift=0.25, divorced_penalty=-0.10,
    )
    assert isinstance(results, pd.DataFrame)
    assert "parameter" in results.columns
    assert "delta_EV" in results.columns
    assert set(results["parameter"].unique()) == {
        "horizon_years", "discount_rate", "divorce_fixed_cost"
    }