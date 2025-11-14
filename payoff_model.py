# payoff_model.py

"""
Expected Value (EV) of Marriage Model

This module computes the expected value of marriage vs. staying single
by combining:
1. Divorce risk predictions from logistic regression
2. Economic effects (income uplifts/penalties)
3. Discounted cash flow calculations

Key assumptions:
- Income effects are constant over the time horizon
- Divorce probability is static (doesn't vary with marriage duration)
- Fixed divorce cost is applied once at divorce
- No remarriage probability considered
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ev_of_marriage_by_covariate import load_ipums_ev_data, get_label_maps
from risk_model import build_logit_dataset, fit_divorce_logit

NUM_COLS = ["AGE", "is_female", "log_incwage", "YRMARR"]

# =========================================================
# 0. Input validation
# =========================================================

def validate_profile(profile: dict) -> None:
    """
    Validate a profile dictionary for completeness and reasonableness.
    
    Raises ValueError if validation fails.
    
    Args:
        profile: Dictionary with keys: age, sex, educ_label, incwage, yrmarr
    """
    required_keys = ["age", "sex", "educ_label", "incwage", "yrmarr"]
    missing = [key for key in required_keys if key not in profile]
    if missing:
        raise ValueError(f"Missing required profile keys: {missing}")
    
    # Validate age
    age = profile["age"]
    if not isinstance(age, (int, float)) or age < 18 or age > 100:
        raise ValueError(f"Invalid age: {age}. Must be between 18 and 100.")
    
    # Validate sex
    sex = profile["sex"]
    if sex not in ["Male", "Female"]:
        raise ValueError(f"Invalid sex: {sex}. Must be 'Male' or 'Female'.")
    
    # Validate education label
    _, _, educ_map, _ = get_label_maps()
    valid_educ_labels = set(educ_map.values())
    educ_label = profile["educ_label"]
    if educ_label not in valid_educ_labels:
        raise ValueError(
            f"Invalid educ_label: '{educ_label}'. "
            f"Valid options: {sorted(valid_educ_labels)}"
        )
    
    # Validate income
    incwage = profile["incwage"]
    if not isinstance(incwage, (int, float)) or incwage < 0:
        raise ValueError(f"Invalid incwage: {incwage}. Must be non-negative.")
    
    # Validate marriage year
    yrmarr = profile["yrmarr"]
    current_year = pd.Timestamp.now().year
    if not isinstance(yrmarr, (int, float)) or yrmarr < 1950 or yrmarr > current_year:
        raise ValueError(
            f"Invalid yrmarr: {yrmarr}. Must be between 1950 and {current_year}."
        )


def get_valid_education_labels() -> List[str]:
    """Return list of valid education labels from the data schema."""
    _, _, educ_map, _ = get_label_maps()
    return sorted(set(educ_map.values()))


# =========================================================
# 1. Train or load risk model
# =========================================================

def train_default_risk_model(
    cache_path: Optional[str] = None,
    force_retrain: bool = False
) -> Tuple[object, object, List[str]]:
    """
    Train the baseline divorce-risk logistic model on ages 25–45,
    ever-married only. Returns model, scaler, and feature column list.
    
    Args:
        cache_path: Optional path to cache trained model. If None, no caching.
        force_retrain: If True, retrain even if cache exists.
    
    Returns:
        Tuple of (model, scaler, feature_cols)
    """
    # Try to load from cache
    if cache_path and os.path.exists(cache_path) and not force_retrain:
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            print(f"✅ Loaded cached model from {cache_path}")
            return cached["model"], cached["scaler"], cached["feature_cols"]
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}. Retraining...")
    
    # Train new model
    print("🔹 Training divorce risk model...")
    df = load_ipums_ev_data()
    X, y = build_logit_dataset(df, age_min=25, age_max=45, ever_married_only=True)
    result = fit_divorce_logit(X, y)

    model = result["model"]
    scaler = result["scaler"]
    feature_cols = X.columns.tolist()
    
    # Cache if path provided
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    "model": model,
                    "scaler": scaler,
                    "feature_cols": feature_cols
                }, f)
            print(f"💾 Cached model to {cache_path}")
        except Exception as e:
            print(f"⚠️ Failed to cache model: {e}")

    return model, scaler, feature_cols


# =========================================================
# 2. Estimate economic parameters from data
# =========================================================

def estimate_income_effects_simple(df: pd.DataFrame,
                                  age_min: int = 25,
                                  age_max: int = 45) -> dict:
    """
    Simple median-based estimation of income effects (original method).
    
    Note: This method doesn't control for confounders. Use estimate_income_effects()
    for more robust estimates.

    Returns:
        {
            'married_uplift': float (e.g. 0.12 = +12%),
            'divorced_penalty': float (e.g. -0.08 = -8%)
        }
    """
    subset = df[(df["AGE"] >= age_min) & (df["AGE"] <= age_max)].copy()
    subset = subset[subset["INCWAGE"].notna() & (subset["INCWAGE"] > 0)]

    medians = subset.groupby("MARST_label")["INCWAGE"].median()

    married = medians.get("Married, spouse present", np.nan)
    never = medians.get("Never married/single", np.nan)
    divorced = medians.get("Divorced", np.nan)

    married_uplift = (married / never - 1.0) if (not np.isnan(married) and not np.isnan(never)) else 0.0
    divorced_penalty = (divorced / married - 1.0) if (not np.isnan(divorced) and not np.isnan(married)) else 0.0

    return {
        "married_uplift": married_uplift,
        "divorced_penalty": divorced_penalty,
        "method": "simple_median"
    }


def estimate_income_effects(df: pd.DataFrame,
                            age_min: int = 25,
                            age_max: int = 45,
                            method: str = "regression") -> dict:
    """
    Estimate married-income uplift and divorced-income penalty from data.
    
    Uses regression to control for age, education, and sex to get more robust
    estimates of the causal effect of marital status on income.

    Args:
        df: Cleaned IPUMS dataframe
        age_min: Minimum age for analysis
        age_max: Maximum age for analysis
        method: 'regression' (default) or 'simple' (median-based)

    Returns:
        {
            'married_uplift': float (e.g. 0.12 = +12%),
            'divorced_penalty': float (e.g. -0.08 = -8%),
            'method': str,
            'n_obs': int
        }
    """
    if method == "simple":
        return estimate_income_effects_simple(df, age_min, age_max)
    
    subset = df[(df["AGE"] >= age_min) & (df["AGE"] <= age_max)].copy()
    subset = subset[
        subset["INCWAGE"].notna() 
        & (subset["INCWAGE"] > 0)
        & subset["EDUC_label"].notna()
    ].copy()
    
    # Log income for regression (more stable)
    subset["log_incwage"] = np.log1p(subset["INCWAGE"])
    
    # Create marital status indicators
    subset["is_married"] = (subset["MARST_label"] == "Married, spouse present").astype(int)
    subset["is_divorced"] = (subset["MARST_label"] == "Divorced").astype(int)
    subset["is_never_married"] = (subset["MARST_label"] == "Never married/single").astype(int)
    
    # Build design matrix
    X_numeric = subset[["AGE", "is_female"]].copy()
    
    # Education dummies
    educ_dummies = pd.get_dummies(
        subset["EDUC_label"],
        prefix="EDUC",
        drop_first=True,
        dummy_na=False
    )
    
    X = pd.concat([X_numeric, educ_dummies], axis=1)
    y = subset["log_incwage"]
    
    # Fit regression: log(income) ~ covariates + marital_status
    # Compare married vs never-married
    mask_married_vs_never = (subset["is_married"] == 1) | (subset["is_never_married"] == 1)
    X_mvn = X.loc[mask_married_vs_never].copy()
    y_mvn = y.loc[mask_married_vs_never]
    marital_mvn = subset.loc[mask_married_vs_never, "is_married"].values
    
    X_mvn_with_marital = X_mvn.copy()
    X_mvn_with_marital["is_married"] = marital_mvn
    
    model_mvn = LinearRegression()
    model_mvn.fit(X_mvn_with_marital, y_mvn)
    # Create coefficient dictionary for easier access
    coef_dict_mvn = dict(zip(X_mvn_with_marital.columns, model_mvn.coef_))
    married_coef = coef_dict_mvn.get("is_married", 0.0)
    married_uplift = np.exp(married_coef) - 1.0  # Convert log difference to percentage
    
    # Compare divorced vs married
    mask_divorced_vs_married = (subset["is_divorced"] == 1) | (subset["is_married"] == 1)
    X_dvm = X.loc[mask_divorced_vs_married].copy()
    y_dvm = y.loc[mask_divorced_vs_married]
    marital_dvm = subset.loc[mask_divorced_vs_married, "is_divorced"].values
    
    X_dvm_with_marital = X_dvm.copy()
    X_dvm_with_marital["is_divorced"] = marital_dvm
    
    model_dvm = LinearRegression()
    model_dvm.fit(X_dvm_with_marital, y_dvm)
    # Create coefficient dictionary for easier access
    coef_dict_dvm = dict(zip(X_dvm_with_marital.columns, model_dvm.coef_))
    divorced_coef = coef_dict_dvm.get("is_divorced", 0.0)
    divorced_penalty = np.exp(divorced_coef) - 1.0  # Convert log difference to percentage
    
    return {
        "married_uplift": married_uplift,
        "divorced_penalty": divorced_penalty,
        "method": "regression",
        "n_obs": len(subset)
    }


# =========================================================
# 3. Build feature row for a single profile
# =========================================================

def build_feature_row(profile: dict,
                      feature_cols: list) -> pd.DataFrame:
    """
    Build a 1-row DataFrame matching the training design matrix.

    Args:
        profile: Dictionary with keys:
            - age: int
            - sex: 'Male' or 'Female'
            - educ_label: string matching EDUC_label values
            - incwage: float, annual wage income
            - yrmarr: int, year of marriage
        feature_cols: List of feature column names from training data
    
    Returns:
        DataFrame with 1 row matching the training design matrix
    
    Raises:
        ValueError: If profile is invalid or education label doesn't match
    """
    # Validate profile first
    validate_profile(profile)
    
    age = profile["age"]
    sex = profile["sex"]
    educ_label = profile["educ_label"]
    incwage = profile["incwage"]
    yrmarr = profile["yrmarr"]

    is_female = 1 if sex == "Female" else 0
    log_incwage = np.log1p(max(0, incwage))

    # Start with zeros for all features
    data = {col: 0.0 for col in feature_cols}

    # Numeric features
    if "AGE" in data:
        data["AGE"] = float(age)
    if "is_female" in data:
        data["is_female"] = float(is_female)
    if "log_incwage" in data:
        data["log_incwage"] = float(log_incwage)
    if "YRMARR" in data:
        data["YRMARR"] = float(yrmarr)

    # Education dummy: EDUC_<label>
    # Note: pandas get_dummies creates columns like "EDUC_4 yrs college"
    educ_col = f"EDUC_{educ_label}"
    if educ_col in data:
        data[educ_col] = 1.0
    else:
        # Education label is baseline category (dropped in get_dummies with drop_first=True)
        # This is correct - all education dummies should be 0
        # But warn if it seems like a mismatch
        educ_cols = [col for col in feature_cols if col.startswith("EDUC_")]
        if educ_cols:  # If there are any education columns, this might be an error
            # Check if it's just the baseline category
            # We can't easily verify this without the original data, so we'll proceed
            pass

    X_profile = pd.DataFrame([data], columns=feature_cols)
    return X_profile


# =========================================================
# 4. Predict P(divorce | covariates)
# =========================================================

def predict_divorce_prob(profile: dict,
                         model,
                         scaler,
                         feature_cols: list) -> float:
    """
    Given a profile and fitted model, return predicted probability of divorce.
    
    Args:
        profile: Dictionary with profile information (see build_feature_row)
        model: Fitted logistic regression model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names from training
    
    Returns:
        Predicted probability of divorce (float between 0 and 1)
    
    Raises:
        ValueError: If profile is invalid or scaling fails
        AttributeError: If model/scaler don't have required methods
    """
    try:
        X_profile = build_feature_row(profile, feature_cols)
    except ValueError as e:
        raise ValueError(f"Invalid profile: {e}")

    # Transform numeric columns to match how the scaler was fit
    X_scaled = X_profile.copy()
    
    # Ensure all required numeric columns exist
    missing = [c for c in NUM_COLS if c not in X_scaled.columns]
    if missing:
        raise ValueError(
            f"Missing required numeric columns for scaling: {missing}. "
            f"Available columns: {list(X_scaled.columns)}"
        )

    try:
        # Scale only the numeric columns that were used in training
        X_scaled[NUM_COLS] = scaler.transform(X_profile[NUM_COLS])
    except Exception as e:
        raise ValueError(
            f"Scaling failed: {e}. "
            f"Ensure scaler was fit on columns: {NUM_COLS}"
        ) from e

    try:
        proba = model.predict_proba(X_scaled)[:, 1][0]
        return float(np.clip(proba, 0.0, 1.0))  # Ensure valid probability
    except Exception as e:
        raise ValueError(f"Model prediction failed: {e}") from e



# =========================================================
# 5. Compute EV of marriage vs staying single
# =========================================================

def compute_ev_of_marriage(profile: dict,
                           p_divorce: float,
                           married_uplift: float,
                           divorced_penalty: float,
                           horizon_years: int = 10,
                           discount_rate: float = 0.03,
                           divorce_fixed_cost: float = 20000.0) -> dict:
    """
    Compute Expected Value (EV) of marriage vs. staying single.
    
    Model assumptions:
    - Baseline income = profile['incwage']
    - If married & not divorced: income * (1 + married_uplift)
    - If divorced: income * (1 + divorced_penalty), minus fixed divorce cost
    - Income effects are constant over the time horizon
    - Divorce happens at the start (not mid-horizon)
    - Fixed divorce cost is applied once at divorce
    - No remarriage probability considered
    
    Formula:
        EV(single) = PV(annuity of base_income)
        EV(marry) = (1 - p_divorce) * PV(married_income) 
                   + p_divorce * (PV(divorced_income) - divorce_cost)
    
    Args:
        profile: Dictionary with 'incwage' key
        p_divorce: Probability of divorce (0-1)
        married_uplift: Income multiplier for married (e.g., 0.12 = +12%)
        divorced_penalty: Income multiplier for divorced (e.g., -0.08 = -8%)
        horizon_years: Time horizon for calculation (default: 10 years)
        discount_rate: Annual discount rate (default: 0.03 = 3%)
        divorce_fixed_cost: One-time cost of divorce (default: $20,000)
    
    Returns:
        Dictionary with:
            - EV_stay_single: Expected value of staying single
            - EV_marry: Expected value of marrying
            - delta_EV_marry_minus_single: Difference (marry - single)
            - p_divorce_used: Divorce probability used in calculation
    
    Example:
        ```python
        profile = {"incwage": 80000}
        ev = compute_ev_of_marriage(
            profile, 
            p_divorce=0.3, 
            married_uplift=0.12, 
            divorced_penalty=-0.08
        )
        # Access EV difference: ev['delta_EV_marry_minus_single']
        ```
    """
    # Validate inputs
    if not 0 <= p_divorce <= 1:
        raise ValueError(f"p_divorce must be between 0 and 1, got {p_divorce}")
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be positive, got {horizon_years}")
    if discount_rate < 0:
        raise ValueError(f"discount_rate must be non-negative, got {discount_rate}")
    if "incwage" not in profile:
        raise ValueError("profile must contain 'incwage' key")
    
    y = horizon_years
    r = discount_rate
    base_income = float(profile["incwage"])
    
    if base_income < 0:
        raise ValueError(f"incwage must be non-negative, got {base_income}")

    def pv_annuity(payment: float) -> float:
        """Calculate present value of an annuity."""
        if r == 0:
            return payment * y
        return payment * (1 - (1 + r) ** -y) / r

    # Scenario 1: stay single
    ev_single = pv_annuity(base_income)

    # Scenario 2: marry
    married_income = base_income * (1 + married_uplift)
    divorced_income = base_income * (1 + divorced_penalty)

    ev_marriage = (
        (1 - p_divorce) * pv_annuity(married_income)
        + p_divorce * (pv_annuity(divorced_income) - divorce_fixed_cost)
    )

    return {
        "EV_stay_single": ev_single,
        "EV_marry": ev_marriage,
        "delta_EV_marry_minus_single": ev_marriage - ev_single,
        "p_divorce_used": p_divorce,
        "horizon_years": horizon_years,
        "discount_rate": discount_rate,
    }


# =========================================================
# 6. Example usage: evaluate a few profiles
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Expected Value of Marriage Analysis")
    print("=" * 60)
    
    # Load data to estimate income uplifts/penalties
    print("\n📊 Loading data and estimating income effects...")
    df = load_ipums_ev_data()
    
    # Try regression method first, fall back to simple if it fails
    try:
        econ_params = estimate_income_effects(df, method="regression")
        print(f"✅ Using regression-based estimates (n={econ_params.get('n_obs', 'unknown')})")
    except Exception as e:
        print(f"⚠️ Regression method failed: {e}. Using simple method.")
        econ_params = estimate_income_effects(df, method="simple")
    
    married_uplift = econ_params["married_uplift"]
    divorced_penalty = econ_params["divorced_penalty"]
    method_used = econ_params.get("method", "unknown")

    print(f"\nEstimated income effects ({method_used}):")
    print(f"  Married uplift:  {married_uplift:.3f} (~{married_uplift*100:+.1f}%)")
    print(f"  Divorced penalty: {divorced_penalty:.3f} (~{divorced_penalty*100:+.1f}%)")

    # Train risk model (with optional caching)
    cache_path = "models/divorce_risk_model.pkl"
    try:
        model, scaler, feature_cols = train_default_risk_model(cache_path=cache_path)
    except Exception as e:
        print(f"⚠️ Model training/caching issue: {e}")
        model, scaler, feature_cols = train_default_risk_model(cache_path=None)

    # Example profiles
    profiles = [
        {
            "label": "Male, 32, 4 yrs college, 80K, married 2020",
            "age": 32,
            "sex": "Male",
            "educ_label": "4 yrs college",
            "incwage": 80000,
            "yrmarr": 2020,
        },
        {
            "label": "Female, 29, 5+ yrs college, 120K, married 2022",
            "age": 29,
            "sex": "Female",
            "educ_label": "5+ yrs college",
            "incwage": 120000,
            "yrmarr": 2022,
        },
    ]

    print("\n" + "=" * 60)
    print("Profile Analysis")
    print("=" * 60)
    
    for p in profiles:
        try:
            p_div = predict_divorce_prob(p, model, scaler, feature_cols)
            ev = compute_ev_of_marriage(
                p,
                p_divorce=p_div,
                married_uplift=married_uplift,
                divorced_penalty=divorced_penalty,
            )

            print(f"\nProfile: {p['label']}")
            print(f"  P(divorce | covariates): {p_div:.3f} ({p_div*100:.1f}%)")
            print(f"  EV(stay single):          ${ev['EV_stay_single']:,.0f}")
            print(f"  EV(marry):                ${ev['EV_marry']:,.0f}")
            print(f"  ΔEV (marry - single):     ${ev['delta_EV_marry_minus_single']:,.0f}")
            
            if ev['delta_EV_marry_minus_single'] > 0:
                print(f"  → Recommendation: Marry (positive EV)")
            else:
                print(f"  → Recommendation: Stay single (negative EV)")
                
        except Exception as e:
            print(f"\n❌ Error analyzing profile '{p.get('label', 'unknown')}': {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
