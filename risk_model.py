import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lifelines

"""
risk_model.py

Divorce risk modeling for EV-of-marriage project.

- Logistic regression: P(divorced | covariates)
- Optional Cox proportional hazards: hazard of divorce over marriage duration
- Plots and artifacts saved to visuals/
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import your loader
from ev_of_marriage_by_covariate import load_ipums_ev_data

# Try to reuse save_visual from your analysis module if available
try:
    from analysis_relationships import save_visual
except ImportError:  # fallback if not imported there
    def save_visual(fig: plt.Figure, filename: str, folder: str = "visuals") -> None:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved visualization: {path}")


# =========================================================
# 0. Utilities
# =========================================================

def _ensure_visuals_dir(folder: str = "visuals") -> None:
    os.makedirs(folder, exist_ok=True)


# =========================================================
# 1. Logistic model dataset construction
# =========================================================

def build_logit_dataset(
    df: pd.DataFrame,
    age_min: int = 25,
    age_max: int = 45,
    ever_married_only: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a modeling dataset for logistic regression:
    P(divorced | covariates) among individuals in a given age range.

    Target:
        is_divorced: 1 if MARST_label == 'Divorced', else 0

    Predictors (example set):
        - AGE
        - is_female
        - log_incwage
        - YRMARR (marriage cohort)
        - EDUC_label (one-hot)
    """
    data = df.copy()

    # Age range
    data = data[(data["AGE"] >= age_min) & (data["AGE"] <= age_max)]

    # Universe restriction
    if ever_married_only:
        data = data[data["is_ever_married"] == 1]

    # Target
    data["is_divorced"] = (data["MARST_label"] == "Divorced").astype(int)

    # Basic filters: drop rows with missing key info
    required_cols = ["AGE", "is_female", "INCWAGE", "YRMARR", "EDUC_label"]
    data = data.dropna(subset=required_cols)

    # Log income to tame skew; zero/negative -> 0 after clip
    data["INCWAGE"] = pd.to_numeric(data["INCWAGE"], errors="coerce").fillna(0)
    data["log_incwage"] = np.log1p(data["INCWAGE"].clip(lower=0))

    # Handle missing / zero YRMARR (should mostly be set for ever-married)
    data["YRMARR"] = pd.to_numeric(data["YRMARR"], errors="coerce")

    # Design matrix:
    #   numeric: AGE, is_female, log_incwage, YRMARR
    #   categorical: EDUC_label (one-hot)
    X_numeric = data[["AGE", "is_female", "log_incwage", "YRMARR"]]

    X_cats = pd.get_dummies(
        data["EDUC_label"],
        prefix="EDUC",
        drop_first=True,
        dummy_na=False,
    )

    X = pd.concat([X_numeric, X_cats], axis=1)
    y = data["is_divorced"]

    return X, y


# =========================================================
# 2. Fit logistic regression
# =========================================================

def fit_divorce_logit(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Fit a logistic regression model for P(divorced | X).

    Returns:
        {
            "model": fitted LogisticRegression,
            "scaler": fitted StandardScaler,
            "X_train", "X_test", "y_train", "y_test",
            "y_pred_proba": predicted probs on test set,
            "roc_auc": ROC AUC on test,
            "brier": Brier score on test
        }
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric columns only (keep dummy cols as-is)
    num_cols = ["AGE", "is_female", "log_incwage", "YRMARR"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # Logistic regression (L2 regularized, reasonably stable)
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
    )
    model.fit(X_train_scaled, y_train)

    # Predictions on test
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_proba": y_proba,
        "roc_auc": roc_auc,
        "brier": brier,
    }


# =========================================================
# 3. Logistic model diagnostics (visuals)
# =========================================================

def plot_logit_roc(result: Dict[str, object]) -> None:
    """
    Plot ROC curve for the fitted logistic model and save to visuals/.
    """
    y_test = result["y_test"]
    y_proba = result["y_pred_proba"]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Divorce Risk Model: ROC Curve")
    ax.legend()
    fig.tight_layout()

    save_visual(fig, "logit_divorce_roc.png")


def plot_logit_calibration_like(result: Dict[str, object], n_bins: int = 10) -> None:
    """
    Simple calibration-style plot:
    bin predicted probabilities, compare with observed divorce rate.
    """
    y_test = result["y_test"].to_numpy()
    y_proba = result["y_pred_proba"]

    # Bin by predicted probability quantiles
    bins = np.quantile(y_proba, np.linspace(0, 1, n_bins + 1))
    # Ensure uniqueness to avoid issues
    bins = np.unique(bins)
    bin_ids = np.digitize(y_proba, bins, right=True)

    bin_centers = []
    avg_pred = []
    avg_obs = []

    for b in np.unique(bin_ids):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_centers.append(y_proba[mask].mean())
        avg_pred.append(y_proba[mask].mean())
        avg_obs.append(y_test[mask].mean())

    fig, ax = plt.subplots()
    ax.plot(avg_pred, avg_obs, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Avg predicted P(divorce)")
    ax.set_ylabel("Observed divorce rate")
    ax.set_title("Divorce Risk Model: Calibration (binned)")
    fig.tight_layout()

    save_visual(fig, "logit_divorce_calibration.png")


# =========================================================
# 4. Optional: Cox proportional hazards model
# =========================================================

def fit_divorce_cox(
    df: pd.DataFrame,
    min_duration: float = 0.5,
) -> "lifelines.CoxPHFitter":
    """
    Fit a Cox PH model on time-to-divorce among married individuals.

    Uses:
        duration: marriage_duration
        event: is_divorced
        covariates: AGE, is_female, EDUC_label dummies, log_incwage, YRMARR

    Requires:
        lifelines to be installed.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        raise ImportError("lifelines is required for Cox model. Install via `pip install lifelines`.")

    data = df.copy()

    # Keep those ever married with marriage_duration info
    mask = (
        data["is_ever_married"] == 1
        & data["marriage_duration"].notna()
        & (data["marriage_duration"] >= min_duration)
    )
    data = data[mask]

    # Event: currently divorced vs not
    data["event_divorce"] = (data["MARST_label"] == "Divorced").astype(int)

    # Income
    data["INCWAGE"] = pd.to_numeric(data["INCWAGE"], errors="coerce").fillna(0)
    data["log_incwage"] = np.log1p(data["INCWAGE"].clip(lower=0))

    # Dummies for education
    educ_dummies = pd.get_dummies(data["EDUC_label"], prefix="EDUC", drop_first=True)

    # Covariates
    covars = pd.concat(
        [
            data[["AGE", "is_female", "log_incwage", "YRMARR"]],
            educ_dummies,
        ],
        axis=1,
    )

    cox_df = pd.concat(
        [
            covars,
            data[["marriage_duration", "event_divorce"]],
        ],
        axis=1,
    ).dropna()

    cph = CoxPHFitter()
    cph.fit(
        cox_df,
        duration_col="marriage_duration",
        event_col="event_divorce",
        show_progress=False,
    )
    return cph


# =========================================================
# 5. Example usage (script mode)
# =========================================================

if __name__ == "__main__":
    _ensure_visuals_dir()

    # Load cleaned data
    df = load_ipums_ev_data()

    # ---------- Logistic model ----------
    print("🔹 Building logistic regression dataset...")
    X, y = build_logit_dataset(df)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Divorce rate in sample: {y.mean():.3f}")

    print("🔹 Fitting logistic regression...")
    result = fit_divorce_logit(X, y)

    print(f"ROC AUC: {result['roc_auc']:.3f}")
    print(f"Brier score: {result['brier']:.4f}")

    print("🔹 Saving logistic model diagnostics...")
    plot_logit_roc(result)
    plot_logit_calibration_like(result)

    # ---------- Cox model (optional) ----------
    try:
        print("🔹 Fitting Cox proportional hazards model...")
        cph = fit_divorce_cox(df)
        print(cph.summary)
        # You can also save baseline hazard / survival curves here if you’d like
    except ImportError as e:
        print(f"(Skipping Cox model: {e})")
