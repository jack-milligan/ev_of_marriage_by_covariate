"""
risk_model.py

Divorce risk modeling for EV-of-marriage project.

- Logistic regression: P(divorced | covariates)
- Optional Cox proportional hazards: hazard of divorce over marriage duration
- Model persistence (save/load)
- Feature importance visualization
- Prediction functions for new data
- Plots and artifacts saved to visuals/
"""

import os
import pickle
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
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
    """Ensure the visuals directory exists."""
    os.makedirs(folder, exist_ok=True)


def _ensure_models_dir(folder: str = "models") -> None:
    """Ensure the models directory exists."""
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

    Args:
        df: Cleaned IPUMS dataframe
        age_min: Minimum age for analysis (default: 25)
        age_max: Maximum age for analysis (default: 45)
        ever_married_only: If True, restrict to ever-married individuals

    Returns:
        Tuple of (X, y) where:
            X: Feature matrix with columns [AGE, is_female, log_incwage, YRMARR, EDUC_*]
            y: Binary target (1 if divorced, 0 otherwise)

    Raises:
        ValueError: If DataFrame is empty or missing required columns

    Target:
        is_divorced: 1 if MARST_label == 'Divorced', else 0

    Predictors:
        - AGE: Age in years
        - is_female: Binary indicator (1 if female, 0 if male)
        - log_incwage: Log of annual wage income (log1p transformation)
        - YRMARR: Year of marriage (cohort effect)
        - EDUC_label: One-hot encoded education categories
    """
    # Input validation
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    required_cols = ["AGE", "is_female", "INCWAGE", "YRMARR", "EDUC_label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if age_min >= age_max:
        raise ValueError(f"age_min ({age_min}) must be less than age_max ({age_max})")

    data = df.copy()
    initial_rows = len(data)

    # Age range
    data = data[(data["AGE"] >= age_min) & (data["AGE"] <= age_max)]
    rows_after_age = len(data)

    # Universe restriction
    if ever_married_only:
        data = data[data["is_ever_married"] == 1]
    rows_after_marital = len(data)

    # Target
    data["is_divorced"] = (data["MARST_label"] == "Divorced").astype(int)

    # Basic filters: drop rows with missing key info
    data = data.dropna(subset=required_cols)
    rows_after_dropna = len(data)
    
    # Report data loss
    if rows_after_dropna < initial_rows * 0.5:
        print(f"⚠️ Warning: Dropped {initial_rows - rows_after_dropna:,} rows "
              f"({(1 - rows_after_dropna/initial_rows)*100:.1f}% of data)")
    
    if rows_after_dropna < 100:
        raise ValueError(
            f"Insufficient data after filtering: {rows_after_dropna} rows. "
            f"Need at least 100 rows for modeling."
        )

    # Log income to tame skew; zero/negative -> 0 after clip
    data["INCWAGE"] = pd.to_numeric(data["INCWAGE"], errors="coerce").fillna(0)
    data["log_incwage"] = np.log1p(data["INCWAGE"].clip(lower=0))

    # Handle missing / zero YRMARR (should mostly be set for ever-married)
    data["YRMARR"] = pd.to_numeric(data["YRMARR"], errors="coerce")
    
    # Validate YRMARR has reasonable values
    if data["YRMARR"].isna().sum() > len(data) * 0.5:
        print(f"⚠️ Warning: {data['YRMARR'].isna().sum():,} rows have missing YRMARR")

    # Design matrix:
    #   numeric: AGE, is_female, log_incwage, YRMARR
    #   categorical: EDUC_label (one-hot)
    X_numeric = data[["AGE", "is_female", "log_incwage", "YRMARR"]].copy()
    
    # Fill any remaining NaN in numeric columns with median
    for col in X_numeric.columns:
        if X_numeric[col].isna().any():
            median_val = X_numeric[col].median()
            X_numeric[col].fillna(median_val, inplace=True)

    X_cats = pd.get_dummies(
        data["EDUC_label"],
        prefix="EDUC",
        drop_first=True,
        dummy_na=False,
    )

    X = pd.concat([X_numeric, X_cats], axis=1)
    y = data["is_divorced"]
    
    print(f"✅ Built dataset: {len(X)} rows, {X.shape[1]} features, "
          f"divorce rate: {y.mean():.3f}")

    return X, y


# =========================================================
# 2. Fit logistic regression
# =========================================================

def fit_divorce_logit(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
    C: float = 1.0,
    max_iter: int = 500,
) -> Dict[str, object]:
    """
    Fit a logistic regression model for P(divorced | X).

    Args:
        X: Feature matrix (DataFrame)
        y: Binary target (Series)
        test_size: Proportion of data for testing (default: 0.25)
        random_state: Random seed for reproducibility (default: 42)
        C: Inverse regularization strength (default: 1.0, smaller = more regularization)
        max_iter: Maximum iterations for solver (default: 500)

    Returns:
        Dictionary with:
            - "model": fitted LogisticRegression
            - "scaler": fitted StandardScaler
            - "X_train", "X_test", "y_train", "y_test": train/test splits
            - "y_pred_proba": predicted probabilities on test set
            - "roc_auc": ROC AUC score on test set
            - "brier": Brier score on test set
            - "feature_names": list of feature names
            - "num_cols": list of numeric column names

    Raises:
        ValueError: If inputs are invalid or model fails to converge
    """
    # Input validation
    if X.empty or y.empty:
        raise ValueError("X and y cannot be empty")
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}")
    
    # Check for sufficient positive class
    if y.sum() < 10:
        raise ValueError(f"Insufficient positive cases: {y.sum()} (need at least 10)")
    
    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        # If stratification fails (e.g., too few positive cases), try without
        print(f"⚠️ Stratification failed: {e}. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    # Scale numeric columns only (keep dummy cols as-is)
    num_cols = ["AGE", "is_female", "log_incwage", "YRMARR"]
    # Only scale columns that exist
    num_cols = [col for col in num_cols if col in X_train.columns]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if num_cols:
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # Logistic regression (L2 regularized, reasonably stable)
    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        raise ValueError(f"Model fitting failed: {e}") from e

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
        "feature_names": X.columns.tolist(),
        "num_cols": num_cols,
    }


# =========================================================
# 2.5. Model persistence
# =========================================================

def save_model(result: Dict[str, object], path: str) -> None:
    """
    Save trained model, scaler, and metadata to disk.
    
    Args:
        result: Dictionary returned by fit_divorce_logit()
        path: Path to save the model (e.g., "models/divorce_model.pkl")
    """
    _ensure_models_dir(os.path.dirname(path) if os.path.dirname(path) else "models")
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(result, f)
        print(f"💾 Saved model to {path}")
    except Exception as e:
        raise IOError(f"Failed to save model to {path}: {e}") from e


def load_model(path: str) -> Dict[str, object]:
    """
    Load trained model, scaler, and metadata from disk.
    
    Args:
        path: Path to the saved model file
    
    Returns:
        Dictionary with model, scaler, and metadata
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        IOError: If loading fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            result = pickle.load(f)
        print(f"✅ Loaded model from {path}")
        return result
    except Exception as e:
        raise IOError(f"Failed to load model from {path}: {e}") from e


# =========================================================
# 2.6. Prediction function
# =========================================================

def predict_divorce_prob(
    X: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
    num_cols: List[str],
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Predict divorce probability for new data.
    
    Args:
        X: Feature matrix (must match training features)
        model: Fitted LogisticRegression model
        scaler: Fitted StandardScaler
        num_cols: List of numeric column names to scale
        feature_names: Optional list of expected feature names (for validation)
    
    Returns:
        Array of predicted probabilities (0-1)
    
    Raises:
        ValueError: If feature mismatch or scaling fails
    """
    if X.empty:
        raise ValueError("X cannot be empty")
    
    # Validate feature names if provided
    if feature_names is not None:
        missing = set(feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        extra = set(X.columns) - set(feature_names)
        if extra:
            print(f"⚠️ Warning: Extra features in X: {extra}")
    
    X_scaled = X.copy()
    
    # Scale numeric columns
    num_cols_present = [col for col in num_cols if col in X_scaled.columns]
    if num_cols_present:
        try:
            X_scaled[num_cols_present] = scaler.transform(X[num_cols_present])
        except Exception as e:
            raise ValueError(f"Scaling failed: {e}") from e
    
    # Predict
    try:
        y_proba = model.predict_proba(X_scaled)[:, 1]
        return np.clip(y_proba, 0.0, 1.0)  # Ensure valid probabilities
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}") from e


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


def plot_feature_importance(result: Dict[str, object], top_n: int = 20) -> None:
    """
    Plot feature importance (coefficients) from logistic regression.
    
    Args:
        result: Dictionary returned by fit_divorce_logit()
        top_n: Number of top features to display (default: 20)
    """
    model = result["model"]
    feature_names = result.get("feature_names", [f"Feature_{i}" for i in range(len(model.coef_[0]))])
    
    coefs = model.coef_[0]
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = ['red' if x < 0 else 'blue' for x in feature_importance["coefficient"]]
    ax.barh(feature_importance["feature"], feature_importance["coefficient"], color=colors)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} Feature Importance (Logistic Regression Coefficients)")
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    
    save_visual(fig, "logit_feature_importance.png")


def plot_confusion_matrix(result: Dict[str, object], threshold: float = 0.5) -> None:
    """
    Plot confusion matrix for the logistic model.
    
    Args:
        result: Dictionary returned by fit_divorce_logit()
        threshold: Probability threshold for classification (default: 0.5)
    """
    y_test = result["y_test"]
    y_proba = result["y_pred_proba"]
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlabel='Predicted',
           ylabel='Actual',
           title=f'Confusion Matrix (threshold={threshold:.2f})')
    ax.set_xticklabels(['Not Divorced', 'Divorced'])
    ax.set_yticklabels(['Not Divorced', 'Divorced'])
    fig.tight_layout()
    
    save_visual(fig, "logit_confusion_matrix.png")


def plot_precision_recall(result: Dict[str, object]) -> None:
    """
    Plot precision-recall curve for the logistic model.
    """
    y_test = result["y_test"]
    y_proba = result["y_pred_proba"]
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Divorce Risk Model: Precision-Recall Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    
    save_visual(fig, "logit_precision_recall.png")


# =========================================================
# 4. Optional: Cox proportional hazards model
# =========================================================

def fit_divorce_cox(
    df: pd.DataFrame,
    min_duration: float = 0.5,
) -> Any:  # Returns lifelines.CoxPHFitter, but lifelines is optional
    """
    Fit a Cox PH model on time-to-divorce among married individuals.

    Args:
        df: Cleaned IPUMS dataframe
        min_duration: Minimum marriage duration in years (default: 0.5)

    Returns:
        Fitted CoxPHFitter model

    Uses:
        duration: marriage_duration
        event: is_divorced
        covariates: AGE, is_female, EDUC_label dummies, log_incwage, YRMARR

    Requires:
        lifelines to be installed.

    Raises:
        ImportError: If lifelines is not installed
        ValueError: If data is insufficient or invalid
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        raise ImportError("lifelines is required for Cox model. Install via `pip install lifelines`.")
    
    # Input validation
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    required_cols = ["is_ever_married", "marriage_duration", "MARST_label", 
                     "AGE", "is_female", "INCWAGE", "YRMARR", "EDUC_label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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

    if len(cox_df) < 100:
        raise ValueError(f"Insufficient data for Cox model: {len(cox_df)} rows (need at least 100)")
    
    cph = CoxPHFitter()
    try:
        cph.fit(
            cox_df,
            duration_col="marriage_duration",
            event_col="event_divorce",
            show_progress=False,
        )
    except Exception as e:
        raise ValueError(f"Cox model fitting failed: {e}") from e
    
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
    plot_feature_importance(result)
    plot_confusion_matrix(result)
    plot_precision_recall(result)

    # ---------- Cox model (optional) ----------
    try:
        print("🔹 Fitting Cox proportional hazards model...")
        cph = fit_divorce_cox(df)
        print(cph.summary)
        # You can also save baseline hazard / survival curves here if you’d like
    except ImportError as e:
        print(f"(Skipping Cox model: {e})")
