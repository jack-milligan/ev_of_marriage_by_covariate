"""
validation_simple.py

Simplified validation for the EV-of-marriage project.
Essential checks only - keeps the project clean and focused.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ev_of_marriage_by_covariate import load_ipums_ev_data
from risk_model import build_logit_dataset, fit_divorce_logit
from payoff_model import estimate_income_effects

def cross_validate_model(X, y, n_splits=5, random_state=42):
    """Simple k-fold cross-validation."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    roc_auc_scores = []
    brier_scores = []
    num_cols = ["AGE", "is_female", "log_incwage", "YRMARR"]
    num_cols = [col for col in num_cols if col in X.columns]
    
    for train_idx, val_idx in kf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = X_train_fold.copy()
        X_val_scaled = X_val_fold.copy()
        if num_cols:
            X_train_scaled[num_cols] = scaler.fit_transform(X_train_fold[num_cols])
            X_val_scaled[num_cols] = scaler.transform(X_val_fold[num_cols])
        
        # Fit model
        model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=500, random_state=random_state)
        model.fit(X_train_scaled, y_train_fold)
        
        # Predict
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Metrics
        roc_auc_scores.append(roc_auc_score(y_val_fold, y_proba))
        brier_scores.append(brier_score_loss(y_val_fold, y_proba))
    
    return {
        "roc_auc_mean": np.mean(roc_auc_scores),
        "roc_auc_std": np.std(roc_auc_scores),
        "brier_mean": np.mean(brier_scores),
        "brier_std": np.std(brier_scores),
    }


def run_validation():
    """Run essential validation checks."""
    print("=" * 60)
    print("Model Validation")
    print("=" * 60)
    
    # Load data
    df = load_ipums_ev_data()
    print(f"\nData loaded: {len(df):,} observations")
    
    # Build dataset
    X, y = build_logit_dataset(df, age_min=25, age_max=45)
    print(f"Modeling dataset: {len(X):,} rows, {X.shape[1]} features")
    print(f"Divorce rate: {y.mean():.3f}")
    
    # Cross-validation
    print("\nCross-Validation (5-fold):")
    cv_results = cross_validate_model(X, y)
    print(f"  ROC AUC: {cv_results['roc_auc_mean']:.3f} ± {cv_results['roc_auc_std']:.3f}")
    print(f"  Brier:   {cv_results['brier_mean']:.4f} ± {cv_results['brier_std']:.4f}")
    
    # Single train-test split
    print("\nTrain-Test Split:")
    result = fit_divorce_logit(X, y)
    print(f"  ROC AUC: {result['roc_auc']:.3f}")
    print(f"  Brier:   {result['brier']:.4f}")
    
    # Economic estimates
    print("\nEconomic Estimates:")
    econ = estimate_income_effects(df, method="regression")
    print(f"  Married uplift:  {econ['married_uplift']:.1%}")
    print(f"  Divorced penalty: {econ['divorced_penalty']:.1%}")
    
    print("\n" + "=" * 60)
    print("✅ Validation complete")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()

