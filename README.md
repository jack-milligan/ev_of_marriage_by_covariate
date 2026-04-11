# Expected Value of Marriage Analysis

A comprehensive data science project analyzing divorce risk and economic outcomes of marriage using IPUMS American Community Survey (ACS) data.

## 🎯 Project Overview

This project builds predictive models to estimate:
- **Divorce risk** based on individual characteristics (age, sex, education, income, marriage cohort)
- **Economic outcomes** of marriage vs. staying single
- **Expected value (EV)** of marriage decisions by demographic group

### Key Questions
- What factors predict divorce risk?
- How does marriage affect income trajectories?
- For which demographic groups is marriage economically beneficial?

## 📊 Data

- **Source**: IPUMS American Community Survey (ACS)
- **Sample Size**: ~15 million observations
- **Time Period**: Multiple years (cohort analysis)
- **Key Variables**: Age, sex, education, income, marital status, marriage year

## 🏗️ Project Structure

```
ev_or_marriage_by_covariate/
├── src/                            # Library package
│   ├── ev_of_marriage_by_covariate.py  # Data loading and preprocessing
│   ├── analysis_relationships.py        # Exploratory data analysis
│   ├── risk_model.py                   # Divorce risk modeling
│   ├── payoff_model.py                 # Economic value calculations
│   ├── validation.py                   # Model validation framework
│   ├── config.py                       # Path configuration
│   └── utils.py                        # Shared utilities
├── tests/                          # Test suite
├── demo.py                         # End-to-end walkthrough
├── eda_narrative.py                # EDA narrative + visualizations
├── generate_report.py              # Report generation
├── outputs/                        # Generated reports/outputs
├── visuals/                        # Generated visualizations
├── models/                         # Saved models
└── README.md                       # This file
```

## 🔧 Technical Stack

- **Python 3.8+**
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, lifelines
- **Data Format**: Parquet (cached), CSV (raw)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ev_or_marriage_by_covariate

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.ev_of_marriage_by_covariate import load_ipums_ev_data
from src.payoff_model import (
    train_default_risk_model,
    predict_divorce_prob_from_profile,
    estimate_income_effects,
    compute_ev_of_marriage,
)

# Load data
df = load_ipums_ev_data()

# Estimate income effects from data
econ = estimate_income_effects(df, method="regression")

# Train divorce risk model
model, scaler, feature_cols = train_default_risk_model()

# Calculate EV of marriage for a profile
profile = {
    "age": 32,
    "sex": "Male",
    "educ_label": "4 yrs college",
    "incwage": 80000,
    "yrmarr": 2020,
}
p_divorce = predict_divorce_prob_from_profile(profile, model, scaler, feature_cols)
ev = compute_ev_of_marriage(
    profile, p_divorce,
    married_uplift=econ["married_uplift"],
    divorced_penalty=econ["divorced_penalty"],
)
print(f"P(divorce): {p_divorce:.3f}")
print(f"Expected value difference: ${ev['delta_EV_marry_minus_single']:,.0f}")
```

### Run Analysis

```bash
# EDA narrative + visualizations
python eda_narrative.py

# Train models and generate diagnostics
python -m src.risk_model

# Run EV calculations
python -m src.payoff_model

# Run comprehensive validation
python -m src.validation

# End-to-end walkthrough
python demo.py
```

## 📈 Key Results

### Model Performance
- **ROC AUC**: 0.687 (acceptable discrimination)
- **Brier Score**: 0.093 (good calibration)
- **Cross-validation stability**: σ = 0.001 (very stable)

### Key Findings
1. **Divorce Risk Factors**:
   - Higher education → Lower divorce risk
   - Women have slightly higher divorce rates
   - More recent marriage cohorts have higher divorce rates
   - Age at marriage matters (older = lower risk)

2. **Economic Effects**:
   - Married individuals earn ~25-37% more than never-married
   - Divorced individuals earn ~11-24% less than married
   - Income benefits of marriage are substantial

3. **Expected Value Analysis**:
   - **Marriage has positive EV for all tested demographics**
   - Break-even divorce probability: ~57% (well above observed rates)
   - Highest risk groups still show positive EV due to income benefits
   - College-educated individuals have highest EV (low risk + high income)

4. **Demographic Patterns**:
   - College-educated: 3-6% divorce risk, highest EV
   - High school only: 6-12% divorce risk, moderate EV
   - Education is strongest predictor of both divorce risk and EV

## 🔍 Validation

The project includes comprehensive validation:
- **Cross-validation**: 5-fold stratified CV
- **Bootstrap validation**: Confidence intervals
- **External validation**: Comparison to known statistics
- **Sensitivity analysis**: Robustness to hyperparameters
- **Prediction reliability**: Edge case testing

Run validation:
```bash
python -m src.validation
```



## 📊 Visualizations

The project generates multiple diagnostic visualizations:
- ROC curves
- Calibration plots
- Feature importance
- Confusion matrices
- Precision-recall curves
- Divorce rate by demographics
- Age at marriage distributions

All saved to `visuals/` directory.

## 🧪 Model Details

### Divorce Risk Model
- **Algorithm**: Logistic Regression (L2 regularized)
- **Features**: Age, sex, log(income), marriage year, education (one-hot)
- **Evaluation**: ROC AUC, Brier score, calibration

### Survival Analysis
- **Model**: Cox Proportional Hazards
- **Outcome**: Time to divorce
- **Covariates**: Same as logistic model

### Economic Model
- **Method**: Regression-based income effects (controls for confounders)
- **Outcome**: Log income
- **Comparison**: Married vs. never-married, divorced vs. married

## 🎓 Data Science Skills Demonstrated

### Data Engineering
- ✅ Large-scale data processing (15M+ rows)
- ✅ ETL pipeline with caching
- ✅ Data quality validation
- ✅ Feature engineering

### Machine Learning
- ✅ Supervised learning (classification)
- ✅ Survival analysis
- ✅ Model evaluation and validation
- ✅ Hyperparameter sensitivity analysis

### Statistical Analysis
- ✅ Regression modeling
- ✅ Causal inference considerations
- ✅ Bootstrap resampling
- ✅ Cross-validation

### Software Engineering
- ✅ Modular code structure
- ✅ Error handling and validation
- ✅ Model persistence
- ✅ Reproducible analysis

## ⚠️ Limitations & Assumptions

1. **Causal Inference**: Income effects may reflect selection bias
2. **Temporal Effects**: Cross-sectional data limits temporal analysis
3. **Missing Data**: ~40% missing marriage year (expected for never-married)
4. **Model Scope**: Focuses on economic outcomes, not other marriage benefits
5. **Generalizability**: Results specific to US population and time period

## 🔮 Future Enhancements

- [ ] Add interaction terms to models
- [ ] Implement alternative models (random forest, XGBoost)
- [ ] Temporal validation (train on earlier years, test on later)
- [ ] Geographic analysis (state-level effects)
- [ ] Web application for predictions
- [x] Unit tests
- [ ] CI/CD pipeline

## 📝 Citation

If using this project, please cite:
- IPUMS USA: Steven Ruggles, Sarah Flood, Matthew Sobek, Danika Brockman, Grace Cooper, Stephanie Richards, and Megan Schouweiler. IPUMS USA: Version 13.0 [dataset]. Minneapolis, MN: IPUMS, 2023. https://doi.org/10.18128/D010.V13.0

## 👤 Author

Jack Milligan


## 🙏 Acknowledgments

- IPUMS for providing the data
- Contributors to open-source libraries used

---

**Note**: This project is for educational/research purposes. Results should not be used for personal financial or relationship decisions.

