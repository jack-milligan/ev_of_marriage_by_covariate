"""
demo.py

Quick demonstration script showing key project capabilities.
Run this to see the project in action.
"""

from ev_of_marriage_by_covariate import load_ipums_ev_data
from risk_model import build_logit_dataset, fit_divorce_logit
from payoff_model import (
    estimate_income_effects,
    train_default_risk_model,
    predict_divorce_prob,
    compute_ev_of_marriage
)

def main():
    print("=" * 70)
    print("Expected Value of Marriage Analysis - Demo")
    print("=" * 70)
    
    # 1. Load data
    print("\n📊 Step 1: Loading data...")
    df = load_ipums_ev_data()
    print(f"   Loaded {len(df):,} observations")
    
    # 2. Estimate economic effects
    print("\n💰 Step 2: Estimating economic effects...")
    econ_params = estimate_income_effects(df, method="regression")
    print(f"   Married income uplift: {econ_params['married_uplift']:.1%}")
    print(f"   Divorced income penalty: {econ_params['divorced_penalty']:.1%}")
    
    # 3. Train model
    print("\n🤖 Step 3: Training divorce risk model...")
    model, scaler, feature_cols = train_default_risk_model(cache_path=None)
    print(f"   Model trained with {len(feature_cols)} features")
    
    # 4. Example predictions
    print("\n🔮 Step 4: Example predictions...")
    
    examples = [
        {
            "label": "Male, 32, College-educated, $80K income",
            "age": 32,
            "sex": "Male",
            "educ_label": "4 yrs college",
            "incwage": 80000,
            "yrmarr": 2020,
        },
        {
            "label": "Female, 29, Advanced degree, $120K income",
            "age": 29,
            "sex": "Female",
            "educ_label": "5+ yrs college",
            "incwage": 120000,
            "yrmarr": 2022,
        },
        {
            "label": "Male, 25, High school, $40K income",
            "age": 25,
            "sex": "Male",
            "educ_label": "Gr 12 (no college)",
            "incwage": 40000,
            "yrmarr": 2023,
        },
    ]
    
    for example in examples:
        try:
            # Predict divorce probability
            p_div = predict_divorce_prob(
                example,
                model,
                scaler,
                feature_cols
            )
            
            # Calculate EV
            ev = compute_ev_of_marriage(
                example,
                p_divorce=p_div,
                married_uplift=econ_params["married_uplift"],
                divorced_penalty=econ_params["divorced_penalty"],
            )
            
            print(f"\n   {example['label']}:")
            print(f"     Divorce risk: {p_div:.1%}")
            print(f"     EV difference: ${ev['delta_EV_marry_minus_single']:,.0f}")
            if ev['delta_EV_marry_minus_single'] > 0:
                print(f"     → Recommendation: Marry (positive EV)")
            else:
                print(f"     → Recommendation: Stay single (negative EV)")
        except Exception as e:
            print(f"   Error with {example['label']}: {e}")
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ Data loaded and processed")
    print("✅ Economic effects estimated")
    print("✅ Divorce risk model trained")
    print("✅ Example predictions generated")
    print("\nFor full analysis, run:")
    print("  python analysis_relationships.py  # EDA")
    print("  python risk_model.py              # Model training")
    print("  python payoff_model.py            # EV calculations")
    print("  python validation.py              # Validation")
    print("=" * 70)

if __name__ == "__main__":
    main()

