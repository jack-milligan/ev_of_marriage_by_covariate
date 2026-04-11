# Expected Value of Marriage: Findings Summary

**Author:** Jack Milligan  
**Data:** IPUMS American Community Survey (ACS), ~15 million observations  
**Scope:** US household population, cross-sectional analysis

---

## Executive Summary

Using logistic regression trained on IPUMS ACS data, this analysis estimates divorce risk and the economic expected value (EV) of marriage by demographic group. The central finding is that **marriage has positive expected economic value across all tested demographic profiles**, driven primarily by a substantial income premium for married individuals. The break-even divorce probability — the rate at which the EV of marrying equals staying single — is approximately 57%, well above any observed group's actual divorce risk.

Education emerges as the dominant predictor of both divorce risk and EV: college-educated individuals face 3–6% divorce risk and command the highest income premium, resulting in the largest positive EV. The weakest case for marriage is among lower-education, lower-income individuals, where the EV is still positive but with a narrower margin.

---

## Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| ROC AUC | 0.687 | Acceptable discrimination; meaningful but not strong |
| Brier Score | 0.093 | Good probabilistic calibration |
| 5-fold CV AUC | 0.687 ± 0.001 | Highly stable — not overfitting |

The logistic regression was benchmarked against a random forest classifier. The AUC gap between the two models was less than 0.01, confirming that the feature set — not the algorithm — is the binding constraint on predictive power. Logistic regression was retained because its coefficients are directly interpretable as log-odds, which is essential for communicating driver importance to non-technical audiences.

---

## Key Findings

### 1. Divorce Risk Drivers

- **Education** is the strongest predictor: college-educated individuals have roughly half the divorce risk of those with a high school diploma only.
- **Women** have slightly higher observed divorce rates than men at the same age and education level.
- **Marriage cohort (YRMARR)** shows a negative cross-sectional relationship with observed divorce — recent marriages appear to have lower divorce rates, but this is a duration artifact (less time to dissolve). The cohort coefficient in the logistic model captures this confound; longitudinal inference requires the Cox PH model.
- **Age** has a modest negative relationship with divorce risk: marrying older is slightly protective.
- **Income** has a negative relationship: higher earners are less likely to be divorced, reflecting both causal stability and selection effects.

### 2. Economic Effects of Marriage

Regression-controlled estimates (controlling for age, sex, and education):

| Comparison | Income Effect |
|---|---|
| Married vs. never-married | +25–37% |
| Divorced vs. married | -11–24% |

These estimates are substantially larger than naive median comparisons because they partially control for the selection of higher earners into marriage. Note that full causal identification is not possible with cross-sectional ACS data; residual selection bias likely remains.

### 3. Expected Value Analysis

EV is computed as a discounted cash flow comparison over a 10-year horizon (3% discount rate, $20,000 divorce cost):

| Profile | P(divorce) | Delta EV (Marry − Single) |
|---|---|---|
| Male, 32, college, $80K | ~6% | Strongly positive |
| Female, 29, advanced degree, $120K | ~5% | Strongly positive |
| Male, 25, high school, $40K | ~12% | Positive, narrower margin |

**Break-even analysis:** At the regression-estimated income effects, marriage has positive EV until the divorce probability exceeds approximately 57%. No demographic group in the data approaches this threshold.

### 4. Sensitivity of EV Conclusions

The positive-EV conclusion is robust to reasonable variation in model parameters:

- **Time horizon (5–30 years):** Delta EV remains positive across all horizons; magnitude scales with horizon.
- **Discount rate (1–7%):** Sign of delta EV does not change; higher rates reduce magnitude.
- **Divorce cost ($0–$50K):** Even at $50K fixed divorce costs, delta EV remains positive for all tested profiles given the income premium.

The conclusion is sensitive to the income effect estimate. If the true causal married premium is near zero (i.e., the observed gap is entirely due to selection), the EV case for marriage weakens substantially.

---

## Limitations

1. **Causal identification:** Income effects may substantially reflect selection rather than a causal effect of marriage. Instrumental variable or difference-in-differences approaches would be required for stronger causal claims — not possible with this cross-sectional dataset.

2. **Static model:** Divorce probability is assumed constant over the horizon. In reality, hazard varies with marriage duration (Cox PH model handles this; logistic regression does not).

3. **Economic scope:** The EV model captures only wage income effects. It excludes non-monetary benefits (companionship, health effects), shared household economies of scale, and wealth accumulation effects. The full EV of marriage is likely larger than these estimates suggest.

4. **Temporal validity:** The model is trained on a cross-section of a single ACS wave. Temporal generalization (e.g., to marriages beginning in 2025) assumes that structural relationships are stable, which may not hold during periods of rapid social change.

5. **Remarriage:** The model does not incorporate the probability of remarriage following divorce, which would reduce the effective economic penalty of a failed marriage.

---

## Recommended Next Steps

- **Temporal validation:** Train on earlier ACS waves, test on later waves to assess out-of-sample generalization.
- **Interaction terms:** Test education × sex and education × income interactions, which may reveal heterogeneous divorce risk not captured by main effects.
- **Household income framing:** Extend the income effect model to household income rather than individual wage income to better capture economies of scale.
- **Geographic disaggregation:** State-level divorce rates vary substantially; a state fixed effect may improve prediction.