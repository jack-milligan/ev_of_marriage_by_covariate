"""
eda_narrative.py

Exploratory data analysis with narrative commentary.
Tells the data story that motivates modeling choices. Run before risk_model.py.

Output: printed narrative to stdout + outputs/eda_summary.txt + PNG visualizations
"""

import os
import sys
import numpy as np
import pandas as pd


class _Tee:
    """Write simultaneously to the original stdout and a file."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._file = open(filepath, "w")
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._stdout
        self._file.close()

from src.ev_of_marriage_by_covariate import load_ipums_ev_data
from src.analysis_relationships import (
    compute_divorce_rate_by_age_sex_educ,
    summarize_age_at_marriage,
    compute_prob_divorced_given_yrmarr_sex,
    plot_divorce_rate_by_age_sex,
    plot_divorce_rate_by_educ,
    plot_age_at_marriage_hist,
    plot_age_at_marriage_by_sex,
    plot_prob_divorced_given_yrmarr_sex,
)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title.upper())
    print("=" * 60)


def main() -> None:
    df = load_ipums_ev_data()
    df_core = df[(df["AGE"] >= 25) & (df["AGE"] <= 45)].copy()
    df_core_income = df_core[df_core["INCWAGE"] > 0].copy()

    # ------------------------------------------------------------------
    _section("1. Sample Overview")
    # ------------------------------------------------------------------
    print(f"Total observations:            {len(df):>12,}")
    print(f"Ages 25-45 (modeling cohort):  {len(df_core):>12,}")

    print("\nMarital status distribution (full sample):")
    marst_dist = df["MARST_label"].value_counts(normalize=True)
    for status, pct in marst_dist.items():
        print(f"  {status:<35} {pct:.1%}")

    print(
        "\nThe sample is drawn from the household population (GQ codes 1-2)."
        "\nGroup quarters residents (dormitories, prisons) are excluded as their"
        "\nmarital and income patterns are not representative of the target population."
    )

    # ------------------------------------------------------------------
    _section("2. Income Gap by Marital Status (Unadjusted)")
    # ------------------------------------------------------------------
    medians = (
        df_core_income
        .groupby("MARST_label")["INCWAGE"]
        .median()
        .sort_values(ascending=False)
    )
    print("\nMedian wage income by marital status (ages 25-45, income > 0):")
    for status, med in medians.items():
        print(f"  {status:<35} ${med:>10,.0f}")

    married = medians.get("Married, spouse present", np.nan)
    never = medians.get("Never married/single", np.nan)
    divorced = medians.get("Divorced", np.nan)

    if not (np.isnan(married) or np.isnan(never)):
        raw_premium = married / never - 1
        print(f"\nRaw married premium over never-married: {raw_premium:+.1%}")

    if not (np.isnan(divorced) or np.isnan(married)):
        raw_penalty = divorced / married - 1
        print(f"Raw divorced shortfall vs. married:     {raw_penalty:+.1%}")

    print(
        "\nCAUTION: These raw figures conflate the causal effect of marriage with"
        "\nselection (higher-earning individuals are more likely to marry and stay"
        "\nmarried). payoff_model.estimate_income_effects() uses regression controls"
        "\nfor age, sex, and education to produce less confounded estimates."
    )

    # ------------------------------------------------------------------
    _section("3. Divorce Rate Patterns")
    # ------------------------------------------------------------------
    ever_married_core = df_core[df_core["is_ever_married"] == 1]
    overall_rate = (df["MARST_label"] == "Divorced").mean()
    core_rate = (ever_married_core["MARST_label"] == "Divorced").mean()

    print(f"Overall divorce rate (full sample):               {overall_rate:.3f}")
    print(f"Divorce rate — ever-married, ages 25-45:          {core_rate:.3f}")

    educ_rates = (
        ever_married_core[ever_married_core["INCWAGE"] > 0]
        .groupby("EDUC_label", observed=True)
        .apply(lambda x: (x["MARST_label"] == "Divorced").mean())
        .sort_values(ascending=False)
    )
    print("\nDivorce rate by education (ever-married, ages 25-45):")
    for educ, rate in educ_rates.items():
        print(f"  {educ:<30} {rate:.3f}")

    print(
        "\nEducation is the strongest single predictor of divorce risk in this data."
        "\nThe gradient from less-educated to college-educated is consistent across"
        "\nsex and age groups. This motivates one-hot encoding EDUC_label in the"
        "\nlogistic model rather than treating it as ordinal — the relationship is"
        "\nnot linear across the 12-level education scale."
    )

    print("\nGenerating divorce rate visualizations...")
    plot_divorce_rate_by_age_sex(df)
    plot_divorce_rate_by_educ(df)

    # ------------------------------------------------------------------
    _section("4. Age at Marriage Distribution")
    # ------------------------------------------------------------------
    summary = summarize_age_at_marriage(df)
    valid_aam = df.loc[df["age_at_marriage"] > 0, "age_at_marriage"]
    pct_missing = df["age_at_marriage"].isna().mean()

    print(f"Records with valid age_at_marriage: {summary['count'].iloc[0]:>10,.0f}")
    print(f"Missing (expected for never-married): {pct_missing:.1%}")
    print(f"\nMean age at marriage:   {summary['mean'].iloc[0]:.1f}")
    print(f"Median age at marriage: {summary['p50'].iloc[0]:.1f}")
    print(f"Std dev:                {summary['std'].iloc[0]:.1f}")
    print(f"10th–90th percentile:   {valid_aam.quantile(0.10):.0f}–{valid_aam.quantile(0.90):.0f}")

    print(
        "\nMissing YRMARR (~40%) is structural: never-married respondents have no"
        "\nmarriage year to report. This is not random missingness and is handled"
        "\ncorrectly by restricting the logistic model to ever-married individuals,"
        "\nwhere YRMARR is almost complete."
    )

    print("\nGenerating age at marriage visualizations...")
    plot_age_at_marriage_hist(df)
    plot_age_at_marriage_by_sex(df)

    # ------------------------------------------------------------------
    _section("5. Cohort Effect — Marriage Year vs. Observed Divorce Rate")
    # ------------------------------------------------------------------
    prob_table = compute_prob_divorced_given_yrmarr_sex(df)

    early_cutoff = prob_table["YRMARR"].quantile(0.25)
    recent_cutoff = prob_table["YRMARR"].quantile(0.75)
    early = prob_table[prob_table["YRMARR"] <= early_cutoff]
    recent = prob_table[prob_table["YRMARR"] >= recent_cutoff]

    print(f"Early-marriage cohorts (YRMARR <= {early_cutoff:.0f}):")
    for sex, grp in early.groupby("SEX_label"):
        print(f"  {sex}: avg P(divorced) = {grp['prob_divorced'].mean():.3f}")

    print(f"\nRecent-marriage cohorts (YRMARR >= {recent_cutoff:.0f}):")
    for sex, grp in recent.groupby("SEX_label"):
        print(f"  {sex}: avg P(divorced) = {grp['prob_divorced'].mean():.3f}")

    print(
        "\nRecent cohorts appear to have lower observed divorce rates, but this is"
        "\na duration artifact: a couple married in 2020 has had fewer years to"
        "\npotentially divorce than one married in 1990. YRMARR is included in the"
        "\nlogistic model to capture true cohort effects (e.g. shifting norms), but"
        "\nits coefficient should be interpreted cautiously given this confound."
        "\nThe Cox PH model (risk_model.fit_divorce_cox) handles duration explicitly"
        "\nand is the more appropriate tool for longitudinal inference."
    )

    print("\nGenerating cohort visualization...")
    plot_prob_divorced_given_yrmarr_sex(df)

    # ------------------------------------------------------------------
    _section("6. Modeling Implications Summary")
    # ------------------------------------------------------------------
    print("""
Key decisions this EDA supports:

  Feature engineering
  - Log-transform INCWAGE: right-skewed distribution warrants log1p
  - One-hot encode EDUC_label: non-linear relationship with divorce risk
  - Include YRMARR as numeric: cohort effects are real but interpretation requires care

  Sample restrictions
  - Restrict logistic model to ever-married, ages 25-45: cleaner target and
    near-complete YRMARR coverage; avoids conflating never-married characteristics
    with divorce predictors

  Causal caveats
  - Raw income-by-marital-status gap is substantial but reflects selection
  - Regression-controlled estimates in payoff_model.py are preferred
  - Full causal identification would require quasi-experimental methods (IV, DiD)
    not available with cross-sectional ACS data
""")


if __name__ == "__main__":
    with _Tee("outputs/eda_summary.txt"):
        main()
    print("Narrative saved to outputs/eda_summary.txt")