import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 0. Utility: save all plots to /visuals folder
# =========================================================

def save_visual(fig: plt.Figure, filename: str, folder: str = "visuals") -> None:
    """
    Save matplotlib Figure to the visuals/ directory.
    Creates the folder if it doesn't exist.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"💾 Saved visualization: {path}")

# =========================================================
# 1. DIVORCE RATE BY AGE, SEX, EDUCATION
# =========================================================

def compute_divorce_rate_by_age_sex_educ(df: pd.DataFrame,
                                         min_n: int = 50) -> pd.DataFrame:
    """
    Compute divorce rate by AGE, SEX_label, and EDUC_label.

    Divorce defined as MARST_label == 'Divorced'.

    Args:
        df: Cleaned IPUMS dataframe.
        min_n: Minimum cell size to keep (for stability).

    Returns:
        DataFrame with columns:
        ['AGE', 'SEX_label', 'EDUC_label', 'divorce_rate', 'n']
    """
    data = df.copy()
    data["is_divorced"] = (data["MARST_label"] == "Divorced").astype(int)

    grouped = (
        data
        .groupby(["AGE", "SEX_label", "EDUC_label"], observed=True)
        .agg(
            divorce_rate=("is_divorced", "mean"),
            n=("is_divorced", "size"),
        )
        .reset_index()
    )

    # filter out tiny cells
    grouped = grouped[grouped["n"] >= min_n]

    return grouped


def plot_divorce_rate_by_age_sex(df: pd.DataFrame,
                                 min_n: int = 100) -> None:
    """
    Line plot: divorce rate by AGE, split by SEX_label.
    Education is pooled (for a clean high-level view).

    Args:
        df: Cleaned IPUMS dataframe.
        min_n: Minimum observations per (AGE, SEX) cell.
    """
    data = df.copy()
    data["is_divorced"] = (data["MARST_label"] == "Divorced").astype(int)

    grouped = (
        data
        .groupby(["AGE", "SEX_label"], observed=True)
        .agg(
            divorce_rate=("is_divorced", "mean"),
            n=("is_divorced", "size"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["n"] >= min_n]

    fig, ax = plt.subplots()
    for sex in grouped["SEX_label"].dropna().unique():
        sub = grouped[grouped["SEX_label"] == sex]
        ax.plot(sub["AGE"], sub["divorce_rate"], label=str(sex))

    ax.set_xlabel("Age")
    ax.set_ylabel("Divorce rate")
    ax.set_title("Divorce rate by age and sex")
    ax.legend()
    fig.tight_layout()

    save_visual(fig, "divorce_rate_by_age_sex.png")


def plot_divorce_rate_by_educ(df: pd.DataFrame,
                              age_min: int = 25,
                              age_max: int = 45,
                              min_n: int = 200) -> None:
    """
    Bar chart: divorce rate by education (restricted to age range).

    Args:
        df: Cleaned dataframe.
        age_min: Min age.
        age_max: Max age.
        min_n: Minimum observations per education bin.
    """
    data = df[(df["AGE"] >= age_min) & (df["AGE"] <= age_max)].copy()
    data["is_divorced"] = (data["MARST_label"] == "Divorced").astype(int)

    grouped = (
        data
        .groupby("EDUC_label", observed=True)
        .agg(
            divorce_rate=("is_divorced", "mean"),
            n=("is_divorced", "size"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["n"] >= min_n]
    grouped = grouped.sort_values("divorce_rate", ascending=False)

    fig, ax = plt.subplots()
    ax.bar(grouped["EDUC_label"], grouped["divorce_rate"])
    ax.set_xticklabels(grouped["EDUC_label"], rotation=45, ha="right")
    ax.set_xlabel("Education")
    ax.set_ylabel("Divorce rate (25–45)")
    ax.set_title("Divorce rate by education (ages 25–45)")
    fig.tight_layout()

    save_visual(fig, "divorce_rate_by_education.png")


# =========================================================
# 2. AGE AT MARRIAGE DISTRIBUTION
# =========================================================

def summarize_age_at_marriage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantify distribution of age_at_marriage for those with valid values.

    Returns a one-row DataFrame with count, mean, stdev, and key quantiles.
    """
    mask = df["age_at_marriage"].notna() & (df["age_at_marriage"] > 0)
    subset = df.loc[mask, "age_at_marriage"]

    return pd.DataFrame({
        "count": [subset.shape[0]],
        "mean": [subset.mean()],
        "std": [subset.std()],
        "min": [subset.min()],
        "p25": [subset.quantile(0.25)],
        "p50": [subset.median()],
        "p75": [subset.quantile(0.75)],
        "max": [subset.max()],
    })


def plot_age_at_marriage_hist(df: pd.DataFrame,
                              bins: int = 30) -> None:
    """
    Histogram of age_at_marriage for all with valid data.
    """
    mask = df["age_at_marriage"].notna() & (df["age_at_marriage"] > 0)
    vals = df.loc[mask, "age_at_marriage"]

    fig, ax = plt.subplots()
    ax.hist(vals, bins=bins)
    ax.set_xlabel("Age at marriage")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of age at marriage")
    fig.tight_layout()

    save_visual(fig, "age_at_marriage_distribution.png")


def plot_age_at_marriage_by_sex(df: pd.DataFrame,
                                bins: int = 30) -> None:
    """
    Overlay histograms of age_at_marriage by sex (quick visual comparison).
    """
    mask = df["age_at_marriage"].notna() & (df["age_at_marriage"] > 0)
    data = df.loc[mask, ["age_at_marriage", "SEX_label"]]

    sexes = data["SEX_label"].dropna().unique()

    fig, ax = plt.subplots()
    for sex in sexes:
        sub = data[data["SEX_label"] == sex]["age_at_marriage"]
        ax.hist(sub, bins=bins, alpha=0.5, label=str(sex))

    ax.set_xlabel("Age at marriage")
    ax.set_ylabel("Count")
    ax.set_title("Age at marriage distribution by sex")
    ax.legend()
    fig.tight_layout()

    save_visual(fig, "age_at_marriage_distribution_by_sex.png")

# =========================================================
# 3. P(DIVORCED | YRMARR, SEX)
# =========================================================

def compute_prob_divorced_given_yrmarr_sex(df: pd.DataFrame,
                                           min_n: int = 50) -> pd.DataFrame:
    """
    Compute probability of being divorced given YRMARR and SEX_label,
    among ever-married individuals.

    Returns:
        DataFrame with ['YRMARR', 'SEX_label', 'prob_divorced', 'n']
    """
    data = df.copy()

    ever_married = data["MARST_label"].isin([
        "Married, spouse present",
        "Married, spouse absent",
        "Separated",
        "Divorced",
        "Widowed"
    ])

    data = data[ever_married & data["YRMARR"].notna()]
    data["is_divorced"] = (data["MARST_label"] == "Divorced").astype(int)

    grouped = (
        data
        .groupby(["YRMARR", "SEX_label"], observed=True)
        .agg(
            prob_divorced=("is_divorced", "mean"),
            n=("is_divorced", "size"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["n"] >= min_n]
    grouped = grouped.sort_values(["YRMARR", "SEX_label"])

    return grouped


def plot_prob_divorced_given_yrmarr_sex(df: pd.DataFrame,
                                        min_n: int = 100) -> None:
    """
    Line plot of P(divorced | YRMARR, SEX_label) over marriage cohort.

    Args:
        df: cleaned dataframe
        min_n: minimum obs per (YRMARR, SEX) cell
    """
    grouped = compute_prob_divorced_given_yrmarr_sex(df, min_n=min_n)

    fig, ax = plt.subplots()
    for sex in grouped["SEX_label"].dropna().unique():
        sub = grouped[grouped["SEX_label"] == sex]
        ax.plot(sub["YRMARR"], sub["prob_divorced"], label=str(sex))

    ax.set_xlabel("Year of marriage (YRMARR)")
    ax.set_ylabel("P(divorced)")
    ax.set_title("Probability of being divorced by marriage year and sex")
    ax.legend()
    fig.tight_layout()

    save_visual(fig, "prob_divorced_given_yrmarr_sex.png")


# =========================================================
# Example usage (if running this module directly)
# =========================================================
if __name__ == "__main__":
    from ev_of_marriage_by_covariate import load_ipums_ev_data  # adjust import to your loader module

    df = load_ipums_ev_data()

    # 1. Divorce rate relationships
    rate_table = compute_divorce_rate_by_age_sex_educ(df)
    print("\nDivorce rate by age, sex, education (head):")
    print(rate_table.head())

    plot_divorce_rate_by_age_sex(df)
    plot_divorce_rate_by_educ(df)

    # 2. Age at marriage distribution
    print("\nAge at marriage summary:")
    print(summarize_age_at_marriage(df))

    plot_age_at_marriage_hist(df)
    plot_age_at_marriage_by_sex(df)

    # 3. P(divorced | YRMARR, SEX)
    prob_table = compute_prob_divorced_given_yrmarr_sex(df)
    print("\nP(divorced | YRMARR, SEX) (head):")
    print(prob_table.head())

    plot_prob_divorced_given_yrmarr_sex(df)
