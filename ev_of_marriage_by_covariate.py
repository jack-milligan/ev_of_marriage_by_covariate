import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pyarrow
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================================
# Config
# =====================================================================

RAW_PATH = "/Users/jackmilligan/Data Sets/2025_11_ev_marriage_data.csv.gz"
CACHE_PATH = "/Users/jackmilligan/Data Sets/marriage_clean.parquet"


# =====================================================================
# Label Maps
# =====================================================================

def get_label_maps() -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], Dict[int, str]]:
    """Return mapping dictionaries for SEX, MARST, EDUC, STATEFIP."""
    sex_map = {
        1: "Male",
        2: "Female",
        9: "Missing",
    }

    marst_map = {
        1: "Married, spouse present",
        2: "Married, spouse absent",
        3: "Separated",
        4: "Divorced",
        5: "Widowed",
        6: "Never married/single",
        9: "Missing",
    }

    educ_map = {
        0: "N/A or no schooling",
        1: "Gr 1–4",
        2: "Gr 5–8",
        3: "Gr 9",
        4: "Gr 10",
        5: "Gr 11",
        6: "Gr 12 (no college)",
        7: "1 yr college",
        8: "2 yrs college",
        9: "3 yrs college",
        10: "4 yrs college",
        11: "5+ yrs college",
        99: "Missing",
    }

    state_map = {
        1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
        8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
        12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
        18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
        23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
        27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
        31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
        35: "New Mexico", 36: "New York", 37: "North Carolina",
        38: "North Dakota", 39: "Ohio", 40: "Oklahoma", 41: "Oregon",
        42: "Pennsylvania", 44: "Rhode Island", 45: "South Carolina",
        46: "South Dakota", 47: "Tennessee", 48: "Texas", 49: "Utah",
        50: "Vermont", 51: "Virginia", 53: "Washington",
        54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
        72: "Puerto Rico",
        97: "Military/Mil. Reservation",
        99: "Not identified",
    }

    return sex_map, marst_map, educ_map, state_map


# =====================================================================
# Single-responsibility helpers
# =====================================================================

def read_raw_ipums(path: str) -> pd.DataFrame:
    """Read raw IPUMS CSV/CSV.GZ into a DataFrame."""
    return pd.read_csv(path)


def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure key numeric fields are typed correctly."""
    df = df.copy()
    df["YRMARR"] = pd.to_numeric(df["YRMARR"], errors="coerce")
    df.loc[df["YRMARR"] == 0, "YRMARR"] = np.nan
    df["INCWAGE"] = pd.to_numeric(df["INCWAGE"], errors="coerce")
    return df


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to the relevant analytical universe:
    - Household population (GQ in [1,2])
    - Valid sex codes (1,2)
    - Valid marital status codes (1-6)
    """
    df = df.copy()
    df = df[df["GQ"].isin([1, 2])]
    df = df[df["SEX"].isin([1, 2])]
    df = df[df["MARST"].isin([1, 2, 3, 4, 5, 6])]
    return df


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply human-readable labels for sex, marital status, education, and state."""
    df = df.copy()
    sex_map, marst_map, educ_map, state_map = get_label_maps()

    df["SEX_label"] = df["SEX"].map(sex_map)
    df["MARST_label"] = df["MARST"].map(marst_map)
    df["EDUC_label"] = df["EDUC"].map(educ_map)
    df["STATE_label"] = df["STATEFIP"].map(state_map)

    return df


def add_derived_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags used in EV-of-marriage modeling."""
    df = df.copy()
    df["is_married_now"] = (df["MARST"] == 1).astype(int)
    df["is_ever_married"] = df["MARST"].isin([1, 2, 3, 4, 5]).astype(int)
    df["is_divorced"] = (df["MARST"] == 4).astype(int)
    df["is_never_married"] = (df["MARST"] == 6).astype(int)
    df["is_female"] = (df["SEX"] == 2).astype(int)
    return df


def add_marriage_timing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute marriage_duration and age_at_marriage where YRMARR is available.
    """
    df = df.copy()
    has_year = df["YRMARR"].notna() & (df["YRMARR"] > 0)

    df.loc[has_year, "marriage_duration"] = df.loc[has_year, "YEAR"] - df.loc[has_year, "YRMARR"]
    df.loc[has_year, "age_at_marriage"] = (
        df.loc[has_year, "AGE"] - (df.loc[has_year, "YEAR"] - df.loc[has_year, "YRMARR"])
    )

    return df


def cache_dataset(df: pd.DataFrame, cache_path: str) -> None:
    """Save cleaned DataFrame as Parquet."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)


def load_cached_dataset(cache_path: str) -> pd.DataFrame:
    """Load cached Parquet dataset."""
    return pd.read_parquet(cache_path)


# =====================================================================
# Public loader (orchestrator) — still "one thing" at call site
# =====================================================================

def load_ipums_ev_data(
    raw_path: str = RAW_PATH,
    cache_path: str = CACHE_PATH,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load analysis-ready IPUMS ACS data for EV-of-marriage modeling.

    - If use_cache and cache exists: load cached Parquet.
    - Else: read raw, clean, label, derive fields, cache, and return.

    This is the single entrypoint other modules should use.
    """
    if use_cache and os.path.exists(cache_path):
        print(f"✅ Loading cached dataset from {cache_path}")
        return load_cached_dataset(cache_path)

    print(f"⚙️ Reading and processing raw IPUMS data from {raw_path}")
    df = read_raw_ipums(raw_path)
    df = cast_numeric(df)
    df = filter_universe(df)
    df = apply_labels(df)
    df = add_derived_flags(df)
    df = add_marriage_timing(df)

    if use_cache:
        cache_dataset(df, cache_path)
        print(f"💾 Saved cleaned dataset to {cache_path}")

    return df

# =====================================================================
# Script-mode: quick sanity checks (this block does reporting only)
# =====================================================================

if __name__ == "__main__":
    df = load_ipums_ev_data()

    df_25_45 = df[(df["AGE"] >= 25) & (df["AGE"] <= 45)]

    print("\nMarital status distribution (filtered universe):")
    print(df["MARST_label"].value_counts(dropna=False))
    print("\nMedian income by marital status (all ages):")
    print(
        df.groupby("MARST_label")["INCWAGE"]
          .median()
          .sort_values(ascending=False)
    )
    print("\nMedian income by marital status (ages 25–45):")
    print(
        df_25_45.groupby("MARST_label")["INCWAGE"]
                .median()
                .sort_values(ascending=False)
    )
    sns.boxplot(x="MARST_label", y="INCWAGE", data=df_25_45)
    plt.xticks(rotation=45)
    plt.title("Income Distribution by Marital Status (Ages 25–45)")
    plt.show()

