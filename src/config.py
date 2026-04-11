"""
config.py

Central configuration for data paths and directories.
Override defaults by setting environment variables before running any script:

    export IPUMS_RAW_PATH="/path/to/your/data.csv.gz"
    export IPUMS_CACHE_PATH="/path/to/cache/marriage_clean.parquet"
    export MODELS_DIR="models"
    export VISUALS_DIR="visuals"
"""

import os

RAW_PATH = os.environ.get(
    "IPUMS_RAW_PATH",
    "/Users/jackmilligan/Data Sets/2025_11_ev_marriage_data.csv.gz",
)
CACHE_PATH = os.environ.get(
    "IPUMS_CACHE_PATH",
    "/Users/jackmilligan/Data Sets/marriage_clean.parquet",
)
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
VISUALS_DIR = os.environ.get("VISUALS_DIR", "visuals")