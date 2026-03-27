"""Central configuration for opioid surveillance pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

# Geographic scope: Chicago metro + surrounding Midwest
CHICAGO_CENTER = (41.8781, -87.6298)
ZIP_CODES_N = 200  # Number of zip codes to simulate
TIME_WINDOW_HOURS = 6  # Grid resolution
HISTORY_DAYS = 540  # 18 months of data
PREDICTION_HORIZONS = [4, 8, 12]  # 24h, 48h, 72h in 6-hour windows

# Data source volumes (per day averages)
EMS_EVENTS_PER_DAY = 150
ED_ADMISSIONS_PER_DAY = 80
NALOXONE_DISTRIBUTIONS_PER_DAY = 200
DEA_SEIZURES_PER_WEEK = 25

# Hotspot parameters
N_HOTSPOT_ZIPS = 30  # Zip codes with elevated overdose risk
HOTSPOT_MULTIPLIER = 6.0  # Risk multiplier for hotspot zips
SURGE_PROBABILITY = 0.12  # Probability of a surge event in any window

# Model parameters
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 7,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

TEST_SIZE = 0.2
