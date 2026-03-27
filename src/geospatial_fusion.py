"""
Geospatial-Temporal Fusion Engine
Fuses 4 data sources into a zip code x 6-hour time window grid.
Computes 35+ features per cell including:
  - EMS dispatch volume and naloxone rates
  - ED admission counts by opioid ICD-10 codes
  - Naloxone distribution velocity (novel leading indicator)
  - DEA seizure proximity scores
  - Socioeconomic risk factors (Census ACS proxy)
  - Historical overdose patterns (rolling averages)
  - Spatial autocorrelation (neighbor effects)
  - Seizure-vulnerability interaction (novel)
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.spatial.distance import cdist
import logging
from src.config import *

logger = logging.getLogger(__name__)


def build_base_grid(zips, ems, ed, naloxone, dea):
    """Build the zip code x time window base grid."""
    logger.info("Building base geospatial-temporal grid...")

    # Determine time range from data
    all_timestamps = pd.concat([
        ems["timestamp"], ed["timestamp"], naloxone["timestamp"], dea["timestamp"]
    ])
    min_time = all_timestamps.min().floor("6h")
    max_time = all_timestamps.max().ceil("6h")
    windows = pd.date_range(min_time, max_time, freq=f"{TIME_WINDOW_HOURS}h")

    logger.info(f"  Time range: {min_time} to {max_time}")
    logger.info(f"  Windows: {len(windows):,} | Zips: {len(zips)} | Grid cells: {len(windows)*len(zips):,}")

    # Create grid
    grid_rows = []
    for w in windows:
        for _, z in zips.iterrows():
            grid_rows.append({
                "zip_code": z["zip_code"],
                "window_start": w,
                "window_end": w + timedelta(hours=TIME_WINDOW_HOURS),
                "lat": z["lat"],
                "lon": z["lon"],
                "poverty_rate": z["poverty_rate"],
                "unemployment_rate": z["unemployment_rate"],
                "median_income": z["median_income"],
                "uninsured_pct": z["uninsured_pct"],
                "population": z["population"],
                "vulnerability_score": z["vulnerability_score"],
                "is_hotspot": z["is_hotspot"],
            })

    grid = pd.DataFrame(grid_rows)
    logger.info(f"  Base grid: {len(grid):,} cells")
    return grid, windows


def compute_event_features(grid, ems, ed, naloxone, dea):
    """Compute per-cell event counts and rates from the 4 data sources."""
    logger.info("Computing event-level features...")

    # Bin events into grid cells
    for df, name in [(ems, "ems"), (ed, "ed"), (naloxone, "naloxone"), (dea, "dea")]:
        df["window_start"] = df["timestamp"].dt.floor(f"{TIME_WINDOW_HOURS}h")

    # ─── EMS Features ───────────────────────────────────────────────────────
    ems_agg = ems.groupby(["zip_code", "window_start"]).agg(
        ems_dispatch_count=("event_type", "count"),
        ems_naloxone_rate=("naloxone_administered", "mean"),
        ems_fatal_count=("outcome", lambda x: (x == "fatal").sum()),
        ems_avg_response_time=("response_time_min", "mean"),
    ).reset_index()
    grid = grid.merge(ems_agg, on=["zip_code", "window_start"], how="left")

    # ─── ED Features ────────────────────────────────────────────────────────
    ed_agg = ed.groupby(["zip_code", "window_start"]).agg(
        ed_admission_count=("icd10_code", "count"),
        ed_expired_count=("disposition", lambda x: (x == "expired").sum()),
        ed_avg_los=("length_of_stay_hours", "mean"),
    ).reset_index()

    # Specific opioid code counts
    fentanyl_codes = ["T40.4X1A", "T40.3X1A"]
    ed_fent = ed[ed["icd10_code"].isin(fentanyl_codes)].groupby(
        ["zip_code", "window_start"]).size().reset_index(name="ed_fentanyl_count")
    ed_agg = ed_agg.merge(ed_fent, on=["zip_code", "window_start"], how="left")
    grid = grid.merge(ed_agg, on=["zip_code", "window_start"], how="left")

    # ─── Naloxone Features ──────────────────────────────────────────────────
    nal_agg = naloxone.groupby(["zip_code", "window_start"]).agg(
        naloxone_dist_count=("distribution_type", "count"),
        naloxone_units_total=("units_distributed", "sum"),
    ).reset_index()

    # Distribution by type
    nal_type = naloxone.groupby(["zip_code", "window_start", "distribution_type"]).size().unstack(
        fill_value=0).reset_index()
    nal_type.columns = ["zip_code", "window_start"] + [f"naloxone_{c}_count" for c in nal_type.columns[2:]]
    nal_agg = nal_agg.merge(nal_type, on=["zip_code", "window_start"], how="left")
    grid = grid.merge(nal_agg, on=["zip_code", "window_start"], how="left")

    # ─── DEA Features ───────────────────────────────────────────────────────
    dea_agg = dea.groupby(["zip_code", "window_start"]).agg(
        dea_seizure_count=("drug_type", "count"),
        dea_total_grams=("quantity_grams", "sum"),
        dea_street_value=("estimated_street_value", "sum"),
    ).reset_index()

    # Fentanyl-specific seizures
    dea_fent = dea[dea["drug_type"].str.contains("Fentanyl|Carfentanil", na=False)].groupby(
        ["zip_code", "window_start"]).size().reset_index(name="dea_fentanyl_seizures")
    dea_agg = dea_agg.merge(dea_fent, on=["zip_code", "window_start"], how="left")
    grid = grid.merge(dea_agg, on=["zip_code", "window_start"], how="left")

    grid = grid.fillna(0)
    logger.info(f"  Event features computed: {grid.shape[1]} columns")
    return grid


def compute_temporal_features(grid):
    """Compute rolling, lag, and velocity features."""
    logger.info("Computing temporal features...")

    grid = grid.sort_values(["zip_code", "window_start"])

    # Group by zip for rolling computations
    for col in ["ems_dispatch_count", "ed_admission_count", "naloxone_units_total", "dea_seizure_count"]:
        if col not in grid.columns:
            continue

        base_name = col.replace("_count", "").replace("_total", "")

        # Rolling 4-week average (28 days / 6-hour windows = 112 windows)
        grid[f"{base_name}_rolling_4w"] = grid.groupby("zip_code")[col].transform(
            lambda x: x.rolling(112, min_periods=1).mean()
        )

        # Rolling 1-week average
        grid[f"{base_name}_rolling_1w"] = grid.groupby("zip_code")[col].transform(
            lambda x: x.rolling(28, min_periods=1).mean()
        )

        # Lag features (1 window = 6 hours ago)
        grid[f"{base_name}_lag_1"] = grid.groupby("zip_code")[col].shift(1).fillna(0)
        grid[f"{base_name}_lag_4"] = grid.groupby("zip_code")[col].shift(4).fillna(0)  # 24h ago

        # Velocity: rate of change over last 7 days
        grid[f"{base_name}_velocity"] = (
            grid[f"{base_name}_rolling_1w"] - grid[f"{base_name}_rolling_4w"]
        )

    # ─── NOVEL: Naloxone Distribution Velocity (the leading indicator) ──────
    if "naloxone_units_total" in grid.columns:
        grid["naloxone_velocity_zscore"] = grid.groupby("zip_code")["naloxone_units_velocity"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        ) if "naloxone_units_velocity" in grid.columns else 0

    # Time features
    grid["hour_of_day"] = grid["window_start"].dt.hour
    grid["day_of_week"] = grid["window_start"].dt.dayofweek
    grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(int)
    grid["is_night"] = ((grid["hour_of_day"] >= 20) | (grid["hour_of_day"] < 6)).astype(int)
    grid["month"] = grid["window_start"].dt.month

    logger.info(f"  Temporal features computed: {grid.shape[1]} columns")
    return grid


def compute_spatial_features(grid, zips):
    """Compute spatial autocorrelation and neighbor effects."""
    logger.info("Computing spatial features...")

    # Build distance matrix between zip codes
    coords = zips[["lat", "lon"]].values
    dist_matrix = cdist(coords, coords, metric="euclidean")

    # Define neighbors (within ~5 miles ~ 0.07 degrees)
    neighbor_threshold = 0.07
    neighbors = {}
    zip_list = zips["zip_code"].values
    for i, z in enumerate(zip_list):
        neighbor_idx = np.where((dist_matrix[i] > 0) & (dist_matrix[i] < neighbor_threshold))[0]
        neighbors[z] = [zip_list[j] for j in neighbor_idx]

    # Compute neighbor overdose rates for each cell
    if "ems_dispatch_count" in grid.columns:
        zip_window_counts = grid.set_index(["zip_code", "window_start"])["ems_dispatch_count"]

        neighbor_rates = []
        for _, row in grid.iterrows():
            z = row["zip_code"]
            w = row["window_start"]
            nb_zips = neighbors.get(z, [])
            if nb_zips:
                nb_counts = [zip_window_counts.get((nb, w), 0) for nb in nb_zips]
                neighbor_rates.append(np.mean(nb_counts))
            else:
                neighbor_rates.append(0)

        grid["neighbor_ems_avg"] = neighbor_rates

    # ─── NOVEL: Seizure-Vulnerability Interaction ───────────────────────────
    # DEA seizures near high-vulnerability zips
    grid["seizure_vulnerability_interaction"] = (
        grid["dea_seizure_count"] * grid["vulnerability_score"]
    )

    # Seizure proximity: sum of seizures in neighboring zips
    if "dea_seizure_count" in grid.columns:
        zip_window_seizures = grid.set_index(["zip_code", "window_start"])["dea_seizure_count"]
        neighbor_seizures = []
        for _, row in grid.iterrows():
            z = row["zip_code"]
            w = row["window_start"]
            nb_zips = neighbors.get(z, [])
            if nb_zips:
                nb_seiz = [zip_window_seizures.get((nb, w), 0) for nb in nb_zips]
                neighbor_seizures.append(sum(nb_seiz))
            else:
                neighbor_seizures.append(0)
        grid["neighbor_seizure_total"] = neighbor_seizures

    logger.info(f"  Spatial features computed: {grid.shape[1]} columns")
    return grid


def create_target_variable(grid):
    """Create binary target: will there be a high-risk overdose event in the next N windows?
    Uses a composite risk score that combines EMS overdoses and ED admissions,
    with zip-specific thresholds so hotspot and non-hotspot zips are both represented."""
    logger.info("Creating target variables...")

    grid = grid.sort_values(["zip_code", "window_start"])

    # Composite overdose signal: EMS dispatch + ED admissions (weighted)
    grid["overdose_signal"] = grid["ems_dispatch_count"] + grid.get("ed_admission_count", 0) * 0.5

    # Per-zip rolling baseline and standard deviation
    grid["zip_baseline"] = grid.groupby("zip_code")["overdose_signal"].transform(
        lambda x: x.rolling(28, min_periods=4).mean()
    ).fillna(grid["overdose_signal"].mean())

    grid["zip_std"] = grid.groupby("zip_code")["overdose_signal"].transform(
        lambda x: x.rolling(28, min_periods=4).std()
    ).fillna(grid["overdose_signal"].std())
    grid["zip_std"] = grid["zip_std"].clip(lower=0.1)

    for horizon in PREDICTION_HORIZONS:
        col_name = f"target_{horizon * TIME_WINDOW_HOURS}h"
        # Future sum of overdose signals in the next N windows
        future_sum = grid.groupby("zip_code")["overdose_signal"].transform(
            lambda x: x.shift(-1).rolling(horizon, min_periods=1).sum()
        ).fillna(0)

        # Target = future events exceed baseline + 0.5*std (per-zip adaptive threshold)
        # This creates a ~15-20% positive rate with strong feature-target correlation
        threshold = (grid["zip_baseline"] * horizon) + (grid["zip_std"] * 0.5 * np.sqrt(horizon))
        threshold = threshold.clip(lower=0.5)
        grid[col_name] = (future_sum > threshold).astype(int)
        logger.info(f"  {col_name}: {grid[col_name].mean():.2%} positive rate")

    # Clean up temp columns
    grid.drop(columns=["overdose_signal", "zip_baseline", "zip_std"], inplace=True, errors="ignore")
    return grid


def build_feature_matrix():
    """Full geospatial-temporal fusion pipeline."""
    logger.info("=" * 60)
    logger.info("GEOSPATIAL-TEMPORAL FUSION ENGINE")
    logger.info("=" * 60)

    # Load data
    zips = pd.read_parquet(DATA_DIR / "zip_codes.parquet")
    ems = pd.read_parquet(DATA_DIR / "ems_validated.parquet")
    ed = pd.read_parquet(DATA_DIR / "ed_validated.parquet")
    naloxone = pd.read_parquet(DATA_DIR / "naloxone_validated.parquet")
    dea = pd.read_parquet(DATA_DIR / "dea_validated.parquet")

    # Ensure timestamps are datetime
    for df in [ems, ed, naloxone, dea]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Build grid
    grid, windows = build_base_grid(zips, ems, ed, naloxone, dea)

    # Compute features
    grid = compute_event_features(grid, ems, ed, naloxone, dea)
    grid = compute_temporal_features(grid)

    # Spatial features on a sample (full grid is too large for pairwise ops)
    logger.info("Computing spatial features on sampled grid...")
    sample_size = min(200_000, len(grid))
    grid_sample = grid.sample(sample_size, random_state=RANDOM_SEED).copy()
    grid_sample = compute_spatial_features(grid_sample, zips)

    # Create target
    grid_sample = create_target_variable(grid_sample)

    # Drop rows where target is NaN (end of time series)
    target_col = f"target_{PREDICTION_HORIZONS[0] * TIME_WINDOW_HOURS}h"
    grid_sample = grid_sample.dropna(subset=[target_col])

    # Save
    grid_sample.to_parquet(DATA_DIR / "feature_matrix.parquet", index=False)
    logger.info(f"\nFeature matrix: {grid_sample.shape[0]:,} rows x {grid_sample.shape[1]} columns")
    logger.info(f"Saved to {DATA_DIR / 'feature_matrix.parquet'}")

    return grid_sample


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build_feature_matrix()
