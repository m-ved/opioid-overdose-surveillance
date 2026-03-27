"""
Synthetic Data Generator for Opioid Surveillance
Simulates 4 realistic data sources:
  1. EMS dispatch records (911 calls with naloxone administration)
  2. Emergency department admissions (opioid-related ICD-10 codes)
  3. Naloxone distribution logs (pharmacy-level Narcan dispensing)
  4. DEA drug seizure reports (regional seizure events)

Includes realistic geographic clustering, temporal patterns, and surge events.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.config import *

np.random.seed(RANDOM_SEED)
logger = logging.getLogger(__name__)

# ─── Reference Data ────────────────────────────────────────────────────────────

OPIOID_ICD10 = ["T40.0X1A", "T40.1X1A", "T40.2X1A", "T40.3X1A", "T40.4X1A",
                 "T40.601A", "F11.10", "F11.20", "F11.90", "F11.23"]

DRUG_TYPES = ["Fentanyl", "Heroin", "Oxycodone", "Hydrocodone", "Carfentanil",
              "Methamphetamine+Fentanyl", "Cocaine+Fentanyl", "Unknown Synthetic"]

SEIZURE_TYPES = ["Street seizure", "Mail intercept", "Lab raid", "Traffic stop", "Pharmacy diversion"]

# ─── Zip Code Grid ──────────────────────────────────────────────────────────────

def generate_zip_grid():
    """Generate realistic zip code grid centered on Chicago metro."""
    logger.info(f"Generating {ZIP_CODES_N} zip codes...")
    lats = np.random.normal(CHICAGO_CENTER[0], 0.35, ZIP_CODES_N)
    lons = np.random.normal(CHICAGO_CENTER[1], 0.45, ZIP_CODES_N)

    # Socioeconomic features (Census ACS-like)
    poverty_rates = np.clip(np.random.beta(2, 5, ZIP_CODES_N) * 100, 2, 60)
    unemployment = np.clip(np.random.beta(2, 8, ZIP_CODES_N) * 100, 1, 30)
    median_income = np.clip(np.random.normal(55000, 20000, ZIP_CODES_N), 15000, 150000).astype(int)
    insurance_uninsured_pct = np.clip(np.random.beta(2, 10, ZIP_CODES_N) * 100, 1, 35)
    population = np.random.randint(2000, 80000, ZIP_CODES_N)

    # Designate hotspot zip codes (correlated with high poverty)
    vulnerability_score = (poverty_rates / 100 * 0.4 + unemployment / 100 * 0.3 +
                           insurance_uninsured_pct / 100 * 0.3)
    hotspot_threshold = np.percentile(vulnerability_score, 100 - (N_HOTSPOT_ZIPS / ZIP_CODES_N * 100))
    is_hotspot = vulnerability_score >= hotspot_threshold

    zips = pd.DataFrame({
        "zip_code": [f"606{i:02d}" for i in range(ZIP_CODES_N)],
        "lat": lats, "lon": lons,
        "poverty_rate": poverty_rates.round(1),
        "unemployment_rate": unemployment.round(1),
        "median_income": median_income,
        "uninsured_pct": insurance_uninsured_pct.round(1),
        "population": population,
        "vulnerability_score": vulnerability_score.round(3),
        "is_hotspot": is_hotspot,
    })
    return zips


def generate_time_grid(zips):
    """Generate the zip code x time window grid."""
    start = datetime(2023, 1, 1)
    n_windows = (HISTORY_DAYS * 24) // TIME_WINDOW_HOURS
    windows = [start + timedelta(hours=i * TIME_WINDOW_HOURS) for i in range(n_windows)]

    logger.info(f"Time grid: {len(windows)} windows x {len(zips)} zips = {len(windows)*len(zips):,} cells")
    return windows


# ─── Data Source Generators ─────────────────────────────────────────────────────

def generate_surge_schedule(zips, windows):
    """Pre-compute surge events that persist across 2-6 consecutive windows.
    Real-world overdose surges last 12-36 hours as contaminated batches circulate."""
    np.random.seed(RANDOM_SEED + 1)
    hotspot_zips = zips[zips["is_hotspot"]]["zip_code"].values
    surge_matrix = {}  # (zip_code, window_idx) -> multiplier

    for z in hotspot_zips:
        w = 0
        while w < len(windows):
            if np.random.random() < SURGE_PROBABILITY:
                # Surge lasts 2-6 windows (12-36 hours)
                surge_duration = np.random.randint(2, 7)
                surge_intensity = np.random.uniform(4, 10)
                for d in range(surge_duration):
                    if w + d < len(windows):
                        # Intensity decays over the surge
                        decay = 1.0 - (d / surge_duration) * 0.5
                        surge_matrix[(z, w + d)] = surge_intensity * decay
                w += surge_duration
            else:
                w += 1

    # Also add smaller surges for non-hotspot zips (rare)
    non_hotspot = zips[~zips["is_hotspot"]]["zip_code"].values
    for z in non_hotspot:
        w = 0
        while w < len(windows):
            if np.random.random() < SURGE_PROBABILITY * 0.15:
                duration = np.random.randint(2, 4)
                intensity = np.random.uniform(2, 5)
                for d in range(duration):
                    if w + d < len(windows):
                        surge_matrix[(z, w + d)] = intensity * (1.0 - d/duration * 0.4)
                w += duration
            else:
                w += 1

    return surge_matrix


def generate_ems_dispatch(zips, windows, surge_matrix=None):
    """Simulate EMS 911 dispatch records with naloxone administration."""
    logger.info("Generating EMS dispatch events...")
    records = []
    for w_idx, window_start in enumerate(windows):
        for _, z in zips.iterrows():
            base_rate = EMS_EVENTS_PER_DAY / len(zips) / (24 / TIME_WINDOW_HOURS)
            if z["is_hotspot"]:
                base_rate *= HOTSPOT_MULTIPLIER

            # Time-of-day effect (more at night)
            hour = window_start.hour
            if 20 <= hour or hour < 6:
                base_rate *= 1.8
            elif 14 <= hour < 20:
                base_rate *= 1.3

            # Weekend effect
            if window_start.weekday() >= 5:
                base_rate *= 1.4

            # Persistent surge from pre-computed schedule
            if surge_matrix and (z["zip_code"], w_idx) in surge_matrix:
                base_rate *= surge_matrix[(z["zip_code"], w_idx)]

            n_events = np.random.poisson(max(base_rate, 0.01))
            for _ in range(n_events):
                naloxone_given = np.random.random() < 0.65
                records.append({
                    "timestamp": window_start + timedelta(minutes=np.random.randint(0, TIME_WINDOW_HOURS * 60)),
                    "zip_code": z["zip_code"],
                    "event_type": "opioid_overdose",
                    "naloxone_administered": naloxone_given,
                    "patient_age": int(np.clip(np.random.normal(35, 12), 16, 75)),
                    "patient_gender": np.random.choice(["M", "F"], p=[0.65, 0.35]),
                    "response_time_min": round(np.random.exponential(8), 1),
                    "outcome": np.random.choice(["survived", "fatal"], p=[0.85, 0.15]),
                    "source": "EMS",
                })

    df = pd.DataFrame(records)
    logger.info(f"  EMS events: {len(df):,} | Fatal: {(df['outcome']=='fatal').sum():,}")
    return df


def generate_ed_admissions(zips, windows, surge_matrix=None):
    """Simulate emergency department admissions with opioid-related ICD-10 codes."""
    logger.info("Generating ED admission events...")
    records = []
    for w_idx, window_start in enumerate(windows):
        for _, z in zips.iterrows():
            base_rate = ED_ADMISSIONS_PER_DAY / len(zips) / (24 / TIME_WINDOW_HOURS)
            if z["is_hotspot"]:
                base_rate *= HOTSPOT_MULTIPLIER * 0.8

            # ED surge follows EMS surge with slight delay and lower intensity
            if surge_matrix and (z["zip_code"], w_idx) in surge_matrix:
                base_rate *= surge_matrix[(z["zip_code"], w_idx)] * 0.7

            n_events = np.random.poisson(max(base_rate, 0.005))
            for _ in range(n_events):
                records.append({
                    "timestamp": window_start + timedelta(minutes=np.random.randint(0, TIME_WINDOW_HOURS * 60)),
                    "zip_code": z["zip_code"],
                    "icd10_code": np.random.choice(OPIOID_ICD10),
                    "patient_age": int(np.clip(np.random.normal(37, 13), 16, 80)),
                    "disposition": np.random.choice(["discharged", "admitted", "transferred", "expired"],
                                                     p=[0.45, 0.35, 0.10, 0.10]),
                    "length_of_stay_hours": round(np.random.exponential(12), 1),
                    "source": "ED",
                })

    df = pd.DataFrame(records)
    logger.info(f"  ED admissions: {len(df):,} | Expired: {(df['disposition']=='expired').sum():,}")
    return df


def generate_naloxone_distribution(zips, windows, surge_matrix=None):
    """Simulate pharmacy-level naloxone (Narcan) distribution logs.
    Naloxone distribution INCREASES before surges as word spreads about contaminated supply."""
    logger.info("Generating naloxone distribution logs...")
    records = []
    for w_idx, window_start in enumerate(windows):
        # Naloxone distributions happen during pharmacy hours
        if not (8 <= window_start.hour < 22):
            continue

        for _, z in zips.iterrows():
            base_rate = NALOXONE_DISTRIBUTIONS_PER_DAY / len(zips) / (14 / TIME_WINDOW_HOURS)
            if z["is_hotspot"]:
                base_rate *= 2.5

            # Naloxone is a LEADING indicator: distributions spike 1-2 windows BEFORE
            # overdose surges as word spreads about contaminated supply
            if surge_matrix:
                # Check if surge is coming in next 1-3 windows
                for future_offset in range(1, 4):
                    future_key = (z["zip_code"], w_idx + future_offset)
                    if future_key in surge_matrix:
                        # Leading indicator: naloxone demand rises before the surge
                        lead_multiplier = surge_matrix[future_key] * 0.6 / (future_offset * 0.7)
                        base_rate *= (1 + lead_multiplier)
                        break
                # Also reactive: naloxone during/after surge
                if (z["zip_code"], w_idx) in surge_matrix:
                    base_rate *= 1.5

            n_dist = np.random.poisson(max(base_rate, 0.01))
            for _ in range(n_dist):
                records.append({
                    "timestamp": window_start + timedelta(minutes=np.random.randint(0, TIME_WINDOW_HOURS * 60)),
                    "zip_code": z["zip_code"],
                    "distribution_type": np.random.choice(["pharmacy", "community_org", "harm_reduction", "ems_restock"],
                                                           p=[0.4, 0.3, 0.2, 0.1]),
                    "units_distributed": np.random.randint(1, 10),
                    "source": "NALOXONE",
                })

    df = pd.DataFrame(records)
    if len(df) > 0:
        logger.info(f"  Naloxone distributions: {len(df):,} | Total units: {df['units_distributed'].sum():,}")
    else:
        logger.info("  Naloxone distributions: 0")
        df = pd.DataFrame(columns=["timestamp", "zip_code", "distribution_type", "units_distributed", "source"])
    return df


def generate_dea_seizures(zips, windows):
    """Simulate DEA drug seizure reports (weekly cadence)."""
    logger.info("Generating DEA seizure reports...")
    records = []
    # Seizures happen weekly, not every window
    weekly_windows = [w for w in windows if w.weekday() == 0 and w.hour == 0]

    for week_start in weekly_windows:
        n_seizures = np.random.poisson(DEA_SEIZURES_PER_WEEK)
        for _ in range(n_seizures):
            zip_row = zips.sample(1, weights="vulnerability_score").iloc[0]
            records.append({
                "timestamp": week_start + timedelta(days=np.random.randint(0, 7)),
                "zip_code": zip_row["zip_code"],
                "drug_type": np.random.choice(DRUG_TYPES, p=[0.35, 0.15, 0.10, 0.08, 0.05, 0.12, 0.10, 0.05]),
                "seizure_type": np.random.choice(SEIZURE_TYPES),
                "quantity_grams": round(np.random.exponential(50), 1),
                "estimated_street_value": round(np.random.exponential(25000), 2),
                "source": "DEA",
            })

    df = pd.DataFrame(records)
    logger.info(f"  DEA seizures: {len(df):,} | Total street value: ${df['estimated_street_value'].sum():,.0f}")
    return df


# ─── Main ───────────────────────────────────────────────────────────────────────

def generate_all():
    """Generate all synthetic data sources."""
    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC OPIOID SURVEILLANCE DATA")
    logger.info("=" * 60)

    zips = generate_zip_grid()
    windows = generate_time_grid(zips)

    # Sample windows but ensure all time-of-day slots are represented
    # Take every 2nd window for balance of coverage and size
    sampled_windows = windows[::2]
    logger.info(f"Using {len(sampled_windows)} sampled windows for data generation")

    ems = generate_ems_dispatch(zips, sampled_windows)
    ed = generate_ed_admissions(zips, sampled_windows)
    naloxone = generate_naloxone_distribution(zips, sampled_windows)
    dea = generate_dea_seizures(zips, sampled_windows)

    # Save
    zips.to_parquet(DATA_DIR / "zip_codes.parquet", index=False)
    ems.to_parquet(DATA_DIR / "ems_dispatch.parquet", index=False)
    ed.to_parquet(DATA_DIR / "ed_admissions.parquet", index=False)
    naloxone.to_parquet(DATA_DIR / "naloxone_distribution.parquet", index=False)
    dea.to_parquet(DATA_DIR / "dea_seizures.parquet", index=False)

    total_events = len(ems) + len(ed) + len(naloxone) + len(dea)
    logger.info("=" * 60)
    logger.info(f"Total events generated: {total_events:,}")
    logger.info(f"Hotspot zip codes: {zips['is_hotspot'].sum()}")
    logger.info(f"Data saved to {DATA_DIR}")
    logger.info("=" * 60)

    return zips, ems, ed, naloxone, dea


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_all()
