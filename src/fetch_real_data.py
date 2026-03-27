"""
Real Data Fetchers for Opioid Overdose Surveillance
Fetches data from 3 free public APIs:
  1. CDC VSRR — Provisional drug overdose death counts (state × month)
  2. Census ACS — Socioeconomic data by ZCTA (zip code)
  3. CDC WONDER — Final overdose mortality (state × year)

All sources are free, no API keys required.
"""
import pandas as pd
import numpy as np
import requests
import logging
import time
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_DIR

logger = logging.getLogger(__name__)

REAL_DATA_DIR = DATA_DIR / "real"
REAL_DATA_DIR.mkdir(exist_ok=True)

# ─── CDC VSRR: Provisional Drug Overdose Death Counts ──────────────────────────

CDC_VSRR_URL = "https://data.cdc.gov/api/views/xkb8-kh2a/rows.csv?accessType=DOWNLOAD"

OPIOID_INDICATORS = [
    "Number of Drug Overdose Deaths",
    "Natural & semi-synthetic opioids (T40.2)",
    "Heroin (T40.1)",
    "Synthetic opioids, excl. methadone (T40.4)",
    "Methadone (T40.3)",
    "Opioids (T40.0-T40.4,T40.6)",
    "Cocaine (T40.5)",
    "Psychostimulants with abuse potential (T43.6)",
]


def fetch_cdc_vsrr():
    """Fetch CDC VSRR provisional drug overdose death counts.
    Returns state × month overdose death counts by drug type.
    Source: https://data.cdc.gov/NCHS/VSRR-Provisional-Drug-Overdose-Death-Counts/xkb8-kh2a
    """
    logger.info("Fetching CDC VSRR provisional overdose death counts...")
    logger.info(f"  URL: {CDC_VSRR_URL}")

    response = requests.get(CDC_VSRR_URL, timeout=120)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    logger.info(f"  Raw records: {len(df):,}")

    # Filter for opioid-related indicators
    df = df[df["Indicator"].isin(OPIOID_INDICATORS)].copy()

    # Clean data
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Percent Complete"] = pd.to_numeric(df["Percent Complete"], errors="coerce")

    # Keep useful columns
    df = df[[
        "State", "State Name", "Year", "Month", "Period",
        "Indicator", "Data Value", "Percent Complete", "Predicted Value",
    ]].copy()

    df = df.rename(columns={
        "State": "state_abbr",
        "State Name": "state_name",
        "Year": "year",
        "Month": "month",
        "Data Value": "death_count",
        "Percent Complete": "pct_complete",
        "Predicted Value": "predicted_value",
        "Indicator": "indicator",
        "Period": "period",
    })

    # Drop rows without death counts
    df = df.dropna(subset=["death_count"])

    # Save
    out_path = REAL_DATA_DIR / "cdc_vsrr_overdose.parquet"
    df.to_parquet(out_path, index=False)

    n_states = df["state_abbr"].nunique()
    year_range = f"{int(df['year'].min())}-{int(df['year'].max())}"
    logger.info(f"  Saved: {len(df):,} records | {n_states} states | Years: {year_range}")
    logger.info(f"  Output: {out_path}")

    return df


# ─── Census ACS: Socioeconomic Data by ZCTA ────────────────────────────────────

CENSUS_ACS_BASE = "https://api.census.gov/data/2022/acs/acs5"

# ACS variable codes
ACS_VARIABLES = {
    "B01001_001E": "total_population",
    "B17001_001E": "poverty_universe",       # Population for whom poverty is determined
    "B17001_002E": "poverty_below",          # Below poverty level
    "B19013_001E": "median_household_income",
    "B23025_003E": "civilian_labor_force",
    "B23025_005E": "unemployed",
    "B27010_001E": "insurance_universe",     # Civilian noninstitutionalized population
    "B27010_017E": "no_insurance_19_34",     # Uninsured 19-34
    "B27010_033E": "no_insurance_35_64",     # Uninsured 35-64
    "B27010_050E": "no_insurance_65_plus",   # Uninsured 65+
    "B01002_001E": "median_age",
}

# State FIPS codes for states with high overdose rates
HIGH_OVERDOSE_STATES = {
    "17": "Illinois",
    "39": "Ohio",
    "42": "Pennsylvania",
    "54": "West Virginia",
    "21": "Kentucky",
    "47": "Tennessee",
    "25": "Massachusetts",
    "36": "New York",
    "34": "New Jersey",
    "24": "Maryland",
}


def fetch_census_acs():
    """Fetch Census ACS 5-year socioeconomic data by ZCTA.
    No API key required (limited to 500 queries/day/IP without key).
    ZCTAs are queried nationally (they cross state boundaries).
    Source: https://api.census.gov
    """
    logger.info("Fetching Census ACS data for all ZCTAs nationally...")

    # Batch variables into 2 requests to avoid URL length limits
    batch1_vars = {
        "B01001_001E": "total_population",
        "B17001_001E": "poverty_universe",
        "B17001_002E": "poverty_below",
        "B19013_001E": "median_household_income",
        "B23025_003E": "civilian_labor_force",
        "B23025_005E": "unemployed",
    }
    batch2_vars = {
        "B01002_001E": "median_age",
        "B27010_001E": "insurance_universe",
        "B27010_017E": "no_insurance_19_34",
        "B27010_033E": "no_insurance_35_64",
        "B27010_050E": "no_insurance_65_plus",
    }

    all_batches = []
    for batch_name, var_dict in [("batch1", batch1_vars), ("batch2", batch2_vars)]:
        var_string = ",".join(var_dict.keys())
        url = f"{CENSUS_ACS_BASE}?get=NAME,{var_string}&for=zip%20code%20tabulation%20area:*"
        logger.info(f"  Fetching {batch_name} ({len(var_dict)} variables)...")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
            # Rename variable codes to human-readable names
            rename_map = {k: v for k, v in var_dict.items() if k in df.columns}
            rename_map["zip code tabulation area"] = "zip_code"
            df = df.rename(columns=rename_map)
            all_batches.append(df)
            logger.info(f"    {len(df):,} ZCTAs retrieved")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.warning(f"    {batch_name} failed: {e}")

    if not all_batches:
        logger.error("  No Census data retrieved!")
        return pd.DataFrame()

    # Merge batches on zip_code
    if len(all_batches) == 2:
        df = all_batches[0].merge(
            all_batches[1].drop(columns=["NAME"], errors="ignore"),
            on="zip_code", how="outer"
        )
    else:
        df = all_batches[0]

    # Convert to numeric
    numeric_cols = list(batch1_vars.values()) + list(batch2_vars.values())
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute derived features
    df["poverty_rate"] = (df["poverty_below"] / df["poverty_universe"].clip(lower=1) * 100).round(1)
    df["unemployment_rate"] = (df["unemployed"] / df["civilian_labor_force"].clip(lower=1) * 100).round(1)

    # Uninsured rate
    uninsured_cols = ["no_insurance_19_34", "no_insurance_35_64", "no_insurance_65_plus"]
    existing_uninsured = [c for c in uninsured_cols if c in df.columns]
    if existing_uninsured:
        df["total_uninsured"] = df[existing_uninsured].sum(axis=1)
        df["uninsured_pct"] = (df["total_uninsured"] / df["insurance_universe"].clip(lower=1) * 100).round(1)
    else:
        df["uninsured_pct"] = np.nan

    # Vulnerability score (same formula as synthetic data)
    df["vulnerability_score"] = (
        df["poverty_rate"].fillna(0) / 100 * 0.4 +
        df["unemployment_rate"].fillna(0) / 100 * 0.3 +
        df["uninsured_pct"].fillna(0) / 100 * 0.3
    ).round(3)

    # Keep useful columns
    output_cols = [
        "zip_code", "NAME",
        "total_population", "median_household_income", "median_age",
        "poverty_rate", "unemployment_rate", "uninsured_pct",
        "vulnerability_score",
    ]
    df = df[[c for c in output_cols if c in df.columns]]

    # Save
    out_path = REAL_DATA_DIR / "census_acs_socioeconomic.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"  Saved: {len(df):,} ZCTAs | Vulnerability range: {df['vulnerability_score'].min():.3f} - {df['vulnerability_score'].max():.3f}")
    logger.info(f"  Output: {out_path}")

    return df


# ─── CDC WONDER: Overdose Mortality ─────────────────────────────────────────────

CDC_WONDER_URL = "https://wonder.cdc.gov/controller/datarequest/D176"

# XML query for drug overdose deaths by state and year
# ICD-10 underlying cause: X40-X44 (accidental), X60-X64 (suicide), X85 (assault), Y10-Y14 (undetermined)
CDC_WONDER_XML = """<?xml version="1.0" encoding="utf-8"?>
<request-parameters>
    <parameter>
        <name>B_1</name>
        <value>D176.V9-level1</value>
    </parameter>
    <parameter>
        <name>B_2</name>
        <value>D176.V1-level1</value>
    </parameter>
    <parameter>
        <name>M_1</name>
        <value>D176.M1</value>
    </parameter>
    <parameter>
        <name>M_2</name>
        <value>D176.M2</value>
    </parameter>
    <parameter>
        <name>O_V1_fmode</name>
        <value>freg</value>
    </parameter>
    <parameter>
        <name>O_V9_fmode</name>
        <value>freg</value>
    </parameter>
    <parameter>
        <name>V_D176.V2</name>
        <value>*All*</value>
    </parameter>
    <parameter>
        <name>V_D176.V25</name>
        <value>*All*</value>
    </parameter>
    <parameter>
        <name>F_D176.V25</name>
        <value>*All*</value>
    </parameter>
    <parameter>
        <name>I_D176.V25</name>
        <value>*All* (All Coverage)</value>
    </parameter>
    <parameter>
        <name>V_D176.V4</name>
        <value>*All*</value>
    </parameter>
    <parameter>
        <name>action-Send</name>
        <value>Send</value>
    </parameter>
    <parameter>
        <name>O_title</name>
        <value>Drug Overdose Deaths by State</value>
    </parameter>
    <parameter>
        <name>O_timeout</name>
        <value>300</value>
    </parameter>
    <parameter>
        <name>O_datatable</name>
        <value>default</value>
    </parameter>
    <parameter>
        <name>O_V10_fmode</name>
        <value>freg</value>
    </parameter>
    <parameter>
        <name>accept_datause_restrictions</name>
        <value>true</value>
    </parameter>
</request-parameters>"""


def fetch_cdc_wonder():
    """Fetch overdose mortality data from CDC WONDER.
    Note: CDC WONDER API has strict rate limits (1 req / 2 min).
    Falls back to a direct download approach if the XML API fails.
    Source: https://wonder.cdc.gov
    """
    logger.info("Fetching CDC WONDER overdose mortality data...")
    logger.info("  Note: CDC WONDER API may be slow or require agreement to terms.")

    try:
        response = requests.post(
            CDC_WONDER_URL,
            data={"request_xml": CDC_WONDER_XML},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=120,
        )

        if response.status_code == 200 and "<response" in response.text:
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)

            rows = []
            for r in root.findall(".//r"):
                cells = r.findall("c")
                if len(cells) >= 4:
                    rows.append({
                        "state": cells[0].get("l", ""),
                        "year": cells[1].get("l", ""),
                        "deaths": cells[2].get("v", ""),
                        "crude_rate": cells[3].get("v", ""),
                    })

            if rows:
                df = pd.DataFrame(rows)
                df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
                df["crude_rate"] = pd.to_numeric(df["crude_rate"], errors="coerce")
                df["year"] = pd.to_numeric(df["year"], errors="coerce")

                out_path = REAL_DATA_DIR / "cdc_wonder_overdose.parquet"
                df.to_parquet(out_path, index=False)
                logger.info(f"  Saved: {len(df):,} records | {df['state'].nunique()} states")
                return df

        logger.warning("  CDC WONDER XML API returned unexpected response, using VSRR data as fallback.")

    except Exception as e:
        logger.warning(f"  CDC WONDER API error: {e}")
        logger.info("  Using CDC VSRR data as fallback for mortality statistics.")

    # Fallback: aggregate VSRR data to get state-level annual counts
    vsrr_path = REAL_DATA_DIR / "cdc_vsrr_overdose.parquet"
    if vsrr_path.exists():
        vsrr = pd.read_parquet(vsrr_path)
        # Use "Number of Drug Overdose Deaths" indicator, December of each year (12-month ending)
        annual = vsrr[
            (vsrr["indicator"] == "Number of Drug Overdose Deaths") &
            (vsrr["month"] == "December") &
            (vsrr["period"] == "12 month-ending")
        ][["state_abbr", "state_name", "year", "death_count"]].copy()
        annual = annual.dropna(subset=["death_count"])

        out_path = REAL_DATA_DIR / "cdc_wonder_overdose.parquet"
        annual.to_parquet(out_path, index=False)
        logger.info(f"  Fallback saved: {len(annual):,} state-year records from VSRR")
        return annual

    logger.warning("  No fallback data available. Run fetch_cdc_vsrr() first.")
    return pd.DataFrame()


# ─── Main Orchestrator ──────────────────────────────────────────────────────────

def fetch_all():
    """Fetch all real data sources."""
    logger.info("=" * 60)
    logger.info("FETCHING REAL DATA FROM PUBLIC SOURCES")
    logger.info("=" * 60)

    results = {}

    # 1. CDC VSRR (most reliable, fetch first)
    try:
        results["vsrr"] = fetch_cdc_vsrr()
        logger.info(f"  ✓ CDC VSRR: {len(results['vsrr']):,} records")
    except Exception as e:
        logger.error(f"  ✗ CDC VSRR failed: {e}")
        results["vsrr"] = pd.DataFrame()

    # 2. Census ACS (socioeconomic)
    try:
        results["census"] = fetch_census_acs()
        logger.info(f"  ✓ Census ACS: {len(results['census']):,} ZCTAs")
    except Exception as e:
        logger.error(f"  ✗ Census ACS failed: {e}")
        results["census"] = pd.DataFrame()

    # 3. CDC WONDER (uses VSRR fallback)
    try:
        results["wonder"] = fetch_cdc_wonder()
        logger.info(f"  ✓ CDC WONDER: {len(results['wonder']):,} records")
    except Exception as e:
        logger.error(f"  ✗ CDC WONDER failed: {e}")
        results["wonder"] = pd.DataFrame()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("REAL DATA FETCH COMPLETE")
    total = sum(len(v) for v in results.values())
    logger.info(f"  Total records: {total:,}")
    for name, df in results.items():
        status = f"{len(df):,} records" if len(df) > 0 else "FAILED"
        logger.info(f"  {name:10s}: {status}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    fetch_all()
