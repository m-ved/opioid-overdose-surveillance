"""
Streaming Ingestion Simulator
Simulates Kafka-style event processing with:
  - Schema validation per source type
  - Event deduplication
  - Dead letter queue for malformed records
  - Ingestion metrics logging
"""
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from src.config import *

logger = logging.getLogger(__name__)

# ─── Schema Definitions (Avro-style) ───────────────────────────────────────────

SCHEMAS = {
    "EMS": {"required": ["timestamp", "zip_code", "event_type", "naloxone_administered", "outcome"],
            "types": {"naloxone_administered": bool}},
    "ED": {"required": ["timestamp", "zip_code", "icd10_code", "disposition"],
           "types": {}},
    "NALOXONE": {"required": ["timestamp", "zip_code", "distribution_type", "units_distributed"],
                  "types": {"units_distributed": (int, np.integer)}},
    "DEA": {"required": ["timestamp", "zip_code", "drug_type", "quantity_grams"],
            "types": {"quantity_grams": (int, float, np.floating)}},
}


def validate_event(event, source):
    """Validate a single event against its schema."""
    schema = SCHEMAS.get(source)
    if not schema:
        return False, f"Unknown source: {source}"

    for field in schema["required"]:
        if field not in event or pd.isna(event.get(field)):
            return False, f"Missing required field: {field}"

    for field, expected_type in schema["types"].items():
        if field in event and not isinstance(event[field], expected_type):
            return False, f"Type mismatch for {field}: expected {expected_type}, got {type(event[field])}"

    return True, "OK"


def process_stream(df, source_name):
    """Process a dataframe as if it were a stream of events."""
    logger.info(f"Processing {source_name} stream ({len(df):,} events)...")

    valid_events = []
    dead_letter = []
    duplicates = 0
    seen_keys = set()

    for _, row in df.iterrows():
        event = row.to_dict()

        # Deduplication key
        dedup_key = f"{event.get('timestamp')}_{event.get('zip_code')}_{source_name}_{hash(str(event))}"
        if dedup_key in seen_keys:
            duplicates += 1
            continue
        seen_keys.add(dedup_key)

        # Validate
        is_valid, msg = validate_event(event, source_name)
        if is_valid:
            valid_events.append(event)
        else:
            event["_error"] = msg
            event["_source"] = source_name
            dead_letter.append(event)

    valid_df = pd.DataFrame(valid_events) if valid_events else pd.DataFrame()
    dlq_df = pd.DataFrame(dead_letter) if dead_letter else pd.DataFrame()

    logger.info(f"  Valid: {len(valid_df):,} | DLQ: {len(dlq_df):,} | Duplicates: {duplicates}")

    # Save ingestion metrics
    metrics = {
        "source": source_name,
        "total_received": len(df),
        "valid": len(valid_df),
        "dead_letter": len(dlq_df),
        "duplicates": duplicates,
        "completeness": len(valid_df) / max(len(df), 1),
        "processed_at": datetime.now().isoformat(),
    }

    return valid_df, dlq_df, metrics


def run_ingestion():
    """Run ingestion for all data sources."""
    logger.info("=" * 60)
    logger.info("RUNNING STREAMING INGESTION")
    logger.info("=" * 60)

    ems = pd.read_parquet(DATA_DIR / "ems_dispatch.parquet")
    ed = pd.read_parquet(DATA_DIR / "ed_admissions.parquet")
    naloxone = pd.read_parquet(DATA_DIR / "naloxone_distribution.parquet")
    dea = pd.read_parquet(DATA_DIR / "dea_seizures.parquet")

    all_metrics = []

    ems_valid, ems_dlq, ems_m = process_stream(ems, "EMS")
    ed_valid, ed_dlq, ed_m = process_stream(ed, "ED")
    nal_valid, nal_dlq, nal_m = process_stream(naloxone, "NALOXONE")
    dea_valid, dea_dlq, dea_m = process_stream(dea, "DEA")

    all_metrics = [ems_m, ed_m, nal_m, dea_m]

    # Save validated data
    ems_valid.to_parquet(DATA_DIR / "ems_validated.parquet", index=False)
    ed_valid.to_parquet(DATA_DIR / "ed_validated.parquet", index=False)
    nal_valid.to_parquet(DATA_DIR / "naloxone_validated.parquet", index=False)
    dea_valid.to_parquet(DATA_DIR / "dea_validated.parquet", index=False)

    # Save DLQ
    dlq_all = pd.concat([ems_dlq, ed_dlq, nal_dlq, dea_dlq], ignore_index=True)
    if len(dlq_all) > 0:
        dlq_all.to_parquet(DATA_DIR / "dead_letter_queue.parquet", index=False)

    # Save metrics
    pd.DataFrame(all_metrics).to_csv(DATA_DIR / "ingestion_metrics.csv", index=False)

    total_valid = sum(m["valid"] for m in all_metrics)
    total_received = sum(m["total_received"] for m in all_metrics)
    logger.info(f"\nOverall completeness: {total_valid}/{total_received} ({total_valid/max(total_received,1):.1%})")

    return all_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_ingestion()
