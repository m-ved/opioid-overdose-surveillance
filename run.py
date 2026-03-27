"""
Pipeline Orchestrator: Opioid Overdose Surveillance
  1. Fetch real data from CDC/Census APIs
  2. Run streaming ingestion (validation, dedup, DLQ)
  3. Geospatial-temporal fusion (35+ features)
  4. Train LightGBM models (24h/48h/72h + SHAP + ablation)
"""
import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
)
logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("=" * 70)
    logger.info("  OPIOID OVERDOSE SURVEILLANCE PIPELINE")
    logger.info("  Real-Time Prediction | Geospatial Fusion | LightGBM + SHAP")
    logger.info("=" * 70)

    logger.info("\n[1/1] FETCHING REAL DATA (CDC VSRR, Census ACS, CDC WONDER)...")
    from src.fetch_real_data import fetch_all
    results = fetch_all()

    elapsed = time.time() - start
    logger.info("\n" + "=" * 70)
    logger.info(f"  PIPELINE COMPLETE in {elapsed:.1f} seconds")
    logger.info("=" * 70)
    logger.info("\n  Launch dashboard: streamlit run dashboards/app.py")

    return results


if __name__ == "__main__":
    main()
