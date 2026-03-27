"""
Pipeline Orchestrator: Opioid Overdose Surveillance
  1. Generate synthetic data (4 sources)
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

    logger.info("\n[1/4] GENERATING SYNTHETIC DATA (4 sources)...")
    from src.generate_data import generate_all
    generate_all()

    logger.info("\n[2/4] RUNNING STREAMING INGESTION...")
    from src.streaming_ingestion import run_ingestion
    run_ingestion()

    logger.info("\n[3/4] GEOSPATIAL-TEMPORAL FUSION (35+ features)...")
    from src.geospatial_fusion import build_feature_matrix
    build_feature_matrix()

    logger.info("\n[4/4] TRAINING LIGHTGBM MODELS (24h/48h/72h)...")
    from src.train_model import train_pipeline
    results = train_pipeline()

    elapsed = time.time() - start
    logger.info("\n" + "=" * 70)
    logger.info(f"  PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    for horizon, r in results.items():
        logger.info(f"  {horizon} AUC-ROC: {r['auc_roc']}")
    logger.info("=" * 70)
    logger.info("\n  Launch dashboard: streamlit run dashboards/app.py")


if __name__ == "__main__":
    main()
