"""
Prediction Engine: LightGBM with SHAP Explainability
Trains models for 24h, 48h, and 72h prediction horizons.
Runs ablation study comparing full features vs. subsets.
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from src.config import *

logger = logging.getLogger(__name__)

# Feature groups for ablation
SOCIOECONOMIC_FEATURES = ["poverty_rate", "unemployment_rate", "median_income",
                           "uninsured_pct", "population", "vulnerability_score"]

TEMPORAL_FEATURES = [c for c in ["hour_of_day", "day_of_week", "is_weekend", "is_night", "month"]]

NOVEL_FEATURES = ["seizure_vulnerability_interaction", "neighbor_seizure_total", "neighbor_ems_avg"]

DROP_COLS = ["zip_code", "window_start", "window_end", "lat", "lon", "is_hotspot",
             "target_24h", "target_48h", "target_72h"]


def prepare_data(feature_matrix, target_col):
    """Prepare train/test split."""
    # Time-based split: train on first 80%, test on last 20%
    feature_matrix = feature_matrix.sort_values("window_start")
    split_idx = int(len(feature_matrix) * (1 - TEST_SIZE))

    train = feature_matrix.iloc[:split_idx]
    test = feature_matrix.iloc[split_idx:]

    feature_cols = [c for c in feature_matrix.columns if c not in DROP_COLS and c != target_col
                    and not c.startswith("target_")]
    feature_cols = [c for c in feature_cols if feature_matrix[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.bool_]]

    X_train = train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_test = test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train[target_col].astype(int)
    y_test = test[target_col].astype(int)

    logger.info(f"  Features: {len(feature_cols)} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"  Target positive rate - Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_cols


def train_lightgbm(X_train, y_train, X_test, y_test, label="full"):
    """Train LightGBM classifier."""
    # Compute scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    params = LIGHTGBM_PARAMS.copy()
    params["scale_pos_weight"] = n_neg / n_pos

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(100)],
    )

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)

    # Precision at top 5%
    top_5_pct = max(int(len(y_test) * 0.05), 1)
    top_idx = np.argsort(y_pred)[-top_5_pct:]
    prec_top5 = y_test.iloc[top_idx].mean()

    logger.info(f"  [{label}] AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f} | P@5%: {prec_top5:.4f}")
    return model, y_pred, auc, ap, prec_top5


def run_ablation(X_train, y_train, X_test, y_test, feature_cols):
    """Run ablation study to quantify value of each feature group."""
    logger.info("\n--- ABLATION STUDY ---")
    results = {}

    # Full model
    _, _, full_auc, _, _ = train_lightgbm(X_train, y_train, X_test, y_test, "FULL")
    results["full"] = full_auc

    # Without socioeconomic features
    socio_cols = [c for c in SOCIOECONOMIC_FEATURES if c in feature_cols]
    if socio_cols:
        non_socio = [c for c in feature_cols if c not in socio_cols]
        _, _, auc_no_socio, _, _ = train_lightgbm(
            X_train[non_socio], y_train, X_test[non_socio], y_test, "NO_SOCIOECONOMIC"
        )
        results["without_socioeconomic"] = auc_no_socio

    # Without novel features
    novel_cols = [c for c in NOVEL_FEATURES if c in feature_cols]
    if novel_cols:
        non_novel = [c for c in feature_cols if c not in novel_cols]
        _, _, auc_no_novel, _, _ = train_lightgbm(
            X_train[non_novel], y_train, X_test[non_novel], y_test, "NO_NOVEL"
        )
        results["without_novel_features"] = auc_no_novel

    # Only event counts (baseline)
    event_cols = [c for c in feature_cols if any(c.startswith(p) for p in ["ems_", "ed_", "naloxone_", "dea_"])
                  and "rolling" not in c and "lag" not in c and "velocity" not in c]
    if event_cols:
        _, _, auc_baseline, _, _ = train_lightgbm(
            X_train[event_cols], y_train, X_test[event_cols], y_test, "EVENT_COUNTS_ONLY"
        )
        results["event_counts_only"] = auc_baseline

    logger.info("\n--- ABLATION RESULTS ---")
    for name, auc in results.items():
        delta = auc - results.get("event_counts_only", auc)
        logger.info(f"  {name:30s}: {auc:.4f} (delta: {delta:+.4f})")

    return results


def train_pipeline():
    """Full model training pipeline."""
    logger.info("=" * 60)
    logger.info("TRAINING PREDICTION MODELS")
    logger.info("=" * 60)

    feature_matrix = pd.read_parquet(DATA_DIR / "feature_matrix.parquet")

    all_results = {}

    for horizon in PREDICTION_HORIZONS:
        hours = horizon * TIME_WINDOW_HOURS
        target_col = f"target_{hours}h"

        if target_col not in feature_matrix.columns:
            logger.warning(f"  Target {target_col} not found, skipping")
            continue

        logger.info(f"\n{'='*40}")
        logger.info(f"  HORIZON: {hours}h prediction")
        logger.info(f"{'='*40}")

        X_train, X_test, y_train, y_test, feature_cols = prepare_data(feature_matrix, target_col)

        # Train full model
        model, y_pred, auc, ap, prec_top5 = train_lightgbm(
            X_train, y_train, X_test, y_test, f"{hours}h_FULL"
        )

        # SHAP
        logger.info(f"  Computing SHAP values for {hours}h model...")
        explainer = shap.TreeExplainer(model)
        shap_sample = X_test[:2000]
        raw_shap = explainer.shap_values(shap_sample)
        shap_values = raw_shap[1] if isinstance(raw_shap, list) else raw_shap

        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        logger.info(f"\n  Top 10 features ({hours}h):")
        for _, row in importance.head(10).iterrows():
            is_novel = "*NOVEL*" if row["feature"] in NOVEL_FEATURES else ""
            logger.info(f"    {row['feature']:40s}: {row['importance']:6.0f} {is_novel}")

        # Run ablation for 24h model only
        if hours == 24:
            ablation_results = run_ablation(X_train, y_train, X_test, y_test, feature_cols)
        else:
            ablation_results = {}

        # Save artifacts
        pickle.dump(model, open(MODEL_DIR / f"model_{hours}h.pkl", "wb"))
        np.save(MODEL_DIR / f"shap_values_{hours}h.npy", shap_values)
        shap_sample.to_parquet(DATA_DIR / f"shap_sample_{hours}h.parquet", index=False)
        importance.to_csv(MODEL_DIR / f"feature_importance_{hours}h.csv", index=False)

        # Save predictions
        test_preds = X_test.copy()
        test_preds["y_true"] = y_test.values
        test_preds["y_pred"] = y_pred
        test_preds.to_parquet(DATA_DIR / f"predictions_{hours}h.parquet", index=False)

        all_results[f"{hours}h"] = {
            "auc_roc": round(auc, 4),
            "avg_precision": round(ap, 4),
            "precision_at_top5pct": round(prec_top5, 4),
            "n_features": len(feature_cols),
            "ablation": {k: round(v, 4) for k, v in ablation_results.items()},
        }

    # Save results summary
    json.dump(all_results, open(MODEL_DIR / "results.json", "w"), indent=2)
    logger.info(f"\n  All models saved to {MODEL_DIR}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    train_pipeline()
