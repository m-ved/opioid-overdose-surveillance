"""Unit tests for opioid surveillance pipeline."""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataGeneration:
    def test_zip_grid(self):
        from src.generate_data import generate_zip_grid
        zips = generate_zip_grid()
        assert len(zips) == 200
        assert "vulnerability_score" in zips.columns
        assert zips["is_hotspot"].sum() > 0
        assert zips["poverty_rate"].between(0, 100).all()

    def test_ems_dispatch(self):
        from src.generate_data import generate_zip_grid, generate_ems_dispatch
        zips = generate_zip_grid()
        windows = [datetime(2024, 1, 1) + timedelta(hours=i*6) for i in range(10)]
        ems = generate_ems_dispatch(zips, windows)
        assert len(ems) > 0
        assert "naloxone_administered" in ems.columns
        assert "outcome" in ems.columns
        assert set(ems["outcome"].unique()).issubset({"survived", "fatal"})

    def test_ed_admissions(self):
        from src.generate_data import generate_zip_grid, generate_ed_admissions
        zips = generate_zip_grid()
        windows = [datetime(2024, 1, 1) + timedelta(hours=i*6) for i in range(10)]
        ed = generate_ed_admissions(zips, windows)
        assert len(ed) > 0
        assert "icd10_code" in ed.columns

    def test_naloxone_distribution(self):
        from src.generate_data import generate_zip_grid, generate_naloxone_distribution
        zips = generate_zip_grid()
        windows = [datetime(2024, 1, 1, 10) + timedelta(hours=i*6) for i in range(10)]
        nal = generate_naloxone_distribution(zips, windows)
        assert len(nal) > 0
        assert "units_distributed" in nal.columns
        assert (nal["units_distributed"] > 0).all()

    def test_dea_seizures(self):
        from src.generate_data import generate_zip_grid, generate_dea_seizures
        zips = generate_zip_grid()
        # Need Monday windows for DEA
        windows = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(60)]
        dea = generate_dea_seizures(zips, windows)
        assert len(dea) > 0
        assert "drug_type" in dea.columns
        assert "quantity_grams" in dea.columns


class TestStreamingIngestion:
    def test_validate_ems_event(self):
        from src.streaming_ingestion import validate_event
        good = {"timestamp": datetime.now(), "zip_code": "60601",
                "event_type": "opioid_overdose", "naloxone_administered": True, "outcome": "survived"}
        valid, msg = validate_event(good, "EMS")
        assert valid is True

    def test_reject_missing_fields(self):
        from src.streaming_ingestion import validate_event
        bad = {"timestamp": datetime.now(), "zip_code": "60601"}
        valid, msg = validate_event(bad, "EMS")
        assert valid is False
        assert "Missing" in msg

    def test_reject_unknown_source(self):
        from src.streaming_ingestion import validate_event
        valid, msg = validate_event({}, "UNKNOWN")
        assert valid is False


class TestFeatureEngineering:
    def test_vulnerability_score(self):
        poverty = np.array([30.0])
        unemployment = np.array([15.0])
        uninsured = np.array([20.0])
        score = poverty/100 * 0.4 + unemployment/100 * 0.3 + uninsured/100 * 0.3
        assert 0 < score[0] < 1

    def test_seizure_vulnerability_interaction(self):
        seizures = np.array([0, 1, 3, 0, 5])
        vulnerability = np.array([0.1, 0.8, 0.2, 0.9, 0.7])
        interaction = seizures * vulnerability
        assert interaction[1] > interaction[0]  # High vuln + seizure > low vuln + no seizure
        assert interaction[4] > interaction[2]  # More seizures * high vuln wins

    def test_rolling_average(self):
        values = pd.Series([0, 0, 0, 5, 10, 3, 0, 0])
        rolling = values.rolling(4, min_periods=1).mean()
        assert rolling.iloc[4] > rolling.iloc[0]


class TestModelTraining:
    def test_time_based_split(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="6h")
        df = pd.DataFrame({"window_start": dates, "value": range(100)})
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        assert train["window_start"].max() < test["window_start"].min()

    def test_prediction_horizons(self):
        from src.config import PREDICTION_HORIZONS, TIME_WINDOW_HOURS
        hours = [h * TIME_WINDOW_HOURS for h in PREDICTION_HORIZONS]
        assert 24 in hours
        assert 48 in hours
        assert 72 in hours


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
