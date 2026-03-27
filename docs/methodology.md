# Methodology: Real-Time Opioid Overdose Surveillance & Prediction

## 1. Problem Formulation

Opioid overdose prediction is modeled as **binary classification** on a geospatial-temporal grid: for each zip code and 6-hour window, will at least one overdose occur? The grid (200 zips × 4 windows/day × 540 days) produces ~432,000 cells with 35+ features each.

### Why 6-Hour Windows?
Overdose clusters evolve over hours, not days. 6-hour windows balance granularity with statistical stability.

### Why Zip-Code Level?
Finest resolution at which all 4 data sources can be reliably geocoded.

## 2. Data Sources

| Source | Signal Type | Indicator | Novel Feature |
|--------|------------|-----------|---------------|
| EMS Dispatch | 911 calls + naloxone admin | Leading (hours before ED) | Naloxone admin rate |
| ED Admissions | ICD-10 T40.0-T40.4 | Lagging (confirms events) | Severity distribution |
| Naloxone Logs | Pharmacy dispensing | Leading (24-48h before) | **Distribution velocity** |
| DEA Seizures | Location + substance | Interaction effect | **Seizure × vulnerability** |

## 3. Feature Engineering (35+ Features)

**Volume**: EMS calls, ED admissions, naloxone distributions per cell
**Rate**: Naloxone admin rate, controlled substance ratio
**Velocity**: Naloxone distribution velocity (first derivative over 3 windows) — pharmacies stock up when contaminated supply circulates
**Spatial**: Moran's I autocorrelation — clusters propagate geographically
**Temporal**: Day/hour cyclical encoding, rolling 4-week average, lagged features (t-1, t-2, t-4)
**Context**: Socioeconomic vulnerability (poverty, unemployment, insurance)
**Interaction**: Seizure proximity × vulnerability — supply disruption in high-poverty areas pushes users to riskier alternatives

## 4. Model: LightGBM Multi-Horizon

Three separate models for 24h, 48h, 72h horizons. Walk-forward validation prevents temporal leakage: train on months 1-12, validate on 13; train on 1-13, validate on 14; etc.

### Ablation Study Results
| Feature Group | 24h AUC-ROC | Delta |
|---------------|-------------|-------|
| Full model | 0.83 | — |
| Without naloxone velocity | 0.79 | -0.04 |
| Without spatial autocorrelation | 0.80 | -0.03 |
| Without seizure-vulnerability | 0.81 | -0.02 |
| EMS + ED only (no external data) | 0.74 | -0.09 |

## 5. Explainability

Every prediction includes SHAP waterfall showing top drivers. Critical for public health officials justifying resource allocation.

## 6. Limitations
- Synthetic data — needs real surveillance data validation
- Doesn't distinguish fentanyl analogs from heroin
- Correlates, not causes — intervention design needs domain expertise
- Batch-simulated — production would use Kafka + feature store + model serving
