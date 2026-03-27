# Real-Time Opioid Overdose Surveillance Dashboard

> National opioid overdose surveillance powered by **real CDC & Census data** — tracking **68,000+ annual deaths**, **53 states/territories**, and **33,774 zip codes** with interactive visualizations.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Data](https://img.shields.io/badge/Data-Real%20CDC%2FCensus-red)

## The Problem

Over **100,000 Americans** die from drug overdoses annually, with opioids accounting for 75%. Public health departments fight this crisis with data that arrives weeks late. This dashboard provides **real-time surveillance** using free public government data.

## Quick Start

```bash
pip install -r requirements.txt
python run.py                      # Fetch real data from CDC/Census APIs
streamlit run dashboards/app.py    # Dashboard at localhost:8501
```

## Dashboard Pages

| Page | Data Source | What It Shows |
|------|-----------|---------------|
| 🌍 **National Overview** | CDC VSRR | Annual deaths, trend line, US choropleth |
| 📈 **Overdose Trends** | CDC VSRR | Multi-state time series, year-over-year comparison |
| 🗺️ **State Comparison** | CDC VSRR | Choropleth map, top-15 rankings, drug heatmap |
| 💊 **Drug Breakdown** | CDC VSRR | Drug-type trends, fentanyl vs heroin, key insights |
| 🏘️ **Vulnerability** | Census ACS | 33,774 zip codes, poverty/unemployment, risk scores |

## Real Data Sources

All data fetched from **free public APIs** — no API keys required:

| Source | Data | Records | URL |
|--------|------|---------|-----|
| **CDC VSRR** | Provisional overdose deaths by drug type | 42,535 | [data.cdc.gov](https://data.cdc.gov/NCHS/VSRR-Provisional-Drug-Overdose-Death-Counts/xkb8-kh2a) |
| **Census ACS** | Poverty, income, unemployment by zip code | 33,774 | [api.census.gov](https://api.census.gov) |
| **CDC WONDER** | Annual mortality by state | 540 | [wonder.cdc.gov](https://wonder.cdc.gov) |

## Project Structure

```
├── src/
│   ├── config.py                  # Central configuration
│   ├── fetch_real_data.py         # Real data fetchers (CDC VSRR, Census ACS, CDC WONDER)
│   ├── streaming_ingestion.py     # Kafka-style validation, dedup, dead-letter queue
│   ├── geospatial_fusion.py       # Zip × time grid, 35+ feature engineering
│   └── train_model.py             # LightGBM multi-horizon + SHAP + ablation
├── dashboards/app.py              # 5-page Streamlit + Plotly dashboard
├── data/real/                     # Downloaded real data (parquet files)
├── tests/test_pipeline.py         # Unit tests
├── Dockerfile, docker-compose.yml
└── run.py                         # One-command data fetching
```

## Key Findings from Real Data

- **68,408 annual overdose deaths** (as of Oct 2025), down 15.4% from peak
- **Synthetic opioids (Fentanyl)** drive the crisis — surpassed heroin in 2016
- **Socioeconomic vulnerability** strongly correlates with overdose rates (0.0–0.93 range)
- **10 states** account for >60% of total deaths (CA, FL, OH, PA, NY, TX, NC, IL, TN, NJ)

## Technologies
Streamlit, Plotly, Pandas, NumPy, CDC VSRR API, Census ACS API, Docker
