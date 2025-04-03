# solar-forecast-dashboard
# Solar Radiation Forecasting Using Bi-LSTM with Target Normalization and Satellite Features

This repository contains the code, data pipeline, and model implementation for the manuscript:

**"A Bidirectional LSTM Approach with Target Normalization and Satellite Context for Solar Radiation Forecasting in Arid Environments"**

## Overview

This study presents a deep learning framework for short-term solar radiation forecasting in Riyadh, Saudi Arabia. The proposed method integrates:

- **Bidirectional LSTM (Bi-LSTM)** model
- **Target normalization**
- **Satellite-derived features** (e.g., surface radiation, solar zenith angle)

The forecasting system was implemented as an interactive dashboard with real-time data visualization.

## Contents

- `data/` – Processed meteorological and satellite input features (normalized)
- `model/` – Bi-LSTM model architecture and training script
- `dashboard/` – Streamlit application code
- `results/` – Evaluation metrics, residual plots, prediction graphs
- `utils/` – Data preprocessing and statistical testing utilities
- `requirements.txt` – Python dependencies
- `run_pipeline.py` – End-to-end pipeline execution

## Key Features

- 34.8% RMSE reduction compared to linear regression baseline
- Real-time forecasting with Open-Meteo API inputs
- Comprehensive **ablation studies** included
- Deployment-ready **Streamlit dashboard**

## Dependencies

```bash
python >= 3.8
tensorflow >= 2.8
streamlit
pandas
numpy
matplotlib
scikit-learn
statsmodels
