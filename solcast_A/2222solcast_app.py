
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

# Disable GPU usage and suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# UI Setup
st.set_page_config(layout="wide")
st.title("☀️ Solar Radiation Forecast Dashboard")
st.write("Train or forecast using Bi-LSTM on solar radiation data.")

# Sidebar toggle for using forecast input and CI
use_forecast_input = st.sidebar.checkbox("🔘 Use Solcast Forecast Input", value=True)
show_ci = st.sidebar.checkbox("📉 Show 95% Confidence Interval", value=True)

# Load data
if use_forecast_input:
    data_path = "data/solcast_forecast_input.csv"
else:
    data_path = "data/riyadh_weather_timeseries.csv"

try:
    df = pd.read_csv(data_path)
    st.info(f"Loaded {'Solcast' if use_forecast_input else 'historical'} input.")
except Exception as e:
    st.error(f"❌ Failed to load input CSV: {e}")
    st.stop()

# Preprocessing
df = df.dropna()
st.success(f"✅ Rows after NaN removal: {len(df)}")

if len(df) < 48:
    st.warning("⚠️ Not enough data to form sequences. Need at least 48 rows.")
    st.stop()

# Feature engineering
try:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 23)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 23)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofyear / 365)
except Exception as e:
    st.error(f"Timestamp or feature error: {e}")
    st.stop()

# Dummy solar position (optional)
df['solar_elevation'] = 45 + 10 * np.sin(2 * np.pi * df.index / len(df))
df['solar_zenith'] = 90 - df['solar_elevation']
df['cloud_index'] = 0.5

# Sequence building
lookback = 48
features = ['temperature_2m', 'cloudcover',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'solar_elevation', 'solar_zenith', 'cloud_index']

X = []
for i in range(len(df) - lookback):
    X.append(df[features].iloc[i:i+lookback].values)

X = np.array(X)

if X.shape[0] == 0:
    st.error("❌ Not enough valid sequences for prediction.")
    st.stop()

# Load model and predict
try:
    model = load_model("solcast_model.keras", compile=False)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

try:
    y_pred_ci = model.predict(X).flatten()
    st.success("✅ Forecast completed using saved model.")
except Exception as e:
    st.error(f"Error during forecast: {e}")
    st.stop()

# Confidence Interval Calculation
try:
    std_residual = np.std(y_pred_ci - np.mean(y_pred_ci))
    ci_upper = y_pred_ci + 1.96 * std_residual
    ci_lower = y_pred_ci - 1.96 * std_residual
except Exception as e:
    st.warning(f"⚠️ CI computation issue: {e}")
    ci_upper = y_pred_ci
    ci_lower = y_pred_ci

# Plotting
fig, ax = plt.subplots()
x_range = np.arange(len(y_pred_ci))

if show_ci and len(y_pred_ci) > 0:
    ax.plot(x_range, y_pred_ci, label="Prediction", color="deepskyblue")
    ax.fill_between(x_range, ci_lower, ci_upper, color="lightgrey", alpha=0.5, label="95% CI")
else:
    ax.plot(x_range, y_pred_ci, label="Prediction", color="deepskyblue")

ax.set_title("📉 Running Inference on Solcast Forecast")
ax.set_xlabel("Time Step")
ax.set_ylabel("Direct Radiation (W/m²)")
ax.legend()
st.pyplot(fig)

# Save outputs
os.makedirs("outputs", exist_ok=True)
fig.savefig("outputs/solcast_forecast_plot.png", dpi=300)
np.savetxt("outputs/solcast_forecast_ci.csv", np.vstack((y_pred_ci, ci_lower, ci_upper)).T,
           delimiter=",", header="Prediction,CI_Lower,CI_Upper", comments="")
