
import os
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("☀️ Solar Radiation Forecast Dashboard")
st.markdown("Runs a Bi-LSTM model on merged Solcast + Open-Meteo data for forecasting.")

# Use merged forecast as default
forecast_path = "data/merged_forecast_input.csv"
if not os.path.exists(forecast_path):
    st.error("❌ Merged forecast CSV not found at data/merged_forecast_input.csv")
    st.stop()

use_ci = st.sidebar.checkbox("Show 95% Confidence Interval", value=True)

# Load and parse
df = pd.read_csv(forecast_path, parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# Fill missing features if needed
required = ['temperature_2m', 'direct_radiation', 'cloudcover', 'humidity', 'wind_speed']
defaults = {'temperature_2m': 30, 'direct_radiation': 600, 'cloudcover': 40, 'humidity': 25, 'wind_speed': 3}

for col in required:
    if col not in df.columns:
        df[col] = defaults[col]
        st.warning(f"⚠️ Column '{col}' missing — filled with default {defaults[col]}.")

# Feature engineering
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 23)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 23)
df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
if 'cloud_index' not in df.columns:
    df['cloud_index'] = df['cloudcover'] / 100

df.dropna(inplace=True)
st.success(f"✅ Loaded data with shape: {df.shape}")

if df.shape[0] < 100:
    st.error("❌ Not enough rows to forecast. Extend your data.")
    st.stop()

features = [
    'temperature_2m', 'cloudcover', 'humidity', 'wind_speed',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'cloud_index'
]

lookback = 24
X, y = [], []
for i in range(len(df) - lookback):
    X.append(df[features].iloc[i:i + lookback].values)
    y.append(df['direct_radiation'].iloc[i + lookback])

X = np.array(X)
y = np.array(y)

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

model_path = "solcast_model.keras"
if not os.path.exists(model_path):
    st.error("❌ Trained model not found. Ensure solcast_model.keras is in the root directory.")
    st.stop()

model = tf.keras.models.load_model(model_path)

pred_scaled = model.predict(X).flatten()
y_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
y_true = y[:len(y_pred)]

non_zero = y_true != 0
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
r2 = r2_score(y_true, y_pred)

metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}
st.subheader("📊 Forecast Metrics")
st.json(metrics)

residuals = y_true - y_pred
std_resid = np.std(residuals)
ci_upper = y_pred + 1.96 * std_resid
ci_lower = y_pred - 1.96 * std_resid

fig, ax = plt.subplots()
ax.plot(y_pred, label='Prediction', color='steelblue')
if use_ci:
    ax.fill_between(range(len(y_pred)), ci_lower, ci_upper, alpha=0.3, color='gray', label='95% CI')
ax.set_title("Direct Radiation Forecast with Confidence Interval")
ax.set_xlabel("Hour")
ax.set_ylabel("W/m²")
ax.legend()
st.pyplot(fig)

os.makedirs("outputs", exist_ok=True)
pd.DataFrame({
    "timestamp": df.index[lookback:lookback + len(y_pred)],
    "y_true": y_true,
    "y_pred": y_pred
}).to_csv("outputs/predictions.csv", index=False)
pd.DataFrame([metrics]).to_csv("outputs/forecast_metrics.csv", index=False)
fig.savefig("outputs/prediction_ci_plot.png", dpi=300)

st.info("📁 Forecast results and plots saved to `outputs/` folder.")
