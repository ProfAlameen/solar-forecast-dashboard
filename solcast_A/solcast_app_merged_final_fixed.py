
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
st.markdown("Train or forecast using Bi-LSTM on solar radiation data.")

# Load merged data
forecast_path = "data/merged_forecast_input.csv"
if not os.path.exists(forecast_path):
    st.warning("⚠️ Merged file not found. Falling back to Solcast forecast input only.")
    forecast_path = "data/solcast_forecast_input.csv"

# Sidebar options
use_ci = st.sidebar.checkbox("📉 Show 95% Confidence Interval", value=True)

# Load Data
try:
    df = pd.read_csv(forecast_path, parse_dates=["timestamp"])
    st.success(f"✅ Loaded forecast data from: {forecast_path}")
except Exception as e:
    st.error(f"❌ Failed to load forecast data: {e}")
    st.stop()

df.set_index("timestamp", inplace=True)

# Fill required features if missing
required_features = ['temperature_2m', 'direct_radiation', 'cloudcover', 'humidity', 'wind_speed']
for col in required_features:
    if col not in df.columns:
        default_val = 30.0 if col == "humidity" else 2.0 if col == "wind_speed" else 0.0
        df[col] = default_val
        st.warning(f"⚠️ Missing feature '{col}' was filled with {default_val}.")

# Feature Engineering
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 23)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 23)
df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
df['cloud_index'] = df['cloudcover'] / 100.0
df.dropna(inplace=True)

st.info(f"📊 Rows after NaN removal: {len(df)}")

if len(df) < 100:
    st.error("❌ Not enough data to forecast. Please provide a longer time series.")
    st.stop()

# Prepare data
features = ['temperature_2m', 'cloudcover', 'humidity', 'wind_speed',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'cloud_index']
X = []
lookback = 24
for i in range(len(df) - lookback):
    X.append(df[features].iloc[i:i + lookback].values)

X = np.array(X)

# Simulate targets
y = df['direct_radiation'].iloc[lookback:].values[:len(X)]

# Scale target
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Load model
model_path = "solcast_model.keras"
if not os.path.exists(model_path):
    st.error("❌ Pre-trained model not found.")
    st.stop()

model = tf.keras.models.load_model(model_path)

# Predict
preds_scaled = model.predict(X, verbose=0).flatten()
y_pred = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
y_true = y[:len(y_pred)]

# Compute metrics
non_zero = y_true != 0
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else np.nan
r2 = r2_score(y_true, y_pred)

metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}
st.success("✅ Forecast completed using saved model.")
st.json(metrics)

# Confidence interval
residuals = y_true - y_pred
std_residual = np.std(residuals) if len(residuals) > 1 else 0
ci_upper = y_pred + 1.96 * std_residual
ci_lower = y_pred - 1.96 * std_residual

# Plot forecast
fig, ax = plt.subplots()
ax.plot(y_pred, label='Prediction', color='cornflowerblue')
if use_ci:
    ax.fill_between(range(len(y_pred)), ci_lower, ci_upper, color='gray', alpha=0.4, label='95% CI')
ax.set_title("Forecast with Confidence Interval")
ax.set_xlabel("Time Step")
ax.set_ylabel("Direct Radiation (W/m²)")
ax.legend()
st.pyplot(fig)
fig.savefig("outputs/prediction_ci_plot.png", dpi=300)

# Forecasting metrics table
st.subheader("📈 Forecasting Metrics Summary")
df_metrics = pd.DataFrame([metrics])
st.dataframe(df_metrics)

# Export
df_metrics.to_csv("outputs/forecast_metrics.csv", index=False)
pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv("outputs/predictions.csv", index=False)

st.info("📁 Forecast data and plots saved to `outputs/` folder.")
