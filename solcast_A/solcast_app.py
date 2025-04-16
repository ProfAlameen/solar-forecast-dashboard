import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("☀️ Solar Radiation Forecast Dashboard")
st.write("Train or forecast using Bi-LSTM on solar radiation data.")

# Sidebar options
use_solcast_input = st.sidebar.checkbox("🔁 Use Solcast Forecast Input", value=True)
show_ci = st.sidebar.checkbox("📊 Show 95% Confidence Interval", value=True)

def prepare_data(use_solcast=False):
    forecast_path = "data/solcast_forecast_input.csv"
    merged_path = "data/merged_forecast_input.csv"

    if use_solcast:
        if os.path.exists(merged_path):
            st.info("🔄 Using merged forecast input with Open-Meteo + Solcast.")
            forecast_path = merged_path
        elif os.path.exists(forecast_path):
            st.warning("⚠️ Merged file not found. Falling back to Solcast forecast input only.")
        else:
            st.error("❌ Forecast file not found in `data/` directory.")
            st.stop()

    try:
        df = pd.read_csv(forecast_path)
    except Exception as e:
        st.error(f"❌ Failed to load forecast data from {forecast_path}: {e}")
        st.stop()

    st.success(f"✅ Loaded forecast data from: `{forecast_path}`")
    df['time'] = pd.to_datetime(df['timestamp'])
    df.set_index('time', inplace=True)
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)

    return df

def calculate_metrics(true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)
    non_zero = true != 0
    return {
        'MAE': mean_absolute_error(true, pred),
        'RMSE': np.sqrt(mean_squared_error(true, pred)),
        'MAPE': np.mean(np.abs((true[non_zero] - pred[non_zero]) / true[non_zero])) * 100 if np.any(non_zero) else np.nan,
        'R²': r2_score(true, pred)
    }

def main():
    df = prepare_data(use_solcast=use_solcast_input)

    # Feature engineering
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 23)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 23)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    for col in ['humidity', 'wind_speed']:
        if col not in df.columns:
            df[col] = 0.0
            st.warning(f"⚠️ Missing feature '{col}' was filled with zeros.")
    if 'cloud_index' not in df.columns:
        df['cloud_index'] = df.get('cloudcover', 0.5)

    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)
    st.info(f"Rows before NaN removal: {rows_before}")
    st.success(f"Rows after NaN removal: {rows_after}")

    lookback = 48
    forecast_horizon = 24
    features = ['temperature_2m', 'cloudcover', 'humidity', 'wind_speed',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'cloud_index']

    X, y = [], []
    for i in range(len(df) - lookback - forecast_horizon):
        X.append(df[features].iloc[i:i+lookback].values)
        y.append(df['direct_radiation'].iloc[i+lookback:i+lookback+forecast_horizon].mean())

    X = np.array(X)
    y = np.array(y)

    if len(y) == 0:
        st.error("❌ Not enough data after preprocessing to perform forecasting.")
        st.stop()

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    try:
        model = tf.keras.models.load_model("solcast_model.keras")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

    pred_scaled = model.predict(X).flatten()
    if not hasattr(y_scaler, 'min_'):
        y_scaler.fit(y.reshape(-1, 1))

    y_pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_true = y

    metrics = calculate_metrics(y_true, y_pred)
    st.success("✅ Forecast completed using saved model.")
    st.json(metrics)

    std_residual = np.std(y_true - y_pred) if len(y_true) == len(y_pred) else np.std(y_pred - np.mean(y_pred))
    ci_upper = y_pred + 1.96 * std_residual
    ci_lower = y_pred - 1.96 * std_residual

    fig, ax = plt.subplots()
    ax.plot(y_pred, label='Prediction', color='deepskyblue')
    if show_ci:
        ax.fill_between(range(len(y_pred)), ci_lower, ci_upper, alpha=0.3, color='gray', label='95% CI')
    ax.set_title("📉 Running Inference on Forecast")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Direct Radiation (W/m²)")
    ax.legend()
    st.pyplot(fig)

    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/prediction_plot_ci.png", dpi=300)
    pd.DataFrame({'prediction': y_pred}).to_csv("outputs/predictions.csv", index=False)

    st.subheader("📈 Forecasting Metrics Summary")
    metrics_df = pd.DataFrame([metrics])
    st.dataframe(metrics_df)
    metrics_df.to_csv("outputs/metrics_summary.csv", index=False)

if __name__ == "__main__":
    main()
