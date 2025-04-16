
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import base64
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")
st.title("☀️ Solar Radiation Forecast Dashboard")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["📊 Forecasting", "📈 Report"])

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

if section == "📊 Forecasting":
    st.header("📊 Forecast Model Inference")
    forecast_path = "data/merged_forecast_input.csv"

    if not os.path.exists(forecast_path):
        st.error(f"❌ Forecast input not found at {forecast_path}")
        st.stop()

    df = pd.read_csv(forecast_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 23)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 23)
    df["day_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df["day_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df["cloud_index"] = df.get("cloudcover", 0) / 100

    # Fill missing expected features
    for col, default in [("humidity", 30.0), ("wind_speed", 2.0)]:
        if col not in df.columns:
            st.warning(f"⚠️ Missing feature '{col}' filled with {default}.")
            df[col] = default

    features = [
        "temperature_2m", "cloudcover", "humidity", "wind_speed",
        "hour_sin", "hour_cos", "day_sin", "day_cos", "cloud_index"
    ]

    df.dropna(subset=features + ["direct_radiation"], inplace=True)

    st.success(f"✅ Loaded forecast data: {df.shape}")

    lookback = 24
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df[features].iloc[i:i + lookback].values)
        y.append(df["direct_radiation"].iloc[i + lookback])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        st.error("❌ Not enough data to forecast. Please provide a longer time series.")
        st.stop()

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    model = load_model("solcast_model.keras")
    preds_scaled = model.predict(X).flatten()
    y_pred = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    residuals = y - y_pred

    metrics = calculate_metrics(y, y_pred)

    st.subheader("📌 Forecasting Metrics Summary")
    st.json(metrics)

    df_metrics = pd.DataFrame([metrics])
    st.dataframe(df_metrics)

    fig, ax = plt.subplots()
    ax.plot(y[:100], label="Actual")
    ax.plot(y_pred[:100], label="Prediction")
    ci = 1.96 * np.std(residuals)
    ax.fill_between(range(100), y_pred[:100] - ci, y_pred[:100] + ci, alpha=0.2, label="95% CI")
    ax.set_title("Forecast vs Actual (First 100 Samples)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Direct Radiation (W/m²)")
    ax.legend()
    st.pyplot(fig)

elif section == "📈 Report":
    st.header("📈 Performance Report")

    metrics_path = "outputs/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.subheader("Model Performance Metrics")
        st.metric("MAE (W/m²)", f"{metrics['MAE']:.2f}")
        st.metric("RMSE (W/m²)", f"{metrics['RMSE']:.2f}")
        st.metric("MAPE (%)", f"{metrics['MAPE']:.2f}")
        st.metric("R²", f"{metrics['R²']:.3f}")
    else:
        st.warning("⚠️ No metrics file found at outputs/metrics.json")

    for label, filename in [
        ("Forecast vs Actual (95% CI)", "forecast_vs_actual_ci.png"),
        ("Residuals Plot", "residuals_plot.png"),
        ("SHAP Feature Importance", "feature_importance_shap.png")
    ]:
        path = os.path.join("outputs", filename)
        if os.path.exists(path):
            st.subheader(label)
            st.image(path, use_column_width=True)
        else:
            st.warning(f"❌ Missing plot: {filename}")

    html_path = "outputs/final_report.html"
    if os.path.exists(html_path):
        st.subheader("📄 Full HTML Report")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
            st.components.v1.html(html, height=800, scrolling=True)

        with open(html_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:file/html;base64,{b64}" download="final_report.html">📥 Download Full Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("❌ final_report.html not found.")
