import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pvlib
import json

# Configuration
st.set_page_config(layout="wide")

if not os.path.exists("data/riyadh_open_meteo.json"):
    st.error("Missing Open-Meteo data. Please run fetch_riyadh_open_meteo.py.")
    st.stop()

def prepare_data():
    df = pd.read_csv("data/riyadh_weather_timeseries.csv")
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour/23)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour/23)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear/365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear/365)
    location = pvlib.location.Location(latitude=24.7136, longitude=46.6753)
    solar_position = location.get_solarposition(df.index)
    df['solar_elevation'] = solar_position['elevation']
    df['solar_zenith'] = solar_position['zenith']
    df = df[df['solar_elevation'] > 0]

    # Load satellite-derived placeholder features
    if os.path.exists("data/satellite_features.csv"):
        sat_df = pd.read_csv("data/satellite_features.csv")
        sat_df['time'] = pd.to_datetime(sat_df['time'])
        sat_df.set_index('time', inplace=True)
        df = df.join(sat_df, how='inner')
        st.info("Satellite-derived features loaded and merged.")
    else:
        df['cloud_index'] = 0.5  # Placeholder constant feature
        st.warning("Satellite-derived features not found. Using constant placeholder.")
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

def build_lstm_model(input_shape, units=128, dropout=0.3, lr=0.001):
    model = Sequential()
    model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units//2)))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
    return model

def generate_html_report(metrics, params):
    html = f"""
    <html><head><title>Solar Forecast Report</title></head><body>
    <h1>Solar Radiation Forecast Report</h1>
    <h2>Model Parameters</h2>
    <ul>
        <li>Units: {params['units']}</li>
        <li>Dropout: {params['dropout']}</li>
        <li>Learning Rate: {params['lr']}</li>
    </ul>
    <h2>Performance Metrics</h2>
    <ul>
        <li>MAE: {metrics['MAE']:.2f}</li>
        <li>RMSE: {metrics['RMSE']:.2f}</li>
        <li>MAPE: {metrics['MAPE']:.2f}</li>
        <li>R²: {metrics['R²']:.3f}</li>
    </ul>
    <h2>Figures</h2>
    <img src="loss_curve.png" width="600"/><br>
    <img src="prediction_vs_actual_ci.png" width="600"/><br>
    <img src="residuals_plot.png" width="600"/>
    </body></html>
    """
    with open("outputs/report.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    st.title("☀️ Solar Radiation Forecast Dashboard")
    st.write("This dashboard uses LSTM models to predict solar radiation and compares with benchmark models.")

    df = prepare_data()
    st.success("Data prepared. Shape: {}".format(df.shape))

    lookback = 48
    forecast_horizon = 24
    features = ['temperature_2m', 'cloudcover', 'humidity', 'wind_speed',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'solar_elevation', 'solar_zenith', 'cloud_index']

    X, y = [], []
    for i in range(len(df) - lookback - forecast_horizon):
        X.append(df[features].iloc[i:i+lookback].values)
        y.append(df['direct_radiation'].iloc[i+lookback:i+lookback+forecast_horizon].mean())

    X = np.array(X)
    y = np.array(y)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    input_shape = X.shape[1:]
    st.subheader(f"Lookback: {lookback} hours")

    best_params = {"units": 128, "dropout": 0.3, "lr": 0.001}
    st.info(f"Training with params: {best_params}")

    model = build_lstm_model(input_shape, **best_params)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("solcast_model.keras", monitor='val_loss', save_best_only=True)

    history = model.fit(X, y_scaled, epochs=50, batch_size=32, verbose=0, validation_split=0.2,
                        callbacks=[early_stop, checkpoint])

    preds_scaled = model.predict(X).flatten()
    y_true = y
    y_pred = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    lstm_metrics = calculate_metrics(y_true, y_pred)

    # Confidence Intervals (95%)
    residuals = y_true - y_pred
    std_residual = np.std(residuals)
    ci_upper = y_pred + 1.96 * std_residual
    ci_lower = y_pred - 1.96 * std_residual

    # Baseline: Persistence model
    persistence_pred = np.roll(y_true, 1)
    persistence_pred[0] = persistence_pred[1]  # fix first value
    persistence_metrics = calculate_metrics(y_true, persistence_pred)

    # Baseline: Linear Regression
    flat_X = X.reshape((X.shape[0], -1))
    linreg = LinearRegression().fit(flat_X, y_true)
    linreg_pred = linreg.predict(flat_X)
    linreg_metrics = calculate_metrics(y_true, linreg_pred)

    # Display metrics
    st.write("## 📊 Forecasting Error Comparison")
    df_metrics = pd.DataFrame({
        'Model': ['LSTM', 'Persistence', 'Linear Regression'],
        'MAE': [lstm_metrics['MAE'], persistence_metrics['MAE'], linreg_metrics['MAE']],
        'RMSE': [lstm_metrics['RMSE'], persistence_metrics['RMSE'], linreg_metrics['RMSE']],
        'MAPE': [lstm_metrics['MAPE'], persistence_metrics['MAPE'], linreg_metrics['MAPE']],
        'R²': [lstm_metrics['R²'], persistence_metrics['R²'], linreg_metrics['R²']]
    })
    st.dataframe(df_metrics.set_index("Model"))

    os.makedirs("outputs", exist_ok=True)
    df_metrics.to_csv("outputs/forecast_metrics.csv", index=False)

    # Loss plot
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curve")
    ax.legend()
    st.pyplot(fig)
    fig.savefig("outputs/loss_curve.png", dpi=300)

    # Actual vs LSTM prediction with confidence intervals
    fig2, ax2 = plt.subplots()
    last_n = min(100, len(y_pred))
    ax2.plot(y_true[-last_n:], label='Actual')
    ax2.plot(y_pred[-last_n:], label='LSTM Prediction')
    ax2.fill_between(range(last_n), ci_lower[-last_n:], ci_upper[-last_n:], color='gray', alpha=0.3, label='95% CI')
    ax2.set_title("Actual vs LSTM Prediction (Last Samples) with Confidence Interval")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Direct Radiation (W/m²)")
    ax2.legend()
    st.pyplot(fig2)
    fig2.savefig("outputs/prediction_vs_actual_ci.png", dpi=300)

    # Residuals Plot
    fig3, ax3 = plt.subplots()
    ax3.plot(residuals[-last_n:], label='Residuals', color='purple')
    ax3.axhline(0, color='black', linestyle='--')
    ax3.set_title("Residuals of LSTM Forecast (Last Samples)")
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Error (W/m²)")
    ax3.legend()
    st.pyplot(fig3)
    fig3.savefig("outputs/residuals_plot.png", dpi=300)

    # Pipeline Diagram (summary plot)
    st.write("### 🔁 Modeling Pipeline Summary")
    st.markdown("""
    - **Data Ingestion** → Riyadh weather time series
    - **Feature Engineering** → Time, trigonometric, solar, satellite (or placeholder)
    - **Sequence Windowing** → Lookback of 48 hours
    - **Target Normalization** → MinMaxScaler
    - **Model** → Bidirectional LSTM + Dropout + Dense
    - **Training** → Adam optimizer, Huber loss
    - **Evaluation** → MAE, RMSE, MAPE, R², Confidence Interval
    - **Baselines** → Persistence, Linear Regression
    """)

    with open("outputs/final_metrics.json", "w") as f:
        json.dump(lstm_metrics, f, indent=2)

    with open("outputs/final_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    generate_html_report(lstm_metrics, best_params)

    st.success("✅ Best model fine-tuned and saved!")
    st.write("### Summary of Final Run")
    st.json(best_params)
    st.json(lstm_metrics)

if __name__ == "__main__":
    main()
