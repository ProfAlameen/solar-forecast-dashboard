
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pvlib

st.set_page_config(layout="wide")

def prepare_data(use_solcast=False):
    if use_solcast:
        df = pd.read_csv("data/solcast_forecast_input.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        st.info("Loaded Solcast forecast input.")
    else:
        df = pd.read_csv("data/riyadh_weather_timeseries.csv")
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        st.info("Loaded historical Riyadh weather timeseries.")

    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 23)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 23)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    if 'solar_elevation' not in df.columns:
        location = pvlib.location.Location(latitude=24.7136, longitude=46.6753)
        solar_position = location.get_solarposition(df.index)
        df['solar_elevation'] = solar_position['elevation']
        df['solar_zenith'] = solar_position['zenith']

    if 'cloud_index' not in df.columns:
        df['cloud_index'] = 0.5
    return df[df['solar_elevation'] > 0]

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

def main():
    st.title("☀️ Solar Radiation Forecast Dashboard")
    st.write("Train or forecast using Bi-LSTM on solar radiation data.")

    use_solcast_input = st.sidebar.checkbox("Use Solcast Forecast Input", value=False)
    df = prepare_data(use_solcast=use_solcast_input)

    st.warning(f"Rows before NaN removal: {len(df)}")
    df = df.dropna()
    st.success(f"Rows after NaN removal: {len(df)}")

    lookback = 48
    forecast_horizon = 24
    features = ['temperature_2m', 'cloudcover', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'solar_elevation', 'solar_zenith', 'cloud_index']
    available_features = [f for f in features if f in df.columns]

    X, y = [], []
    for i in range(len(df) - lookback - forecast_horizon):
        X.append(df[available_features].iloc[i:i+lookback].values)
        y.append(df['direct_radiation'].iloc[i+lookback:i+lookback+forecast_horizon].mean())

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0 or len(y) == 0:
        st.warning("Not enough data to run model. Try a longer input series.")
        return

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    input_shape = X.shape[1:]
    best_params = {"units": 128, "dropout": 0.3, "lr": 0.001}

    if use_solcast_input:
        st.subheader("📡 Running Inference on Solcast Forecast")
        try:
            model = tf.keras.models.load_model("solcast_model.keras")
            preds_scaled = model.predict(X).flatten()
            y_pred = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            st.line_chart(y_pred)
            st.success("✅ Forecast completed using saved model.")
        except Exception as e:
            st.error(f"⚠️ Failed to load model: {e}")
        return

    st.subheader("🧪 Training Mode")
    model = build_lstm_model(input_shape, **best_params)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint("solcast_model.keras", monitor='val_loss', save_best_only=True)

    history = model.fit(X, y_scaled, epochs=30, batch_size=32, verbose=0, validation_split=0.2,
                        callbacks=[early_stop, checkpoint])

    preds_scaled = model.predict(X).flatten()
    y_pred = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    metrics = calculate_metrics(y, y_pred)
    st.json(metrics)

if __name__ == "__main__":
    main()
