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
import pvlib

# Configuration
st.set_page_config(layout="wide")

if not os.path.exists("data/riyadh_open_meteo.json"):
    st.error("⚠️ Missing Open-Meteo data. Please run fetch_riyadh_open_meteo.py.")
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

def build_lstm_model(input_shape, units=64, dropout=0.2, lr=0.001):
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
    st.write("This dashboard uses LSTM models to predict solar radiation and compares with benchmark models.")

    df = prepare_data()
    st.success("✅ Data prepared. Shape: {}".format(df.shape))

    lookbacks = [48, 72, 96]  # Can add Streamlit selectbox here
    forecast_horizon = 24
    features = ['temperature_2m', 'cloudcover', 'humidity', 'wind_speed',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'solar_elevation', 'solar_zenith']

    for lookback in lookbacks:
        X, y = [], []
        for i in range(len(df) - lookback - forecast_horizon):
            X.append(df[features].iloc[i:i+lookback].values)
            y.append(df['direct_radiation'].iloc[i+lookback:i+lookback+forecast_horizon].mean())

        X = np.array(X)
        y = np.array(y)
        input_shape = X.shape[1:]

        st.subheader(f"🔁 Lookback: {lookback} hours")

        param_grid = [
            {"units": 128, "dropout": 0.3, "lr": 0.001},
        ]

        results = []
        for params in param_grid:
            st.info(f"Training with params: {params}")
            model = build_lstm_model(input_shape, **params)
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            checkpoint = ModelCheckpoint("solcast_model.keras", monitor='val_loss', save_best_only=True)
            history = model.fit(X, y, epochs=50, batch_size=32, verbose=0, validation_split=0.2,
                                callbacks=[early_stop, checkpoint])

            preds = model.predict(X)
            metrics = calculate_metrics(y, preds.flatten())
            results.append((params, metrics))
            st.write(f"✅ Done: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.3f}")

            # Plot training and validation loss
            fig, ax = plt.subplots()
            ax.plot(history.history['loss'], label='Training Loss')
            ax.plot(history.history['val_loss'], label='Validation Loss')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss Curve")
            ax.legend()
            st.pyplot(fig)

        st.success("🎯 Best model fine-tuned and saved!")
        st.write("### Summary of Final Run")
        for p, m in results:
            st.write(p, m)

if __name__ == "__main__":
    main()
