
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

# Load cleaned input
df = pd.read_csv("data/merged_forecast_input.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# Ensure all expected columns exist
required = ['temperature_2m', 'direct_radiation', 'cloudcover', 'humidity', 'wind_speed']
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Add cyclical time features
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 23)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 23)
df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
df['cloud_index'] = df['cloudcover'] / 100

df.dropna(inplace=True)

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

# Normalize target
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Load original model
model = tf.keras.models.load_model("solcast_model.keras")

# Fine-tune settings
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.Huber())
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# Fit on new data (no validation split to maximize learning from limited input)
model.fit(X, y_scaled, epochs=5, batch_size=32, callbacks=[early_stop])

# Save fine-tuned model
model.save("solcast_model_finetuned.keras")
print("✅ Fine-tuned model saved to solcast_model_finetuned.keras")
