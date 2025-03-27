import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
import pvlib
from datetime import datetime

# Load the data
df = pd.read_csv("data/riyadh_weather_timeseries.csv")
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Add cyclical time features
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour/23)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour/23)
df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear/365)
df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear/365)

# Add solar geometry features
df = df.copy()
location = pvlib.location.Location(latitude=24.7136, longitude=46.6753)
solar_position = location.get_solarposition(df.index)
df['solar_elevation'] = solar_position['elevation']
df['solar_zenith'] = solar_position['zenith']

# Define features and target
features = ['temperature_2m', 'cloudcover', 'humidity', 'wind_speed',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'solar_elevation', 'solar_zenith']
target = 'direct_radiation'

# Scale the data
feature_scaler = RobustScaler()
target_scaler = RobustScaler()

scaled_features = feature_scaler.fit_transform(df[features])
scaled_target = target_scaler.fit_transform(df[[target]])

# Save scalers
joblib.dump({
    'feature': feature_scaler,
    'target': target_scaler,
    'features': features,
    'target_column': target,
    'last_training_date': df.index[-1]
}, 'scaler.pkl')

print("\u2705 Scaler successfully created with:")
print(f"- Features: {features}")
print(f"- Target: {target}")
print(f"- Last training date: {df.index[-1]}")
