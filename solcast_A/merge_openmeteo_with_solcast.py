import pandas as pd
import json
import os

# File paths
solcast_path = "data/solcast_forecast_input.csv"
meteo_path = "data/riyadh_open_meteo.json"
merged_path = "data/merged_forecast_input.csv"

# Load Solcast CSV
solcast_df = pd.read_csv(solcast_path, parse_dates=["timestamp"])
solcast_df.columns = solcast_df.columns.str.strip().str.lower()

# Load Open-Meteo JSON
with open(meteo_path, "r") as f:
    data = json.load(f)

hourly = data.get("hourly", {})
if not hourly:
    raise ValueError("❌ No 'hourly' data in JSON!")

meteo_df = pd.DataFrame(hourly)
meteo_df["timestamp"] = pd.to_datetime(meteo_df.pop("time"))

# Rename existing fields
meteo_df.rename(columns={
    "temperature_2m": "temperature_2m",
    "cloudcover": "cloudcover",
    "direct_radiation": "direct_radiation"
}, inplace=True)

# Insert dummy fields if missing
if "relativehumidity_2m" not in hourly:
    meteo_df["humidity"] = 30.0  # assume moderate humidity
    print("⚠️ 'humidity' not in Open-Meteo → filled with 30.0")

if "windspeed_10m" not in hourly:
    meteo_df["wind_speed"] = 2.0  # default light breeze
    print("⚠️ 'wind_speed' not in Open-Meteo → filled with 2.0")

# Merge by timestamp
merged_df = pd.merge(solcast_df, meteo_df, on="timestamp", how="left")

# Fill any remaining NaNs
merged_df.fillna(method="ffill", inplace=True)
merged_df.fillna(method="bfill", inplace=True)

# Save merged forecast
os.makedirs("data", exist_ok=True)
merged_df.to_csv(merged_path, index=False)
print(f"✅ Merged forecast saved to: {merged_path}")
