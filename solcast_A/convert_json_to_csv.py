import json
import pandas as pd
import os

# Path to the forecast file
input_path = "data/forecast.json"
output_path = "data/solcast_forecast_input.csv"

# Load the JSON content
with open(input_path, "r") as f:
    data = json.load(f)

# Extract hourly data
hourly = data.get("hourly", {})

# Required fields
required_fields = ["time", "temperature_2m", "direct_radiation", "cloudcover"]
missing_fields = [field for field in required_fields if field not in hourly]

if missing_fields:
    raise ValueError(f"Missing fields in JSON: {missing_fields}")

# Build DataFrame
df = pd.DataFrame({
    "timestamp": hourly["time"],
    "temperature_2m": hourly["temperature_2m"],
    "direct_radiation": hourly["direct_radiation"],
    "cloudcover": hourly["cloudcover"]
})

# Save as CSV
df.to_csv(output_path, index=False)
print(f"✅ CSV saved to {output_path}")
