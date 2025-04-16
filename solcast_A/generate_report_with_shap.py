
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/merged_forecast_input.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 23)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 23)
df["day_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
df["day_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)
df["cloud_index"] = df["cloudcover"] / 100

df.dropna(inplace=True)

features = [
    "temperature_2m", "cloudcover", "humidity", "wind_speed",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "cloud_index"
]

lookback = 24
X, y = [], []
for i in range(len(df) - lookback):
    X.append(df[features].iloc[i:i + lookback].values)
    y.append(df["direct_radiation"].iloc[i + lookback])

X = np.array(X)
y = np.array(y)

# Scale target
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Load model
model = load_model("solcast_model.keras")

# Predict
y_pred_scaled = model.predict(X).flatten()
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
residuals = y - y_pred

# Metrics
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals**2))
mape = np.mean(np.abs(residuals / np.maximum(y, 1e-2))) * 100
r2 = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R²": r2}
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# SHAP
explainer = shap.DeepExplainer(model, X[:100])
shap_values = explainer.shap_values(X[:100])[0]

shap.summary_plot(shap_values, features=np.array(features), show=False)
plt.tight_layout()
plt.savefig("outputs/feature_importance_shap.png", dpi=300)
plt.close()

# Forecast vs Actual
plt.figure(figsize=(10, 4))
plt.plot(y[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.fill_between(range(100), y_pred[:100] - 1.96*np.std(residuals), y_pred[:100] + 1.96*np.std(residuals),
                 alpha=0.2, label="95% CI")
plt.title("Forecast vs Actual (First 100 samples)")
plt.xlabel("Sample")
plt.ylabel("Direct Radiation (W/m²)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/forecast_vs_actual_ci.png", dpi=300)
plt.close()

# Residuals Plot
plt.figure(figsize=(10, 4))
plt.plot(residuals[:100], color="purple")
plt.axhline(0, linestyle="--", color="black")
plt.title("Forecast Residuals (First 100 samples)")
plt.xlabel("Sample")
plt.ylabel("Error (W/m²)")
plt.tight_layout()
plt.savefig("outputs/residuals_plot.png", dpi=300)
plt.close()

# HTML report
html = f"""
<html><head><title>Solar Forecast Report</title></head><body>
<h1>Solar Radiation Forecast Report</h1>
<h2>Model Performance</h2>
<ul>
  <li><b>MAE:</b> {mae:.2f} W/m²</li>
  <li><b>RMSE:</b> {rmse:.2f} W/m²</li>
  <li><b>MAPE:</b> {mape:.2f}%</li>
  <li><b>R²:</b> {r2:.3f}</li>
</ul>
<h2>Visualizations</h2>
<h3>Forecast vs Actual + Confidence Interval</h3>
<img src="forecast_vs_actual_ci.png" width="700">
<h3>Residuals</h3>
<img src="residuals_plot.png" width="700">
<h3>SHAP Feature Importance</h3>
<img src="feature_importance_shap.png" width="700">
</body></html>
"""

with open("outputs/final_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Report and SHAP visualizations saved to outputs/")
