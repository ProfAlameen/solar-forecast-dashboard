import os
import requests
import json

os.makedirs("data", exist_ok=True)

url = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude=24.7136&longitude=46.6753"
    "&hourly=temperature_2m,direct_radiation,cloudcover"
    "&timezone=Asia%2FRiyadh"
)

try:
    response = requests.get(url)
    if response.status_code == 200:
        with open("data/riyadh_open_meteo.json", "w") as f:
            json.dump(response.json(), f, indent=2)
        print("✅ Saved riyadh_open_meteo.json")
    else:
        print(f"❌ Failed to fetch data: {response.status_code}")
except Exception as e:
    print(f"❌ Error fetching Open-Meteo data: {str(e)}")
