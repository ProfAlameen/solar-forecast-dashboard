import pandas as pd
import numpy as np

def generate_cloud_index_csv(start="2025-03-15", end="2025-03-28"):
    rng = pd.date_range(start=start, end=end, freq="H")
    np.random.seed(42)
    cloud_index = np.clip(np.random.normal(loc=0.5, scale=0.2, size=len(rng)), 0, 1)
    df = pd.DataFrame({
        "time": rng,
        "cloud_index": cloud_index
    })
    df.to_csv("data/satellite_features.csv", index=False)
    print("✅ satellite_features.csv generated with shape:", df.shape)

if __name__ == "__main__":
    generate_cloud_index_csv()
