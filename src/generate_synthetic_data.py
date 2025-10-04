import pandas as pd
import numpy as np
from pathlib import Path

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    n = 100

    df = pd.DataFrame({
        "age": np.random.randint(60, 90, size=n),
        "hr_mean": np.random.normal(75, 10, size=n),
        "hrv_rmssd": np.random.normal(30, 8, size=n),
        "spo2_min": np.random.normal(94, 2, size=n),
        "steps_24h": np.random.randint(0, 10000, size=n),
        "sleep_efficiency": np.random.uniform(0.6, 0.95, size=n),
        "night_movements": np.random.randint(0, 20, size=n),
        "infection_flag": np.random.randint(0, 2, size=n),
        "med_change_flag": np.random.randint(0, 2, size=n),
        "fall_history": np.random.randint(0, 2, size=n),
        "hydration_level": np.random.uniform(0.4, 1.0, size=n),
        # 🔑 Three outcome columns instead of one
        "deterioration_72h": np.random.randint(0, 2, size=n),
        "fall_within_7d": np.random.randint(0, 2, size=n),
        "dehydration_48h": np.random.randint(0, 2, size=n),
    })

    out_path = data_dir / "synthetic_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote synthetic dataset to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    main()
