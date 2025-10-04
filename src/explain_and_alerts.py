import pandas as pd
import joblib
from pathlib import Path
import numpy as np

# Paths
reports = Path("reports")
artifacts = Path("artifacts")
data_path = Path("data") / "synthetic_dataset.csv"
reports.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Outcomes
outcomes = ["deterioration_72h", "fall_within_7d", "dehydration_48h"]

def top_drivers_logreg(model, X, n=3):
    """Return top drivers with coefficient impact for each row."""
    contributions = []
    for i in range(X.shape[0]):
        row = X.iloc[i].values
        imp = row * model.coef_[0]
        top_idx = np.argsort(np.abs(imp))[-n:][::-1]
        top_feats = [
            f"{X.columns[j]} ({imp[j]:+.3f})" for j in top_idx
        ]
        contributions.append(top_feats)
    return contributions

alerts_all = []

for outcome in outcomes:
    print(f"🔎 Explaining {outcome}...")

    X = df.drop(columns=outcomes)
    rf_model = joblib.load(artifacts / f"model_{outcome}.pkl")
    log_model = joblib.load(artifacts / f"logreg_{outcome}.pkl")

    probs = rf_model.predict_proba(X)[:, 1]

    drivers = top_drivers_logreg(log_model, X, n=3)

    alerts = pd.DataFrame({
        "target": outcome,
        "risk_score": probs,
        "priority": pd.cut(
            probs, bins=[-0.01, 0.6, 0.85, 1.0], labels=["Green", "Amber", "Red"]
        ),
        "drv_1": [d[0] for d in drivers],
        "drv_2": [d[1] for d in drivers],
        "drv_3": [d[2] for d in drivers],
    })

    alerts_all.append(alerts)

alerts_df = pd.concat(alerts_all, ignore_index=True)
alerts_df.to_csv(reports / "alerts.csv", index=False)

print("✅ Alerts generated at reports/alerts.csv")
