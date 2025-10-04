import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Paths
data_path = Path("data") / "synthetic_dataset.csv"
reports = Path("reports")
artifacts = Path("artifacts")
reports.mkdir(exist_ok=True)
artifacts.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Outcomes
outcomes = ["deterioration_72h", "fall_within_7d", "dehydration_48h"]

metrics = {}

for outcome in outcomes:
    print(f"🔹 Training {outcome}...")

    X = df.drop(columns=outcomes)
    y = df[outcome]

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X, y)
    rf_probs = rf_model.predict_proba(X)[:, 1]
    rf_preds = rf_model.predict(X)

    rf_metrics = {
        "auroc": roc_auc_score(y, rf_probs),
        "auprc": average_precision_score(y, rf_probs),
        "precision": precision_score(y, rf_preds, zero_division=0),
        "recall": recall_score(y, rf_preds, zero_division=0),
        "f1": f1_score(y, rf_preds, zero_division=0),
        "accuracy": accuracy_score(y, rf_preds),
    }

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=2000)
    log_reg.fit(X, y)
    lr_probs = log_reg.predict_proba(X)[:, 1]
    lr_preds = log_reg.predict(X)

    lr_metrics = {
        "auroc": roc_auc_score(y, lr_probs),
        "auprc": average_precision_score(y, lr_probs),
        "precision": precision_score(y, lr_preds, zero_division=0),
        "recall": recall_score(y, lr_preds, zero_division=0),
        "f1": f1_score(y, lr_preds, zero_division=0),
        "accuracy": accuracy_score(y, lr_preds),
    }

    # Save models
    joblib.dump(rf_model, artifacts / f"model_{outcome}.pkl")
    joblib.dump(log_reg, artifacts / f"logreg_{outcome}.pkl")

    # Save feature importances (RF)
    rf_importances = dict(zip(X.columns, rf_model.feature_importances_))

    metrics[outcome] = {
        "random_forest": rf_metrics,
        "logistic_regression": lr_metrics,
        "feature_importances_rf": rf_importances,
    }

with open(reports / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Training complete. Models + metrics saved.")
