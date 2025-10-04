# Healthcare Synthetic Pilot (Stage 0)  
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
*A Stage 0 synthetic data pilot for a healthcare risk-alert system.*

---

## 📌 Overview
This project demonstrates a **Synthetic Data Pilot** for a healthcare risk-alert system.  
It:
- Generates synthetic patient data
- Trains baseline ML models
- Produces risk alerts
- Builds dashboards (static + interactive)

---

## 📂 Project Structure
- `data/synthetic_dataset.csv` → synthetic patient records  
- `src/` → scripts for data generation, training, explaining, and dashboards  
- `artifacts/` → saved models (.pkl) and training outputs  
- `reports/` → metrics, alerts, and dashboards  
- `notebooks/` → optional end-to-end Jupyter notebook  

---

## 🚀 Quick Start

### 1) Create a Python environment
**Option A: Conda**
```bash
conda create -n health_pilot python=3.10 -y
conda activate health_pilot


Option B: venv

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

2) Install dependencies
pip install -r requirements.txt

3) Run the pipeline

From the project root:

python src/generate_synthetic_data.py
python src/train_baseline.py
python src/explain_and_alerts.py
python src/build_static_dashboard.py

4) Explore outputs

reports/metrics.json → model metrics (AUROC, AUPRC, precision, recall, F1)

reports/alerts.csv → combined alerts (deterioration, falls, dehydration)

reports/dashboard.html → static dashboard (open in browser)

src/app.py → interactive Dash app (python src/app.py → http://127.0.0.1:8050
)

📊 Sample Dashboard

📊 Sample Dashboard

![Dashboard Screenshot](docs/dashboard_sample.png)


🧠 Models

RandomForest → risk scoring (probabilities)

Logistic Regression → explainability (top drivers)

Alert thresholds:

🔴 Red: ≥ 0.85

🟠 Amber: ≥ 0.60

🟢 Green: otherwise

📝 Notes

Data is synthetic, for demonstration only

Thresholds configurable in src/explain_and_alerts.py

Dashboard supports filtering & sorting

📜 License

This project is licensed under the MIT License — see the LICENSE
 file.