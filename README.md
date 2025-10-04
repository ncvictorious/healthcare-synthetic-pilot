# Healthcare Synthetic Pilot (Stage 0)

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Dependencies](https://img.shields.io/badge/dependencies-scikit--learn%2C%20pandas%2C%20dash%2C%20plotly-green)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project demonstrates the Stage 0 **Synthetic Data Pilot** for a Healthcare risk-alert system.  
It generates synthetic data, trains baseline machine learning models, produces alerts, and builds dashboards (both static and interactive).  

---

## 📂 Project Structure

- **data/synthetic_dataset.csv** — synthetic patient records
- **src/** — scripts for data generation, training, explaining, and dashboards
- **artifacts/** — saved models (`.pkl`) and training outputs
- **reports/** — metrics, alerts, and dashboards
- **notebooks/** — optional end-to-end notebook

---

## 🚀 Quick Start

### 1. Create a Python environment
- **Option A (Conda)**
  ```bash
  conda create -n health_pilot python=3.10 -y
  conda activate health_pilot
Option B (venv)

bash
Copy code
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Run the pipeline (scripts)
From the project root:

bash
Copy code
python src/generate_synthetic_data.py
python src/train_baseline.py
python src/explain_and_alerts.py
python src/build_static_dashboard.py
4. Explore the outputs
reports/metrics.json — model evaluation (AUROC, AUPRC, precision, recall, F1)

reports/alerts.csv — combined alerts (deterioration, falls, dehydration)

reports/dashboard.html — static dashboard view

src/app.py — interactive Dash app (run python src/app.py and open http://127.0.0.1:8050)

📊 Sample Dashboard

(Replace with your own screenshot from reports/dashboard.html or the running Dash app.)

🧠 Models
RandomForest → risk scoring (probabilities)

Logistic Regression → explainability (top feature drivers)

Alerts are prioritized by thresholds:

Red ≥ 0.85

Amber ≥ 0.60

Green otherwise

📝 Notes
Data is synthetic, for demonstration purposes only.

Thresholds can be tuned in src/explain_and_alerts.py.

Dashboard supports basic filtering and sorting.

📜 License
This project is licensed under the MIT License — see the LICENSE file for details.