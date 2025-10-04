# Healthcare Synthetic Pilot (Stage 0)

This is a self-contained project to demonstrate the Stage 0 **Synthetic Data Pilot** for the Healthcare risk-alert system.

## What you get
- **data/synthetic_healthcare.csv** — 100 synthetic rows
- **src/** — scripts for preprocessing, training, explaining, and alert generation
- **reports/** — evaluation metrics and a static dashboard HTML
- **artifacts/** — saved models and generated alerts
- **notebooks/** — one end-to-end notebook (optional)

## Quick Start (recommended)
### 1) Create a Python environment
- **Option A (Conda)**  
  ```bash
  conda create -n health_pilot python=3.10 -y
  conda activate health_pilot
  ```
- **Option B (venv)**  
  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate
  # macOS/Linux:
  source .venv/bin/activate
  ```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run the pipeline (scripts)
From the project root:
```bash
python src/generate_synthetic_data.py
python src/train_baseline.py
python src/explain_and_alerts.py
python src/build_static_dashboard.py
```

### 4) Inspect outputs
- **reports/metrics.json** — model metrics (AUROC, AUPRC, precision, recall, F1)
- **artifacts/alerts_deterioration.json** — alert cards for deterioration (similarly for fall & dehydration)
- **reports/dashboard.html** — open in your browser for a static dashboard view

## End-to-end in a Notebook (optional)
Open `notebooks/stage0_healthcare_pipeline.ipynb` and run all cells.

## Notes
- Baseline models use scikit-learn (RandomForest + Logistic Regression for explainability).
- Alert thresholds: **Red ≥ 0.85**, **Amber ≥ 0.6**, else **Green**.
- You can tweak thresholds in `src/explain_and_alerts.py`.
