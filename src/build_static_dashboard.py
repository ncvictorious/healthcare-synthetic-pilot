import pandas as pd
from pathlib import Path
import plotly.express as px

reports = Path("reports")
alerts_file = reports / "alerts.csv"

# Load alerts
alerts_df = pd.read_csv(alerts_file)

# Build HTML dashboard
html_path = reports / "dashboard.html"

sections = []
for outcome, group in alerts_df.groupby("target"):
    # Count by priority
    counts = group["priority"].value_counts().reindex(["Red", "Amber", "Green"], fill_value=0)

    # Chart
    fig = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Priority", "y": "Count"},
        title=f"Alert distribution for {outcome}",
        color=counts.index,
        color_discrete_map={"Red": "red", "Amber": "orange", "Green": "green"},
    )
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Table
    table_html = group.to_html(index=False)

    sections.append(f"<h2>{outcome}</h2>{chart_html}{table_html}")

# Write final HTML
with open(html_path, "w", encoding="utf-8") as f:
    f.write("<html><head><title>Healthcare Synthetic Pilot — Deterioration Alerts</title></head><body>")
    f.write("<h1>Healthcare Synthetic Pilot — Deterioration Alerts</h1>")
    f.write("<p>Synthetic dataset; priorities driven by RandomForest probability. Drivers from Logistic Regression.</p>")
    f.write("<hr/>".join(sections))
    f.write("<p><b>Legend:</b> Red ≥ 0.85; Amber ≥ 0.60; else Green.</p>")
    f.write("</body></html>")

print(f"✅ Dashboard with charts written to {html_path}")
