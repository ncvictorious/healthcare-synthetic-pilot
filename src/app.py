import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px
from pathlib import Path

# Load alerts
alerts_path = Path("reports/alerts.csv")
df = pd.read_csv(alerts_path)

# Initialise Dash app
app = dash.Dash(__name__)
app.title = "Healthcare Synthetic Pilot"

# Priorities order
priority_order = ["Red", "Amber", "Green"]

# App layout
app.layout = html.Div([
    html.H1("Healthcare Synthetic Pilot — Deterioration Alerts"),
    html.P("Synthetic dataset; priorities driven by RandomForest probability. Drivers from Logistic Regression."),

    # Tabs for each outcome
    dcc.Tabs(id="outcome-tabs", value="deterioration_72h", children=[
        dcc.Tab(label="Deterioration (72h)", value="deterioration_72h"),
        dcc.Tab(label="Falls (7d)", value="fall_within_7d"),
        dcc.Tab(label="Dehydration (48h)", value="dehydration_48h"),
    ]),

    # Filters
    html.Div([
        html.Label("Filter by Priority:"),
        dcc.Dropdown(
            id="priority-filter",
            options=[{"label": p, "value": p} for p in priority_order],
            value=[],
            multi=True,
            placeholder="Select priority (optional)"
        ),

        html.Label("Filter by Risk Score Range:"),
        dcc.RangeSlider(
            id="risk-slider",
            min=0, max=1, step=0.05,
            marks={i/10: str(i/10) for i in range(0, 11)},
            value=[0, 1]
        ),
    ], style={"margin": "20px 0"}),

    # DataTable
    dash_table.DataTable(
        id="alerts-table",
        columns=[{"name": col, "id": col} for col in df.columns],
        page_size=10,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_data_conditional=[
            {"if": {"filter_query": "{priority} = Red"}, "backgroundColor": "#FF9999"},
            {"if": {"filter_query": "{priority} = Amber"}, "backgroundColor": "#FFE699"},
            {"if": {"filter_query": "{priority} = Green"}, "backgroundColor": "#C6EFCE"},
        ],
    ),

    # Risk distribution chart
    dcc.Graph(id="risk-chart")
])

# Callbacks
@app.callback(
    [Output("alerts-table", "data"),
     Output("risk-chart", "figure")],
    [Input("outcome-tabs", "value"),
     Input("priority-filter", "value"),
     Input("risk-slider", "value")]
)
def update_dashboard(selected_outcome, selected_priorities, risk_range):
    # Filter dataframe
    dff = df[df["target"] == selected_outcome]

    if selected_priorities:
        dff = dff[dff["priority"].isin(selected_priorities)]

    dff = dff[(dff["risk_score"] >= risk_range[0]) & (dff["risk_score"] <= risk_range[1])]

    # Create risk distribution chart
    fig = px.histogram(
        dff, x="risk_score", color="priority",
        category_orders={"priority": priority_order},
        nbins=20, barmode="overlay",
        title=f"Risk Score Distribution for {selected_outcome}"
    )

    return dff.to_dict("records"), fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
