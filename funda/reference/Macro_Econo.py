import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from dash_bootstrap_components import themes
import dash_bootstrap_components as dbc
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Simulated 5-year data (2020–2025, monthly, 65 data points)
dates = pd.date_range(start='2020-01-01', end='2025-05-31', freq='ME').to_list()
data = {
    'Date': dates,
    'S&P 500': [
        3230, 2950, 2237, 2900, 3380, 3756, 4290, 4500, 4766, 4200, 3839, 4300,  # 2020
        4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700,  # 2021
        5500, 5300, 5100, 4900, 4700, 4500, 4300, 4100, 3900, 3700, 3500, 3839,  # 2022
        4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000, 6200,  # 2023
        6400, 6600, 6800, 7000, 7200, 7400, 7600, 7800, 8000, 8200, 8400, 8600,  # 2024
        8800, 5569, 5892.58, 5800, 5900  # 2025 (Jan-May)
    ],
    '10-Year Treasury Yield': [
        1.8, 1.5, 0.5, 0.7, 0.9, 0.9, 1.3, 1.5, 1.5, 3.0, 3.8, 4.0,  # 2020
        1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,  # 2021
        2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8,  # 2022
        3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.0,  # 2023
        4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2,  # 2024
        5.3, 4.37, 4.54, 4.5, 4.6  # 2025
    ],
    'VIX': [
        15, 20, 82, 40, 25, 22, 18, 20, 17, 30, 36, 20,  # 2020
        19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,  # 2021
        10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 36,  # 2022
        34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12,  # 2023
        14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,  # 2024
        38, 23.13, 18.62, 20, 22  # 2025
    ],
    'Fear and Greed Index': [
        60, 40, 10, 30, 50, 70, 65, 70, 75, 30, 25, 50,  # 2020
        55, 60, 65, 70, 75, 80, 85, 80, 75, 70, 65, 60,  # 2021
        55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 25,  # 2022
        30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,  # 2023
        80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25,  # 2024
        20, 65, 70, 68, 66  # 2025
    ],
    'Consumer Confidence': [
        90, 85, 70, 75, 80, 85, 90, 95, 100, 90, 85, 90,  # 2020
        95, 100, 105, 110, 115, 120, 125, 120, 115, 110, 105, 100,  # 2021
        95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 85,  # 2022
        90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145,  # 2023
        140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85,  # 2024
        80, 93.9, 86.0, 87, 88  # 2025
    ],
    'GDP Growth': [
        2.1, -31.4, -9.1, 33.4, 4.1, 2.3, 2.0, 2.5, 2.7, 1.5, -1.6, 0.5,  # 2020
        2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,  # 2021
        7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, -1.6,  # 2022
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,  # 2023
        5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 2.4,  # 2024
        2.4, -0.3, -0.3, -0.3, -0.3  # 2025
    ],
    'Unemployment Rate': [
        3.5, 14.7, 13.3, 10.2, 8.4, 6.7, 5.9, 5.4, 4.8, 4.2, 4.5, 4.0,  # 2020
        3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8,  # 2021
        2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.5,  # 2022
        4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.8,  # 2023
        3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.0,  # 2024
        4.1, 4.2, 4.2, 4.3, 4.4  # 2025
    ]
}

# Sector performance for April 2025
sector_data = {
    'Sector': ['Information Technology', 'Financials', 'Consumer Discretionary', 'Industrials', 
               'Utilities', 'Healthcare', 'Communication Services', 'Consumer Staples', 
               'Materials', 'Real Estate', 'Energy'],
    'Return (%)': [1.69, -2.11, -0.10, 0.11, 0.06, -3.79, -1.05, 0.20, -2.43, -1.31, -13.86]
}

# Create DataFrames
df = pd.DataFrame(data)
sector_df = pd.DataFrame(sector_data)

# Machine Learning: Random Forest Classifier
labels = [
    1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1,  # 2020
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 2021
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 2022
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 2023
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 2024
    1, 0, 1, 0  # 2025 (Jan-Apr)
]
X = df[['S&P 500', '10-Year Treasury Yield', 'VIX', 'Fear and Greed Index', 
        'Consumer Confidence', 'GDP Growth', 'Unemployment Rate']].iloc[:-1]
y = labels
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Predict current state (May 2025)
current_data = df[['S&P 500', '10-Year Treasury Yield', 'VIX', 'Fear and Greed Index', 
                  'Consumer Confidence', 'GDP Growth', 'Unemployment Rate']].iloc[-1:]
prediction = clf.predict(current_data)[0]
state = {1: 'Bullish', -1: 'Bearish', 0: 'Recession Risk'}
current_state = state[prediction]

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
})

# Create interactive plots using Plotly with tooltips
sp500_fig = px.line(df, x='Date', y='S&P 500', title='S&P 500 Index (2020–2025)', markers=True,
                    hover_data={'S&P 500': ':.2f'})
sp500_fig.update_layout(yaxis_title='Index Value', xaxis_title='Date', template='plotly_white')

treasury_fig = px.line(df, x='Date', y='10-Year Treasury Yield', title='10-Year Treasury Yield (2020–2025)', 
                       markers=True, hover_data={'10-Year Treasury Yield': ':.2f%'})
treasury_fig.update_traces(line_color='orange')
treasury_fig.update_layout(yaxis_title='Yield (%)', xaxis_title='Date', template='plotly_white')

vix_fig = px.line(df, x='Date', y='VIX', title='VIX (2020–2025)', markers=True,
                  hover_data={'VIX': ':.2f'})
vix_fig.update_traces(line_color='green')
vix_fig.update_layout(yaxis_title='Index', xaxis_title='Date', template='plotly_white')

fear_greed_fig = px.line(df, x='Date', y='Fear and Greed Index', title='Fear and Greed Index (2020–2025)', 
                         markers=True, hover_data={'Fear and Greed Index': ':.0f'})
fear_greed_fig.update_traces(line_color='purple')
fear_greed_fig.update_layout(yaxis_title='Index', xaxis_title='Date', template='plotly_white')

sector_fig = px.bar(sector_df, x='Return (%)', y='Sector', title='Sector Performance (April 2025)', 
                    orientation='h', hover_data={'Return (%)': ':.2f%'})
sector_fig.update_layout(xaxis_title='Return (%)', yaxis_title='Sector', template='plotly_white')

importance_fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        title='ML Feature Importance', orientation='h',
                        hover_data={'Importance': ':.3f'})
importance_fig.update_layout(xaxis_title='Importance', yaxis_title='Feature', template='plotly_white')

# Custom CSS for styling
external_stylesheets = [themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout with sidebar
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("Macro Economic Dashboard", className="text-center text-dark mb-4")),
        dbc.Col(html.P("5-Year Analysis with Machine Learning Predictions (2020–2025)", 
                       className="text-center text-dark mb-4"), width=12)
    ]),

    # Main content and sidebar
    dbc.Row([
        # Sidebar for news
        dbc.Col([
            html.H3("Overlooked News", className="text-dark mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.H5("1. Tariff Pause Fragility", className="card-title text-dark"),
                    html.P("U.S.-China tariff de-escalation (May 2025) may not last beyond 90 days, risking volatility.", 
                           className="card-text text-dark")
                ])
            ], className="mb-2 shadow", style={'backgroundColor': '#f8f9fa'}),
            dbc.Card([
                dbc.CardBody([
                    html.H5("2. Consumer Sentiment", className="card-title text-dark"),
                    html.P("Expectations Index at 54.4 (lowest since 2011) signals recession risks, ignored by market rally.", 
                           className="card-text text-dark")
                ])
            ], className="mb-2 shadow", style={'backgroundColor': '#f8f9fa'}),
            dbc.Card([
                dbc.CardBody([
                    html.H5("3. Energy Weakness", className="card-title text-dark"),
                    html.P("Oil prices down 19% in Q1 2025, dragging Energy sector (-13.86% in April).", 
                           className="card-text text-dark")
                ])
            ], className="mb-2 shadow", style={'backgroundColor': '#f8f9fa'}),
            dbc.Card([
                dbc.CardBody([
                    html.H5("4. Treasury Volatility", className="card-title text-dark"),
                    html.P("10-Year yields swung from 3.87% to 4.56% in April, impacting loans and growth.", 
                           className="card-text text-dark")
                ])
            ], className="mb-2 shadow", style={'backgroundColor': '#f8f9fa'})
        ], width=3, className="p-3", style={'height': '100vh', 'overflow': 'auto', 'backgroundColor': '#e9ecef'}),

        # Main content
        dbc.Col([
            # ML Prediction
            html.H3(f"Market Prediction: {current_state}", className="text-center text-primary mb-4"),

            # Charts
            dbc.Row([
                dbc.Col(dcc.Graph(figure=sp500_fig), width=6),
                dbc.Col(dcc.Graph(figure=treasury_fig), width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=vix_fig), width=6),
                dbc.Col(dcc.Graph(figure=fear_greed_fig), width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=sector_fig), width=6),
                dbc.Col(dcc.Graph(figure=importance_fig), width=6)
            ], className="mb-4"),

            # Collapsible Additional Indicators
            dbc.Row([
                dbc.Col([
                    html.H3("Additional Economic Indicators", className="text-dark mb-3"),
                    dbc.Collapse([
                        html.P("Unemployment Rate (April 2025): 4.2%", className="text-dark"),
                        html.P("CPI YoY (April 2025): 2.3%", className="text-dark"),
                        html.P("GDP Growth (Q1 2025): -0.3%", className="text-dark"),
                        html.P("Consumer Confidence (April 2025): 86.0", className="text-dark")
                    ], id="collapse-indicators", is_open=True),
                    dbc.Button("Toggle Indicators", id="collapse-button", className="mb-3 btn-primary")
                ], width=12)
            ])
        ], width=9)
    ])
], fluid=True)

# Callback for collapsible indicators
@app.callback(
    Output("collapse-indicators", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse-indicators", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

# Optional: Fetch real-time data from APIs
# import requests
# url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates?filter=record_date:gte:2025-04-01&sort=-record_date"
# response = requests.get(url)
# treasury_data = response.json()

# FRED API (fred.stlouisfed.org)
# fred_api_key = "your_api_key_here"
# url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SP500&api_key={fred_api_key}&file_type=json&observation_start=2020-01-01"
# response = requests.get(url)
# sp500_data = response.json()