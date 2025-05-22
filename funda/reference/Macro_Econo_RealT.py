import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from dash_bootstrap_components import themes
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import requests
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from dotenv import load_dotenv
import os
import yfinance as yf

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Validate API keys (Alpha Vantage key no longer needed)
if not FRED_API_KEY or not NEWS_API_KEY:
    raise ValueError("Missing API keys. Ensure FRED_API_KEY and NEWS_API_KEY are set in the .env file.")

# Initialize NewsAPI and Sentiment Analyzer
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

# Function to fetch real-time economic data from FRED
def fetch_economic_data():
    # Define the 5-year span (May 2020 to May 2025)
    end_date = datetime(2025, 5, 15)
    start_date = end_date - timedelta(days=5*365)
    
    # Create a unified date range (daily)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range_str = [d.strftime('%Y-%m-%d') for d in date_range]
    
    # Initialize data dictionary with the date range
    data = {'Date': date_range_str}
    
    # Series IDs
    series_ids = {
        'S&P 500': 'SP500',
        '10-Year Treasury Yield': 'DGS10',  # Using FRED's DGS10 for 10-Year Treasury Constant Maturity Rate
        'VIX': 'VIXCLS',
        'Consumer Confidence': 'UMCSENT',
        'GDP Growth': 'A191RL1Q225SBEA',  # Real GDP growth rate (percent change, quarterly)
        'Unemployment Rate': 'UNRATE'
    }
    
    # Fetch data for each series and map to the unified date range
    for indicator, series_id in series_ids.items():
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start_date.strftime('%Y-%m-%d')}&observation_end={end_date.strftime('%Y-%m-%d')}"
        try:
            response = requests.get(url).json()
            # Create a dictionary of date-to-value mappings
            value_dict = {obs['date']: float(obs['value']) if obs['value'] != '.' else np.nan for obs in response['observations']}
            # Map values to the unified date range
            data[indicator] = [value_dict.get(date, np.nan) for date in date_range_str]
        except Exception as e:
            print(f"Error fetching {indicator} data: {e}")
            data[indicator] = [np.nan] * len(date_range_str)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Fill missing values
    df = df.ffill().bfill()
    
    # Simulate Fear and Greed Index (based on VIX)
    df['Fear and Greed Index'] = 50 + (100 - df['VIX']) / 2
    
    return df

# Function to fetch sector performance using yfinance (Yahoo Finance)
def fetch_sector_performance():
    # Define sector ETFs and corresponding sector names
    sector_etfs = {
        'Information Technology': 'XLK',
        'Financials': 'XLF',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Utilities': 'XLU',
        'Healthcare': 'XLV',
        'Communication Services': 'XLC',
        'Consumer Staples': 'XLP',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Energy': 'XLE'
    }
    
    # Define date range for 1-month performance (last 30 days up to May 15, 2025)
    end_date = datetime(2025, 5, 15)
    start_date = end_date - timedelta(days=30)
    
    sectors = []
    returns = []
    
    for sector, ticker in sector_etfs.items():
        try:
            # Fetch historical data for the ETF
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty and len(hist) > 1:
                # Calculate 1-month return (from start to end of period)
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                return_pct = ((end_price - start_price) / start_price) * 100
                sectors.append(sector)
                returns.append(return_pct)
            else:
                print(f"No data available for {sector} ({ticker}), using fallback value.")
                sectors.append(sector)
                returns.append(0.0)
        except Exception as e:
            print(f"Error fetching data for {sector} ({ticker}): {e}")
            sectors.append(sector)
            returns.append(0.0)
    
    # Create DataFrame
    sector_df = pd.DataFrame({
        'Sector': sectors,
        'Return (%)': returns
    })
    
    # Fallback to hardcoded values if no data is fetched successfully
    if sector_df['Return (%)'].eq(0).all():
        print("No sector performance data fetched, using hardcoded fallback values.")
        return pd.DataFrame({
            'Sector': ['Information Technology', 'Financials', 'Consumer Discretionary', 'Industrials', 
                       'Utilities', 'Healthcare', 'Communication Services', 'Consumer Staples', 
                       'Materials', 'Real Estate', 'Energy'],
            'Return (%)': [1.69, -2.11, -0.10, 0.11, 0.06, -3.79, -1.05, 0.20, -2.43, -1.31, -13.86]
        })
    
    return sector_df

# Function to fetch daily news and compute sentiment
def fetch_news():
    try:
        articles = newsapi.get_everything(q='economy OR tariffs OR consumer sentiment OR treasury yields',
                                          language='en', sort_by='relevancy', page_size=4)
        news_items = []
        for article in articles['articles']:
            title = article['title']
            description = article['description'] or article['content'][:200] or "No description available."
            sentiment = analyzer.polarity_scores(description)['compound']
            news_items.append({
                'title': title,
                'description': description,
                'sentiment': sentiment
            })
        return news_items
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Fetch data and news
df = fetch_economic_data()
news_items = fetch_news()
sector_df = fetch_sector_performance()

# Add news sentiment as a feature
df['News Sentiment'] = np.mean([item['sentiment'] for item in news_items]) if news_items else 0

# Machine Learning: Random Forest Classifier
# Generate labels based on S&P 500 returns (Bullish if return > 0, Bearish if < 0, Recession Risk if GDP < 0)
labels = []
for i in range(1, len(df)):
    sp500_return = (df['S&P 500'].iloc[i] - df['S&P 500'].iloc[i-1]) / df['S&P 500'].iloc[i-1] if not np.isnan(df['S&P 500'].iloc[i-1]) else 0
    gdp = df['GDP Growth'].iloc[i]
    if not np.isnan(gdp) and gdp < 0:
        label = 0  # Recession Risk
    elif sp500_return > 0:
        label = 1  # Bullish
    else:
        label = -1  # Bearish
    labels.append(label)

if len(labels) > 0:
    X = df[['S&P 500', '10-Year Treasury Yield', 'VIX', 'Fear and Greed Index', 
            'Consumer Confidence', 'GDP Growth', 'Unemployment Rate', 'News Sentiment']].iloc[:-1]
    # Remove rows with NaN values
    X = X.dropna()
    y = labels[:len(X)]
    if len(X) > 0 and len(y) > 0:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        
        # Predict current state
        current_data = df[['S&P 500', '10-Year Treasury Yield', 'VIX', 'Fear and Greed Index', 
                           'Consumer Confidence', 'GDP Growth', 'Unemployment Rate', 'News Sentiment']].iloc[-1:]
        current_data = current_data.dropna()
        if not current_data.empty:
            prediction = clf.predict(current_data)[0]
            state = {1: 'Bullish', -1: 'Bearish', 0: 'Recession Risk'}
            current_state = state[prediction]
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': clf.feature_importances_
            })
        else:
            print("Current data contains NaN values, cannot predict.")
            current_state = "Insufficient data for prediction"
            feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': [0] * len(X.columns)})
    else:
        print("Insufficient training data after cleaning.")
        current_state = "Insufficient data for prediction"
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': [0] * len(X.columns)})
else:
    print("No labels generated due to insufficient data.")
    current_state = "Insufficient data for prediction"
    feature_importance = pd.DataFrame({'Feature': ['S&P 500', '10-Year Treasury Yield', 'VIX', 'Fear and Greed Index', 
                                                   'Consumer Confidence', 'GDP Growth', 'Unemployment Rate', 'News Sentiment'], 
                                       'Importance': [0] * 8})

# Create interactive plots
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

# Sector performance chart with dynamic title
current_month_year = datetime.now().strftime('%B %Y')
sector_fig = px.bar(sector_df, x='Return (%)', y='Sector', title=f'Sector Performance ({current_month_year})', 
                    orientation='h', hover_data={'Return (%)': ':.2f%'})
sector_fig.update_layout(xaxis_title='Return (%)', yaxis_title='Sector', template='plotly_white')

importance_fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        title='ML Feature Importance', orientation='h',
                        hover_data={'Importance': ':.3f'})
importance_fig.update_layout(xaxis_title='Importance', yaxis_title='Feature', template='plotly_white')

# Initialize Dash app
external_stylesheets = [themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("Macro Economic Dashboard", className="text-center text-dark mb-4")),
        dbc.Col(html.P("Real-Time 5-Year Analysis with Machine Learning Predictions", 
                       className="text-center text-dark mb-4"), width=12),
        dbc.Col(html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       className="text-center text-dark mb-4"), width=12)
    ]),

    # Main content and sidebar
    dbc.Row([
        # Sidebar for news
        dbc.Col([
            html.H3("Latest Economic News", className="text-dark mb-3"),
            *[dbc.Card([
                dbc.CardBody([
                    html.H5(item['title'], className="card-title text-dark"),
                    html.P(item['description'], className="card-text text-dark")
                ])
            ], className="mb-2 shadow", style={'backgroundColor': '#f8f9fa'}) for item in news_items]
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
                    html.H3("Latest Economic Indicators", className="text-dark mb-3"),
                    dbc.Collapse([
                        html.P(f"Unemployment Rate: {df['Unemployment Rate'].iloc[-1]:.1f}%", className="text-dark"),
                        html.P("CPI YoY (April 2025): 2.3%", className="text-dark"),
                        html.P(f"GDP Growth: {df['GDP Growth'].iloc[-1]:.1f}%", className="text-dark"),
                        html.P(f"Consumer Confidence: {df['Consumer Confidence'].iloc[-1]:.1f}", className="text-dark")
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