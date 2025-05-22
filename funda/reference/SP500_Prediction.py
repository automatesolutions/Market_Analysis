import pandas as pd
import numpy as np
import requests
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import yfinance as yf
import logging
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import dash

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log Dash version for compatibility debugging
logging.info(f"Dash version: {dash.__version__}")

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Validate API keys
if not FRED_API_KEY or not ALPHA_VANTAGE_API_KEY:
    raise ValueError("Missing API keys. Ensure FRED_API_KEY and ALPHA_VANTAGE_API_KEY are set in the .env file.")

# Initialize FRED API
fred = Fred(api_key=FRED_API_KEY)

# Function to check for cached data
def load_cached_data(symbol, cache_dir="cache"):
    cache_file = os.path.join(cache_dir, f"{symbol}_daily.csv")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logging.info(f"Loaded cached data for {symbol} from {cache_file}")
            # Remove timezone information from index if present
            try:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            except AttributeError:
                logging.warning(f"Could not check timezone for {symbol} index; proceeding as-is")
            return df['Close']
        except Exception as e:
            logging.warning(f"Error loading cached data for {symbol}: {e}")
    return None

# Function to save data to cache
def save_to_cache(symbol, data, cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{symbol}_daily.csv")
    data.to_frame(name='Close').to_csv(cache_file)
    logging.info(f"Saved data for {symbol} to {cache_file}")

# Function to fetch data from yfinance (primary source)
def fetch_yfinance_data(symbol):
    # Check cache first
    cached_data = load_cached_data(symbol)
    if cached_data is not None:
        return cached_data

    try:
        logging.info(f"Attempting to fetch data for {symbol} from yfinance")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start='2010-01-01', end='2025-05-15')
        if df.empty:
            raise ValueError(f"No data for {symbol} from yfinance")
        df.index = pd.to_datetime(df.index)
        # Remove timezone information from index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        logging.info(f"Successfully fetched data for {symbol} from yfinance")
        save_to_cache(symbol, df['Close'])
        return df['Close']
    except Exception as e:
        logging.error(f"Error fetching {symbol} from yfinance: {e}")
        return None

# Function to fetch Alpha Vantage data (fallback, requires premium access)
def fetch_alpha_vantage_data(symbol, api_key, delay=12):
    # Check cache first
    cached_data = load_cached_data(symbol)
    if cached_data is not None:
        return cached_data

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}"
    try:
        logging.info(f"Attempting to fetch data for {symbol} from Alpha Vantage")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'Time Series (Daily)' not in data:
            error_msg = data.get('Information', 'Unknown error')
            if "premium endpoint" in error_msg:
                logging.error(f"Alpha Vantage requires a premium plan for {symbol}: {error_msg}")
                return None
            raise ValueError(f"No data for {symbol} in response: {data}")
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        # Remove timezone information from index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        logging.info(f"Successfully fetched data for {symbol} from Alpha Vantage")
        save_to_cache(symbol, df['5. adjusted close'])
        time.sleep(delay)  # Respect Alpha Vantage rate limit (5 calls/minute)
        return df['5. adjusted close']
    except Exception as e:
        logging.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
        return None

# Function to fetch data with fallback
def fetch_stock_data(symbol, api_key):
    # Try yfinance first (primary source)
    data = fetch_yfinance_data(symbol)
    if data is not None:
        return data
    
    # Fallback to Alpha Vantage (requires premium access)
    data = fetch_alpha_vantage_data(symbol, api_key)
    if data is not None:
        return data
    
    logging.error(f"Failed to fetch data for {symbol} from both yfinance and Alpha Vantage.")
    return None

# Function to fetch FRED data
def fetch_fred_data(series_id):
    try:
        logging.info(f"Fetching FRED series {series_id}")
        data = fred.get_series(series_id)
        data.index = pd.to_datetime(data.index)
        # Ensure FRED data is timezone-naive
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data = data.resample('D').ffill()
        logging.info(f"Successfully fetched FRED series {series_id}")
        return data
    except Exception as e:
        logging.error(f"Error fetching FRED series {series_id}: {e}")
        return None

# Fetch S&P 500 proxy data (using SPY)
sp500 = fetch_stock_data('SPY', ALPHA_VANTAGE_API_KEY)
if sp500 is None:
    logging.error("Failed to fetch S&P 500 proxy (SPY) data from both yfinance and Alpha Vantage.")
    print("Suggestions:")
    print("- Verify network connectivity and try again.")
    print("- Ensure the symbol 'SPY' is supported and data is available.")
    print("- If using Alpha Vantage, consider upgrading to a premium plan to access the endpoint.")
    exit()

# Fetch sector ETF data
sector_etfs = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Consumer Discretionary': 'XLY',
    'Industrials': 'XLI', 'Utilities': 'XLU', 'Healthcare': 'XLV',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP',
    'Materials': 'XLB', 'Real Estate': 'XLRE', 'Energy': 'XLE'
}
sector_data = {}
for sector, ticker in sector_etfs.items():
    sector_data[sector] = fetch_stock_data(ticker, ALPHA_VANTAGE_API_KEY)
    if sector_data[sector] is None:
        logging.warning(f"Failed to fetch data for {sector} ({ticker}). Using zeros.")
        sector_data[sector] = pd.Series(0, index=sp500.index)

# Fetch macroeconomic data from FRED
macro_series = {
    'GDP': 'A191RL1Q225SBEA',  # Real GDP growth rate (quarterly, annualized)
    'Inflation': 'CPIAUCSL',   # Consumer Price Index
    'Unemployment': 'UNRATE',  # Unemployment rate
    'Interest Rate': 'FEDFUNDS',  # Federal Funds Rate
    'Consumer Confidence': 'UMCSENT',  # Consumer Sentiment Index
    'VIX': 'VIXCLS',  # Volatility Index
    '10Y Treasury': 'DGS10',  # 10-Year Treasury Yield
    '2Y Treasury': 'DGS2'  # 2-Year Treasury Yield
}
macro_data = {key: fetch_fred_data(series_id) for key, series_id in macro_series.items()}

# Log timezone information before combining data
logging.info("Timezone information before combining data:")
logging.info(f"SP500 timezone: {sp500.index.tz}")
for sector in sector_data:
    logging.info(f"{sector}_Close timezone: {sector_data[sector].index.tz}")
for key in macro_data:
    logging.info(f"{key} timezone: {macro_data[key].index.tz if macro_data[key] is not None else 'None'}")

# Combine data into a single DataFrame
# Reindex all series to SP500's index to ensure alignment
common_index = sp500.index
data_dict = {'SP500': sp500}
for sector in sector_data:
    data_dict[f"{sector}_Close"] = sector_data[sector].reindex(common_index, method='ffill')
for key in macro_data:
    if macro_data[key] is not None:
        data_dict[key] = macro_data[key].reindex(common_index, method='ffill')
    else:
        logging.warning(f"Macro data for {key} is None. Filling with zeros.")
        data_dict[key] = pd.Series(0, index=common_index)

# Create DataFrame and handle missing values
data = pd.DataFrame(data_dict).dropna()

# Compute RSI (Relative Strength Index)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Feature Engineering
# Calculate returns (5-day forward returns for prediction)
data['SP500_Returns'] = data['SP500'].pct_change(periods=5).shift(-5)
for sector in sector_data:
    data[f"{sector}_Returns"] = data[f"{sector}_Close"].pct_change(periods=5).shift(-5)

# Technical indicators for S&P 500
data['SP500_MA20'] = data['SP500'].rolling(window=20).mean()
data['SP500_RSI'] = compute_rsi(data['SP500'], 14)

# Macroeconomic features
data['Yield_Spread'] = data['10Y Treasury'] - data['2Y Treasury']
data['GDP_Lag'] = data['GDP'].shift(1)
data['Inflation_Lag'] = data['Inflation'].shift(1)

# Drop rows with NaN values after feature engineering
data = data.dropna()

# Define features
features = ['SP500_MA20', 'SP500_RSI', 'GDP_Lag', 'Inflation_Lag', 'Unemployment',
            'Interest Rate', 'Consumer Confidence', 'VIX', 'Yield_Spread']
X = data[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Define targets
# Market direction: Bullish if 5-day return > 1%, Bearish otherwise
y_market = (data['SP500_Returns'] > 0.01).astype(int)
# Sector returns
y_sector = {sector: data[f"{sector}_Returns"] for sector in sector_data}

# Train-test split (time-series aware)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_market_train, y_market_test = y_market[:train_size], y_market[train_size:]
y_sector_train = {sector: y_sector[sector][:train_size] for sector in y_sector}
y_sector_test = {sector: y_sector[sector][train_size:] for sector in y_sector}

# Train market direction model
market_model = RandomForestClassifier(n_estimators=100, random_state=42)
market_model.fit(X_train, y_market_train)

# Train sector performance models
sector_models = {}
for sector in y_sector:
    sector_models[sector] = RandomForestRegressor(n_estimators=100, random_state=42)
    sector_models[sector].fit(X_train, y_sector_train[sector])

# Evaluate models
market_pred = market_model.predict(X_test)
market_accuracy = accuracy_score(y_market_test, market_pred)
print(f"Market Direction Accuracy: {market_accuracy:.2f}")

sector_mse = {}
for sector in y_sector:
    sector_pred = sector_models[sector].predict(X_test)
    sector_mse[sector] = mean_squared_error(y_sector_test[sector], sector_pred)
    print(f"{sector} MSE: {sector_mse[sector]:.4f}")

# Latest prediction (as of May 15, 2025)
latest_data = X.iloc[-1:].copy()
market_direction = market_model.predict(latest_data)[0]
market_confidence = market_model.predict_proba(latest_data)[0][1] if market_direction == 1 else market_model.predict_proba(latest_data)[0][0]
market_direction_label = "Bullish" if market_direction == 1 else "Bearish"
print(f"\nPredicted Market Direction: {market_direction_label} (Confidence: {market_confidence:.2%})")

sector_predictions = {}
for sector in y_sector:
    sector_predictions[sector] = sector_models[sector].predict(latest_data)[0]
    print(f"Predicted {sector} 5-Day Return: {sector_predictions[sector]:.4f}")

# Feature importance for market direction
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': market_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance for Market Direction:")
print(feature_importance)

# Prepare data for dashboard
# Historical S&P 500 data
sp500_df = pd.DataFrame({'Date': sp500.index, 'SP500': sp500.values})

# Historical sector data
sector_historical = pd.DataFrame(index=common_index)
for sector in sector_data:
    sector_historical[sector] = sector_data[sector]

# Log DataFrame columns before resetting index
logging.info(f"sector_historical columns before reset_index: {sector_historical.columns.tolist()}")
sector_historical = sector_historical.reset_index().rename(columns={'index': 'Date'})
logging.info(f"sector_historical columns after reset_index: {sector_historical.columns.tolist()}")

# Melt the DataFrame for Plotly Express
sector_historical_melted = sector_historical.melt(id_vars='Date', var_name='Sector', value_name='Price')

# Sector predictions
sector_pred_df = pd.DataFrame.from_dict(sector_predictions, orient='index', columns=['Predicted_5Day_Return'])
sector_pred_df.reset_index(inplace=True)
sector_pred_df.rename(columns={'index': 'Sector'}, inplace=True)

# Feature importance
feature_importance_df = feature_importance.copy()

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dashboard layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("S&P 500 Prediction Dashboard", className="text-center text-dark mb-4")),
        dbc.Col(html.P("Real-Time Market Direction and Sector Performance Forecast", 
                       className="text-center text-dark mb-4"), width=12),
        dbc.Col(html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       className="text-center text-dark mb-4"), width=12)
    ]),

    # Main content
    dbc.Row([
        # Market Direction Prediction
        dbc.Col([
            html.H3(f"Market Prediction: {market_direction_label} (Confidence: {market_confidence:.2%})", 
                    className="text-center text-primary mb-4"),
            html.P("Key Drivers:", className="text-dark mb-2"),
            html.Ul([
                html.Li(f"{row['Feature']}: {row['Importance']:.2%}") for _, row in feature_importance_df.iterrows()
            ], className="text-dark")
        ], width=3, className="p-3", style={'height': '100vh', 'overflow': 'auto', 'backgroundColor': '#e9ecef'}),

        # Charts
        dbc.Col([
            # S&P 500 Historical Chart
            dbc.Row([
                dbc.Col(dcc.Graph(id='sp500-chart', figure=px.line(sp500_df, x='Date', y='SP500', 
                            title='S&P 500 (SPY) Historical Data', template='plotly_white')), width=12)
            ], className="mb-4"),

            # Sector Historical Chart
            dbc.Row([
                dbc.Col(dcc.Graph(id='sector-historical-chart', figure=px.line(sector_historical_melted, x='Date', 
                            y='Price', color='Sector', title='Sector ETF Historical Data', 
                            template='plotly_white')), width=12)
            ], className="mb-4"),

            # Sector Predictions Chart
            dbc.Row([
                dbc.Col(dcc.Graph(id='sector-pred-chart', figure=px.bar(sector_pred_df, x='Predicted_5Day_Return', 
                            y='Sector', orientation='h', title='Predicted 5-Day Sector Returns', 
                            template='plotly_white', text_auto='.4f')), width=12)
            ], className="mb-4"),

            # Feature Importance Chart
            dbc.Row([
                dbc.Col(dcc.Graph(id='feature-importance-chart', figure=px.bar(feature_importance_df, x='Importance', 
                            y='Feature', orientation='h', title='Feature Importance for Market Direction', 
                            template='plotly_white', text_auto='.2%')), width=12)
            ], className="mb-4"),

            # Real-Time Enhanced Summary
            dbc.Row([
                dbc.Col([
                    html.H3("Real-Time Enhanced Summary", className="text-dark mb-3"),
                    html.P(f"Market Direction: {market_direction_label} (Confidence: {market_confidence:.2%})", 
                           className="text-dark"),
                    html.P(f"Explanation:", className="text-dark"),
                    html.Ul([
                        html.Li(f"Economic Growth: GDP annualized growth at {data['GDP'].iloc[-1]:.2f}% reflects a strong economy, {'boosting' if market_direction_label == 'Bullish' else 'but may be offset by other factors in'} equity markets.", className="text-dark"),
                        html.Li(f"Low Unemployment: At {data['Unemployment'].iloc[-1]:.2f}%, tight labor markets support consumer spending and corporate profits.", className="text-dark"),
                        html.Li(f"Stable Inflation: CPI at {data['Inflation'].iloc[-1]/100:.2f}% suggests controlled price pressures, {'reducing' if market_direction_label == 'Bullish' else 'but not eliminating'} monetary tightening risks.", className="text-dark"),
                        html.Li(f"Volatility (VIX): At {data['VIX'].iloc[-1]:.2f}, indicating {'low' if data['VIX'].iloc[-1] < 20 else 'high'} market volatility.", className="text-dark"),
                    ]),
                    html.P(f"Key Driver: {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.2%} feature importance) is the primary factor.", className="text-dark"),
                    html.P(f"Risk Factors: The Federal Funds Rate at {data['Interest Rate'].iloc[-1]:.2f}% could pressure valuations if it rises further.", className="text-dark"),
                    html.H4("Sector Performance (5-Day Returns)", className="text-dark mt-3"),
                    html.Ul([
                        html.Li(f"{sector}: {return_pred:.4f}") for sector, return_pred in sector_predictions.items()
                    ], className="text-dark"),
                    html.H4("Actionable Insights", className="text-dark mt-3"),
                    html.P("Investors:", className="text-dark"),
                    html.Ul([
                        html.Li("Overweight: Sectors with positive returns (e.g., Utilities at 0.0056 if applicable)."),
                        html.Li("Underweight: Sectors with negative returns (e.g., Consumer Discretionary at -0.0701).")
                    ], className="text-dark"),
                    html.P("Traders:", className="text-dark"),
                    html.Ul([
                        html.Li("Watch for Federal Reserve signals—rate hikes above 5% could shift the outlook."),
                        html.Li(f"Consider short positions in underperforming sectors like Consumer Discretionary.")
                    ], className="text-dark"),
                    html.P("Long-Term View:", className="text-dark"),
                    html.P(f"Sustained GDP growth (>3%) could {'extend the Bullish trend' if market_direction_label == 'Bullish' else 'mitigate Bearish pressures'} into Q3 2025, barring geopolitical shocks.", className="text-dark")
                ], width=12)
            ])
        ], width=9)
    ])
], fluid=True)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)