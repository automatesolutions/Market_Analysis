import pandas as pd
import numpy as np
import requests
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import yfinance as yf
import logging
import dash
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import torch
import torch.nn as nn
import json
import warnings

# Suppress All-NaN warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log versions
logging.info(f"Dash version: {dash.__version__}")
logging.info(f"dash_bootstrap_components version: {dbc.__version__}")
logging.info(f"PyTorch version: {torch.__version__}")

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not FRED_API_KEY or not ALPHA_VANTAGE_API_KEY:
    raise ValueError("Missing API keys in .env file.")

# Initialize FRED API
fred = Fred(api_key=FRED_API_KEY)

# Define LSTM model for daily predictions
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3, output_size=30, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(1, 0, 2)
        attn_output, _ = self.attention(out, out, out)
        context = attn_output[-1]
        out = self.fc(context)
        return out

# Define CNN-LSTM model for 1-minute predictions
class CNN_LSTM(nn.Module):
    def __init__(self, price_levels, channels, scalar_features_size, hidden_size=100, num_layers=3, output_size=30, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.cnn_residual = nn.Conv1d(channels, 64, kernel_size=1)
        self.cnn_output_size = 64
        self.lstm = nn.LSTM(input_size=self.cnn_output_size + scalar_features_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, order_footprint, scalar_features):
        batch_size, seq_length, price_levels, channels = order_footprint.shape
        order_footprint = order_footprint.view(batch_size * seq_length, channels, price_levels)
        cnn_main = self.cnn(order_footprint)
        cnn_residual = self.cnn_residual(order_footprint)
        cnn_residual = nn.functional.adaptive_max_pool1d(cnn_residual, 1)
        cnn_out = cnn_main + cnn_residual
        cnn_out = cnn_out.squeeze(-1)
        cnn_out = cnn_out.view(batch_size, seq_length, self.cnn_output_size)
        if scalar_features.size(-1) != 9:
            raise ValueError(f"Expected scalar_features with last dim 9, got {scalar_features.size(-1)}")
        lstm_input = torch.cat([cnn_out, scalar_features], dim=2)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(lstm_input.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(lstm_input.device)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out

# Load LSTM models
try:
    daily_model = StockLSTM(input_size=14, output_size=30)  # Updated for new features
    daily_model_path = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\model", 'lstm_daily_model.pt')
    daily_model.load_state_dict(torch.load(daily_model_path, weights_only=True))
    daily_model.eval()
    logging.info("Successfully loaded daily LSTM model.")
except Exception as e:
    logging.error(f"Error loading daily LSTM model: {e}")
    raise

try:
    minute_model = CNN_LSTM(price_levels=10, channels=2, scalar_features_size=9, output_size=30)  # Updated for new features
    minute_model_path = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\model", 'lstm_1minute_model.pt')
    minute_model.load_state_dict(torch.load(minute_model_path, weights_only=True))
    minute_model.eval()
    logging.info("Successfully loaded 1-minute CNN-LSTM model.")
except Exception as e:
    logging.error(f"Error loading 1-minute CNN-LSTM model: {e}")
    raise

# Function to check for cached data
def load_cached_data(symbol, interval='1d', cache_dir="cache"):
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval.replace(' ', '_')}.csv")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logging.info(f"Loaded cached data for {symbol} (interval={interval})")
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if set(['Open', 'High', 'Low', 'Close', 'Volume']).issubset(df.columns):
                if interval == '1d' and len(df) < 252:
                    logging.info(f"Cached data for {symbol} has only {len(df)} rows; re-fetching.")
                    return None
                return df
            logging.warning(f"Cached data for {symbol} does not contain expected columns.")
            return None
        except Exception as e:
            logging.warning(f"Error loading cached data for {symbol}: {e}")
    return None

# Function to save data to cache
def save_to_cache(symbol, data, interval='1d', cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval.replace(' ', '_')}.csv")
    if isinstance(data, pd.Series):
        data = data.to_frame(name='Close')
    data.to_csv(cache_file)
    logging.info(f"Saved data for {symbol} (interval={interval}) to {cache_file}")

# Function to fetch stock and sector data
def fetch_yfinance_data(symbol, period='10y', interval='1d'):
    cached_data = load_cached_data(symbol, interval=interval)
    if cached_data is not None:
        return cached_data
    try:
        logging.info(f"Fetching data for {symbol} from yfinance with period={period}, interval={interval}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[expected_columns]
        min_rows = 60 if interval == '1d' else 30
        if len(df) < min_rows and period != 'max':
            df = ticker.history(period='max', interval=interval)
            if df.empty:
                raise ValueError(f"No data for {symbol} with period='max'")
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[expected_columns]
        if interval == '1m':
            df = df.between_time('09:30', '16:00')
        df['Sector_Close'] = df['Close']  # Placeholder; will be updated in ticker analysis
        df['Order_Flow'] = 0  # Placeholder
        logging.info(f"Fetched data for {symbol}. Shape: {df.shape}")
        save_to_cache(symbol, df, interval=interval)
        return df
    except Exception as e:
        logging.error(f"Error fetching {symbol} from yfinance: {e}")
        return None

# Function to fetch order footprint data
def fetch_order_footprint(ticker, period, interval, expected_rows):
    try:
        logging.info(f"Fetching order footprint for {ticker}")
        price_levels = 10
        channels = 2
        footprint = np.random.rand(expected_rows, price_levels, channels) * 100  # Dummy data
        return footprint
    except Exception as e:
        logging.warning(f"Error fetching order footprint for {ticker}: {e}. Returning dummy data.")
        return np.zeros((expected_rows, price_levels, channels))

# Function to fetch Alpha Vantage data
def fetch_alpha_vantage_data(symbol, api_key, delay=12):
    cached_data = load_cached_data(symbol, interval='1d')
    if cached_data is not None:
        return cached_data
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}"
    try:
        logging.info(f"Fetching data for {symbol} from Alpha Vantage")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"No data for {symbol}: {data.get('Information', 'Unknown error')}")
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().astype(float)
        logging.info(f"Fetched data for {symbol} from Alpha Vantage")
        save_to_cache(symbol, df['5. adjusted close'], interval='1d')
        time.sleep(delay)
        return df['5. adjusted close']
    except Exception as e:
        logging.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
        return None

# Function to fetch stock data
def fetch_stock_data(symbol, api_key, period='10y', interval='1d'):
    data = fetch_yfinance_data(symbol, period, interval)
    if data is not None:
        return data
    data = fetch_alpha_vantage_data(symbol, api_key)
    if data is not None:
        return data
    logging.error(f"Failed to fetch data for {symbol}")
    return None

# Function to fetch FRED data
def fetch_fred_data(series_id):
    try:
        logging.info(f"Fetching FRED series {series_id}")
        data = fred.get_series(series_id)
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data = data.resample('D').ffill().bfill()
        return data
    except Exception as e:
        logging.error(f"Error fetching FRED series {series_id}: {e}")
        return None

# Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute Realized Volatility
def compute_realized_volatility(df, window=30):
    returns = df['Close'].pct_change().dropna()
    return np.sqrt((returns**2).rolling(window=window).sum())

# Fetch S&P 500 and sector data
sp500 = fetch_stock_data('SPY', ALPHA_VANTAGE_API_KEY)
if sp500 is None:
    logging.error("Failed to fetch S&P 500 proxy (SPY) data.")
    exit()

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

# Fetch macroeconomic data
macro_series = {
    'GDP': 'A191RL1Q225SBEA', 'Inflation': 'CPIAUCSL', 'Unemployment': 'UNRATE',
    'Interest Rate': 'FEDFUNDS', 'Consumer Confidence': 'UMCSENT', 'VIX': 'VIXCLS',
    '10Y Treasury': 'DGS10', '2Y Treasury': 'DGS2'
}
macro_data = {key: fetch_fred_data(series_id) for key, series_id in macro_series.items()}

# Combine data
common_index = sp500.index
data_dict = {'SP500': sp500['Close'] if isinstance(sp500, pd.DataFrame) else sp500}
for sector in sector_data:
    data_dict[f"{sector}_Close"] = sector_data[sector]['Close'] if isinstance(sector_data[sector], pd.DataFrame) else sector_data[sector]
for key in macro_data:
    if macro_data[key] is not None:
        data_dict[key] = macro_data[key].reindex(common_index, method='ffill').bfill()
    else:
        logging.warning(f"Macro data for {key} is None. Filling with zeros.")
        data_dict[key] = pd.Series(0, index=common_index)

data = pd.DataFrame(data_dict).dropna()

# Feature Engineering
data['SP500_Returns'] = data['SP500'].pct_change(periods=5).shift(-5)
for sector in sector_data:
    data[f"{sector}_Returns"] = data[f"{sector}_Close"].pct_change(periods=5).shift(-5)
data['SP500_MA20'] = data['SP500'].rolling(window=20).mean()
data['SP500_RSI'] = compute_rsi(data['SP500'], 14)
data['Yield_Spread'] = data['10Y Treasury'] - data['2Y Treasury']
data['GDP_Lag'] = data['GDP'].shift(1)
data['Inflation_Lag'] = data['Inflation'].shift(1)
data = data.dropna()

# Define features
features = ['SP500_MA20', 'SP500_RSI', 'GDP_Lag', 'Inflation_Lag', 'Unemployment',
            'Interest Rate', 'Consumer Confidence', 'VIX', 'Yield_Spread']
X = data[features]
nan_columns = X.columns[X.isna().all()]
if not nan_columns.empty:
    logging.warning(f"Columns with all NaN values: {nan_columns.tolist()}")
    X = X.drop(columns=nan_columns)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Define targets
y_market = (data['SP500_Returns'] > 0.01).astype(int)
y_sector = {sector: data[f"{sector}_Returns"] for sector in sector_data}

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_market_train, y_market_test = y_market[:train_size], y_market[train_size:]
y_sector_train = {sector: y_sector[sector][:train_size] for sector in y_sector}
y_sector_test = {sector: y_sector[sector][train_size:] for sector in y_sector}

# Train models
market_model = RandomForestClassifier(n_estimators=100, random_state=42)
market_model.fit(X_train, y_market_train)
sector_models = {}
for sector in y_sector:
    sector_models[sector] = RandomForestRegressor(n_estimators=100, random_state=42)
    sector_models[sector].fit(X_train, y_sector_train[sector])

# Evaluate models
market_pred = market_model.predict(X_test)
market_accuracy = accuracy_score(y_market_test, market_pred)
logging.info(f"Market Direction Accuracy: {market_accuracy:.2f}")

sector_mse = {}
for sector in y_sector:
    sector_pred = sector_models[sector].predict(X_test)
    sector_mse[sector] = mean_squared_error(y_sector_test[sector], sector_pred)
    logging.info(f"{sector} MSE: {sector_mse[sector]:.4f}")

# Latest S&P 500 prediction
latest_data = X.iloc[-1:].copy()
market_direction = market_model.predict(latest_data)[0]
market_confidence = market_model.predict_proba(latest_data)[0][1] if market_direction == 1 else market_model.predict_proba(latest_data)[0][0]
market_direction_label = "Bullish" if market_direction == 1 else "Bearish"
logging.info(f"Predicted Market Direction: {market_direction_label} (Confidence: {market_confidence:.2%})")

sector_predictions = {}
for sector in y_sector:
    sector_predictions[sector] = sector_models[sector].predict(latest_data)[0]
    logging.info(f"Predicted {sector} 5-Day Return: {sector_predictions[sector]:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': market_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Prepare data for dashboard
sp500_df = pd.DataFrame({'Date': sp500.index, 'SP500': sp500['Close'] if isinstance(sp500, pd.DataFrame) else sp500})
sector_historical = pd.DataFrame(index=common_index)
for sector in sector_data:
    sector_historical[sector] = sector_data[sector]['Close'] if isinstance(sector_data[sector], pd.DataFrame) else sector_data[sector]
sector_historical = sector_historical.reset_index().rename(columns={'index': 'Date'})
sector_historical_melted = sector_historical.melt(id_vars='Date', var_name='Sector', value_name='Price')
sector_pred_df = pd.DataFrame.from_dict(sector_predictions, orient='index', columns=['Predicted_5Day_Return'])
sector_pred_df.reset_index(inplace=True)
sector_pred_df.rename(columns={'index': 'Sector'}, inplace=True)
feature_importance_df = feature_importance.copy()

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dashboard layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("S&P 500 and Stock Prediction Dashboard", className="text-center text-dark mb-4")),
        dbc.Col(html.P("Real-Time Market Direction, Sector Performance, and Custom Ticker Forecast", 
                       className="text-center text-dark mb-4"), width=12),
        dbc.Col(html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       className="text-center text-dark mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H3("Custom Ticker Analysis", className="text-dark mb-3"),
            dcc.Input(id='ticker-input', type='text', placeholder='Enter NYSE/NASDAQ ticker (e.g., AAPL)', 
                      style={'width': '300px', 'marginRight': '10px'}),
            dcc.Dropdown(
                id='timeframe-dropdown',
                options=[
                    {'label': 'Daily', 'value': 'daily'},
                    {'label': '1-Minute', 'value': '1-minute'},
                ],
                value='daily',
                style={'width': '200px', 'display': 'inline-block', 'marginRight': '10px'}
            ),
            html.Button('Generate Chart', id='generate-button', n_clicks=0, style={'marginLeft': '10px'}),
            html.Div(id='ticker-error', style={'color': 'red'})
        ], width=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='ticker-chart', figure={})
        ], width=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            html.H3(id='market-prediction', className="text-center text-primary mb-4"),
            html.P("Key Drivers:", className="text-dark mb-2"),
            html.Ul(id='key-drivers', className="text-dark")
        ], width=3, className="p-3", style={'height': '100vh', 'overflow': 'auto', 'backgroundColor': '#e9ecef'}),
        dbc.Col([
            dbc.Row([dbc.Col(dcc.Graph(id='sp500-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='sector-historical-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='sector-pred-chart'), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='feature-importance-chart'), width=12)], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.H3("Real-Time Enhanced Summary", className="text-dark mb-3"),
                    html.P(id='market-direction', className="text-dark"),
                    html.P("Explanation:", className="text-dark"),
                    html.Ul(id='explanation', className="text-dark"),
                    html.P(id='key-driver', className="text-dark"),
                    html.P(id='risk-factors', className="text-dark"),
                    html.H4("Sector Performance (5-Day Returns)", className="text-dark mt-3"),
                    html.Ul(id='sector-performance', className="text-dark"),
                    html.H4("Actionable Insights", className="text-dark mt-3"),
                    html.P("Investors:", className="text-dark"),
                    html.Ul(id='investors-insights', className="text-dark"),
                    html.P("Traders:", className="text-dark"),
                    html.Ul(id='traders-insights', className="text-dark"),
                    html.P("Long-Term View:", className="text-dark"),
                    html.P(id='long-term-view', className="text-dark")
                ], width=12)
            ])
        ], width=9)
    ])
], fluid=True)

# Callback to update S&P 500 and sector predictions
@app.callback(
    [
        Output('market-prediction', 'children'),
        Output('key-drivers', 'children'),
        Output('sp500-chart', 'figure'),
        Output('sector-historical-chart', 'figure'),
        Output('sector-pred-chart', 'figure'),
        Output('feature-importance-chart', 'figure'),
        Output('market-direction', 'children'),
        Output('explanation', 'children'),
        Output('key-driver', 'children'),
        Output('risk-factors', 'children'),
        Output('sector-performance', 'children'),
        Output('investors-insights', 'children'),
        Output('traders-insights', 'children'),
        Output('long-term-view', 'children')
    ],
    Input('generate-button', 'n_clicks')
)
def update_sp500_dashboard(n_clicks):
    market_pred_text = f"Market Prediction: {market_direction_label} (Confidence: {market_confidence:.2%})"
    key_drivers = [html.Li(f"{row['Feature']}: {row['Importance']:.2%}") for _, row in feature_importance_df.iterrows()]
    sp500_fig = px.line(sp500_df, x='Date', y='SP500', title='S&P 500 (SPY) Historical Data', template='plotly_white')
    sector_hist_fig = px.line(sector_historical_melted, x='Date', y='Price', color='Sector', 
                              title='Sector ETF Historical Data', template='plotly_white')
    sector_pred_fig = px.bar(sector_pred_df, x='Predicted_5Day_Return', y='Sector', orientation='h', 
                             title='Predicted 5-Day Sector Returns', template='plotly_white', text_auto='.4f')
    feature_fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', 
                         title='Feature Importance for Market Direction', template='plotly_white', text_auto='.2%')
    market_direction_text = f"Market Direction: {market_direction_label} (Confidence: {market_confidence:.2%})"
    explanation = [
        html.Li(f"Economic Growth: GDP at {data['GDP'].iloc[-1]:.2f}% supports {'bullish' if market_direction_label == 'Bullish' else 'bearish'} markets."),
        html.Li(f"Unemployment: {data['Unemployment'].iloc[-1]:.2f}% indicates {'strong' if data['Unemployment'].iloc[-1] < 5 else 'weak'} labor markets."),
        html.Li(f"Inflation: CPI at {data['Inflation'].iloc[-1]/100:.2f}% suggests {'stable' if data['Inflation'].iloc[-1]/100 < 3 else 'high'} price pressures."),
        html.Li(f"VIX: {data['VIX'].iloc[-1]:.2f}, indicating {'low' if data['VIX'].iloc[-1] < 20 else 'high'} volatility.")
    ]
    key_driver_text = f"Key Driver: {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.2%})"
    risk_factors_text = f"Risk Factors: Federal Funds Rate at {data['Interest Rate'].iloc[-1]:.2f}% may impact valuations."
    sector_performance = [html.Li(f"{sector}: {return_pred:.4f}") for sector, return_pred in sector_predictions.items()]
    investors_insights = [
        html.Li("Overweight: Sectors with positive returns (e.g., Utilities)."),
        html.Li("Underweight: Sectors with negative returns (e.g., Consumer Discretionary).")
    ]
    traders_insights = [
        html.Li("Monitor Fed signals—rate hikes above 5% could shift outlook."),
        html.Li("Consider shorts in underperforming sectors.")
    ]
    long_term_view = f"Sustained GDP growth (>3%) could {'extend bullish trend' if market_direction_label == 'Bullish' else 'mitigate bearish pressures'} into Q3 2025."

    return (market_pred_text, key_drivers, sp500_fig, sector_hist_fig, sector_pred_fig, feature_fig,
            market_direction_text, explanation, key_driver_text, risk_factors_text, sector_performance,
            investors_insights, traders_insights, long_term_view)

# Callback to handle ticker input and generate chart
@app.callback(
    [Output('ticker-chart', 'figure'),
     Output('ticker-error', 'children')],
    Input('generate-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('timeframe-dropdown', 'value')
)
def generate_ticker_charts(n_clicks, ticker, timeframe):
    if n_clicks == 0 or not ticker or not timeframe:
        return {}, "Please enter a ticker symbol and select a timeframe."

    test_data = fetch_yfinance_data(ticker, period='5d', interval='1d')
    if test_data is None:
        return {}, f"Error: Unable to fetch data for ticker '{ticker}'."

    timeframe_configs = {
        'daily': {'interval': '1d', 'period': '1y', 'label': 'Daily', 'default_days': 180},
        '1-minute': {'interval': '1m', 'period': '7d', 'label': '1-Minute', 'default_days': 7},
    }

    config = timeframe_configs.get(timeframe)
    if not config:
        return {}, "Invalid timeframe selected."

    interval = config['interval']
    period = config['period']
    label = config['label']
    default_days = config['default_days']

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return {}, f"No data available for {ticker} on {label} timeframe."
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        sector_ticker = stock.info.get('sector', 'SPY')
        sector_ticker_map = {
            'Technology': 'XLK', 'Financial Services': 'XLF', 'Consumer Cyclical': 'XLY',
            'Industrials': 'XLI', 'Utilities': 'XLU', 'Healthcare': 'XLV',
            'Communication Services': 'XLC', 'Consumer Defensive': 'XLP',
            'Basic Materials': 'XLB', 'Real Estate': 'XLRE', 'Energy': 'XLE'
        }
        sector_ticker = sector_ticker_map.get(sector_ticker, 'SPY')
        sector_df = yf.Ticker(sector_ticker).history(period=period, interval=interval)
        sector_df.index = pd.to_datetime(sector_df.index)
        if sector_df.index.tz is not None:
            sector_df.index = sector_df.index.tz_localize(None)
        sector_df = sector_df.reindex(df.index, method='ffill').bfill()
        df['Sector_Close'] = sector_df['Close']
        df['Order_Flow'] = 0  # Placeholder
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Sector_Close', 'Order_Flow']]

        end_date = df.index[-1]
        if timeframe == 'daily':
            start_date = end_date - pd.Timedelta(days=180)
            df = df.loc[start_date:end_date]
        else:
            df = df[df.index.date == df.index[-1].date()]
            df = df.between_time('09:30', '16:00')
            if df.empty:
                return {}, f"No data for {ticker} on the last trading day (9:30 AM - 4:00 PM)."

        if len(df) < 2:
            return {}, f"Insufficient data for {ticker} on {label} timeframe."

        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=f'{label} Chart'
        )
        fig = go.Figure(data=[candlestick])
        fig.update_layout(
            title=f"{ticker} {label} Chart",
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            xaxis=dict(type='date', tickformat='%Y-%m-%d %H:%M', showticklabels=True),
            yaxis=dict(autorange=True, title="Price (USD)")
        )

        # Generate predictions
        if timeframe == 'daily' and len(df) >= 60:
            try:
                df['Price_Change'] = df['Close'].pct_change()
                logging.info(f'Rows after Price_Change: {df.shape[0]}')
                df['Volatility'] = df['Close'].rolling(window=5).std()
                logging.info(f'Rows after Volatility: {df.shape[0]}')
                df['Sector_Volatility'] = df['Sector_Close'].rolling(window=5).std()
                logging.info(f'Rows after Sector_Volatility: {df.shape[0]}')
                df['Realized_Vol'] = compute_realized_volatility(df, window=5)
                logging.info(f'Rows after Realized_Vol: {df.shape[0]}')
                df['Vol_Ratio'] = df['Volatility'] / df['Volatility'].rolling(window=10).mean()
                logging.info(f'Rows after Vol_Ratio: {df.shape[0]}')
                df['SMA_20'] = df['Close'].rolling(window=5).mean()
                logging.info(f'Rows after SMA_20: {df.shape[0]}')
                df['RSI'] = compute_rsi(df['Close'], 5)
                logging.info(f'Rows after RSI: {df.shape[0]}')
                df['MACD'] = df['Close'].ewm(span=6, adjust=False).mean() - df['Close'].ewm(span=13, adjust=False).mean()
                logging.info(f'Rows after MACD: {df.shape[0]}')
                df['Upper_BB'] = df['SMA_20'] + 2 * df['Close'].rolling(window=5).std()
                df['Lower_BB'] = df['SMA_20'] - 2 * df['Close'].rolling(window=5).std()
                logging.info(f'Rows after Bollinger Bands: {df.shape[0]}')
                df['ATR'] = (df['High'].rolling(window=5).max() - df['Low'].rolling(window=5).min())
                logging.info(f'Rows after ATR: {df.shape[0]}')
                df['Imbalance'] = df['High'] - df['Low']
                features = ['Close', 'Volume', 'Price_Change', 'Volatility', 'Sector_Volatility', 'Realized_Vol', 'Vol_Ratio', 'SMA_20', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB', 'ATR', 'Order_Flow']
                df_processed = df[features].dropna()
                logging.info(f'Rows after dropna: {df_processed.shape[0]}')
                if df_processed.shape[0] < 60:
                    return fig, f"Not enough processed data for {ticker} to make daily predictions. Need at least 60 rows, got {df_processed.shape[0]}."
                seq = df_processed.iloc[-60:][features].values
                if seq.shape[0] < 1:
                    return fig, f"No valid data for {ticker} after feature engineering."
                feature_scaler = MinMaxScaler()
                feature_scaler.fit(df_processed[features])
                seq_scaled = feature_scaler.transform(seq)
                target_scaler = MinMaxScaler()
                close_values = df_processed['Close'].values.reshape(-1, 1)
                target_scaler.fit(close_values)
                seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
                logging.info(f"Scaled input to model (daily): {seq_scaled}")
                with torch.no_grad():
                    predictions = daily_model(seq_tensor).numpy().flatten()
                logging.info(f"Raw model output (daily): {predictions}")
                predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                logging.info(f"Last 5 closing prices for {ticker} (Daily): {df['Close'].iloc[-5:].tolist()}")
                logging.info(f"Daily Predictions for {ticker} (first 5): {predictions[:5].tolist()}")
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
                pred_open = [df['Close'].iloc[-1]] + list(predictions[:-1])
                pred_close = predictions
                pred_high = np.maximum(pred_open, pred_close) + predictions * 0.005
                pred_low = np.minimum(pred_open, pred_close) - predictions * 0.005
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Open': pred_open,
                    'Predicted_High': pred_high,
                    'Predicted_Low': pred_low,
                    'Predicted_Close': pred_close,
                    'Actual_Close': [None] * len(future_dates)
                })
                excel_filename = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\data", f"predictions_{ticker}_daily.xlsx")
                pred_df.to_excel(excel_filename, index=False)
                logging.info(f"Saved daily predictions for {ticker} to {excel_filename}")
                pred_candlestick = go.Candlestick(
                    x=future_dates,
                    open=pred_open,
                    high=pred_high,
                    low=pred_low,
                    close=pred_close,
                    name='Predicted (30 Days)',
                    increasing_line_color='blue',
                    decreasing_line_color='orange'
                )
                fig.add_trace(pred_candlestick)
                fig.update_xaxes(range=[df.index[0], future_dates[-1]])
                # Add MAE if applicable
                actual_close = df['Close'].iloc[-30:].values
                if len(actual_close) == len(predictions):
                    mae = mean_absolute_error(actual_close, predictions[:len(actual_close)])
                    fig.add_annotation(text=f"MAE: {mae:.4f}", xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False)
            except Exception as e:
                logging.error(f"Error generating daily predictions for {ticker}: {e}")
                return fig, f"Chart generated, but error in daily predictions: {str(e)}"
        elif timeframe == '1-minute' and len(df) >= 30:
            try:
                df['Price_Change'] = df['Close'].pct_change()
                df['Volatility'] = df['Close'].rolling(window=10).std()
                df['Sector_Volatility'] = df['Sector_Close'].rolling(window=10).std()
                df['Realized_Vol'] = compute_realized_volatility(df, window=10)
                df['Vol_Ratio'] = df['Volatility'] / df['Volatility'].rolling(window=20).mean()
                df['Momentum'] = df['Close'].diff(3)
                df['Volume_Spike'] = (df['Volume'] / df['Volume'].rolling(window=10).mean()) - 1
                features = ['Close', 'Volume', 'Price_Change', 'Volatility', 'Sector_Volatility', 'Realized_Vol', 'Vol_Ratio', 'Momentum', 'Volume_Spike']
                df_processed = df[features].dropna()
                logging.info(f'Rows after dropna (1-minute): {df_processed.shape[0]}')
                if df_processed.shape[0] < 30:
                    return fig, f"Not enough processed data for {ticker} to make 1-minute predictions. Need at least 30 rows, got {df_processed.shape[0]}."
                order_footprint = fetch_order_footprint(ticker, period='7d', interval='1m', expected_rows=len(df_processed))
                order_footprint = order_footprint[-30:]
                seq = df_processed.iloc[-30:][features].values
                if seq.shape[0] < 1:
                    return fig, f"No valid data for {ticker} after feature engineering."
                feature_scaler = MinMaxScaler()
                feature_scaler.fit(df_processed[features])
                seq_scaled = feature_scaler.transform(seq)
                target_scaler = MinMaxScaler()
                close_values = df_processed['Close'].values.reshape(-1, 1)
                target_scaler.fit(close_values)
                seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
                order_tensor = torch.tensor(order_footprint, dtype=torch.float32).unsqueeze(0)
                logging.info(f"Scaled input to model (1-minute): {seq_scaled}")
                with torch.no_grad():
                    predictions = minute_model(order_tensor, seq_tensor).numpy().flatten()
                logging.info(f"Raw model output (1-minute): {predictions}")
                predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                logging.info(f"Last 5 closing prices for {ticker} (1-Minute): {df['Close'].iloc[-5:].tolist()}")
                logging.info(f"1-Minute Predictions for {ticker} (first 5): {predictions[:5].tolist()}")
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=30, freq='1min')
                pred_open = [df['Close'].iloc[-1]] + list(predictions[:-1])
                pred_close = predictions
                pred_high = np.maximum(pred_open, pred_close) + predictions * 0.001
                pred_low = np.minimum(pred_open, pred_close) - predictions * 0.001
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Open': pred_open,
                    'Predicted_High': pred_high,
                    'Predicted_Low': pred_low,
                    'Predicted_Close': pred_close,
                    'Actual_Close': [None] * len(future_dates)
                })
                excel_filename = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\data", f"predictions_{ticker}_1minute.xlsx")
                pred_df.to_excel(excel_filename, index=False)
                logging.info(f"Saved 1-minute predictions for {ticker} to {excel_filename}")
                pred_candlestick = go.Candlestick(
                    x=future_dates,
                    open=pred_open,
                    high=pred_high,
                    low=pred_low,
                    close=pred_close,
                    name='Predicted (30 Minutes)',
                    increasing_line_color='blue',
                    decreasing_line_color='orange'
                )
                fig.add_trace(pred_candlestick)
                fig.update_xaxes(range=[df.index[0], future_dates[-1]])
                actual_close = df['Close'].iloc[-30:].values
                if len(actual_close) == len(predictions):
                    mae = mean_absolute_error(actual_close, predictions[:len(actual_close)])
                    fig.add_annotation(text=f"MAE: {mae:.4f}", xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False)
            except Exception as e:
                logging.error(f"Error generating 1-minute predictions for {ticker}: {e}")
                return fig, f"Chart generated, but error in 1-minute predictions: {str(e)}"
        elif timeframe == '1-minute' and len(df) < 30:
            return fig, f"Chart generated, but insufficient data for 1-minute predictions: {len(df)} rows."

        return fig, ""

    except Exception as e:
        logging.error(f"Error generating chart for {ticker} ({label}): {e}")
        return {}, f"Error generating chart for {ticker} ({label}): {str(e)}."

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)