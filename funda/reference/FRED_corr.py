import pandas as pd
import plotly.graph_objects as go
import numpy as np
from fredapi import Fred
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get FRED API key from .env
api_key = os.getenv('FRED_API_KEY')
if not api_key or len(api_key) != 32 or not api_key.isalnum():
    print("Error: FRED_API_KEY is missing or invalid in .env file. It must be a 32-character alphanumeric string.")
    print("Get a key at https://fred.stlouisfed.org/docs/api/api_key.html and add it to .env as FRED_API_KEY=your_key")
    exit(1)

# Initialize FRED API
try:
    fred = Fred(api_key=api_key)
except ValueError as e:
    print(f"Error initializing FRED API: {e}")
    exit(1)

# Define date range (from 2020-01-01 to present, May 12, 2025)
start_date = '2020-01-01'
end_date = datetime(2025, 5, 12).strftime('%Y-%m-%d')

# Fetch FRED data series
series = {
    'NASDAQCOM': 'NASDAQ Composite Index',
    'T10Y2Y': '10-Year Minus 2-Year Treasury Spread',
    'M2SL': 'M2 Money Supply',
    'TEDRATE': 'TED Spread',
    'VIXCLS': 'CBOE Volatility Index'
}
fred_data = {}
for code, name in series.items():
    try:
        fred_data[name] = fred.get_series(code, start_date=start_date, end_date=end_date)
        print(f"Successfully fetched {name}")
    except Exception as e:
        print(f"Error fetching {name}: {e}")

# Combine all data into a single DataFrame
df_list = [pd.DataFrame(data).rename(columns={0: name}) for name, data in fred_data.items() if not data.empty]

# Check if there’s data to process
if not df_list:
    print("No data available to process. Check API key and internet connection.")
    exit(1)

# Concatenate and clean data
try:
    df = pd.concat(df_list, axis=1)
    df = df.ffill().dropna()
    if df.empty:
        print("DataFrame is empty after cleaning. Check date range or data availability.")
        exit(1)
except Exception as e:
    print(f"Error concatenating data: {e}")
    exit(1)

# Detect convergence and divergence
try:
    # Calculate M2SL annual growth (12-month percentage change, annualized)
    m2sl_growth = df['M2 Money Supply'].pct_change(252).mean() * 100  # Approx. 1 year of trading days
    t10y2y_latest = df['10-Year Minus 2-Year Treasury Spread'].iloc[-1]
    tedrate_latest = df['TED Spread'].iloc[-1]
    vixcls_latest = df['CBOE Volatility Index'].iloc[-1]

    print("\nIndicator Trends:")
    print(f"M2SL Annual Growth: {m2sl_growth:.2f}%")
    print(f"T10Y2Y Latest: {t10y2y_latest:.2f}%")
    print(f"TEDRATE Latest: {tedrate_latest:.2f}%")
    print(f"VIXCLS Latest: {vixcls_latest:.2f}")

    # Bullish convergence: Rising M2SL + Positive T10Y2Y
    if m2sl_growth > 5 and t10y2y_latest > 0.5:
        print("ALERT: Bullish Convergence Detected!")
        print("Rising M2SL (>5%) and Positive T10Y2Y (>0.5%) suggest strong NASDAQ upside.")
        print("Action: Buy NASDAQ (e.g., QQQ ETF, tech stocks).")

    # Bearish convergence: Falling M2SL + Negative T10Y2Y + High TEDRATE
    elif m2sl_growth < 2 and t10y2y_latest < 0 and tedrate_latest > 0.5:
        print("ALERT: Bearish Convergence Detected!")
        print("Falling M2SL (<2%), Negative T10Y2Y (<0), and High TEDRATE (>0.5%) signal NASDAQ downside.")
        print("Action: Short NASDAQ (e.g., SQQQ ETF) or buy NDX puts.")

    # Divergence: Rising M2SL vs. Negative T10Y2Y
    elif m2sl_growth > 5 and t10y2y_latest < 0:
        print("ALERT: Divergence Detected!")
        print("Rising M2SL (>5%) suggests bullishness, but Negative T10Y2Y (<0) warns of recession risks.")
        print("Action: Tighten stop-losses on NASDAQ longs; hedge with NDX puts.")

    # Divergence: Rising M2SL vs. High TEDRATE
    elif m2sl_growth > 5 and tedrate_latest > 0.5:
        print("ALERT: Divergence Detected!")
        print("Rising M2SL (>5%) supports NASDAQ, but High TEDRATE (>0.5%) signals financial stress.")
        print("Action: Reduce NASDAQ exposure; consider VIX calls (e.g., VXX).")

    # Divergence: Low VIXCLS vs. Negative T10Y2Y
    elif vixcls_latest < 15 and t10y2y_latest < 0:
        print("ALERT: Divergence Detected!")
        print("Low VIXCLS (<15) suggests complacency, but Negative T10Y2Y (<0) warns of risks.")
        print("Action: Avoid aggressive NASDAQ longs; consider protective NDX puts.")

    else:
        print("No significant convergence or divergence detected. Monitor trends.")
except Exception as e:
    print(f"Error analyzing convergence/divergence: {e}")

# Calculate Pearson correlations
try:
    correlations_nasdaq = df.corr()['NASDAQ Composite Index'].drop('NASDAQ Composite Index', errors='ignore')
    print("\nCorrelations with NASDAQ Composite Index:")
    print(correlations_nasdaq)
except Exception as e:
    print(f"Error calculating correlations: {e}")

# Normalize data for visualization
df_normalized = (df - df.min()) / (df.max() - df.min())

# Create interactive Plotly chart
fig = go.Figure()
for column in df_normalized.columns:
    fig.add_trace(go.Scatter(
        x=df_normalized.index,
        y=df_normalized[column],
        mode='lines',
        name=column,
        hovertemplate='%{x|%Y-%m-%d}<br>%{y:.2f}<extra>%{fullData.name}</extra>'
    ))

fig.update_layout(
    title='Normalized Time Series: NASDAQ and FRED Indicators',
    xaxis_title='Date',
    yaxis_title='Normalized Value',
    hovermode='x unified',
    showlegend=True,
    template='plotly_dark'  # Optional: dark theme for better visibility
)

# Display the interactive chart
fig.show()

# Optionally save as HTML
try:
    fig.write_html('market_correlations.html')
    print("Interactive chart saved as 'market_correlations.html'")
except Exception as e:
    print(f"Error saving HTML chart: {e}")