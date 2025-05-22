import io
import requests
import yfinance as yf
import pandas as pd
import plotly.express as px  # Use Plotly for interactive plotting
from scipy.stats import zscore
import time
import os
import logging
from dotenv import load_dotenv

# Load environment variables with explicit path
env_path = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\outlier\.env"
load_dotenv(env_path)
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
print(f"Loaded API Key: {api_key}")  # Debug: Print the API key
if not api_key:
    raise ValueError("Alpha Vantage API key not found in environment variables.")

# Configure logging
logging.basicConfig(filename='swing_debug.log', level=logging.INFO)

# Step 1: Fetch NASDAQ Stock Tickers via Alpha Vantage API
url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}"

print("Fetching NASDAQ tickers from Alpha Vantage...")
response = requests.get(url)

if response.status_code == 200:
    nasdaq_data = pd.read_csv(io.StringIO(response.text))
    
    # Filter for NASDAQ and active tickers
    if 'exchange' in nasdaq_data.columns and 'status' in nasdaq_data.columns:
        nasdaq_tickers = nasdaq_data[
            (nasdaq_data['exchange'] == 'NASDAQ') & 
            (nasdaq_data['status'] == 'Active')
        ]['symbol'].tolist()
    else:
        nasdaq_tickers = nasdaq_data['symbol'].tolist()

    print(f"Number of active NASDAQ tickers found: {len(nasdaq_tickers)}")
else:
    print(f"Failed to fetch data, status code: {response.status_code}")
    nasdaq_tickers = []

# Step 2: Filter valid tickers
def filter_valid_tickers(tickers):
    """Filter out tickers with non-alphabetic characters or unusual lengths."""
    valid = []
    for t in tickers:
        if pd.isna(t):
            continue
        t = str(t)
        if t.isalpha() and 1 < len(t) <= 5:
            valid.append(t)
    return valid

if nasdaq_tickers:
    print("Filtering valid tickers...")
    nasdaq_tickers = filter_valid_tickers(nasdaq_tickers)
    print(f"Number of valid active NASDAQ tickers: {len(nasdaq_tickers)}")

# Step 3: Pre-filter tickers for 1M+ 10-day avg volume and 2B+ market cap
def filter_high_volume_tickers(tickers, batch_size=50, min_volume=1000000, min_market_cap=2e9):
    """Filter tickers based on 10-day average volume and market cap."""
    filtered_tickers = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Screening batch {i//batch_size + 1}/{(len(tickers)//batch_size) + 1} for volume and market cap...")
        for ticker in batch:
            print(f"Processing ticker: {ticker}")  # Debug: Track individual ticker processing
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                avg_volume = info.get('averageDailyVolume10Day', 0)  # 10-day avg volume
                market_cap = info.get('marketCap', 0)
                print(f"  {ticker}: Avg Volume = {avg_volume}, Market Cap = {market_cap/1e9:.2f}B")  # Debug: Print ticker details
                if avg_volume >= min_volume and market_cap >= min_market_cap:
                    filtered_tickers.append(ticker)
                else:
                    print(f"  Skipping {ticker} (Avg Volume: {avg_volume}, Market Cap: {market_cap/1e9:.2f}B)")
            except Exception as e:
                print(f"  Error screening {ticker}: {e}")
                logging.info(f"Error for {ticker}: {e}")
        time.sleep(1)  # Increased delay to avoid rate limits
    return filtered_tickers

if nasdaq_tickers:
    print("Pre-filtering tickers for 1M+ 10-day avg volume and 2B+ market cap...")
    nasdaq_tickers = filter_high_volume_tickers(
        nasdaq_tickers,
        batch_size=50,
        min_volume=1000000,          # 10-day avg volume > 1 million shares
        min_market_cap=2e9           # Market cap > $2 billion
    )
    print(f"Number of NASDAQ tickers that are active with 1M+ 10-day avg volume and 2B+ market cap: {len(nasdaq_tickers)}")

# Step 4: Fetch historical data for filtered tickers
def fetch_yahoo_data(tickers, batch_size=50):
    """Fetch historical stock data from Yahoo Finance in batches."""
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Fetching batch {i//batch_size + 1}/{(len(tickers)//batch_size) + 1}...")
        try:
            batch_data = yf.download(batch, period="1y", interval="1d", group_by="ticker")
            for ticker in batch:
                if ticker in batch_data and "Close" in batch_data[ticker]:
                    all_data[ticker] = batch_data[ticker][["Close"]]
        except Exception as e:
            print(f"Error fetching batch {i//batch_size + 1}: {e}")
        time.sleep(1)  # Delay to avoid rate limits
    return all_data

if nasdaq_tickers:
    print("Fetching historical stock data from Yahoo Finance...")
    stock_data = fetch_yahoo_data(nasdaq_tickers)

    # Step 5: Calculate performance metrics (6-month, 3-month, 1-month)
    print("Calculating performance metrics...")
    performance = {}
    for ticker, df in stock_data.items():
        try:
            if len(df) < 126:  # Ensure at least 6 months (126 trading days) of data
                print(f"Skipping {ticker} (Insufficient data)")
                continue
            performance[ticker] = {
                "6m": (df["Close"].iloc[-1] - df["Close"].iloc[-126]) / df["Close"].iloc[-126] * 100,
                "3m": (df["Close"].iloc[-1] - df["Close"].iloc[-63]) / df["Close"].iloc[-63] * 100,
                "1m": (df["Close"].iloc[-1] - df["Close"].iloc[-21]) / df["Close"].iloc[-21] * 100,
            }
        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")

    performance_df = pd.DataFrame(performance).T
    print("\nPerformance Metrics:\n", performance_df.head())

    # Step 6: Create interactive quadrant charts with Plotly
    performance_df_reset = performance_df.reset_index().rename(columns={'index': 'Ticker'})
    
    # Chart 1: 6-month vs 3-month
    print("\nCreating interactive 6-month vs 3-month quadrant chart (zoomable)...")
    fig1 = px.scatter(
        performance_df_reset,
        x="6m",
        y="3m",
        text="Ticker",
        title="6-Month vs 3-Month Performance (Interactive)",
        labels={"6m": "6-Month Performance (%)", "3m": "3-Month Performance (%)"}
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="black")
    fig1.add_vline(x=0, line_dash="dash", line_color="black")
    fig1.update_traces(textposition="top center")
    fig1.update_layout(
        width=1200,
        height=800,
        showlegend=False,
        hovermode="closest",
        dragmode="zoom",
        template="plotly_white"
    )
    fig1.show()

    # Chart 2: 3-month vs 1-month
    print("\nCreating interactive 3-month vs 1-month quadrant chart (zoomable)...")
    fig2 = px.scatter(
        performance_df_reset,
        x="3m",
        y="1m",
        text="Ticker",
        title="3-Month vs 1-Month Performance (Interactive)",
        labels={"3m": "3-Month Performance (%)", "1m": "1-Month Performance (%)"}
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="black")
    fig2.add_vline(x=0, line_dash="dash", line_color="black")
    fig2.update_traces(textposition="top center")
    fig2.update_layout(
        width=1200,
        height=800,
        showlegend=False,
        hovermode="closest",
        dragmode="zoom",
        template="plotly_white"
    )
    fig2.show()

    # Step 7: Identify outliers using Z-scores
    print("\nIdentifying outliers...")
    z_scores = performance_df.apply(zscore)
    outliers = z_scores[(z_scores.abs() > 2).any(axis=1)]
    print("\nOutliers:\n", outliers)

    # Step 8: Save results to CSV files
    print("\nSaving results to CSV files...")
    output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\outlier\data"
    os.makedirs(output_dir, exist_ok=True)
    performance_df.to_csv(os.path.join(output_dir, "nasdaq_swing_performance_metrics.csv"))
    outliers.to_csv(os.path.join(output_dir, "nasdaq_swing_outliers.csv"))
    print(f"Results saved to:\n- {os.path.join(output_dir, 'nasdaq_swing_performance_metrics.csv')}\n- {os.path.join(output_dir, 'nasdaq_swing_outliers.csv')}")
else:
    print("No NASDAQ tickers were fetched. Analysis cannot be performed.")