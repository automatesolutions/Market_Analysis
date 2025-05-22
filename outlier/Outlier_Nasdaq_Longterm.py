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
logging.basicConfig(filename='longterm_debug.log', level=logging.INFO)

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

# Step 3: Pre-filter tickers for 1M+ 10-day avg volume and 10B+ market cap
def filter_high_volume_tickers(tickers, batch_size=50, min_volume=1000000, min_market_cap=10e9):
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
    print("Pre-filtering tickers for 1M+ 10-day avg volume and 10B+ market cap...")
    nasdaq_tickers = filter_high_volume_tickers(
        nasdaq_tickers,
        batch_size=50,
        min_volume=1000000,          # 10-day avg volume > 1 million shares
        min_market_cap=10e9          # Market cap > $10 billion
    )
    print(f"Number of NASDAQ tickers that are active with 1M+ 10-day avg volume and 10B+ market cap: {len(nasdaq_tickers)}")

# Step 4: Fetch historical data for filtered tickers
def fetch_yahoo_data(tickers, batch_size=50):
    """Fetch historical stock data from Yahoo Finance in batches."""
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Fetching batch {i//batch_size + 1}/{(len(tickers)//batch_size) + 1}...")
        try:
            batch_data = yf.download(batch, period="2y", interval="1d", group_by="ticker")  # Extended to 2 years
            for ticker in batch:
                if ticker in batch_data and "Close" in batch_data[ticker]:
                    df = batch_data[ticker][["Close"]]
                    print(f"  Fetched {len(df)} days of data for {ticker}")  # Debug: Log number of data points
                    all_data[ticker] = df
                else:
                    print(f"  No data fetched for {ticker}")
                    logging.info(f"No data fetched for {ticker}")
        except Exception as e:
            print(f"Error fetching batch {i//batch_size + 1}: {e}")
            logging.info(f"Error fetching batch {i//batch_size + 1}: {e}")
        time.sleep(1)  # Delay to avoid rate limits
    return all_data

if nasdaq_tickers:
    print("Fetching historical stock data from Yahoo Finance...")
    stock_data = fetch_yahoo_data(nasdaq_tickers)

    # Step 5: Calculate performance metrics (1-year, 6-month)
    print("Calculating performance metrics...")
    performance = {}
    for ticker, df in stock_data.items():
        try:
            if len(df) < 252:  # Ensure at least 1 year (252 trading days) of data
                print(f"Skipping {ticker} (Insufficient data: {len(df)} days)")
                logging.info(f"Skipping {ticker} (Insufficient data: {len(df)} days)")
                continue
            performance[ticker] = {
                "1y": (df["Close"].iloc[-1] - df["Close"].iloc[-252]) / df["Close"].iloc[-252] * 100,
                "6m": (df["Close"].iloc[-1] - df["Close"].iloc[-126]) / df["Close"].iloc[-126] * 100,
            }
        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")
            logging.info(f"Skipping {ticker} due to error: {e}")

    performance_df = pd.DataFrame(performance).T
    print("\nPerformance Metrics:\n", performance_df.head())

    # Step 6: Create interactive 1-year vs 6-month quadrant chart with Plotly
    if not performance_df.empty:
        print("\nCreating interactive 1-year vs 6-month quadrant chart (zoomable)...")
        performance_df_reset = performance_df.reset_index().rename(columns={'index': 'Ticker'})
        
        # Create scatter plot with Plotly
        fig = px.scatter(
            performance_df_reset,
            x="1y",
            y="6m",
            text="Ticker",
            title="1-Year vs 6-Month Performance (Interactive)",
            labels={"1y": "1-Year Performance (%)", "6m": "6-Month Performance (%)"}
        )
        
        # Add horizontal and vertical lines at 0
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        # Update text position and layout
        fig.update_traces(textposition="top center")
        fig.update_layout(
            width=1200,
            height=800,
            showlegend=False,
            hovermode="closest",
            dragmode="zoom",  # Default mode is zoom
            template="plotly_white"
        )
        
        # Show the plot (opens in browser or default viewer)
        fig.show()

        # Step 7: Identify outliers using Z-scores
        print("\nIdentifying outliers...")
        z_scores = performance_df.apply(zscore)
        outliers = z_scores[(z_scores.abs() > 2).any(axis=1)]
        print("\nOutliers:\n", outliers)

        # Step 8: Save results to CSV files
        print("\nSaving results to CSV files...")
        output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\outlier\data"
        os.makedirs(output_dir, exist_ok=True)
        performance_df.to_csv(os.path.join(output_dir, "nasdaq_longterm_performance_metrics.csv"))
        outliers.to_csv(os.path.join(output_dir, "nasdaq_longterm_outliers.csv"))
        print(f"Results saved to:\n- {os.path.join(output_dir, 'nasdaq_longterm_performance_metrics.csv')}\n- {os.path.join(output_dir, 'nasdaq_longterm_outliers.csv')}")
    else:
        print("No performance metrics calculated. Skipping plotting, outlier detection, and CSV saving.")
else:
    print("No NASDAQ tickers were fetched. Analysis cannot be performed.")