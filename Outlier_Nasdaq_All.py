import io
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import time

# Step 1: Fetch NASDAQ Stock Tickers via Alpha Vantage API
api_key = "5URFSW0JFPSROO2D"  # Replace with your Alpha Vantage API key
url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}"

print("Fetching NASDAQ tickers from Alpha Vantage...")
response = requests.get(url)

if response.status_code == 200:
    nasdaq_data = pd.read_csv(io.StringIO(response.text))
    
    # Check if the 'exchange' column exists
    if 'exchange' in nasdaq_data.columns:
        nasdaq_tickers = nasdaq_data[nasdaq_data['exchange'] == 'NASDAQ']['symbol'].tolist()
    else:
        nasdaq_tickers = nasdaq_data['symbol'].tolist()

    print(f"Number of NASDAQ tickers found: {len(nasdaq_tickers)}")
else:
    print(f"Failed to fetch data, status code: {response.status_code}")
    nasdaq_tickers = []

# Step 2: Fetch historical data for NASDAQ stocks using Yahoo Finance (Batch Processing)
def fetch_yahoo_data(tickers, batch_size=50):
    """
    Fetch historical stock data from Yahoo Finance in batches to avoid request failures.
    """
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

        time.sleep(2)  # Avoid overloading the Yahoo Finance API
    return all_data

if nasdaq_tickers:
    print("Fetching historical stock data from Yahoo Finance...")
    stock_data = fetch_yahoo_data(nasdaq_tickers)

    # Step 3: Calculate performance metrics
    print("Calculating performance metrics...")
    performance = {}

    for ticker, df in stock_data.items():
        try:
            if len(df) < 126:  # Ensure there is enough data
                print(f"Skipping {ticker} (Insufficient data)")
                continue

            performance[ticker] = {
                "1y": (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100,
                "6m": (df["Close"].iloc[-1] - df["Close"].iloc[-126]) / df["Close"].iloc[-126] * 100,
                "3m": (df["Close"].iloc[-1] - df["Close"].iloc[-63]) / df["Close"].iloc[-63] * 100,
                "1m": (df["Close"].iloc[-1] - df["Close"].iloc[-21]) / df["Close"].iloc[-21] * 100,
                "1w": (df["Close"].iloc[-1] - df["Close"].iloc[-5]) / df["Close"].iloc[-5] * 100,
            }
        except Exception as e:
            print(f"Skipping {ticker} due to error: {e}")

    # Convert to DataFrame for easier analysis
    performance_df = pd.DataFrame(performance).T
    print("\nPerformance Metrics:\n", performance_df.head())

    # Step 4: Create 4-Quadrant Charts with Annotations
    print("\nCreating 4-quadrant charts with annotations...")
    quadrant_pairs = [
        ("1y", "6m"),
        ("6m", "3m"),
        ("3m", "1m"),
        ("1m", "1w"),
    ]

    for x, y in quadrant_pairs:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=performance_df, x=x, y=y)

        # Annotate each point with the stock ticker
        for index, row in performance_df.iterrows():
            plt.text(row[x], row[y], index, fontsize=9, ha="center", va="bottom")

        plt.axhline(0, color="black", linestyle="--")  # Horizontal line at y=0
        plt.axvline(0, color="black", linestyle="--")  # Vertical line at x=0
        plt.title(f"{x} vs {y} Performance")
        plt.xlabel(f"{x} Performance (%)")
        plt.ylabel(f"{y} Performance (%)")
        plt.show()

    # Step 5: Identify Outliers using Z-scores
    print("\nIdentifying outliers...")
    z_scores = performance_df.apply(zscore)  # Calculate Z-scores for each performance metric
    outliers = z_scores[(z_scores.abs() > 2).any(axis=1)]  # Outliers have Z-scores > 2 or < -2
    print("\nOutliers:\n", outliers)

    # Step 6: Save Results to CSV Files
    print("\nSaving results to CSV files...")
    performance_df.to_csv("nasdaq_performance_metrics.csv")  # Save performance metrics
    outliers.to_csv("nasdaq_outliers.csv")  # Save outliers
    print("Results saved to 'nasdaq_performance_metrics.csv' and 'nasdaq_outliers.csv'.")

else:
    print("No NASDAQ tickers were fetched. Analysis cannot be performed.")

