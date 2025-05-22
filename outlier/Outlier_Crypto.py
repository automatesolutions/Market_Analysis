import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os  # Add os import

# Step 1: Fetch Cryptocurrency List from CoinGecko API
print("Fetching cryptocurrency list from CoinGecko API...")
coingecko_url = "https://api.coingecko.com/api/v3/coins/markets"
params = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 30,  # Fetch top 30 cryptos
    "page": 1,
    "sparkline": False
}
response = requests.get(coingecko_url, params=params)

# Check if the response was successful
if response.status_code == 200:
    crypto_data = response.json()
    crypto_tickers = [crypto['symbol'].upper() + "-USD" for crypto in crypto_data]  # Format for Yahoo Finance
    print(f"Fetched {len(crypto_tickers)} cryptocurrency tickers: {crypto_tickers}")
else:
    print(f"Failed to fetch data, status code: {response.status_code}")
    crypto_tickers = []  # Fallback if the request fails

# Step 2: Fetch historical data for all cryptocurrencies
if crypto_tickers:
    print("Fetching historical data from Yahoo Finance...")
    data = yf.download(crypto_tickers, period="1y", interval="1d", group_by='ticker')

    # Step 3: Calculate performance metrics
    print("Calculating performance metrics...")
    performance = {}

    for ticker in crypto_tickers:
        try:
            df = data[ticker]
            performance[ticker] = {
                "1y": (df['Close'][-1] - df['Close'][0]) / df['Close'][0] * 100,  # Yearly performance
                "6m": (df['Close'][-1] - df['Close'][-126]) / df['Close'][-126] * 100,  # 6-month performance
                "3m": (df['Close'][-1] - df['Close'][-63]) / df['Close'][-63] * 100,  # 3-month performance
                "1m": (df['Close'][-1] - df['Close'][-21]) / df['Close'][-21] * 100,  # 1-month performance
                "1w": (df['Close'][-1] - df['Close'][-5]) / df['Close'][-5] * 100,  # 1-week performance
            }
        except KeyError:
            print(f"Skipping {ticker} due to missing data.")

    # Convert to DataFrame for easier analysis
    performance_df = pd.DataFrame(performance).T
    print("\nPerformance Metrics:")
    print(performance_df)

    # Step 4: Create 4-Quadrant Charts with Annotations
    print("\nCreating 4-quadrant charts with annotations...")
    quadrant_pairs = [
        ("1y", "6m"),
        ("6m", "3m"),
        ("3m", "1m"),
        ("1m", "1w")
    ]

    for x, y in quadrant_pairs:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=performance_df, x=x, y=y)

        # Annotate each point with the crypto ticker
        for index, row in performance_df.iterrows():
            plt.text(row[x], row[y], index, fontsize=9, ha='center', va='bottom')

        plt.axhline(0, color='black', linestyle='--')  # Horizontal line at y=0
        plt.axvline(0, color='black', linestyle='--')  # Vertical line at x=0
        plt.title(f"{x} vs {y} Performance (Cryptos)")
        plt.xlabel(f"{x} Performance (%)")
        plt.ylabel(f"{y} Performance (%)")
        plt.show()

    # Step 5: Identify Outliers using Z-scores
    print("\nIdentifying outliers...")
    z_scores = performance_df.apply(zscore)  # Calculate Z-scores for each performance metric
    outliers = z_scores[(z_scores.abs() > 2).any(axis=1)]  # Outliers have Z-scores > 2 or < -2
    print("\nOutliers:")
    print(outliers)

    # Step 6: Save Results to CSV Files
    print("\nSaving results to CSV files...")
    # Create the output directory if it doesn't exist
    output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\outlier\data"
    os.makedirs(output_dir, exist_ok=True)

    # Save the files in the specified directory
    performance_df.to_csv(os.path.join(output_dir, "crypto_performance_metrics.csv"))  # Save performance metrics
    outliers.to_csv(os.path.join(output_dir, "crypto_outliers.csv"))  # Save outliers
    print(f"Results saved to:\n- {os.path.join(output_dir, 'crypto_performance_metrics.csv')}\n- {os.path.join(output_dir, 'crypto_outliers.csv')}")

else:
    print("No cryptocurrency tickers were fetched. Analysis cannot be performed.")
