import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Step 1: Define a list of major commodities tracked by Yahoo Finance
commodity_tickers = [
    "GC=F",  # Gold
    "SI=F",  # Silver
    "PL=F",  # Platinum
    "HG=F",  # Copper
    "PA=F",  # Palladium
    "CL=F",  # Crude Oil (WTI)
    "BZ=F",  # Brent Crude Oil
    "NG=F",  # Natural Gas
    "ZC=F",  # Corn
    "ZW=F",  # Wheat
    "ZS=F",  # Soybeans
    "KC=F",  # Coffee
    "CT=F",  # Cotton
    "SB=F",  # Sugar
    "OJ=F"   # Orange Juice
]

print(f"Using {len(commodity_tickers)} Commodities: {commodity_tickers}")

# Step 2: Fetch historical data for Commodities
print("Fetching historical commodity data from Yahoo Finance...")
data = yf.download(commodity_tickers, period="1y", interval="1d", group_by='ticker')

# Step 3: Calculate performance metrics
print("Calculating performance metrics...")
performance = {}

for ticker in commodity_tickers:
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

    # Annotate each point with the commodity ticker
    for index, row in performance_df.iterrows():
        plt.text(row[x], row[y], index, fontsize=9, ha='center', va='bottom')

    plt.axhline(0, color='black', linestyle='--')  # Horizontal line at y=0
    plt.axvline(0, color='black', linestyle='--')  # Vertical line at x=0
    plt.title(f"{x} vs {y} Performance (Commodities)")
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
performance_df.to_csv("commodity_performance_metrics.csv")  # Save performance metrics
outliers.to_csv("commodity_outliers.csv")  # Save outliers
print("Results saved to 'commodity_performance_metrics.csv' and 'commodity_outliers.csv'.")

print("\nAnalysis complete!")
