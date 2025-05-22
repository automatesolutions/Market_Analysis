# stock_performance.py
import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Ask the user for the instrument ticker
ticker = input("Enter the stock ticker symbol (default: TSLA): ").strip().upper()
if not ticker:
    ticker = "TSLA"

# Step 2: Fetch 3 years of daily historical data
print(f"Fetching 5 years of daily data for {ticker}...")
data = yf.download(ticker, period="5y", interval="1d")

# Step 3: Calculate performance metrics
def calculate_performance(df, days, label):
    df[label] = (df['Close'] - df['Close'].shift(days)) / df['Close'].shift(days) * 100

# Apply calculations
calculate_performance(data, 252, "Yearly Performance (%)")
calculate_performance(data, 126, "6M Performance (%)")
calculate_performance(data, 63, "3M Performance (%)")
calculate_performance(data, 21, "1M Performance (%)")
calculate_performance(data, 5, "1W Performance (%)")

# Drop rows with NaN values (first N days won't have performance data)
data = data.dropna()

# Step 4: Display the first few rows of the data
print("\nFirst few rows of the data with performance metrics:")
print(data.head())

# Step 5: Plot separate subplots for each performance metric
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

metrics = {
    "Yearly Performance (%)": "green",
    "6M Performance (%)": "blue",
    "3M Performance (%)": "orange",
    "1M Performance (%)": "red",
    "1W Performance (%)": "purple"
}

for ax, (metric, color) in zip(axes, metrics.items()):
    ax.plot(data.index, data[metric], label=metric, color=color)
    ax.set_title(metric, fontsize=14)
    ax.set_ylabel("Performance (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)  # Add horizontal line at 0%
    ax.legend()

plt.xlabel("Date", fontsize=12)
plt.suptitle(f"{ticker} Performance on a Daily Basis (Last 5 Years)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Step 6: Save the data to a CSV file inside a folder
output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\Historical_ML\Data"
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

csv_filename = os.path.join(output_dir, f"{ticker}_5y_performance_metrics.csv")
data.to_csv(csv_filename)
print(f"\nData saved to {csv_filename}")

