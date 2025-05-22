import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm

# Step 1: Ask the user for the instrument ticker
ticker = input("Enter the stock ticker symbol (default: TSLA): ").strip().upper()
if not ticker:
    ticker = "TSLA"

# Step 2: Fetch 5 years of daily historical data
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

# Step 4: Calculate additional metrics (Sharpe Ratio, Sortino Ratio, CVaR, Standard Deviation)
def calculate_metrics(returns, risk_free_rate=0.0):
    mean_return = returns.mean()
    std_dev = returns.std()
    
    # Calculate downside standard deviation (only for negative returns)
    downside_returns = returns[returns < 0]
    downside_std_dev = downside_returns.std() if not downside_returns.empty else np.nan
    
    # Ensure downside_std_dev is a scalar
    if isinstance(downside_std_dev, pd.Series):
        downside_std_dev = downside_std_dev.item()
    
    # Sharpe Ratio
    sharpe_ratio = (mean_return - risk_free_rate) / std_dev
    
    # Sortino Ratio
    if pd.notna(downside_std_dev) and downside_std_dev != 0:
        sortino_ratio = (mean_return - risk_free_rate) / downside_std_dev
    else:
        sortino_ratio = np.nan
    
    # CVaR (Conditional Value at Risk)
    cvar = returns[returns <= returns.quantile(0.05)].mean()
    
    return {
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "CVaR (5%)": cvar,
        "Standard Deviation": std_dev
    }

# Calculate metrics for each timeframe
timeframes = {
    "Yearly": data["Yearly Performance (%)"] / 100,
    "6M": data["6M Performance (%)"] / 100,
    "3M": data["3M Performance (%)"] / 100,
    "1M": data["1M Performance (%)"] / 100,
    "1W": data["1W Performance (%)"] / 100
}

metrics_summary = {}
for timeframe, returns in timeframes.items():
    metrics_summary[timeframe] = calculate_metrics(returns)

# Print metrics summary before plotting
print("\nMetrics Summary:")
print(metrics_summary)

# Convert metrics summary to a DataFrame for better visualization
metrics_df = pd.DataFrame(metrics_summary).T
print("\nMetrics DataFrame:")
print(metrics_df)

# Ensure columns are numeric before plotting
metrics_df = metrics_df.apply(pd.to_numeric, errors='coerce')

# Check for any missing (NaN) values in the DataFrame
if metrics_df.isnull().values.any():
    print("\nWarning: The DataFrame contains NaN values. These will be ignored in the plot.")

# Step 5: Visual Summary
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Try plotting only if data is valid
if not metrics_df["Sharpe Ratio"].isnull().all():
    metrics_df["Sharpe Ratio"].plot(kind="bar", ax=axes[0, 0], title="Sharpe Ratio", color="blue")
else:
    print("Sharpe Ratio has no valid data to plot.")

if not metrics_df["Sortino Ratio"].isnull().all():
    metrics_df["Sortino Ratio"].plot(kind="bar", ax=axes[0, 1], title="Sortino Ratio", color="green")
else:
    print("Sortino Ratio has no valid data to plot.")

if not metrics_df["CVaR (5%)"].isnull().all():
    metrics_df["CVaR (5%)"].plot(kind="bar", ax=axes[1, 0], title="CVaR (5%)", color="red")
else:
    print("CVaR (5%) has no valid data to plot.")

if not metrics_df["Standard Deviation"].isnull().all():
    metrics_df["Standard Deviation"].plot(kind="bar", ax=axes[1, 1], title="Standard Deviation", color="purple")
else:
    print("Standard Deviation has no valid data to plot.")

for ax in axes.flatten():
    ax.grid(True, linestyle="--", alpha=0.7)

plt.suptitle(f"{ticker} Risk and Performance Metrics", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Step 6: Decision-Making for each timeframe
def make_decision(metrics):
    # Ensure we extract scalar values
    sharpe = metrics.get("Sharpe Ratio", np.nan)
    sortino = metrics.get("Sortino Ratio", np.nan)
    cvar = metrics.get("CVaR (5%)", np.nan)
    std_dev = metrics.get("Standard Deviation", np.nan)
    
    # Convert to scalar if they are Series (just in case)
    if isinstance(sharpe, pd.Series):
        sharpe = sharpe.iloc[0]
    if isinstance(sortino, pd.Series):
        sortino = sortino.iloc[0]
    if isinstance(cvar, pd.Series):
        cvar = cvar.iloc[0]
    if isinstance(std_dev, pd.Series):
        std_dev = std_dev.iloc[0]
    
    # Now we can check for NaN values and make the decision
    if pd.notna(sharpe) and pd.notna(sortino) and pd.notna(cvar) and pd.notna(std_dev):
        if sharpe > 1 and sortino > 1 and cvar > -0.1 and std_dev < 0.2:
            return "Good for investment (Low risk, High return)"
        elif sharpe > 0.5 and sortino > 0.5 and cvar > -0.2 and std_dev < 0.3:
            return "Moderate risk, Consider investment"
        else:
            return "High risk, Avoid investment"
    else:
        return "Invalid or missing metrics data"

# Make the decision for each timeframe
for timeframe in timeframes:
    decision = make_decision(metrics_summary.get(timeframe, {}))
    print(f"\nDecision for {timeframe}: {decision}")

# Step 7: Save the data to a CSV file inside a folder
output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\Historical_ML\Data"
os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

# Save performance metrics data
metrics_file = os.path.join(output_dir, f"{ticker}_5y_performance_metrics.csv")
data.to_csv(metrics_file)

# Save risk metrics data
risk_metrics_file = os.path.join(output_dir, f"{ticker}_risk_metrics.csv")
metrics_df.to_csv(risk_metrics_file)

print(f"\nData saved to:\n- {metrics_file}\n- {risk_metrics_file}")


