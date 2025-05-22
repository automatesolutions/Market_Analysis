import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam

# Step 1: Ask the user for the instrument ticker and date range
ticker = input("Enter the stock ticker symbol (default: TSLA): ").strip().upper()
if not ticker:
    ticker = "TSLA"
years = int(input("Enter number of years of data to fetch (default: 5): ").strip() or 5)

# Step 2: Fetch the historical data
try:
    print(f"Fetching {years} years of daily data for {ticker}...")
    data = yf.download(ticker, period=f"{years}y", interval="1d")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Step 3: Calculate performance metrics
def calculate_performance(df, days, label):
    df[label] = (df['Close'] - df['Close'].shift(days)) / df['Close'].shift(days) * 100

# Apply calculations for various time periods
calculate_performance(data, 252, "Yearly Performance (%)")  # 1 year = 252 trading days
calculate_performance(data, 126, "6M Performance (%)")  # 6 months = 126 trading days
calculate_performance(data, 63, "3M Performance (%)")  # 3 months = 63 trading days
calculate_performance(data, 21, "1M Performance (%)")  # 1 month = 21 trading days
calculate_performance(data, 5, "1W Performance (%)")  # 1 week = 5 trading days

# Drop rows with NaN values (first N days won't have performance data)
data = data.dropna()

# Step 4: Prepare the data for the GRU model
# Use the current performance metrics as input features
X = data[["Yearly Performance (%)", "6M Performance (%)", "3M Performance (%)", "1M Performance (%)", "1W Performance (%)"]]

# Now create labels for the performance metrics for the next periods
y = pd.DataFrame({
    "Yearly Performance (%)": data["Yearly Performance (%)"].shift(-252),  # Predict 1 year ahead
    "6M Performance (%)": data["6M Performance (%)"].shift(-126),  # Predict 6 months ahead
    "3M Performance (%)": data["3M Performance (%)"].shift(-63),  # Predict 3 months ahead
    "1M Performance (%)": data["1M Performance (%)"].shift(-21),  # Predict 1 month ahead
    "1W Performance (%)": data["1W Performance (%)"].shift(-5)  # Predict 1 week ahead
})

# Drop rows with NaN values caused by shifting
X = X[:-252]  # Drop the last 252 rows since they don't have the target values for 1 year
y = y[:-252]  # Drop the last 252 rows

# Step 5: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Reshape the data into 3D for the GRU model
# GRU model expects input shape: (samples, timesteps, features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])  # 1 timestep per sample

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Step 8: Define the GRU model
model = Sequential()
model.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='linear'))  # 5 outputs for the 5 return predictions

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Step 9: Train the model
print("Training the GRU model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Step 10: Evaluate the model
print("Evaluating the model...")
loss = model.evaluate(X_test, y_test)
print(f"Model loss: {loss}")

# Step 11: Make predictions
predictions = model.predict(X_test)

# Step 12: Visualize the predictions vs actual values
plt.figure(figsize=(14, 8))
for i, metric in enumerate(y.columns):
    plt.subplot(3, 2, i+1)
    plt.plot(y_test.index, y_test.iloc[:, i], label="Actual")
    plt.plot(y_test.index, predictions[:, i], label="Predicted", linestyle="--")
    plt.title(f"{metric} Prediction vs Actual")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Step 13: Save the predictions and model
# Create the output directory if it doesn't exist
output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\Historical_ML\Data"
os.makedirs(output_dir, exist_ok=True)

# Save the predictions
output_file = os.path.join(output_dir, f"{ticker}_predictions_GRU.csv")
predictions_df = pd.DataFrame(predictions, columns=y.columns, index=y_test.index)
predictions_df.to_csv(output_file)
print(f"Predictions saved to {output_file}")

