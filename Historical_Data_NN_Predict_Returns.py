import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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

# Step 4: Prepare input features (X) and target variables (Y)
# We will use performance metrics as input features
X = data[["Yearly Performance (%)", "6M Performance (%)", "3M Performance (%)", "1M Performance (%)", "1W Performance (%)"]]

# Target variables: We want to predict the performance for the next 5 timeframes (shifted for future periods)
y = pd.DataFrame({
    "Next Year Performance (%)": data['Close'].shift(-252),  # 252 trading days ~ 1 year
    "Next 6 Months Performance (%)": data['Close'].shift(-126),  # 126 trading days ~ 6 months
    "Next 3 Months Performance (%)": data['Close'].shift(-63),  # 63 trading days ~ 3 months
    "Next Month Performance (%)": data['Close'].shift(-21),  # 21 trading days ~ 1 month
    "Next Week Performance (%)": data['Close'].shift(-5)  # 5 trading days ~ 1 week
})

# Drop rows with NaN values caused by shifting
X = X[:-252]  # Drop last 252 rows since they don't have 1 year of future data
y = y[:-252]

# Step 5: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Step 7: Define the Neural Network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='linear'))  # 5 outputs for the 5 return predictions

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Step 8: Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Step 9: Evaluate the model
print("Evaluating the model...")
loss = model.evaluate(X_test, y_test)
print(f"Model loss: {loss}")

# Step 10: Make predictions
predictions = model.predict(X_test)

# Step 11: Visualize the predictions vs actual values
plt.figure(figsize=(14, 8))
for i, metric in enumerate(y.columns):
    plt.subplot(3, 2, i+1)
    plt.plot(y_test.index, y_test.iloc[:, i], label="Actual")  # Actual future performance
    plt.plot(y_test.index, predictions[:, i], label="Predicted", linestyle="--")  # Predicted future performance
    plt.title(f"{metric} Prediction vs Actual")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Step 12: Save the predictions and model
predictions_df = pd.DataFrame(predictions, columns=y.columns, index=y_test.index)
predictions_df.to_csv(f"{ticker}_predictions.csv")
print(f"Predictions saved to {ticker}_predictions.csv")
