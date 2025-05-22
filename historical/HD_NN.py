import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
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

# Target variables: We want to predict the *future performance* (percentage return), not the future price
y = pd.DataFrame({
    "Next Year Performance (%)": pd.Series(((data['Close'].shift(-252) - data['Close']) / data['Close'] * 100).values.flatten(), index=data.index),  # 252 trading days ~ 1 year
    "Next 6 Months Performance (%)": pd.Series(((data['Close'].shift(-126) - data['Close']) / data['Close'] * 100).values.flatten(), index=data.index),  # 126 trading days ~ 6 months
    "Next 3 Months Performance (%)": pd.Series(((data['Close'].shift(-63) - data['Close']) / data['Close'] * 100).values.flatten(), index=data.index),  # 63 trading days ~ 3 months
    "Next Month Performance (%)": pd.Series(((data['Close'].shift(-21) - data['Close']) / data['Close'] * 100).values.flatten(), index=data.index),  # 21 trading days ~ 1 month
    "Next Week Performance (%)": pd.Series(((data['Close'].shift(-5) - data['Close']) / data['Close'] * 100).values.flatten(), index=data.index)  # 5 trading days ~ 1 week
})

# Concatenate X and y, drop all rows with any NaNs, then split back into X and y to ensure perfect alignment
full = pd.concat([X, y], axis=1).dropna()
print("Columns in 'full':", full.columns.tolist())  # Print column names for verification
X = full[[("Yearly Performance (%)", ""), ("6M Performance (%)", ""), ("3M Performance (%)", ""), ("1M Performance (%)", ""), ("1W Performance (%)", "")]]
y = full[["Next Year Performance (%)", "Next 6 Months Performance (%)", "Next 3 Months Performance (%)", "Next Month Performance (%)", "Next Week Performance (%)"]]

# Step 5: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Step 7: Define the Neural Network model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
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
test_dates = data.index[-len(y_test):]  # Use the correct dates for the test set
for i, metric in enumerate(y.columns):
    plt.subplot(3, 2, i+1)
    plt.plot(test_dates, y_test.iloc[:, i], label="Actual")
    plt.plot(test_dates, predictions[:, i], label="Predicted", linestyle="--")
    plt.title(f"{metric} Prediction vs Actual")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Step 12: Save the predictions and model
# Create the output directory if it doesn't exist
output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\Historical_ML\Data"
os.makedirs(output_dir, exist_ok=True)

# Save the predictions
output_file = os.path.join(output_dir, f"{ticker}_predictions_NN.csv")
predictions_df = pd.DataFrame(predictions, columns=y.columns, index=y_test.index)
predictions_df.to_csv(output_file)
print(f"Predictions saved to {output_file}")
