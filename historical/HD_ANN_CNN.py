import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Step 1: Fetch and preprocess data
ticker = input("Enter the stock ticker symbol (default: TSLA): ").strip().upper()
if not ticker:
    ticker = "TSLA"

print(f"Fetching 5 years of daily data for {ticker}...")
data = yf.download(ticker, period="5y", interval="1d")

# Calculate performance metrics
def calculate_performance(df, days, label):
    df[label] = (df['Close'] - df['Close'].shift(days)) / df['Close'].shift(days) * 100

calculate_performance(data, 252, "Yearly Performance (%)")
calculate_performance(data, 126, "6M Performance (%)")
calculate_performance(data, 63, "3M Performance (%)")
calculate_performance(data, 21, "1M Performance (%)")
calculate_performance(data, 5, "1W Performance (%)")

# Drop rows with NaN values
data = data.dropna()

# Step 2: Define Risk Profile based on Volatility
data['Volatility'] = data['Close'].pct_change().rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
data['Risk Level'] = pd.cut(data['Volatility'], bins=[-np.inf, 0.1, 0.2, np.inf], labels=[0, 1, 2])  # 0: Low, 1: Medium, 2: High

# Drop rows with NaN values (due to rolling calculations)
data = data.dropna()

# Step 3: Prepare data for Deep Learning
features = ["Yearly Performance (%)", "6M Performance (%)", "3M Performance (%)", "1M Performance (%)", "1W Performance (%)"]
X = data[features].values
y = to_categorical(data['Risk Level'].values)  # One-hot encode risk levels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for CNN (if using CNN)
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 4: Build a Deep Learning Model
def build_ann_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # 3 output classes for risk levels
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 output classes for risk levels
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Choose ANN or CNN
model_type = input("Choose model type (ANN or CNN): ").strip().upper()
if model_type == "CNN":
    model = build_cnn_model((X_train_cnn.shape[1], 1))
    X_train_final, X_test_final = X_train_cnn, X_test_cnn
else:
    model = build_ann_model((X_train.shape[1],))
    X_train_final, X_test_final = X_train, X_test

# Step 5: Train the Model
history = model.fit(X_train_final, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 6: Evaluate the Model
loss, accuracy = model.evaluate(X_test_final, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 7: Predict Risk Profile
predictions = model.predict(X_test_final)
predicted_risk_levels = np.argmax(predictions, axis=1)

# Display predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Actual: {np.argmax(y_test[i])}, Predicted: {predicted_risk_levels[i]}")

# Classification Report
print("\nClassification Report:")
print(classification_report(np.argmax(y_test, axis=1), predicted_risk_levels))

# Step 8: Create output directory and save results
output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\Historical_ML\Data"
os.makedirs(output_dir, exist_ok=True)

# Save the Model
model_path = os.path.join(output_dir, f"{ticker}_risk_profile_model_{model_type.lower()}.keras")
model.save(model_path)
print(f"Model saved to {model_path}")

# Save predictions and actual values
results_df = pd.DataFrame({
    'Actual_Risk_Level': np.argmax(y_test, axis=1),
    'Predicted_Risk_Level': predicted_risk_levels,
    'Low_Risk_Probability': predictions[:, 0],
    'Medium_Risk_Probability': predictions[:, 1],
    'High_Risk_Probability': predictions[:, 2]
})
predictions_path = os.path.join(output_dir, f"{ticker}_risk_predictions_{model_type.lower()}.csv")
results_df.to_csv(predictions_path)
print(f"Predictions saved to {predictions_path}")

# Save training history
history_df = pd.DataFrame({
    'Train_Accuracy': history.history['accuracy'],
    'Val_Accuracy': history.history['val_accuracy'],
    'Train_Loss': history.history['loss'],
    'Val_Loss': history.history['val_loss']
})
history_path = os.path.join(output_dir, f"{ticker}_training_history_{model_type.lower()}.csv")
history_df.to_csv(history_path)
print(f"Training history saved to {history_path}")

# Step 9: Visualize Training History
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 10: Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
