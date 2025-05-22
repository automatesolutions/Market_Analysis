import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# Step 1: Ask for the stock ticker symbol
ticker = input("Enter the stock ticker symbol (default: TSLA): ").strip().upper()
if not ticker:
    ticker = "TSLA"

# Step 2: Fetch 5 years of daily historical data
print(f"Fetching 5 years of daily data for {ticker}...")
data = yf.download(ticker, period="5y", interval="1d")

# Step 3: Calculate performance metrics
def calculate_performance(df, days, label):
    df[label] = (df['Close'] - df['Close'].shift(days)) / df['Close'].shift(days) * 100

calculate_performance(data, 252, "Yearly Performance (%)")
calculate_performance(data, 126, "6M Performance (%)")
calculate_performance(data, 63, "3M Performance (%)")
calculate_performance(data, 21, "1M Performance (%)")
calculate_performance(data, 5, "1W Performance (%)")

# Drop rows with NaN values
data = data.dropna()

# Step 4: Define the trading environment using Gym
class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()

        self.data = data
        self.current_step = 0
        
        # Define action space: Buy (0), Hold (1), Sell (2)
        self.action_space = spaces.Discrete(3)

        # Define observation space: stock performance metrics
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step][['Yearly Performance (%)', '6M Performance (%)', '3M Performance (%)', '1M Performance (%)', '1W Performance (%)']].values

    def step(self, action):
        # Define rewards for actions
        current_data = self.data.iloc[self.current_step]
        perf = current_data['Yearly Performance (%)']
        if isinstance(perf, pd.Series):
            perf = perf.iloc[0]
        if action == 0:  # Buy
            reward = float(perf)
        elif action == 2:  # Sell
            reward = -float(perf)
        else:  # Hold
            reward = 0.0  # No reward for holding

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Return the next state, reward, and whether the episode is done
        next_state = self.data.iloc[self.current_step][['Yearly Performance (%)', '6M Performance (%)', '3M Performance (%)', '1M Performance (%)', '1W Performance (%)']].values
        return next_state, reward, done, {}

    def render(self):
        pass  # Optionally implement rendering

# Step 5: Train the PPO model
env = StockTradingEnv(data)

# Define PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train PPO for 10,000 timesteps (reduced for faster testing)
model.learn(total_timesteps=10000)

# Step 6: Evaluate the trained model
obs = env.reset()
total_reward = 0
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

print(f"Total Reward from the model: {total_reward}")

# Step 7: Save the trained model
# Create the output directory if it doesn't exist
output_dir = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\Historical_ML\Data"
os.makedirs(output_dir, exist_ok=True)

# Save the model
model_path = os.path.join(output_dir, f"{ticker}_ppo_trained_model")
model.save(model_path)
print(f"Model saved to {model_path}.zip")
