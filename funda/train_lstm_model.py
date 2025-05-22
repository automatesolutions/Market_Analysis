import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging
import os
from backtesting import Backtest, Strategy
from torch.nn import MultiheadAttention
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (only for FRED and Alpha Vantage)
load_dotenv()

# Define LSTM model with bidirectional and multi-head attention for daily predictions
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3, output_size=30, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(1, 0, 2)  # For attention: [seq_len, batch, hidden]
        attn_output, _ = self.attention(out, out, out)
        context = attn_output[-1]  # Take the last time step
        out = self.fc(context)
        return out

# Define CNN-LSTM model with deeper CNN and residual connections for 1-minute predictions
class CNN_LSTM(nn.Module):
    def __init__(self, price_levels, channels, scalar_features_size, hidden_size=100, num_layers=3, output_size=30, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.cnn_residual = nn.Conv1d(channels, 64, kernel_size=1)
        self.cnn_output_size = 64
        self.lstm = nn.LSTM(input_size=self.cnn_output_size + scalar_features_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, order_footprint, scalar_features):
        batch_size, seq_length, price_levels, channels = order_footprint.shape
        # Log input shapes
        logging.debug(f"order_footprint shape: {order_footprint.shape}")
        logging.debug(f"scalar_features shape: {scalar_features.shape}")
        # Reshape for CNN: [batch_size * seq_length, channels, price_levels]
        order_footprint = order_footprint.view(batch_size * seq_length, channels, price_levels)
        # CNN processing
        cnn_main = self.cnn(order_footprint)  # [batch_size * seq_length, 64, 1]
        cnn_residual = self.cnn_residual(order_footprint)  # [batch_size * seq_length, 64, price_levels]
        # Pool residual to match main path
        cnn_residual = nn.functional.adaptive_max_pool1d(cnn_residual, 1)  # [batch_size * seq_length, 64, 1]
        # Add main and residual
        cnn_out = cnn_main + cnn_residual  # [batch_size * seq_length, 64, 1]
        cnn_out = cnn_out.squeeze(-1)  # [batch_size * seq_length, 64]
        # Log shape after CNN
        logging.debug(f"cnn_out shape after squeeze: {cnn_out.shape}")
        # Reshape back to [batch_size, seq_length, 64]
        cnn_out = cnn_out.view(batch_size, seq_length, self.cnn_output_size)
        # Log shape after reshape
        logging.debug(f"cnn_out shape after view: {cnn_out.shape}")
        # Ensure scalar_features has correct shape
        if scalar_features.size(-1) != 9:  # Expected scalar features
            raise ValueError(f"Expected scalar_features with last dim 9, got {scalar_features.size(-1)}")
        # Concatenate CNN output and scalar features
        lstm_input = torch.cat([cnn_out, scalar_features], dim=2)  # [batch_size, seq_length, 64 + 9]
        # Log final LSTM input shape
        logging.debug(f"lstm_input shape: {lstm_input.shape}")
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(lstm_input.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(lstm_input.device)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out

# Custom loss function with trend penalty
def trend_loss(output, target):
    mse = nn.MSELoss()(output, target)
    trend = torch.sign(target[:, 1:] - target[:, :-1]) == torch.sign(output[:, 1:] - output[:, :-1])
    trend_penalty = torch.mean((1 - trend.float()) * 0.1)
    return mse + trend_penalty

# Function to fetch stock and sector data
def fetch_stock_data(ticker, period, interval):
    try:
        logging.info(f"Fetching {interval} data for {ticker} with period={period}")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        sector_ticker = stock.info.get('sector', 'SPY')
        sector_ticker_map = {
            'Technology': 'XLK', 'Financial Services': 'XLF', 'Consumer Cyclical': 'XLY',
            'Industrials': 'XLI', 'Utilities': 'XLU', 'Healthcare': 'XLV',
            'Communication Services': 'XLC', 'Consumer Defensive': 'XLP',
            'Basic Materials': 'XLB', 'Real Estate': 'XLRE', 'Energy': 'XLE'
        }
        sector_ticker = sector_ticker_map.get(sector_ticker, 'SPY')
        sector_df = yf.Ticker(sector_ticker).history(period=period, interval=interval)
        sector_df.index = pd.to_datetime(sector_df.index)
        if sector_df.index.tz is not None:
            sector_df.index = sector_df.index.tz_localize(None)
        sector_df = sector_df.reindex(df.index, method='ffill').bfill()
        df['Sector_Close'] = sector_df['Close']
        df['Order_Flow'] = 0  # Placeholder
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Sector_Close', 'Order_Flow']]
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to fetch order footprint data
def fetch_order_footprint(ticker, period, interval, expected_rows):
    logging.info(f"Fetching order footprint for {ticker}")
    price_levels = 10
    channels = 2  # Buy and sell volumes
    footprint = np.random.rand(expected_rows, price_levels, channels) * 100  # Dummy data
    return footprint

# Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute Realized Volatility
def compute_realized_volatility(df, window=30):
    returns = df['Close'].pct_change().dropna()
    return np.sqrt((returns**2).rolling(window=window).sum())

# Feature engineering
def engineer_features(df, interval, order_footprint=None):
    df = df.copy()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=20 if interval == '1d' else 30).std()
    df['Sector_Volatility'] = df['Sector_Close'].rolling(window=20 if interval == '1d' else 30).std()
    df['Realized_Vol'] = compute_realized_volatility(df)
    df['Vol_Ratio'] = df['Volatility'] / df['Volatility'].rolling(window=100).mean()
    if interval == '1d':
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['Imbalance'] = df['High'] - df['Low']
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Upper_BB'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
        df['Lower_BB'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
        df['ATR'] = (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
        features = ['Close', 'Volume', 'Price_Change', 'Volatility', 'Sector_Volatility', 'Realized_Vol', 'Vol_Ratio', 'SMA_20', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB', 'ATR', 'Order_Flow']
        df_processed = df[features].dropna()
        return df_processed, None, features, df  # Return both processed and original DataFrame
    else:
        df['Momentum'] = df['Close'].diff(5)
        df['Volume_Spike'] = (df['Volume'] / df['Volume'].rolling(window=30).mean()) - 1
        features = ['Close', 'Volume', 'Price_Change', 'Volatility', 'Sector_Volatility', 'Realized_Vol', 'Vol_Ratio', 'Momentum', 'Volume_Spike']
        df_processed = df[features].dropna()
        return df_processed, order_footprint, features, df  # Return both processed and original DataFrame

# Prepare sequences for LSTM
def prepare_sequences(df, order_footprint, features, seq_length, pred_horizon):
    data = df[features].values
    feature_scaler = MinMaxScaler()
    data_scaled = feature_scaler.fit_transform(data)
    
    # Separate scaler for the target (Close price)
    target_scaler = MinMaxScaler()
    target_data = df['Close'].values.reshape(-1, 1)
    target_scaled = target_scaler.fit_transform(target_data)
    
    X_scalar, y = [], []
    X_order_footprint = [] if order_footprint is not None else None
    num_rows = len(data_scaled) - seq_length - pred_horizon + 1
    if order_footprint is not None:
        if len(order_footprint) != len(df):
            logging.warning(f"Order footprint length ({len(order_footprint)}) does not match DataFrame length ({len(df)}). Truncating/padding.")
            if len(order_footprint) > len(df):
                order_footprint = order_footprint[:len(df)]
            else:
                padding = np.zeros((len(df) - len(order_footprint), order_footprint.shape[1], order_footprint.shape[2]))
                order_footprint = np.vstack([order_footprint, padding])
    
    for i in range(num_rows):
        X_scalar.append(data_scaled[i:i + seq_length])
        if order_footprint is not None:
            seq = order_footprint[i:i + seq_length]
            if len(seq) < seq_length:
                padding = np.zeros((seq_length - len(seq), seq.shape[1], seq.shape[2]))
                seq = np.vstack([seq, padding])
            X_order_footprint.append(seq)
        y.append(target_scaled[i + seq_length:i + seq_length + pred_horizon, 0])
    
    X_scalar = np.array(X_scalar)
    y = np.array([yi for yi in y if yi.shape == (pred_horizon,)])
    X_scalar = X_scalar[:len(y)]
    if order_footprint is not None:
        X_order_footprint = np.array(X_order_footprint)[:len(y)]
    return X_order_footprint, X_scalar, y, feature_scaler, target_scaler

# Train the LSTM or CNN-LSTM model
def train_lstm_model(X_order_footprint, X_scalar, y, input_size, output_size, epochs=100, batch_size=32, fine_tune=False, use_cnn=False, price_levels=10, channels=2):
    if use_cnn:
        model = CNN_LSTM(price_levels=price_levels, channels=channels, scalar_features_size=input_size, output_size=output_size)
    else:
        model = StockLSTM(input_size=input_size, output_size=output_size)
    if fine_tune:
        logging.info("Fine-tuning model...")
        model_path = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\model", 'lstm_daily_model.pt' if not use_cnn else 'lstm_1minute_model.pt')
        model.load_state_dict(torch.load(model_path))
    criterion = trend_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001 if fine_tune else 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    if use_cnn:
        X_order_footprint = torch.tensor(X_order_footprint, dtype=torch.float32)
    X_scalar = torch.tensor(X_scalar, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_order_footprint, X_scalar, y_tensor) if use_cnn else TensorDataset(X_scalar, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            if use_cnn:
                batch_X_order, batch_X_scalar, batch_y = batch
                output = model(batch_X_order, batch_X_scalar)
            else:
                batch_X_scalar, batch_y = batch
                output = model(batch_X_scalar)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model_path = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\model", 'lstm_daily_model.pt' if not use_cnn else 'lstm_1minute_model.pt')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break
    return model

# Backtesting strategy
class LSTMStrategy(Strategy):
    def init(self):
        self.predictions = self.data.Predicted_Close
        self.i = 0
    def next(self):
        if self.i < len(self.predictions) and self.data.Close[-1] < self.predictions[self.i]:
            self.buy()
        elif self.i < len(self.predictions):
            self.sell()
        self.i += 1

# Evaluate model
def evaluate_model(model, df_processed, df_original, features, seq_length, pred_horizon, feature_scaler, target_scaler, use_cnn=False, order_footprint=None):
    X_order, X_scalar, y, _, _ = prepare_sequences(df_processed, order_footprint, features, seq_length, pred_horizon)
    model.eval()
    with torch.no_grad():
        if use_cnn:
            X_order = torch.tensor(X_order, dtype=torch.float32)
            predictions = model(X_order, torch.tensor(X_scalar, dtype=torch.float32)).numpy()
        else:
            predictions = model(torch.tensor(X_scalar, dtype=torch.float32)).numpy()
    # Inverse-transform using target scaler
    y_actual = target_scaler.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    mae = mean_absolute_error(y_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_actual, predictions))
    directional_accuracy = np.mean(np.sign(y_actual[:, 1:] - y_actual[:, :-1]) == np.sign(predictions[:, 1:] - predictions[:, :-1]))
    logging.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Directional Accuracy: {directional_accuracy:.4f}")
    
    # Backtesting using original DataFrame
    pred_df = pd.DataFrame({
        'Date': df_original.index[seq_length:seq_length+len(predictions)],
        'Open': df_original['Open'].iloc[seq_length:seq_length+len(predictions)],
        'High': df_original['High'].iloc[seq_length:seq_length+len(predictions)],
        'Low': df_original['Low'].iloc[seq_length:seq_length+len(predictions)],
        'Close': df_original['Close'].iloc[seq_length:seq_length+len(predictions)],
        'Predicted_Close': predictions[:, -1]
    })
    bt = Backtest(pred_df, LSTMStrategy, cash=100000, commission=.002)
    stats = bt.run()
    logging.info(f"Backtest Sharpe Ratio: {stats['Sharpe Ratio']:.2f}, Return: {stats['Return [%]']:.2f}%")
    return mae, rmse, directional_accuracy, stats

# Main training function
def main():
    # Main training tickers - diverse set of stocks from different sectors
    tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'INTC',
        # Financial
        'JPM', 'BAC', 'GS', 'V', 'MA',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'MRK',
        # Consumer
        'AMZN', 'WMT', 'PG', 'KO',
        # Energy
        'XOM', 'CVX', 'COP',
        # Market Indicators
        'SPY', 'QQQ', 'DIA', 'IWM',
        # Sector ETFs
        'XLK', 'XLF', 'XLV', 'XLY', 'XLE'
    ]

    # Fine-tuning tickers - focus on specific stocks
    tickers_to_finetune = [
        # Large Cap Stable Stocks
        'AAPL', 'MSFT', 'JPM', 'JNJ', 'PG',
        # Growth Stocks
        'NVDA', 'AMD', 'TSLA',
        # Sector Leaders
        'XLK', 'XLF', 'XLV', 'XLY', 'XLE',
        # Market Indicators
        'SPY', 'QQQ', 'DIA'
    ]

    daily_seq_length = 60
    minute_seq_length = 30
    price_levels = 10
    channels = 2

    # Pre-train daily LSTM model
    logging.info("Pre-training daily LSTM model...")
    daily_data = []
    for ticker in tickers:
        df = fetch_stock_data(ticker, period='5y', interval='1d')
        if df is not None:
            df_processed, _, features, df_original = engineer_features(df, interval='1d')
            X_order, X_scalar, y, feature_scaler, target_scaler = prepare_sequences(df_processed, None, features, daily_seq_length, pred_horizon=30)
            daily_data.append((X_scalar, y, feature_scaler, target_scaler))
    X_daily = np.concatenate([data[0] for data in daily_data])
    y_daily = np.concatenate([data[1] for data in daily_data])
    feature_scaler = daily_data[0][2]  # Use the first ticker's feature scaler
    target_scaler = daily_data[0][3]   # Use the first ticker's target scaler
    logging.info(f"Daily training data shape: X={X_daily.shape}, y={y_daily.shape}")
    daily_model = train_lstm_model(None, X_daily, y_daily, input_size=len(features), output_size=30, fine_tune=False, use_cnn=False)
    
    # Evaluate daily model
    for ticker in tickers:
        df = fetch_stock_data(ticker, period='1y', interval='1d')
        if df is not None:
            df_processed, _, features, df_original = engineer_features(df, interval='1d')
            evaluate_model(daily_model, df_processed, df_original, features, daily_seq_length, 30, feature_scaler, target_scaler, use_cnn=False)

    # Pre-train 1-minute CNN-LSTM model
    logging.info("Pre-training 1-minute CNN-LSTM model...")
    minute_data = []
    for ticker in tickers:
        df = fetch_stock_data(ticker, period='7d', interval='1m')
        if df is not None:
            df_processed, order_footprint, features, df_original = engineer_features(df, interval='1m')
            expected_rows = len(df_processed)
            order_footprint = fetch_order_footprint(ticker, period='7d', interval='1m', expected_rows=expected_rows)
            X_order, X_scalar, y, feature_scaler, target_scaler = prepare_sequences(df_processed, order_footprint, features, minute_seq_length, pred_horizon=30)
            minute_data.append((X_order, X_scalar, y, feature_scaler, target_scaler))
    X_minute_order = np.concatenate([data[0] for data in minute_data])
    X_minute_scalar = np.concatenate([data[1] for data in minute_data])
    y_minute = np.concatenate([data[2] for data in minute_data])
    feature_scaler = minute_data[0][3]  # Use the first ticker's feature scaler
    target_scaler = minute_data[0][4]   # Use the first ticker's target scaler
    logging.info(f"1-Minute training data shape: X_order={X_minute_order.shape}, X_scalar={X_minute_scalar.shape}, y={y_minute.shape}")
    minute_model = train_lstm_model(X_minute_order, X_minute_scalar, y_minute, input_size=len(features), output_size=30, fine_tune=False, use_cnn=True, price_levels=price_levels, channels=channels)
    
    # Evaluate 1-minute model
    for ticker in tickers:
        df = fetch_stock_data(ticker, period='7d', interval='1m')
        if df is not None:
            df_processed, order_footprint, features, df_original = engineer_features(df, interval='1m')
            expected_rows = len(df_processed)
            order_footprint = fetch_order_footprint(ticker, period='7d', interval='1m', expected_rows=expected_rows)
            evaluate_model(minute_model, df_processed, df_original, features, minute_seq_length, 30, feature_scaler, target_scaler, use_cnn=True, order_footprint=order_footprint)

    # Fine-tune on ticker-specific data
    for ticker in tickers_to_finetune:
        logging.info(f"Fine-tuning daily model for {ticker}...")
        df = fetch_stock_data(ticker, period='1y', interval='1d')
        if df is not None:
            df_processed, _, features, df_original = engineer_features(df, interval='1d')
            X_order, X_scalar, y, feature_scaler, target_scaler = prepare_sequences(df_processed, None, features, daily_seq_length, pred_horizon=30)
            train_lstm_model(None, X_scalar, y, input_size=len(features), output_size=30, epochs=20, fine_tune=True, use_cnn=False)

        logging.info(f"Fine-tuning 1-minute model for {ticker}...")
        df = fetch_stock_data(ticker, period='7d', interval='1m')
        if df is not None:
            df_processed, order_footprint, features, df_original = engineer_features(df, interval='1m')
            expected_rows = len(df_processed)
            order_footprint = fetch_order_footprint(ticker, period='7d', interval='1m', expected_rows=expected_rows)
            X_order, X_scalar, y, feature_scaler, target_scaler = prepare_sequences(df_processed, order_footprint, features, minute_seq_length, pred_horizon=30)
            train_lstm_model(X_order, X_scalar, y, input_size=len(features), output_size=30, epochs=20, fine_tune=True, use_cnn=True, price_levels=price_levels, channels=channels)

if __name__ == "__main__":
    main()