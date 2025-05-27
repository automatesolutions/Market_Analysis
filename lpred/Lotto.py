import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow oneDNN warnings
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Input as TFInput
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from statsmodels.tsa.arima.model import ARIMA
from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objs as go

# File path
data_path = r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\lpred\data\Ultra_Lotto.xlsx"

# Load and preprocess data
try:
    df = pd.read_excel(data_path, sheet_name='Sheet1')
except FileNotFoundError:
    print(f"File not found at {data_path}. Please check the path and file name.")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Ensure required columns
required_columns = ['COMBINATIONS', 'DRAW DATE', 'JACKPOT (PHP)', 'WINNERS']
if not all(col in df.columns for col in required_columns):
    print("Excel file must contain columns: COMBINATIONS, DRAW DATE, JACKPOT (PHP), WINNERS")
    exit()

df['DRAW DATE'] = pd.to_datetime(df['DRAW DATE'], errors='coerce')
df = df.sort_values('DRAW DATE').copy().dropna(subset=['DRAW DATE', 'COMBINATIONS'])
df['numbers'] = df['COMBINATIONS'].str.split('-').apply(lambda x: [int(num) for num in x if num.strip().isdigit()])

# After splitting numbers
all_numbers = df['numbers'].explode().astype(int)
min_number = all_numbers.min()
max_number = all_numbers.max()
num_balls = df['numbers'].apply(len).mode()[0]

# Filter valid rows
df = df[df['numbers'].apply(len) == num_balls]
invalid_numbers = df['numbers'].apply(lambda x: any(num < min_number or num > max_number for num in x))
if invalid_numbers.any():
    print(f"Warning: Some draws contain numbers outside {min_number}-{max_number}. These will be ignored.")
    df = df[~invalid_numbers]

# Check for duplicate dates
if df['DRAW DATE'].duplicated().any():
    print("Duplicate draw dates detected. Aggregating jackpots by taking the maximum per day for ARIMA.")

# Statistical Frequency
numbers = df['numbers'].explode().astype(int)
frequency = numbers.value_counts().sort_index()
for i in range(min_number, max_number + 1):
    if i not in frequency.index:
        frequency[i] = 0
frequency = frequency.sort_index()
freq_df = pd.DataFrame({'Number': frequency.index, 'Frequency': frequency.values}).sort_values('Frequency', ascending=False)
top_freq = freq_df['Number'].head(num_balls).tolist()

# Prepare features and labels for ML models
freq = pd.Series(0, index=range(min_number, max_number + 1))
features = []
labels = []
for i in range(len(df)):
    current_numbers = df['numbers'].iloc[i]
    label = np.zeros(max_number - min_number + 1)
    for num in current_numbers:
        label[num - min_number] = 1
    labels.append(label)
    feature = freq.copy().values
    features.append(feature)
    for num in current_numbers:
        freq[num] += 1
features = np.array(features)
labels = np.array(labels)

# Ensure all numbers are represented
if not np.all(np.any(labels == 1, axis=0)):
    print("Some numbers missing in training data. Adding synthetic data.")
    missing = np.where(np.sum(labels, axis=0) == 0)[0]
    for m in missing:
        synthetic_label = np.zeros(max_number - min_number + 1)
        synthetic_label[m] = 1
        synthetic_feature = freq.copy().values
        labels = np.vstack([labels, synthetic_label])
        features = np.vstack([features, synthetic_feature])

# Split data
train_size = int(0.8 * len(features))
X_train = features[:train_size]
y_train = labels[:train_size]
X_test = features[train_size:]
y_test = labels[train_size:]

# Neural Network (FFNN)
model_ffnn = Sequential([
    TFInput(shape=(max_number - min_number + 1,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(max_number - min_number + 1, activation='sigmoid')
])
model_ffnn.compile(optimizer='adam', loss='binary_crossentropy')
model_ffnn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Random Forest
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(eval_metric='logloss')
multi_xgb = MultiOutputClassifier(xgb)
multi_xgb.fit(X_train, y_train)

# ARIMA for jackpot
jackpot = df.groupby('DRAW DATE')['JACKPOT (PHP)'].max().astype(float)
try:
    jackpot = jackpot.asfreq('D', method='ffill')
except Exception as e:
    print(f"Warning: Failed to set daily frequency for ARIMA: {e}. Using original series.")
model_arima = ARIMA(jackpot, order=(1,1,1))
model_fit = model_arima.fit()
predicted_jackpot = model_fit.forecast(steps=1).iloc[0]

# LSTM
T = 8
X_lstm = []
y_lstm = []
for i in range(T, len(labels)):
    X_lstm.append(labels[i-T:i].copy())
    y_lstm.append(labels[i].copy())
X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)
model_lstm = None
if len(X_lstm) > 0:
    model_lstm = Sequential([
        TFInput(shape=(T, max_number - min_number + 1)),
        LSTM(50),
        Dense(max_number - min_number + 1, activation='sigmoid')
    ])
    model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
    model_lstm.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)

# Predictions
latest_freq = freq.copy()
latest_feature = latest_freq.values.reshape(1, -1)

# FFNN prediction
probabilities_ffnn = model_ffnn.predict(latest_feature, verbose=0)[0]
ffnn_df = pd.DataFrame({
    'Number': range(min_number, max_number + 1),
    'Probability': probabilities_ffnn
}).sort_values('Probability', ascending=True)
top_ffnn = ffnn_df['Number'].tail(num_balls).tolist()[::-1]

# Random Forest prediction
probas = rf.predict_proba(latest_feature)
probabilities_rf = [proba[0, 1] for proba in probas]
rf_df = pd.DataFrame({
    'Number': range(min_number, max_number + 1),
    'Probability': probabilities_rf
}).sort_values('Probability', ascending=True)
top_rf = rf_df['Number'].tail(num_balls).tolist()[::-1]

# XGBoost prediction
probas = multi_xgb.predict_proba(latest_feature)
probabilities_xgb = [proba[0, 1] for proba in probas]
xgb_df = pd.DataFrame({
    'Number': range(min_number, max_number + 1),
    'Probability': probabilities_xgb
}).sort_values('Probability', ascending=True)
top_xgb = xgb_df['Number'].tail(num_balls).tolist()[::-1]

# LSTM prediction
lstm_df = pd.DataFrame({'Number': [], 'Probability': []})
top_lstm = []
if model_lstm and len(labels) >= T:
    last_T = labels[-T:].reshape(1, T, max_number - min_number + 1)
    probabilities_lstm = model_lstm.predict(last_T, verbose=0)[0]
    lstm_df = pd.DataFrame({
        'Number': range(min_number, max_number + 1),
        'Probability': probabilities_lstm
    }).sort_values('Probability', ascending=True)
    top_lstm = lstm_df['Number'].tail(num_balls).tolist()[::-1]

# Prepare data for line chart (Winning Numbers Over Time)
line_traces = []
for idx, row in df.iterrows():
    date = row['DRAW DATE']
    numbers = sorted(row['numbers'])
    jackpot = row['JACKPOT (PHP)']
    winners = row['WINNERS']
    line_traces.append(
        go.Scatter(
            x=[date] * len(numbers),
            y=numbers,
            mode='lines+markers',
            line={'color': '#39FF14', 'width': 1},
            marker={'size': 8, 'color': '#39FF14'},
            name=str(date.date()),
            text=[f"Jackpot: {jackpot:,.2f} PHP<br>Winners: {winners}"] * len(numbers),
            hoverinfo='text+y+x',
            showlegend=False
        )
    )

# Prepare data for 3D Area Chart (Number Frequency)
x_area = [0, 1]
y_area = frequency.index
z_area = np.array([frequency.values, frequency.values])

# Prepare data for 3D Heatmap (combined predictions over time)
models = {'FFNN': probabilities_ffnn, 'Random Forest': probabilities_rf, 'XGBoost': probabilities_xgb}
if len(top_lstm) > 0:
    models['LSTM'] = probabilities_lstm
iterations = list(range(5))
numbers_range = list(range(min_number, max_number + 1))
z_heatmap = np.zeros((len(iterations), len(numbers_range)))
for i, iteration in enumerate(iterations):
    for j, num in enumerate(numbers_range):
        avg_prob = np.mean([models[model][num-min_number] for model in models])
        z_heatmap[i, j] = avg_prob

# Dash App
app = Dash(__name__)

# CSS styles for text
text_style = {
    'fontFamily': 'Arial, sans-serif',
    'color': '#FFFFE0',
    'textShadow': '1px 1px 2px black'
}
h1_style = {**text_style, 'fontSize': '36px', 'marginBottom': '10px'}
h2_style = {**text_style, 'fontSize': '24px', 'marginTop': '20px', 'marginBottom': '10px'}
h3_style = {**text_style, 'fontSize': '18px', 'marginTop': '15px', 'marginBottom': '5px'}
p_style = {**text_style, 'fontSize': '14px', 'marginBottom': '5px'}

# Layout
app.layout = html.Div([
    html.H1("LOOM OF FAITH", className="dotdigital-title", style={
        'fontSize': '60px',
        'color': '#FF7043',
        'letterSpacing': '0.05em',
        'textTransform': 'uppercase',
        'marginBottom': '10px',
        'textShadow': 'none'
    }),
    html.H3(
        "A decade's worth of data analyzed using Artificial Intelligence and Machine Learning. ",
        style={**p_style, 'color': '#7CFC00', 'fontStyle': 'bold'}
    ),
    html.H2("Data Summary", style=h2_style),
    html.P(f"Number of Draws: {len(df)}", style=p_style),
    html.P(f"Average Winners per Draw: {df['WINNERS'].mean():.2f}", style=p_style),
    html.P(f"Number Range: {min_number} to {max_number}", style=p_style),
    html.P(f"Balls per Draw: {num_balls}", style=p_style),
    html.H2("Algorithm Predicted Numbers", style=h2_style),
    html.P(f"Statistical Frequency - Simplicity                 : {top_freq}", style=p_style),
    html.P(f"Neural Network (FFNN) - Complex Pattern Recognition: {top_ffnn}", style=p_style),
    html.P(f"Random Forest - for Ensemble Learning              : {top_rf}", style=p_style),
    html.P(f"XGBoost: - for Ensemble Learning                   : {top_xgb}", style=p_style),
    html.P(f"LSTM: - for Time Series Analysis                   : {top_lstm if top_lstm else 'Insufficient data (requires at least 8 draws)'}", style=p_style),
    html.P(
        "FOR REFERENCE ONLY "
        "No method guarantees",
        style={**p_style, 'color': '#E57373', 'fontStyle': 'italic'}
    ),
    dcc.Graph(
        figure=go.Figure(
            data=line_traces,
            layout=go.Layout(
                title= {
                'text': 'The Loom Thread',
                'font': {
                    'color': '#39FF14',
                  }
                },
                xaxis={
                    'title': 'Date',
                    'titlefont': {'color': '#39FF14'},
                    'gridcolor': 'lightgray',
                    'gridwidth': 1,
                    'tickangle': 45,
                    'tickformat': '%Y-%m-%d',
                    'tickfont': {'color': 'white'}
                },
                yaxis={
                    'title': f'Number ({min_number}–{max_number})',
                    'titlefont': {'color': '#39FF14'},
                    'range': [min_number-1, max_number+1],
                    'dtick': 1,
                    'gridcolor': 'lightgray',
                    'gridwidth': 1,
                    'tickfont': {'color': 'white'}
                },
                plot_bgcolor='black',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                height=800
            )
        )
    ),
    dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=freq_df['Number'],
                    y=freq_df['Frequency'],
                    mode='lines',
                    fill='tozeroy',
                    line={'color': '#39FF14', 'width': 2},
                    stackgroup='one',
                    text=freq_df['Frequency'],
                    hoverinfo='x+y+text'
                )
            ],
            layout=go.Layout(
                title='Statistical Frequency',
                xaxis={'title': 'Number'},
                yaxis={'title': 'Frequency', 'gridcolor': 'lightgray'},
                plot_bgcolor='black',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        )
    ),
    dcc.Graph(
        figure=go.Figure(
            data=[
                go.Surface(
                    x=iterations,
                    y=numbers_range,
                    z=z_heatmap,
                    colorscale=[[0, '#FF0000'], [1, '#00FF00']],
                    showscale=True
                )
            ],
            layout=go.Layout(
                title='Model Predictions Heatmap',
                scene={
                    'xaxis': {'title': 'Iteration'},
                    'yaxis': {'title': 'Number'},
                    'zaxis': {'title': 'Probability'}
                },
                plot_bgcolor='black',
                paper_bgcolor='rgba(0,0,0,0)',
                height=600
            )
        )
    ),
    html.Div([
        html.Img(
            src='/assets/logo.png',
            style={
                'height': '80px',
                'marginRight': '10px',
                'verticalAlign': 'middle'
            }
        ),
        html.Span(
            "Partnership with Kumpooni",
            style={
                'color': '#fff',
                'fontSize': '18px',
                'verticalAlign': 'middle',
                'fontFamily': 'Arial, sans-serif',
                'textShadow': '1px 1px 2px black'
            }
        )
    ], style={
        'position': 'fixed',
        'left': '20px',
        'bottom': '20px',
        'zIndex': '1000',
        'display': 'flex',
        'alignItems': 'center',
        'background': 'rgba(0,0,0,0.5)',
        'padding': '8px 16px',
        'borderRadius': '12px'
    })
], style={
    'backgroundColor': '#000000',
    'minHeight': '100vh',
    'padding': '20px'
})

if __name__ == '__main__':
    app.run_server(debug=True)