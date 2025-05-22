import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os
from fredapi import Fred

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
if FRED_API_KEY is None:
    raise ValueError("FRED_API_KEY not found in .env file")

# Step 1: Fetch NASDAQ Composite Data (Daily)
def fetch_nasdaq_data(start_date='2010-01-01', end_date='2025-05-12'):
    nasdaq = yf.download('^IXIC', start=start_date, end=end_date, interval='1d', auto_adjust=True)
    nasdaq = nasdaq[['Close']].reset_index()
    nasdaq.columns = ['date', 'Close']
    nasdaq['date'] = pd.to_datetime(nasdaq['date'])
    return nasdaq

# Step 2: Fetch Daily 10-Year Treasury Yield from FRED
def fetch_treasury_yields(start_date='2010-01-01', api_key=FRED_API_KEY):
    fred = Fred(api_key=api_key)
    series_id = 'DGS10'
    data = fred.get_series(series_id, start_date)
    treasury_data = pd.DataFrame({'date': data.index, 'yield_10y': data.values})
    treasury_data['date'] = pd.to_datetime(treasury_data['date'])
    treasury_data = treasury_data.dropna()
    return treasury_data

# Step 3: Fetch Federal Surplus/Deficit from FRED
def fetch_deficit_surplus_data(start_date='2010-01-01', api_key=FRED_API_KEY):
    fred = Fred(api_key=api_key)
    series_id = 'MTSDS133FMS'
    data = fred.get_series(series_id, start_date)
    df = pd.DataFrame({'date': data.index, 'deficit_surplus_amt': data.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()
    return df

# Step 4: Fetch Monthly Statement of the Public Debt from Fiscal Data
def fetch_mspd_data(start_date='2010-01-01'):
    url = (
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
        "v2/accounting/od/debt_to_penny?"
        "fields=record_date,tot_pub_debt_out_amt&"
        f"filter=record_date:gte:{start_date}&"
        "sort=-record_date"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch MSPD data: {response.status_code} {response.text}")
    data = response.json()['data']
    df = pd.DataFrame(data)
    df = df.rename(columns={'tot_pub_debt_out_amt': 'total_public_debt_outstanding'})
    df['record_date'] = pd.to_datetime(df['record_date'])
    df['total_public_debt_outstanding'] = df['total_public_debt_outstanding'].astype(float)
    return df

# Step 5: Fetch Mortgage Delinquency Rate from FRED
def fetch_deliquency_data(start_date='2010-01-01', api_key=FRED_API_KEY):
    fred = Fred(api_key=api_key)
    series_id = 'DRSFRMACBS'
    data = fred.get_series(series_id, start_date)
    df = pd.DataFrame({'date': data.index, 'delinquency_rate': data.values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()
    return df

# Step 6: Preprocess and Merge Data
def preprocess_data(nasdaq, treasury, deficit, mspd, delinquency):
    # Set date as index for interpolation
    nasdaq = nasdaq.set_index('date')
    treasury = treasury.set_index('date')
    deficit = deficit.set_index('date')
    mspd = mspd.rename(columns={'record_date': 'date'}).set_index('date')
    delinquency = delinquency.set_index('date')
    
    # Create a daily date range
    date_range = pd.date_range(start='2010-01-01', end='2025-05-12', freq='D')
    merged_data = pd.DataFrame(index=date_range)
    
    # Merge datasets
    merged_data = merged_data.join(nasdaq[['Close']], how='left')
    merged_data = merged_data.join(treasury[['yield_10y']], how='left')
    merged_data = merged_data.join(deficit[['deficit_surplus_amt']], how='left')
    merged_data = merged_data.join(mspd[['total_public_debt_outstanding']], how='left')
    merged_data = merged_data.join(delinquency[['delinquency_rate']], how='left')
    
    # Interpolate missing values
    merged_data = merged_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    # Normalize data
    scaler = MinMaxScaler()
    merged_data['Close_scaled'] = scaler.fit_transform(merged_data[['Close']])
    merged_data['yield_10y_scaled'] = scaler.fit_transform(merged_data[['yield_10y']])
    merged_data['deficit_surplus_scaled'] = scaler.fit_transform(merged_data[['deficit_surplus_amt']])
    merged_data['debt_scaled'] = scaler.fit_transform(merged_data[['total_public_debt_outstanding']])
    merged_data['delinquency_scaled'] = scaler.fit_transform(merged_data[['delinquency_rate']])
    
    return merged_data.reset_index().rename(columns={'index': 'date'})

# Step 7: Calculate Correlations
def calculate_correlations(data):
    indicators = ['yield_10y', 'deficit_surplus_amt', 'total_public_debt_outstanding', 'delinquency_rate']
    for indicator in indicators:
        correlation = data['Close'].corr(data[indicator])
        print(f"Correlation between NASDAQ and {indicator}: {correlation:.4f}")

# Step 8: Analyze Convergence and Divergence
def analyze_convergence_divergence(data):
    data['diff_yield'] = data['Close_scaled'] - data['yield_10y_scaled']
    data['diff_deficit'] = data['Close_scaled'] - data['deficit_surplus_scaled']
    data['diff_debt'] = data['Close_scaled'] - data['debt_scaled']
    data['diff_delinquency'] = data['Close_scaled'] - data['delinquency_scaled']
    
    data['diff_yield_ma'] = data['diff_yield'].rolling(window=252, min_periods=1).mean()
    data['diff_deficit_ma'] = data['diff_deficit'].rolling(window=252, min_periods=1).mean()
    data['diff_debt_ma'] = data['diff_debt'].rolling(window=252, min_periods=1).mean()
    data['diff_delinquency_ma'] = data['diff_delinquency'].rolling(window=252, min_periods=1).mean()
    
    return data

# Step 9: Plot Interactive Trends with Plotly
def plot_trends(data):
    # Multi-subplot for time series
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'NASDAQ vs. 10Y Yield',
            'NASDAQ vs. Deficit/Surplus',
            'NASDAQ vs. Public Debt',
            'NASDAQ vs. Mortgage Delinquency',
            'Convergence/Divergence'
        )
    )
    
    # NASDAQ vs. Yield
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['Close_scaled'], name='NASDAQ (Scaled)', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['yield_10y_scaled'], name='10Y Yield (Scaled)', line=dict(color='red')),
        row=1, col=1
    )
    
    # NASDAQ vs. Deficit
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['Close_scaled'], name='NASDAQ (Scaled)', line=dict(color='blue'), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['deficit_surplus_scaled'], name='Deficit/Surplus (Scaled)', line=dict(color='green')),
        row=2, col=1
    )
    
    # NASDAQ vs. Debt
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['Close_scaled'], name='NASDAQ (Scaled)', line=dict(color='blue'), showlegend=False),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['debt_scaled'], name='Public Debt (Scaled)', line=dict(color='purple')),
        row=3, col=1
    )
    
    # NASDAQ vs. Delinquency
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['Close_scaled'], name='NASDAQ (Scaled)', line=dict(color='blue'), showlegend=False),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['delinquency_scaled'], name='Delinquency Rate (Scaled)', line=dict(color='orange')),
        row=4, col=1
    )
    
    # Convergence/Divergence
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['diff_yield_ma'], name='Diff MA (NASDAQ - Yield)', line=dict(color='red')),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['diff_deficit_ma'], name='Diff MA (NASDAQ - Deficit)', line=dict(color='green')),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['diff_debt_ma'], name='Diff MA (NASDAQ - Debt)', line=dict(color='purple')),
        row=5, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['diff_delinquency_ma'], name='Diff MA (NASDAQ - Delinquency)', line=dict(color='orange')),
        row=5, col=1
    )
    fig.add_hline(y=0, line_dash='dash', line_color='black', row=5, col=1)
    
    fig.update_layout(
        title='NASDAQ vs. Fiscal Indicators (2010-2025)',
        height=1200,
        showlegend=True,
        hovermode='x unified'
    )
    fig.update_xaxes(title_text='Date', row=5, col=1)
    fig.update_yaxes(title_text='Scaled Value', row=1, col=1)
    fig.update_yaxes(title_text='Scaled Value', row=2, col=1)
    fig.update_yaxes(title_text='Scaled Value', row=3, col=1)
    fig.update_yaxes(title_text='Scaled Value', row=4, col=1)
    fig.update_yaxes(title_text='Difference (Scaled)', row=5, col=1)
    fig.write_html('nasdaq_fiscal_indicators.html')

# Main Execution
if __name__ == "__main__":
    print("Fetching NASDAQ data...")
    nasdaq_data = fetch_nasdaq_data()
    print("Fetching Treasury Yield data...")
    treasury_data = fetch_treasury_yields()
    print("Fetching Deficit/Surplus data...")
    deficit_data = fetch_deficit_surplus_data()
    print("Fetching Public Debt data...")
    mspd_data = fetch_mspd_data()
    print("Fetching Mortgage Delinquency data...")
    delinquency_data = fetch_deliquency_data()
    
    print("Preprocessing data...")
    merged_data = preprocess_data(nasdaq_data, treasury_data, deficit_data, mspd_data, delinquency_data)
    
    print("Calculating correlations...")
    calculate_correlations(merged_data)
    
    print("Analyzing convergence and divergence...")
    merged_data = analyze_convergence_divergence(merged_data)
    
    print("Generating interactive plots...")
    plot_trends(merged_data)
    
    print("Plot saved as 'nasdaq_fiscal_indicators.html'")
    merged_data.to_csv('nasdaq_fiscal_analysis.csv', index=False)
    print("Data saved to 'nasdaq_fiscal_analysis.csv'")