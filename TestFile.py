import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit interface for parameter tuning
st.title("Momentum-Based Trading Strategy")

# User inputs for strategy parameters
tranche_size = st.sidebar.slider("Tranche Size (Proportion of Portfolio)", min_value=0.1, max_value=1.0, step=0.1, value=0.2)
lookback_period = st.sidebar.slider("Lookback Period (Days)", min_value=5, max_value=50, step=1, value=14)
initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, step=1000, value=1000000)

# Load your Nifty data (replace with your data source)
# Ensure data has columns: ['Date', 'Close']
try:
    data = pd.read_csv("nifty_data.csv")
except FileNotFoundError:
    st.error("The file 'nifty_data.csv' was not found. Please ensure the file is in the correct directory.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("The file 'nifty_data.csv' is empty. Please provide a valid dataset.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the file: {e}")
    st.stop()

# Ensure the Close column is numeric
try:
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    if data['Close'].isna().any():
        st.error("The 'Close' column contains invalid or missing values. Please clean your dataset and try again.")
        st.stop()
except Exception as e:
    st.error(f"An error occurred while processing the 'Close' column: {e}")
    st.stop()

# Parse the Date column
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
if data['Date'].isna().any():
    st.error("The 'Date' column contains invalid or missing values. Please clean your dataset and try again.")
    st.stop()

data.set_index('Date', inplace=True)

# Calculate momentum
data['Momentum'] = data['Close'].diff(lookback_period)

# Generate signals
portfolio_alloc = 0  # Current portfolio allocation as a fraction of capital
data['Signal'] = np.zeros(len(data))  # Preallocate Signal column for performance

for i in range(lookback_period, len(data)):
    if data['Momentum'].iloc[i] > 0 and portfolio_alloc < 1:
        # Momentum is positive, enter trade in tranches
        data.at[data.index[i], 'Signal'] = tranche_size
        portfolio_alloc = min(portfolio_alloc + tranche_size, 1)  # Prevent exceeding full allocation
    elif data['Momentum'].iloc[i] <= 0 and portfolio_alloc > 0:
        # Momentum turned negative, exit trade in tranches
        data.at[data.index[i], 'Signal'] = -tranche_size
        portfolio_alloc = max(portfolio_alloc - tranche_size, 0)  # Prevent negative allocation

# Calculate daily portfolio returns
data['Position'] = data['Signal'].cumsum()
data['Position'] = data['Position'].clip(upper=1, lower=0)  # Ensure allocation stays within bounds
data['Daily_Return'] = data['Position'] * data['Close'].pct_change()

# Portfolio stats
cumulative_returns = (1 + data['Daily_Return']).cumprod()
data['Portfolio_Value'] = cumulative_returns * initial_capital
trading_days = len(data[data['Daily_Return'].notna()])  # Adjust for actual trading days
cagr = ((data['Portfolio_Value'].iloc[-1] / initial_capital) ** (252 / trading_days)) - 1
max_drawdown = (data['Portfolio_Value'] / data['Portfolio_Value'].cummax() - 1).min()
equity_curve = data['Portfolio_Value']

# Signal Stats
num_trades = abs(data['Signal']).sum()
win_trades = data[data['Daily_Return'] > 0]['Daily_Return'].count()
win_ratio = win_trades / num_trades if num_trades > 0 else 0
avg_gain = data[data['Daily_Return'] > 0]['Daily_Return'].mean() * 100
avg_holding_period = len(data[data['Position'] > 0]) / num_trades if num_trades > 0 else 0

# Display stats on Streamlit
st.subheader("Strategy Performance Metrics")
st.write(f"CAGR: {cagr:.2%}")
st.write(f"Max Drawdown: {max_drawdown:.2%}")
st.write(f"Number of Trades: {num_trades}")
st.write(f"Win Ratio: {win_ratio:.2%}")
st.write(f"Average Gain: {avg_gain:.2f}%")
st.write(f"Average Holding Period: {avg_holding_period:.2f} days")

# Plot equity curve
st.subheader("Portfolio Equity Curve")
st.line_chart(data['Portfolio_Value'])

# Save results
try:
    data.to_csv("strategy_results.csv")
    st.success("Results saved to 'strategy_results.csv'")
except Exception as e:
    st.error(f"An error occurred while saving the results: {e}")
