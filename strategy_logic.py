import pandas as pd
import numpy as np


def generate_signals(data: pd.DataFrame, lookback_period: int, tranche_size: float,
                     sharpe_threshold: float, enable_crossover: bool) -> pd.DataFrame:
    """
    Generates trading signals based on the Sharpe ratio and moving average crossovers.

    The function calculates the Sharpe ratio for both n-period and 2n-period windows, computes
    the average Sharpe ratio, and uses it to determine portfolio allocations. If enabled, it also
    checks for moving average crossovers as an additional confirmation before deploying capital.

    Args:
        data (pd.DataFrame): DataFrame containing 'Close' prices.
        lookback_period (int): The period for calculating the Sharpe ratio and moving averages.
        tranche_size (float): The proportion of the portfolio to allocate on each signal.
        sharpe_threshold (float): The minimum Sharpe ratio to trigger a trade signal.
        enable_crossover (bool): Whether to use moving average crossovers for trade confirmation.

    Returns:
        pd.DataFrame: Updated DataFrame with the 'Signal' column representing portfolio allocation.
    """
    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()

    # Calculate rolling means and standard deviations for Sharpe ratio
    rolling_mean_n = data['Returns'].rolling(lookback_period).mean()
    rolling_std_n = data['Returns'].rolling(lookback_period).std()
    rolling_mean_2n = data['Returns'].rolling(2 * lookback_period).mean()
    rolling_std_2n = data['Returns'].rolling(2 * lookback_period).std()

    # Sharpe ratio calculation
    sharpe_n = rolling_mean_n / rolling_std_n
    sharpe_2n = rolling_mean_2n / rolling_std_2n
    data['Sharpe_avg'] = (sharpe_n + sharpe_2n) / 2

    # Initialize signals and portfolio allocation
    portfolio_alloc = 0
    data['Signal'] = 0

    # Iterate through the data and generate signals based on Sharpe ratio and crossover confirmation
    for i in range(2 * lookback_period, len(data)):
        current_sharpe = data['Sharpe_avg'].iloc[i]

        # Check if Sharpe ratio exceeds threshold and deploy tranches accordingly
        if current_sharpe > sharpe_threshold:
            if enable_crossover:
                # Calculate moving averages for trend confirmation
                data['SMA_short'] = data['Close'].rolling(lookback_period).mean()
                data['SMA_long'] = data['Close'].rolling(2 * lookback_period).mean()
                # Check for crossover confirmation (short-term above long-term SMA)
                if data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i]:
                    if portfolio_alloc < 1:
                        data.at[data.index[i], 'Signal'] = tranche_size
                        portfolio_alloc = min(portfolio_alloc + tranche_size, 1)
            else:
                if portfolio_alloc < 1:
                    data.at[data.index[i], 'Signal'] = tranche_size
                    portfolio_alloc = min(portfolio_alloc + tranche_size, 1)

        # Exit condition: Reduce allocation if Sharpe ratio falls below threshold
        elif current_sharpe <= sharpe_threshold and portfolio_alloc > 0:
            data.at[data.index[i], 'Signal'] = -tranche_size
            portfolio_alloc = max(portfolio_alloc - tranche_size, 0)

    return data


def calculate_portfolio(data: pd.DataFrame, initial_capital: float) -> tuple[pd.DataFrame, float, float]:
    """
    Calculate the portfolio value, CAGR, and maximum drawdown based on generated signals.

    Args:
        data (pd.DataFrame): DataFrame containing 'Close', 'Signal', and 'Returns' columns.
        initial_capital (float): Initial portfolio capital.

    Returns:
        tuple: Updated DataFrame, CAGR, and maximum drawdown.
    """
    # Shift signals to ensure returns start being calculated on the day after the signal
    data['Position'] = data['Signal'].shift(1).cumsum()
    data['Position'] = data['Position'].clip(upper=1, lower=0)

    # Daily portfolio returns
    data['Daily_Return'] = data['Position'] * data['Returns']

    # Equity curves
    data['Nifty'] = (1 + data['Returns']).cumprod() * initial_capital
    cumulative_returns = (1 + data['Daily_Return']).cumprod()
    data['Portfolio_Value'] = cumulative_returns * initial_capital

    # Performance metrics
    trading_days = len(data[data['Daily_Return'].notna()])
    cagr = ((data['Portfolio_Value'].iloc[-1] / initial_capital) ** (252 / trading_days)) - 1
    max_drawdown = (data['Portfolio_Value'] / data['Portfolio_Value'].cummax() - 1).min()

    return data, cagr, max_drawdown


def calculate_metrics(data: pd.DataFrame) -> dict:
    """
    Calculate additional strategy performance metrics, including Sharpe ratio and Probabilistic Sharpe Ratio (PSR).

    Args:
        data (pd.DataFrame): DataFrame containing 'Signal', 'Daily_Return', 'Returns', and 'Close' columns.

    Returns:
        dict: Dictionary containing performance metrics like number of trades, win ratio, Sharpe ratio, and PSR.
    """
    # Calculate total trades (number of non-zero signals)
    num_trades = data['Signal'][data['Signal'] != 0].count()

    # Calculate winning trades (trades with positive returns)
    winning_trades = data[(data['Signal'].shift(1) != 0) & (data['Daily_Return'] > 0)]['Daily_Return'].count()

    # Win ratio
    win_ratio = winning_trades / num_trades if num_trades > 0 else 0

    # Average gain on winning trades
    avg_gain = data[data['Daily_Return'] > 0]['Daily_Return'].mean() * 100

    # Average holding period
    avg_holding_period = len(data[data['Position'] > 0]) / num_trades if num_trades > 0 else 0

    # Sharpe ratio for the strategy
    sharpe_ratio = (data['Daily_Return'].mean() / data['Daily_Return'].std()) * np.sqrt(252) \
        if data['Daily_Return'].std() > 0 else 0

    # Calculate the Sharpe ratio for Nifty (used as threshold Sharpe)
    nifty_returns = data['Returns'].dropna()  # Ensure no NaN values
    nifty_sharpe_ratio = (nifty_returns.mean() / nifty_returns.std()) * np.sqrt(252) \
        if nifty_returns.std() > 0 else 0

    # Return calculated metrics
    return {
        "num_trades": num_trades,
        "win_ratio": win_ratio,
        "avg_gain": avg_gain,
        "avg_holding_period": avg_holding_period,
        "sharpe_ratio": sharpe_ratio,
        "nifty_sharpe_ratio": nifty_sharpe_ratio  # Include the Nifty Sharpe ratio in the metrics
    }
