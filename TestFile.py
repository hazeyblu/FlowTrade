import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import erf
import plotly.express as px

# Enable wide mode in Streamlit by default
st.set_page_config(layout="wide")


def create_sidebar() -> tuple[float, int, int]:
    """
    Creates a Streamlit sidebar for parameter tuning with brief explanations for each input.

    Returns:
        tuple: A tuple containing tranche size, lookback period, and initial capital.
    """
    tranche_size = st.sidebar.slider(
        "Tranche Size (Proportion of Portfolio)",
        min_value=0.25, max_value=1.0, step=0.25, value=0.5,
        help="Defines the allocation per tranche as a proportion of the total portfolio."
    )
    lookback_period = st.sidebar.slider(
        "Lookback Period (Days)",
        min_value=5, max_value=50, step=1, value=20,
        help="Specifies the number of days to use when calculating momentum."
    )
    initial_capital = st.sidebar.number_input(
        "Initial Capital",
        min_value=1000, step=1000, value=1000000,
        help="The initial amount of money available for the portfolio."
    )
    return tranche_size, lookback_period, initial_capital


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the dataset from the given file path.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("The file was not found. Please ensure the file is in the correct directory.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("The file is empty. Please provide a valid dataset.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        st.stop()

    # Ensure 'Close' is numeric
    try:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        if data['Close'].isna().any():
            st.error("The 'Close' column contains invalid or missing values. Please clean your dataset and try again.")
            st.stop()
    except Exception as e:
        st.error(f"An error occurred while processing the 'Close' column: {e}")
        st.stop()

    # Parse 'Date' column
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
    if data['Date'].isna().any():
        st.error("The 'Date' column contains invalid or missing values. Please clean your dataset and try again.")
        st.stop()

    data.set_index('Date', inplace=True)
    return data


def generate_signals(data: pd.DataFrame, lookback_period: int, tranche_size: float) -> pd.DataFrame:
    """
    Generate trading signals based on average Sharpe ratios for two periods (n and 2n days).

    Args:
        data (pd.DataFrame): DataFrame containing 'Close' prices.
        lookback_period (int): Lookback period for calculating momentum.
        tranche_size (float): Proportion of the portfolio allocated for each signal.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'Signal' column.
    """
    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()

    # Sharpe ratio for n days
    data['Sharpe_n'] = (
        data['Returns'].rolling(lookback_period).mean() /
        data['Returns'].rolling(lookback_period).std()
    )

    # Sharpe ratio for 2n days
    data['Sharpe_2n'] = (
        data['Returns'].rolling(2 * lookback_period).mean() /
        data['Returns'].rolling(2 * lookback_period).std()
    )

    # Average Sharpe momentum
    data['Momentum'] = (data['Sharpe_n'] + data['Sharpe_2n']) / 2

    # Generate signals based on Momentum
    portfolio_alloc = 0
    data['Signal'] = 0

    for i in range(2 * lookback_period, len(data)):
        if data['Momentum'].iloc[i] > 0 and portfolio_alloc < 1:
            data.at[data.index[i], 'Signal'] = tranche_size
            portfolio_alloc = min(portfolio_alloc + tranche_size, 1)
        elif data['Momentum'].iloc[i] <= 0 < portfolio_alloc:
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
    nifty_sharpe_ratio = nifty_returns.mean() / nifty_returns.std() if nifty_returns.std() > 0 else 0

    # Return calculated metrics
    return {
        "num_trades": num_trades,
        "win_ratio": win_ratio,
        "avg_gain": avg_gain,
        "avg_holding_period": avg_holding_period,
        "sharpe_ratio": sharpe_ratio,
        "nifty_sharpe_ratio": nifty_sharpe_ratio  # Include the Nifty Sharpe ratio in the metrics
    }


def create_expander(data: pd.DataFrame, observed_sharpe: float, nifty_sharpe: float) -> None:
    """
    Create the Streamlit expander with detailed Probabilistic Sharpe Ratio (PSR) information.

    Args:
        data (pd.DataFrame): DataFrame containing 'Daily_Return' column.
        observed_sharpe (float): Observed Sharpe ratio for the strategy.
        nifty_sharpe (float): Nifty Sharpe ratio to be used as the threshold.
    """
    # Probabilistic Sharpe Ratio (PSR) calculation
    psr = 0
    if data['Daily_Return'].std() > 0:
        psr = 0.5 * (1 + erf(
            (observed_sharpe - nifty_sharpe) / (data['Daily_Return'].std() / np.sqrt(len(data['Daily_Return'])))))

    # Display the PSR expander
    with st.expander("Probabilistic Sharpe Ratio Details"):
        st.write(f"Observed Sharpe Ratio: {observed_sharpe:.4f}")
        st.write(f"Nifty Sharpe Ratio (Benchmark): {nifty_sharpe:.4f}")
        st.write(f"Probabilistic Sharpe Ratio (PSR): {psr:.4f}")


def create_download_section(file_list: list[tuple[str, str]]):
    """
    Create a download section in the sidebar for provided files.

    Args:
        file_list (list of tuples): List of tuples containing file paths and MIME types.
    """
    st.sidebar.subheader("Download Results")
    for file_path, mime_type in file_list:
        file_name = file_path.split("/")[-1]  # Extract file name from path
        with open(file_path, "rb") as file:
            st.sidebar.download_button(
                label=f"Download {file_name}",
                data=file,
                file_name=file_name,
                mime=mime_type
            )


def main():
    st.title("Momentum-Based Trading Strategy")

    # Create sidebar elements (tranche_size, lookback_period, initial_capital)
    tranche_size, lookback_period, initial_capital = create_sidebar()

    # Load and preprocess data
    data = load_and_preprocess_data("nifty_data.csv")

    # Generate signals for strategy
    data = generate_signals(data, lookback_period, tranche_size)

    # Calculate portfolio performance
    data, cagr, max_drawdown = calculate_portfolio(data, initial_capital)

    # Calculate strategy and Nifty metrics
    metrics = calculate_metrics(data)

    # Create a dictionary with all the performance metrics
    metrics_dict = {
        "CAGR": f"{cagr:.2%}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Number of Trades": metrics['num_trades'],
        "Win Ratio": f"{metrics['win_ratio']:.2%}",
        "Average Gain": f"{metrics['avg_gain']:.2f}%",
        "Average Holding Period": f"{metrics['avg_holding_period']:.2f} days",
        "Sharpe Ratio (Strategy)": f"{metrics['sharpe_ratio']:.4f}",
        "Sharpe Ratio (Nifty)": f"{metrics['nifty_sharpe_ratio']:.4f}"
    }

    # Display performance metrics as a table
    st.subheader("Strategy Performance Metrics")
    st.dataframe(pd.DataFrame(metrics_dict, index=[0]))

    # Plot equity curve
    st.subheader("Portfolio Equity Curve")
    st.plotly_chart(px.line(data, x=data.index, y=['Portfolio_Value', 'Nifty']), use_container_width=True)

    # Display Probabilistic Sharpe Ratio (PSR) details
    create_expander(data, metrics['sharpe_ratio'], metrics['nifty_sharpe_ratio'])

    # Drop specified columns and rename 'Close' to 'Nifty'
    data = data.drop(columns=["Returns", "Sharpe_n", "Sharpe_2n", "Momentum", "Signal", "Position", "Daily_Return"])
    data = data.rename(columns={"Close": "Nifty"})

    # Optionally save the results to CSV
    try:
        data.to_csv("strategy_results.csv")
        file_list = [("strategy_results.csv", "text/csv"),("FlowTrade.pdf", "application/pdf")]
        create_download_section(file_list)
    except Exception as e:
        st.error(f"An error occurred while saving the results: {e}")


if __name__ == "__main__":
    main()
