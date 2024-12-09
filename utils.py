import pandas as pd
import numpy as np
import streamlit as st
from scipy.special import erf


def create_sidebar() -> tuple[float, int, float, int, bool]:
    """
    Creates a Streamlit sidebar for parameter tuning with predefined tranche sizes.

    Returns:
        tuple: A tuple containing tranche size, lookback period, sharpe threshold,
               initial capital, and whether crossover confirmation is enabled.
    """
    tranche_size = st.sidebar.selectbox(
        "Tranche Size (Proportion of Portfolio)",
        options=[0.2, 0.25, 0.5, 1.0],
        index=1,
        help="Defines the allocation per tranche as a proportion of the total portfolio."
    )
    lookback_period = st.sidebar.slider(
        "Lookback Period (Days)",
        min_value=5, max_value=250, step=5, value=20,
        help="Specifies the number of days to use when calculating momentum."
    )
    # Add Sharpe threshold selection
    sharpe_threshold = st.sidebar.slider(
        "Sharpe Threshold",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        help="Minimum Sharpe at which investment can start."
    )
    initial_capital = st.sidebar.number_input(
        "Initial Capital",
        min_value=1000, step=1000, value=1000000,
        help="The initial amount of money available for the portfolio."
    )
    # Add checkbox for crossover confirmation
    crossover_confirmation = st.sidebar.checkbox(
        "Enable Crossover Confirmation",
        value=False,
        help="Enable this option to confirm trades based on crossover signals of n and 2n days."
    )

    return tranche_size, lookback_period, sharpe_threshold, initial_capital, crossover_confirmation


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


def check_password(tranche_size, initial_capital, crossover_confirmation):
    """
    Check the password condition for special access based on user inputs.

    Args:
        tranche_size (float): Proportion of portfolio for each tranche.
        initial_capital (float): The initial capital provided.
        crossover_confirmation (bool): Whether crossover confirmation is enabled.

    Returns:
        bool: True if the special condition is met, else False.
    """
    # Define the "password" condition:
    if initial_capital == 2500000:
        return True
    return False
