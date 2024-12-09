import streamlit as st
import pandas as pd
import plotly.express as px
from strategy import load_data, process_signals, calculate_performance_metrics
from utils import create_sidebar, create_expander, create_download_section, check_password

# Enable wide mode in Streamlit by default
st.set_page_config(layout="wide")


def main():
    st.title("FlowTrade")

    # Create sidebar elements (tranche_size, lookback_period, initial_capital)
    tranche_size, lookback_period, sharpe_threshold, initial_capital, crossover_confirmation = create_sidebar()

    # Load and preprocess data
    data = load_data("nifty_data.csv")

    # Generate signals for strategy
    data = process_signals(data, lookback_period, tranche_size, sharpe_threshold, crossover_confirmation)

    # Calculate portfolio performance and strategy metrics
    data, cagr, max_drawdown, metrics = calculate_performance_metrics(data, initial_capital)

    # Display performance metrics as a table
    st.subheader("FlowTrade Performance Metrics")
    st.dataframe(pd.DataFrame({
        "CAGR": f"{cagr:.2%}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Number of Trades": metrics['num_trades'],
        "Win Ratio": f"{metrics['win_ratio']:.2%}",
        "Average Gain": f"{metrics['avg_gain']:.2f}%",
        "Average Holding Period": f"{metrics['avg_holding_period']:.2f} days",
        "Sharpe Ratio (Strategy)": f"{metrics['sharpe_ratio']:.4f}",
        "Sharpe Ratio (Nifty)": f"{metrics['nifty_sharpe_ratio']:.4f}"
    }, index=[0]))

    # Plot equity curve
    st.subheader("Portfolio Equity Curve")
    st.plotly_chart(px.line(data, x=data.index, y=['Portfolio_Value', 'Nifty']), use_container_width=True)

    # Display Probabilistic Sharpe Ratio (PSR) details
    create_expander(data, metrics['sharpe_ratio'], metrics['nifty_sharpe_ratio'])

    if check_password(tranche_size, initial_capital, crossover_confirmation):
        st.markdown("[Click here to access strategy code]"
                    "(https://github.com/hazeyblu/FlowTrade/blob/24e9ba1b66a36b4a00b578bd8b1f92c148756c52/strategy_logic.py)")

    # Optionally save the results to CSV
    try:
        data.to_csv("strategy_results.csv")
        file_list = [("strategy_results.csv", "text/csv"), ("FlowTrade.pdf", "application/pdf"), ("Strategy_Excel_Dash.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")]
        create_download_section(file_list)
    except Exception as e:
        st.error(f"An error occurred while saving the results: {e}")


if __name__ == "__main__":
    main()
