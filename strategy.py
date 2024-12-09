from strategy_logic import generate_signals, calculate_portfolio, calculate_metrics
from utils import load_and_preprocess_data


def load_data(file_path: str):
    data = load_and_preprocess_data(file_path)
    return data


def process_signals(data, lookback_period, tranche_size, sharpe_threshold, crossover_confirmation):
    data = generate_signals(data, lookback_period, tranche_size, sharpe_threshold, crossover_confirmation)
    return data


def calculate_performance_metrics(data, initial_capital):
    data, cagr, max_drawdown = calculate_portfolio(data, initial_capital)
    metrics = calculate_metrics(data)
    return data, cagr, max_drawdown, metrics
