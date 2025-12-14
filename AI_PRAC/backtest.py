# backtest.py

"""
Backtesting framework for evaluating ML trading strategies.

This module simulates realistic trading scenarios:
- Start with $10,000 (or any initial capital)
- Make buy/sell decisions based on model predictions
- Track portfolio value over time
- Calculate performance metrics (returns, Sharpe ratio, drawdown)
- Compare against buy-and-hold baseline

The key insight: Accuracy alone doesn't matter in trading - what matters
is whether your predictions lead to profitable trades.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

from data_prep import (
    download_price_data,
    add_technical_features,
    add_labels,
    build_feature_matrix_and_labels,
    split_train_test_by_date,
)

# Set plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


def backtest_model(
    model,
    model_name: str,
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 10000,
    confidence_threshold: float = 0.5,
    transaction_cost: float = 0.001,  # 0.1% per trade
) -> Dict:
    """
    Backtest a trained model on historical data.
    
    Strategy:
    - Predict next day's direction (up/down)
    - If prediction is UP with confidence > threshold: BUY (go long)
    - If prediction is DOWN: SELL/stay in cash
    - Track daily portfolio value
    
    Args:
        model: Trained sklearn model (with predict and predict_proba)
        model_name: Name for reporting (e.g., "XGBoost", "Logistic Regression")
        ticker: Stock ticker to backtest on
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        initial_capital: Starting cash ($10,000 default)
        confidence_threshold: Only trade if P(up) > this value
        transaction_cost: Trading fee as fraction (0.1% = 0.001)
    
    Returns:
        Dictionary with:
        - portfolio_values: Daily portfolio value
        - dates: Corresponding dates
        - trades: List of trade signals (1=long, 0=cash)
        - metrics: Performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Backtesting {model_name} on {ticker}")
    print(f"{'='*60}")
    
    # Download price data
    df = download_price_data(ticker, start=start, end=end)
    df = add_technical_features(df)
    df = add_labels(df)
    
    # Split into train/test (we'll only backtest on test period)
    df_train, df_test = split_train_test_by_date(df, test_size_years=2)
    
    X_test, y_test = build_feature_matrix_and_labels(df_test)
    
    # Get the actual closing prices for the test period
    # (need to align with X_test after dropna)
    df_test_clean = df_test.dropna(subset=[
        "daily_return", "ret_5", "ret_10", "ma_5", "ma_10", "ma_20",
        "std_5", "std_10", "volume_zscore_20", "rsi_14", "macd",
        "macd_signal", "bb_width", "lag_ret_1", "lag_ret_2", "lag_vol_1", "target"
    ])
    
    prices = df_test_clean['Close'].values
    dates = df_test_clean.index
    
    # Get model predictions
    predictions = model.predict(X_test)
    
    # Get probabilities (if available)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)[:, 1]  # P(up)
    else:
        # For models without predict_proba, use predictions directly
        probabilities = predictions.astype(float)
    
    # Initialize tracking variables
    cash = initial_capital
    shares = 0
    portfolio_values = []
    trades = []  # 1 = holding stock, 0 = holding cash
    trade_log = []  # detailed trade history
    
    print(f"\nStarting backtest with ${initial_capital:,.2f}")
    print(f"Confidence threshold: {confidence_threshold:.1%}")
    print(f"Transaction cost: {transaction_cost:.2%}\n")
    
    # Simulate trading day by day
    for i in range(len(prices) - 1):  # -1 because we need next day's price
        current_price = prices[i]
        next_price = prices[i + 1]
        current_date = dates[i]
        
        pred = predictions[i]
        confidence = probabilities[i]
        
        # Current portfolio value
        portfolio_value = cash + shares * current_price
        portfolio_values.append(portfolio_value)
        
        # Trading logic
        if pred == 1 and confidence >= confidence_threshold and shares == 0:
            # BUY signal with sufficient confidence, and we're not already holding
            shares_to_buy = (cash * (1 - transaction_cost)) / current_price
            shares = shares_to_buy
            cash = 0
            trades.append(1)
            trade_log.append({
                'date': current_date,
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'confidence': confidence,
                'portfolio_value': portfolio_value
            })
            
        elif pred == 0 and shares > 0:
            # SELL signal, exit position
            cash = shares * current_price * (1 - transaction_cost)
            shares = 0
            trades.append(0)
            trade_log.append({
                'date': current_date,
                'action': 'SELL',
                'price': current_price,
                'cash': cash,
                'confidence': confidence,
                'portfolio_value': portfolio_value
            })
        else:
            # Hold current position
            trades.append(1 if shares > 0 else 0)
    
    # Final portfolio value
    final_price = prices[-1]
    final_value = cash + shares * final_price
    portfolio_values.append(final_value)
    trades.append(trades[-1] if trades else 0)
    
    # Calculate performance metrics
    total_return = (final_value - initial_capital) / initial_capital
    
    # Daily returns
    portfolio_values_array = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values_array) / portfolio_values_array[:-1]
    
    # Sharpe ratio (annualized)
    avg_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    sharpe_ratio = (avg_return / (std_return + 1e-9)) * np.sqrt(252)
    
    # Maximum drawdown
    peak = np.maximum.accumulate(portfolio_values_array)
    drawdown = (peak - portfolio_values_array) / peak
    max_drawdown = np.max(drawdown)
    
    # Win rate (percentage of profitable trades)
    if trade_log:
        profitable_trades = 0
        total_trades = 0
        for i in range(len(trade_log) - 1):
            if trade_log[i]['action'] == 'BUY' and trade_log[i+1]['action'] == 'SELL':
                buy_price = trade_log[i]['price']
                sell_price = trade_log[i+1]['price']
                if sell_price > buy_price:
                    profitable_trades += 1
                total_trades += 1
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    else:
        win_rate = 0
        total_trades = 0
    
    # Number of days in market
    days_in_market = sum(trades)
    total_days = len(trades)
    market_exposure = days_in_market / total_days
    
    # Print summary
    print(f"Results for {model_name}:")
    print(f"  Final Portfolio Value: ${final_value:,.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"  Maximum Drawdown: {max_drawdown:.2%}")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Number of Trades: {total_trades}")
    print(f"  Market Exposure: {market_exposure:.1%}")
    print(f"  Days in Market: {days_in_market}/{total_days}")
    
    return {
        'model_name': model_name,
        'portfolio_values': portfolio_values,
        'dates': dates,
        'trades': trades,
        'trade_log': trade_log,
        'prices': prices,
        'metrics': {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': total_trades,
            'market_exposure': market_exposure,
        }
    }


def benchmark_buy_and_hold(
    ticker: str,
    start: str,
    end: str,
    initial_capital: float = 10000
) -> Dict:
    """
    Baseline strategy: Buy on day 1, hold until the end.
    
    This is what you'd get if you just bought the stock and forgot about it.
    ML models should ideally beat this!
    """
    print(f"\n{'='*60}")
    print(f"Buy & Hold Baseline for {ticker}")
    print(f"{'='*60}")
    
    df = download_price_data(ticker, start=start, end=end)
    df = add_technical_features(df)
    df = add_labels(df)
    
    df_train, df_test = split_train_test_by_date(df, test_size_years=2)
    
    # Clean data (same as backtest)
    df_test_clean = df_test.dropna(subset=[
        "daily_return", "ret_5", "ret_10", "ma_5", "ma_10", "ma_20",
        "std_5", "std_10", "volume_zscore_20", "rsi_14", "macd",
        "macd_signal", "bb_width", "lag_ret_1", "lag_ret_2", "lag_vol_1", "target"
    ])
    
    prices = df_test_clean['Close'].values
    dates = df_test_clean.index
    
    # Buy on day 1
    first_price = prices[0]
    shares = initial_capital / first_price
    
    # Track portfolio value each day
    portfolio_values = shares * prices
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(252)
    
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown)
    
    print(f"\nBuy & Hold Results:")
    print(f"  Final Portfolio Value: ${final_value:,.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"  Maximum Drawdown: {max_drawdown:.2%}")
    
    return {
        'model_name': 'Buy & Hold',
        'portfolio_values': portfolio_values.tolist(),
        'dates': dates,
        'trades': [1] * len(prices),  # always in market
        'prices': prices,
        'metrics': {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': None,
            'num_trades': 0,
            'market_exposure': 1.0,
        }
    }


def plot_backtest_results(results_list: List[Dict], ticker: str, save_path: str = "backtest_results.png"):
    """
    Create comprehensive visualization of backtest results.
    
    Creates 3 subplots:
    1. Portfolio value over time (all strategies)
    2. Drawdown over time
    3. Performance metrics comparison (bar chart)
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Color palette
    colors = sns.color_palette("husl", len(results_list))
    
    # 1. Portfolio Value Over Time
    ax1 = axes[0]
    for i, result in enumerate(results_list):
        dates = result['dates']
        values = result['portfolio_values']
        name = result['model_name']
        
        ax1.plot(dates, values, label=name, color=colors[i], linewidth=2)
    
    ax1.set_title(f'Portfolio Value Over Time - {ticker}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Drawdown Over Time
    ax2 = axes[1]
    for i, result in enumerate(results_list):
        dates = result['dates']
        values = np.array(result['portfolio_values'])
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak * 100  # as percentage
        name = result['model_name']
        
        ax2.fill_between(dates, 0, -drawdown, alpha=0.3, color=colors[i], label=name)
    
    ax2.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Metrics Comparison
    ax3 = axes[2]
    
    metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown']
    metric_labels = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']
    
    x = np.arange(len(results_list))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = [r['metrics'][metric] * (100 if metric in ['total_return', 'max_drawdown'] else 1) 
                  for r in results_list]
        offset = width * (i - 1)
        ax3.bar(x + offset, values, width, label=metric_labels[i], alpha=0.8)
    
    ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Strategy', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([r['model_name'] for r in results_list], rotation=15, ha='right')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {save_path}")
    plt.show()


def print_comparison_table(results_list: List[Dict]):
    """
    Print a formatted comparison table of all strategies.
    """
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON TABLE")
    print(f"{'='*80}")
    
    # Create DataFrame for easy formatting
    data = []
    for result in results_list:
        metrics = result['metrics']
        data.append({
            'Strategy': result['model_name'],
            'Final Value': f"${metrics['final_value']:,.2f}",
            'Total Return': f"{metrics['total_return']:.2%}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
            'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
            'Win Rate': f"{metrics['win_rate']:.2%}" if metrics['win_rate'] is not None else "N/A",
            'Num Trades': metrics['num_trades'],
            'Market Exposure': f"{metrics['market_exposure']:.1%}",
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    
    # Identify best performers
    best_return_idx = np.argmax([r['metrics']['total_return'] for r in results_list])
    best_sharpe_idx = np.argmax([r['metrics']['sharpe_ratio'] for r in results_list])
    
    print("🏆 Best Total Return:", results_list[best_return_idx]['model_name'])
    print("🏆 Best Risk-Adjusted Return (Sharpe):", results_list[best_sharpe_idx]['model_name'])


def main():
    """
    Run complete backtesting analysis.
    """
    ticker = "SPY"
    start = "2015-01-01"
    end = "2024-12-31"
    initial_capital = 10000
    
    print(f"\n{'#'*80}")
    print(f"BACKTESTING ANALYSIS: ML Trading Strategies vs Buy & Hold")
    print(f"Ticker: {ticker} | Period: {start} to {end}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"{'#'*80}")
    
    # Load trained models
    models = [
        (joblib.load("logreg_model.pkl"), "Logistic Regression"),
        (joblib.load("xgb_model.pkl"), "XGBoost"),
        (joblib.load("mlp_model.pkl"), "MLP Neural Network"),
    ]
    
    # Run backtests
    results = []
    
    for model, name in models:
        result = backtest_model(
            model=model,
            model_name=name,
            ticker=ticker,
            start=start,
            end=end,
            initial_capital=initial_capital,
            confidence_threshold=0.55,  # Only trade when 55%+ confident
        )
        results.append(result)
    
    # Add buy & hold baseline
    baseline = benchmark_buy_and_hold(ticker, start, end, initial_capital)
    results.append(baseline)
    
    # Visualize and compare
    plot_backtest_results(results, ticker)
    print_comparison_table(results)
    
    print("\n✅ Backtesting complete!")
    print("\nKey Insights:")
    print("- Look at Total Return: Did any ML model beat Buy & Hold?")
    print("- Look at Sharpe Ratio: Which strategy had best risk-adjusted returns?")
    print("- Look at Max Drawdown: Which strategy lost least during downturns?")
    print("- Look at Win Rate: What % of trades were profitable?")


if __name__ == "__main__":
    main()