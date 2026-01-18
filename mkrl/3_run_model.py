"""
Script to run the trained model on last 10% of prices.txt
Loads the saved model and generates trading report.
"""

import sys
from pathlib import Path
import numpy as np
import time
import argparse
import webbrowser

# Add parent directory to Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from mkrl.env import TradingEnv
from mkrl.utils import run_strategy, calculate_metrics
from mkrl.web import create_static_html
from mkrl.settings import (
    initial_capital, min_notional, min_size, trading_fee_rate,
    default_prices_file, default_model_file, train_split_ratio
)


def load_prices(prices_file):
    """Load prices from file (one price per line)."""
    prices_path = Path(prices_file)
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices file not found: {prices_path}")
    
    prices = []
    with open(prices_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    prices.append(float(line))
                except ValueError:
                    continue
    
    if len(prices) == 0:
        raise ValueError(f"No valid prices found in {prices_path}")
    
    return np.array(prices)


def main():
    """Run the model on last 10% of prices."""
    parser = argparse.ArgumentParser(description='Run trained RL trading model')
    parser.add_argument('--prices', '-p', type=str, default=default_prices_file,
                        help=f'Input prices file (default: {default_prices_file})')
    parser.add_argument('--model', '-m', type=str, default=default_model_file,
                        help=f'Input model file (default: {default_model_file})')
    parser.add_argument('--split', type=float, default=train_split_ratio,
                        help=f'Training split ratio - test uses remaining (default: {train_split_ratio})')
    
    args = parser.parse_args()
    
    # Load prices
    print(f"Loading prices from {args.prices}...")
    all_prices = load_prices(args.prices)
    print(f"  Loaded {len(all_prices)} price points")
    
    # Split into test set (last 10%)
    split_idx = int(len(all_prices) * args.split)
    test_prices = all_prices[split_idx:]
    print(f"  Using last {len(test_prices)} prices ({(1-args.split)*100:.0f}%) for testing")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(str(model_path))
    print("✓ Model loaded successfully")
    
    # Create test environment
    print("\nCreating test environment...")
    env_test = TradingEnv(
        test_prices,
        initial_capital=initial_capital,
        min_notional=min_notional,
        min_size=min_size,
        trading_fee_rate=trading_fee_rate
    )
    
    # Run strategy
    print("\nRunning strategy on test data...")
    ts = time.time()
    
    log_filename = 'trading_report.log'
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("=" * 90 + "\n")
        log_file.write("TRADING REPORT\n")
        log_file.write("=" * 90 + "\n\n")
        actions, portfolio_values, trades = run_strategy(env_test, model, log_file=log_file)
        
        # Summary statistics
        log_file.write("\n" + "=" * 90 + "\n")
        log_file.write("ACTION SUMMARY\n")
        log_file.write("=" * 90 + "\n")
        action_counts = {0: 0, 1: 0, 2: 0}
        for action in actions:
            action_counts[action] += 1
        log_file.write(f"HOLD actions: {action_counts[0]} ({action_counts[0]/len(actions)*100:.1f}%)\n")
        log_file.write(f"BUY actions:  {action_counts[1]} ({action_counts[1]/len(actions)*100:.1f}%)\n")
        log_file.write(f"SELL actions: {action_counts[2]} ({action_counts[2]/len(actions)*100:.1f}%)\n")
        
        log_file.write(f"\nTotal trades executed: {len(trades)}\n")
        if trades:
            log_file.write("\nTrade Details:\n")
            log_file.write("-" * 90 + "\n")
            for trade in trades:
                log_file.write(f"Step {trade['step']:>5}: {trade['type']:<4} {trade['shares']:.4f} shares @ ${trade['price']:.2f}\n")
    
    execution_time = time.time() - ts
    print(f"✓ Execution complete! Took {round(execution_time, 3)} seconds")
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_values, initial_capital)
    
    print(f"\nResults:")
    print(f"Initial Capital: ${metrics['initial_capital']:.2f}")
    print(f"Final Capital: ${metrics['final_capital']:.2f}")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    
    # Print action statistics
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in actions:
        action_counts[action] += 1
    print(f"\nAction Distribution:")
    print(f"  HOLD: {action_counts[0]} ({action_counts[0]/len(actions)*100:.1f}%)")
    print(f"  BUY:  {action_counts[1]} ({action_counts[1]/len(actions)*100:.1f}%)")
    print(f"  SELL: {action_counts[2]} ({action_counts[2]/len(actions)*100:.1f}%)")
    print(f"Total trades executed: {len(trades)}")
    print(f"\nTrading report saved to: {log_filename}")
    
    # Create static HTML file
    print("\nGenerating static HTML report...")
    html_file = create_static_html(test_prices, actions, portfolio_values, metrics)
    html_path = Path(html_file).resolve()
    
    print(f"HTML report saved to: {html_path}")
    print("Opening in browser...")
    
    # Open in external browser
    webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
