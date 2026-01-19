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
from mkrl.utils import calculate_metrics
from mkrl.web import create_static_html
from mkrl.settings import (
    initial_capital, min_notional, min_size, trading_fee_rate, lot_size,
    default_prices_file, default_model_file, train_split_ratio
)


def run_strategy(env, model, log_file=None):
    """Run the trained model on the environment and log all actions."""
    obs, info = env.reset()
    actions = []
    portfolio_values = []
    trades = []  # Store trade details
    prev_price = None  # Track previous price for price move calculations
    force_sell_index = None  # Track the last force-sell step
    
    total_fees = 0.0  # Track total trading fees paid
    
    if log_file:
        log_file.write(f"{'Step':<6} {'Price':<10} {'Move %':<10} {'Move $':<10} {'Action':<7} {'Flag':<7} {'Qty':<10} {'Cash':<12} {'Position':<12} {'Fee':<10} {'Portfolio':<12} {'AEP':<10} {'Note':<50}\n")
        log_file.write("-" * 170 + "\n")
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        # ACTION MASKING: Filter invalid actions
        # Observation contains can_buy (index 6) and can_sell (index 7) flags
        can_buy = obs[6] > 0.5
        can_sell = obs[7] > 0.5
        
        original_action = action
        if action == 1 and not can_buy:  # BUY but can't afford it
            action = 0  # Convert to HOLD
        elif action == 2 and not can_sell:  # SELL but no holdings
            action = 0  # Convert to HOLD
        
        actions.append(action)
        
        # Record state before action
        price = env.prices[env.current_step]
        cash_before = env.cash
        holdings_before = env.holdings
        avg_entry_before = env.avg_entry_price if env.avg_entry_price > 0 else 0.0  # Get entry price before step
        
        # Calculate price move
        if prev_price is not None:
            price_move_dollar = price - prev_price
            price_move_pct = (price_move_dollar / prev_price) * 100 if prev_price > 0 else 0.0
        else:
            price_move_dollar = 0.0
            price_move_pct = 0.0
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Check if position management rules triggered a trade
        position_mgmt_triggered = info.get('position_mgmt_triggered', False)
        position_mgmt_action = info.get('position_mgmt_action', None)
        
        # Check if force-sell happened at the end
        force_sell_executed = info.get('force_sell_executed', False)
        force_sell_shares = info.get('force_sell_shares', 0.0)
        force_sell_price = info.get('force_sell_price', price)
        
        # Record state after action
        cash_after = env.cash
        holdings_after = env.holdings
        position = holdings_after  # Position is the same as holdings
        # Use current price (step() increments current_step, so use the price at current_step if valid, otherwise last price)
        current_price_idx = min(env.current_step, len(env.prices) - 1)
        portfolio_after = env.cash + env.holdings * env.prices[current_price_idx]
        portfolio_values.append(portfolio_after)
        
        # Update previous price for next iteration
        prev_price = price
        
        # Calculate fees for any trades executed (always track, even if not logging)
        fee_this_step = 0.0
        qty_this_step = 0.0
        
        # Check if force-sell executed
        if force_sell_executed:
            qty_this_step = -force_sell_shares  # Negative for sells
            notional = force_sell_shares * force_sell_price
            fee_this_step = notional * trading_fee_rate
            total_fees += fee_this_step
        # Check if position management rules executed a trade
        elif position_mgmt_triggered and position_mgmt_action:
            if "PROFIT_TAKE" in position_mgmt_action or "SELL" in position_mgmt_action:
                shares_sold = holdings_before - holdings_after
                if shares_sold > 0:
                    qty_this_step = -shares_sold  # Negative for sells
                    notional = shares_sold * price
                    fee_this_step = notional * trading_fee_rate
                    total_fees += fee_this_step
            elif "DCA" in position_mgmt_action or "BUY" in position_mgmt_action:
                shares_bought_auto = holdings_after - holdings_before
                if shares_bought_auto > 0:
                    qty_this_step = shares_bought_auto  # Positive for buys
                    notional = shares_bought_auto * price
                    fee_this_step = notional * trading_fee_rate
                    total_fees += fee_this_step
        # Model actions
        elif action == 1 and cash_after < cash_before:  # BUY executed
            shares_bought = holdings_after - holdings_before
            if shares_bought > 0:
                qty_this_step = shares_bought  # Positive for buys
                notional = shares_bought * price
                fee_this_step = notional * trading_fee_rate
                total_fees += fee_this_step
        elif action == 2 and holdings_after < holdings_before:  # SELL executed
            shares_sold = holdings_before - holdings_after
            if shares_sold > 0:
                qty_this_step = -shares_sold  # Negative for sells
                notional = shares_sold * price
                fee_this_step = notional * trading_fee_rate
                total_fees += fee_this_step
        
        # Log action details
        if log_file:
            # Determine flag: "FORCE" for force-sell, "AUTO" for auto trades, empty otherwise
            flag = ""
            qty = qty_this_step  # Use calculated quantity
            fee = fee_this_step  # Use calculated fee
            note = ""  # Explanation of why action was taken
            
            # Check if force-sell executed (highest priority for logging)
            if force_sell_executed:
                action_name = "SELL"  # Force-sell should show as SELL, not HOLD
                flag = "FORCE"
                qty = qty_this_step  # Already calculated
                fee = fee_this_step  # Already calculated
                note = "End of episode - force sell remaining position"
                force_sell_index = len(actions) - 1  # Track the force-sell step
                trades.append({"step": len(actions)-1, "type": "SELL (FORCE)", "price": force_sell_price, "shares": force_sell_shares})
            # Check if position management rules executed a trade
            elif position_mgmt_triggered and position_mgmt_action:
                flag = "AUTO"
                # Determine action name based on position management type
                if "PROFIT_TAKE" in position_mgmt_action or "SELL" in position_mgmt_action:
                    action_name = "SELL"
                    # Extract shares sold from holdings change
                    shares_sold = holdings_before - holdings_after
                    if shares_sold > 0:
                        qty = qty_this_step  # Already calculated
                        fee = fee_this_step  # Already calculated
                        # Extract reason from position_mgmt_action (e.g., "PROFIT_TAKE: Sold ...")
                        note = position_mgmt_action if position_mgmt_action else "Profit-taking rule triggered"
                        trades.append({"step": len(actions)-1, "type": "SELL (AUTO)", "price": price, "shares": shares_sold})
                elif "DCA" in position_mgmt_action or "BUY" in position_mgmt_action:
                    action_name = "BUY"
                    # Extract shares bought from holdings change
                    shares_bought_auto = holdings_after - holdings_before
                    if shares_bought_auto > 0:
                        qty = qty_this_step  # Already calculated
                        fee = fee_this_step  # Already calculated
                        # Extract reason from position_mgmt_action (e.g., "DCA: Added ...")
                        note = position_mgmt_action if position_mgmt_action else "DCA rule triggered"
                        trades.append({"step": len(actions)-1, "type": "BUY (AUTO)", "price": price, "shares": shares_bought_auto})
                else:
                    action_name = ["HOLD", "BUY", "SELL"][action]
            # Show if action was filtered (converted from invalid action to HOLD)
            elif original_action != action:
                action_name = "HOLD"  # Action was filtered to HOLD
                # Determine why action was filtered
                if original_action == 1:  # Tried to BUY but couldn't
                    note = "Invalid BUY filtered - insufficient cash for minimum trade"
                elif original_action == 2:  # Tried to SELL but couldn't
                    note = "Invalid SELL filtered - no holdings to sell"
                else:
                    note = "Action filtered - invalid conditions"
            elif action == 1:  # Buy
                action_name = "BUY"
                if cash_after < cash_before:  # Trade executed (cash decreased)
                    shares_bought = holdings_after - holdings_before
                    qty = qty_this_step  # Already calculated
                    fee = fee_this_step  # Already calculated
                    
                    # Create descriptive note
                    price_dir = "rising" if price_move_pct > 0 else "falling" if price_move_pct < 0 else "stable"
                    if holdings_before > 0 and avg_entry_before > 0:
                        # Adding to existing position
                        entry_ratio = price / avg_entry_before
                        if entry_ratio < 0.99:  # Buying below entry (DCA-like)
                            note = f"Model buy - adding to position (price {price_dir}, {entry_ratio*100-100:.2f}% vs entry)"
                        else:
                            note = f"Model buy - adding to position (price {price_dir})"
                    elif holdings_before > 0:
                        note = f"Model buy - adding to position (price {price_dir})"
                    else:
                        # New position entry
                        note = f"Model buy - entering position (price {price_dir})"
                    
                    trades.append({"step": len(actions)-1, "type": "BUY", "price": price, "shares": shares_bought})
                else:
                    qty = 0.0
                    fee = 0.0
                    note = "Model buy - attempted but not executed"
            elif action == 2:  # Sell
                action_name = "SELL"
                if holdings_after < holdings_before:  # Trade executed (holdings decreased)
                    shares_sold = holdings_before - holdings_after
                    qty = qty_this_step  # Already calculated
                    fee = fee_this_step  # Already calculated
                    
                    # Create descriptive note
                    price_dir = "rising" if price_move_pct > 0 else "falling" if price_move_pct < 0 else "stable"
                    if holdings_before > 0 and avg_entry_before > 0:
                        entry_ratio = price / avg_entry_before
                        profit_status = f"{entry_ratio*100-100:+.2f}% vs entry"
                        if entry_ratio > 1.001:  # More than 0.1% profit
                            note = f"Model sell - taking profit (price {price_dir}, {profit_status})"
                        elif entry_ratio < 0.999:  # More than 0.1% loss
                            note = f"Model sell - cutting losses (price {price_dir}, {profit_status})"
                        else:
                            note = f"Model sell - reducing position (price {price_dir}, {profit_status})"
                    elif holdings_before > 0:
                        note = f"Model sell - reducing position (price {price_dir})"
                    else:
                        note = f"Model sell - exiting position (price {price_dir})"
                    
                    trades.append({"step": len(actions)-1, "type": "SELL", "price": price, "shares": shares_sold})
                else:
                    qty = 0.0
                    fee = 0.0
                    note = "Model sell - attempted but not executed"
            else:  # HOLD
                action_name = "HOLD"
                # Create more descriptive note for HOLD
                price_dir = "rising" if price_move_pct > 0 else "falling" if price_move_pct < 0 else "stable"
                if holdings_before > 0 and avg_entry_before > 0:
                    entry_ratio = price / avg_entry_before
                    if entry_ratio > 1.01:
                        note = f"Holding position - price {price_dir}, {entry_ratio*100-100:+.2f}% profit"
                    elif entry_ratio < 0.99:
                        note = f"Holding position - price {price_dir}, {entry_ratio*100-100:+.2f}% loss"
                    else:
                        note = f"Holding position - price {price_dir}, near entry"
                elif holdings_before > 0:
                    note = f"Holding position - price {price_dir}"
                else:
                    note = f"Model hold - no position, price {price_dir}"
            
            # Get average entry price for AEP column (0.0 if no position)
            avg_entry_price = env.avg_entry_price if env.holdings > 0 else 0.0
            
            log_file.write(
                f"{len(actions)-1:<6} ${price:<9.2f} "
                f"{price_move_pct:>+8.4f}% "
                f"${price_move_dollar:>+9.2f} "
                f"{action_name:<7} "
                f"{flag:<7} "
                f"{qty:>+10.4f} "
                f"${cash_after:<11.2f} "
                f"{position:<12.4f} "
                f"${fee:<10.6f} "
                f"${portfolio_after:<11.2f} "
                f"${avg_entry_price:<9.2f} "
                f"{note:<50}\n"
            )
    
    # Verify position is zero at the end
    final_position = env.holdings
    if final_position > 1e-6:  # Small tolerance for floating point
        if log_file:
            log_file.write(f"\nWARNING: Position not fully closed at end. Remaining position: {final_position:.6f} shares\n")
    
    return actions, portfolio_values, trades, force_sell_index, total_fees


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
        trading_fee_rate=trading_fee_rate,
        lot_size=lot_size
    )
    
    # Run strategy
    print("\nRunning strategy on test data...")
    ts = time.time()
    
    log_filename = 'trading_report.log'
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("=" * 90 + "\n")
        log_file.write("TRADING REPORT\n")
        log_file.write("=" * 90 + "\n\n")
        actions, portfolio_values, trades, force_sell_index, total_fees = run_strategy(env_test, model, log_file=log_file)
        
        # Calculate metrics
        metrics = calculate_metrics(portfolio_values, initial_capital)
        
        # Performance metrics
        log_file.write("\n" + "=" * 90 + "\n")
        log_file.write("PERFORMANCE METRICS\n")
        log_file.write("=" * 90 + "\n")
        log_file.write(f"Initial Capital: ${metrics['initial_capital']:.2f}\n")
        log_file.write(f"Final Capital:   ${metrics['final_capital']:.2f}\n")
        log_file.write(f"Total P&L:       ${metrics['total_pnl']:.2f}\n")
        log_file.write(f"Total Return:    {metrics['total_return']:.2f}%\n")
        log_file.write(f"Total Fees:      ${total_fees:.2f}\n")
        log_file.write(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%\n")
        log_file.write(f"Volatility:      {metrics['volatility']:.2f}%\n")
        
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
    
    # Calculate metrics (for console output)
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
    html_file = create_static_html(test_prices, actions, portfolio_values, metrics, force_sell_index=force_sell_index)
    html_path = Path(html_file).resolve()
    
    print(f"HTML report saved to: {html_path}")
    print("Opening in browser...")
    
    # Open in external browser
    webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
