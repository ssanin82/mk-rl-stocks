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
import re

# Add parent directory to Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from alpharl.env import TradingEnv
from alpharl.utils import calculate_metrics
from alpharl.web import create_static_html
import alpharl.settings as settings_module


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


def _find_model_file(model_file: str) -> Path:
    """
    Find a model file, checking models/ folder first, then root directory.
    
    Args:
        model_file: Path to model file (may be relative or absolute)
    
    Returns:
        Path to the found model file
    
    Raises:
        FileNotFoundError: If the model file is not found
    """
    model_path = Path(model_file)
    
    # If absolute path, use as-is
    if model_path.is_absolute():
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try models/ folder first
    models_dir = Path(__file__).parent.parent / "models"
    models_path = models_dir / model_path.name
    if models_path.exists():
        return models_path
    
    # Try root directory
    root_path = Path(__file__).parent.parent / model_path.name
    if root_path.exists():
        return root_path
    
    # Try as-is (might be relative to current working directory)
    if model_path.exists():
        return model_path.resolve()
    
    # Try with models/ prefix
    models_prefixed = models_dir / model_path
    if models_prefixed.exists():
        return models_prefixed
    
    raise FileNotFoundError(f"Model file not found: {model_file} (checked models/{model_path.name}, {model_path.name}, and {model_path})")


def extract_config_name_from_model(model_path: str) -> str:
    """
    Extract config name from model filename.
    E.g., 'model_base.zip' -> 'base', 'model_half_aggressive.zip' -> 'half_aggressive'
    """
    model_file = Path(model_path).name
    # Remove .zip extension
    model_base = model_file.replace('.zip', '')
    # Remove 'model_' prefix
    if model_base.startswith('model_'):
        config_name = model_base[6:]  # len('model_') = 6
    else:
        # Fallback: try to extract from any pattern
        match = re.match(r'model_(.+)', model_base)
        if match:
            config_name = match.group(1)
        else:
            config_name = "base"  # Default
    return config_name


def find_settings_file_for_config(config_name: str) -> str:
    """
    Find the settings file for a given config name.
    Returns the settings file path (looks in config/ folder).
    """
    if config_name == "base":
        return "config/settings.json"
    else:
        # Try config/<config_name>.json (new naming without "settings_" prefix)
        settings_file = f"config/{config_name}.json"
        settings_path = Path(settings_file)
        if settings_path.exists():
            return str(settings_path)
        
        # Fallback: Try config/settings_<config_name>.json (old naming for backward compatibility)
        settings_file_old = f"config/settings_{config_name}.json"
        settings_path_old = Path(settings_file_old)
        if settings_path_old.exists():
            return str(settings_path_old)
        
        # Final fallback to base settings
        print(f"  ⚠ Warning: Settings file {settings_file} or {settings_file_old} not found, using config/settings.json")
        return "config/settings.json"


def run_model_for_config(model_file: str, prices_file: str, split: float):
    """
    Run a single model and generate report.
    
    Args:
        model_file: Path to model zip file
        prices_file: Path to prices file
        split: Train/test split ratio
    
    Returns:
        Tuple of (config_name, html_path, log_path)
    """
    # Extract config name from model filename
    config_name = extract_config_name_from_model(model_file)
    
    print(f"\n{'='*70}")
    print(f"Running model: {config_name}")
    print(f"Model file: {model_file}")
    print(f"{'='*70}")
    
    # Load settings for this config
    settings_file = find_settings_file_for_config(config_name)
    print(f"Loading settings from {settings_file}...")
    settings_module.load_settings(settings_file)
    
    # Import settings after reload
    from alpharl.settings import (
        initial_capital, min_notional, min_size, trading_fee_rate, lot_size,
        default_prices_file, train_split_ratio,
        curriculum_forced_buy_delay, curriculum_forced_buy_size
    )
    
    # Load prices
    print(f"Loading prices from {prices_file}...")
    all_prices = load_prices(prices_file)
    print(f"  Loaded {len(all_prices)} price points")
    
    # Split into test set (last 10%)
    split_idx = int(len(all_prices) * split)
    test_prices = all_prices[split_idx:]
    print(f"  Using last {len(test_prices)} prices ({(1-split)*100:.0f}%) for testing")
    
    # Load model (find in models/ folder first)
    model_path = _find_model_file(model_file)
    
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
    
    log_filename = f'trading_report_{config_name}.log'
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("=" * 90 + "\n")
        log_file.write(f"TRADING REPORT - Config: {config_name}\n")
        log_file.write("=" * 90 + "\n\n")
        actions, portfolio_values, trades, force_sell_index, total_fees = run_strategy(
            env=env_test,
            model=model,
            log_file=log_file,
            force_initial_buys=True,  # Enable forced buys during evaluation
            forced_buy_delay=curriculum_forced_buy_delay,
            forced_buy_size=curriculum_forced_buy_size
        )
    
    # Calculate metrics (assume 1-minute data frequency) - outside file context
    metrics = calculate_metrics(portfolio_values, initial_capital, n_periods=len(test_prices), data_frequency_minutes=1)
    
    # Append metrics to log file
    with open(log_filename, 'a', encoding='utf-8') as log_file:
        # Performance metrics
        log_file.write("\n" + "=" * 90 + "\n")
        log_file.write("PERFORMANCE METRICS\n")
        log_file.write("=" * 90 + "\n")
        log_file.write(f"Initial Capital:     ${metrics['initial_capital']:.2f}\n")
        log_file.write(f"Final Capital:       ${metrics['final_capital']:.2f}\n")
        log_file.write(f"Total P&L:           ${metrics['total_pnl']:.2f}\n")
        log_file.write(f"Total Return:        {metrics['total_return']:.2f}%\n")
        log_file.write(f"Annualized Return:   {metrics['annualized_return']:.2f}%\n")
        log_file.write(f"Total Fees:          ${total_fees:.2f}\n")
        log_file.write(f"Max Drawdown:        {metrics['max_drawdown']:.2f}%\n")
        log_file.write(f"Volatility:          {metrics['volatility']:.2f}%\n")
        log_file.write(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.4f} (risk-free rate = 0)\n")
        log_file.write(f"Calmar Ratio:        {metrics['calmar_ratio']:.4f}\n")
        log_file.write(f"\nNote: Metrics assume 1-minute data frequency ({metrics['n_periods']} periods = {metrics['n_periods']/60:.2f} hours)\n")
        
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
    
    # Metrics already calculated above, reuse them
    print(f"\nResults for {config_name}:")
    print(f"Initial Capital:     ${metrics['initial_capital']:.2f}")
    print(f"Final Capital:       ${metrics['final_capital']:.2f}")
    print(f"Total P&L:           ${metrics['total_pnl']:.2f}")
    print(f"Total Return:        {metrics['total_return']:.2f}%")
    print(f"Annualized Return:   {metrics['annualized_return']:.2f}%")
    print(f"Max Drawdown:        {metrics['max_drawdown']:.2f}%")
    print(f"Volatility:          {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.4f} (risk-free rate = 0)")
    print(f"Calmar Ratio:        {metrics['calmar_ratio']:.4f}")
    
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
    
    # Create static HTML file (with config name in filename)
    print("\nGenerating static HTML report...")
    html_filename = f"trading_results_{config_name}.html"
    html_file = create_static_html(test_prices, actions, portfolio_values, metrics, 
                                    output_file=html_filename, force_sell_index=force_sell_index)
    html_path = Path(html_file).resolve()
    
    print(f"HTML report saved to: {html_path}")
    print("Opening in browser...")
    
    # Open in external browser
    webbrowser.open(html_path.as_uri())
    
    return config_name, html_path, Path(log_filename)


def run_strategy(env, model, log_file=None, force_initial_buys=True, forced_buy_delay=10, forced_buy_size=0.001):
    """
    Run the trading strategy using the trained model.
    
    Args:
        env: Trading environment
        model: Trained PPO model
        log_file: Optional file handle to write trading log
        force_initial_buys: Whether to force initial buys if no trade occurred
        forced_buy_delay: Steps before forcing a buy
        forced_buy_size: Size of forced buy
    
    Returns:
        Tuple of (actions, portfolio_values, trades, force_sell_index, total_fees)
    """
    # Gymnasium reset() returns (obs, info), handle both cases for compatibility
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, reset_info = reset_result
    else:
        obs = reset_result
        reset_info = {}
    done = False
    actions = []
    portfolio_values = []
    trades = []
    total_fees = 0.0
    force_sell_index = None
    
    prev_price = None
    prev_norm_price = None
    step_count = 0
    
    # Get normalization method abbreviation (max 8 characters)
    norm_method = env.normalization_method
    norm_abbrev_map = {
        "percentage_changes": "PctChg",
        "log_returns": "LogRet",
        "z-score": "ZScore",
        "price_ratio": "PriceRt"
    }
    norm_column_name = norm_abbrev_map.get(norm_method.value, "Norm")
    
    # Get DCA step value for current normalization method
    dca_step_map = {
        "percentage_changes": env._get_dca_step_value() if env.normalization_method.value == "percentage_changes" else None,
        "log_returns": env._get_dca_step_value() if env.normalization_method.value == "log_returns" else None,
        "z-score": env._get_dca_step_value() if env.normalization_method.value == "z-score" else None,
        "price_ratio": env._get_dca_step_value() if env.normalization_method.value == "price_ratio" else None,
    }
    current_dca_step = dca_step_map.get(norm_method.value, 0.001)
    
    if log_file:
        # Column header: show method name and format hint
        norm_header = f"{norm_column_name:<18}"  # Allow space for "ratio (move)" format
        dca_header = f"{'DCA':<12}"  # Allow space for DCA step value
        log_file.write(f"{'Step':<6} {'Price':<10} {norm_header} {dca_header} {'Action':<7} {'Flag':<7} {'Qty':<10} {'Cash':<12} {'Position':<12} {'Fee':<10} {'Portfolio':<12} {'AEP':<10} {'Note':<50}\n")
        log_file.write("-" * 170 + "\n")
    
    done = False
    forced_buy_executed = False  # Track if forced buy was executed this step
    while not done:
        # Record state BEFORE any action (including forced buy)
        price = env.prices[env.current_step]
        norm_price = env.normalized_prices[env.current_step]  # Get normalized price
        cash_before = env.cash
        holdings_before = env.holdings  # Record BEFORE forced buy (if any)
        
        # FORCED BUY LOGIC: Force initial buys during first N steps if no trade occurred
        forced_buy_executed = False  # Reset each iteration
        if force_initial_buys and step_count > 0 and step_count % forced_buy_delay == 0:
            if not env.has_ever_traded and env.holdings == 0:
                # Force a buy action
                shares = max(forced_buy_size, env.min_size)
                shares = env._round_to_lot_size(shares)
                notional = shares * price
                
                # FIXED: Improved auto-increase logic to ensure trades aren't blocked
                # If below min_notional, increase to meet requirement
                if notional < env.min_notional and price > 0:
                    min_shares_for_notional = env.min_notional / price
                    min_shares_for_notional = env._round_to_lot_size(min_shares_for_notional)
                    shares = max(shares, min_shares_for_notional)
                    notional = shares * price
                
                fee = notional * env.trading_fee_rate
                total_cost = notional + fee
                
                if env.cash >= total_cost:
                    # Execute forced buy (holdings_before already recorded above)
                    env.holdings += shares
                    env.cash -= total_cost
                    env.cash = max(0, env.cash)
                    cost_this_trade = notional
                    env.total_cost_basis += cost_this_trade
                    if env.holdings > 0:
                        env.avg_entry_price = env.total_cost_basis / env.holdings
                        current_norm_price = env.normalized_prices[env.current_step]
                        if env.normalized_entry_price == 0.0:
                            env.normalized_entry_price = current_norm_price
                        else:
                            # Volume-weighted average
                            old_shares = holdings_before
                            if old_shares > 0:
                                env.normalized_entry_price = (
                                    (env.normalized_entry_price * old_shares) + 
                                    (current_norm_price * shares)
                                ) / env.holdings
                            else:
                                env.normalized_entry_price = current_norm_price
                    env.has_ever_traded = True
                    env.last_trade_step = env.current_step
                    env.steps_since_last_trade = 0
                    env.steps_without_trade = 0  # Reset accumulating penalty
                    # FIXED: Set action to HOLD to prevent double execution in env.step()
                    action = 0  # HOLD - forced buy already executed, don't execute again
                    original_action = 0  # Track that this was a forced buy
                    forced_buy_executed = True  # Mark that forced buy happened
                else:
                    # Can't afford forced buy, proceed with model prediction
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                    original_action = action
            else:
                # Already traded or has position, proceed with model prediction
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                original_action = action
        else:
            # Normal model prediction
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            original_action = action
        
        # ACTION MASKING: REMOVED - Let environment handle invalid actions
        # This allows the model to learn from penalties for invalid actions
        # The environment will handle invalid actions appropriately and provide feedback
        # Observation contains can_buy (index 6) and can_sell (index 7) flags for model to learn
        can_buy = obs[6] > 0.5
        can_sell = obs[7] > 0.5
        
        # Only keep forced buys - don't filter model predictions
        # Let the model learn from trying invalid actions (they get penalties)
        
        step_count += 1
        
        # FIXED: Store executed action for chart, not attempted action
        # Only append SELL (2) if it will actually execute (has holdings)
        # Otherwise, store as HOLD (0) so chart doesn't show invalid SELL signals
        if action == 2 and holdings_before == 0:
            # Invalid SELL attempt - store as HOLD for chart
            actions.append(0)  # Store as HOLD
        else:
            actions.append(action)
        avg_entry_before = env.avg_entry_price if env.avg_entry_price > 0 else 0.0  # Get entry price before step
        
        # Calculate normalized price move
        if prev_norm_price is not None:
            norm_move = norm_price - prev_norm_price
        else:
            norm_move = 0.0
        
        # Calculate price move (for notes/descriptions)
        if prev_price is not None:
            price_move_pct = ((price - prev_price) / prev_price) * 100 if prev_price > 0 else 0.0
        else:
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
        
        # Update previous price and normalized price for next iteration
        prev_price = price
        prev_norm_price = norm_price
        
        # Calculate fees for any trades executed (always track, even if not logging)
        fee_this_step = 0.0
        qty_this_step = 0.0
        
        # Check if force-sell executed
        if force_sell_executed:
            qty_this_step = -force_sell_shares  # Negative for sells
            notional = force_sell_shares * force_sell_price
            fee_this_step = notional * env.trading_fee_rate
            total_fees += fee_this_step
        # Check if position management rules executed a trade
        elif position_mgmt_triggered and position_mgmt_action:
            if "PROFIT_TAKE" in position_mgmt_action or "SELL" in position_mgmt_action:
                shares_sold = holdings_before - holdings_after
                if shares_sold > 0:
                    qty_this_step = -shares_sold  # Negative for sells
                    notional = shares_sold * price
                    fee_this_step = notional * env.trading_fee_rate
                    total_fees += fee_this_step
            elif "DCA" in position_mgmt_action or "BUY" in position_mgmt_action:
                shares_bought_auto = holdings_after - holdings_before
                if shares_bought_auto > 0:
                    qty_this_step = shares_bought_auto  # Positive for buys
                    notional = shares_bought_auto * price
                    fee_this_step = notional * env.trading_fee_rate
                    total_fees += fee_this_step
        # Model actions
        elif action == 1 and cash_after < cash_before:  # BUY executed
            shares_bought = holdings_after - holdings_before
            if shares_bought > 0:
                qty_this_step = shares_bought  # Positive for buys
                notional = shares_bought * price
                fee_this_step = notional * env.trading_fee_rate
                total_fees += fee_this_step
        elif action == 2 and holdings_after < holdings_before:  # SELL executed
            shares_sold = holdings_before - holdings_after
            if shares_sold > 0:
                qty_this_step = -shares_sold  # Negative for sells
                notional = shares_sold * price
                fee_this_step = notional * env.trading_fee_rate
                total_fees += fee_this_step
        
        # Log action details
        if log_file:
            # Determine flag: "FORCE" for force-sell, "AUTO" for auto trades, empty otherwise
            flag = ""
            qty = qty_this_step  # Use calculated quantity
            fee = fee_this_step  # Use calculated fee
            note = ""  # Explanation of why action was taken
            
            # Check if forced buy executed (highest priority for logging)
            if forced_buy_executed:
                action_name = "BUY"  # Forced buy should show as BUY
                flag = "FORCE"
                # Calculate qty from holdings change (forced buy shares)
                forced_buy_shares = holdings_after - holdings_before
                qty = forced_buy_shares  # Positive for buys
                if forced_buy_shares > 0:
                    notional = forced_buy_shares * price
                    fee = notional * env.trading_fee_rate
                    total_fees += fee  # Add to total fees
                note = f"Forced buy - initial entry ({forced_buy_shares:.4f} shares)"
                trades.append({"step": len(actions)-1, "type": "BUY (FORCE)", "price": price, "shares": forced_buy_shares})
            # Check if force-sell executed (highest priority for logging)
            elif force_sell_executed:
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
                # FIXED: Check if SELL was actually executed before setting action_name
                if holdings_after < holdings_before:  # Trade executed (holdings decreased)
                    action_name = "SELL"  # Only set to SELL if actually executed
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
                    # FIXED: Invalid SELL attempt - show as HOLD, not SELL
                    action_name = "HOLD"  # Changed from SELL to HOLD for invalid attempts
                    qty = 0.0
                    fee = 0.0
                    note = "INVALID SELL - attempted but no holdings (position = 0)"
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
            
            # Format normalized price ratio and move based on method
            if norm_method.value == "percentage_changes":
                # Percentage changes: show as percentage with move
                norm_display = f"{norm_price*100:>+8.4f}%"
                norm_move_display = f"{norm_move*100:>+7.4f}%"
            elif norm_method.value == "log_returns":
                # Log returns: show as percentage (scaled) with move
                norm_display = f"{norm_price*100:>+8.4f}%"
                norm_move_display = f"{norm_move*100:>+7.4f}%"
            elif norm_method.value == "z-score":
                # Z-score: show as decimal with move
                norm_display = f"{norm_price:>+9.4f}"
                norm_move_display = f"{norm_move:>+8.4f}"
            elif norm_method.value == "price_ratio":
                # Price ratio: show as ratio with move
                norm_display = f"{norm_price:>+9.4f}"
                norm_move_display = f"{norm_move:>+8.4f}"
            else:
                # Fallback
                norm_display = f"{norm_price:>+9.4f}"
                norm_move_display = f"{norm_move:>+8.4f}"
            
            # Format: show ratio (move) - e.g., "+0.0152% (+0.0152%)" or "+1.0002 (+0.0002)"
            norm_column_value = f"{norm_display} ({norm_move_display})"
            
            # Get current DCA step value
            current_dca_step = env._get_dca_step_value()
            
            # Format DCA step value based on normalization method
            if norm_method.value == "percentage_changes":
                dca_display = f"{current_dca_step*100:.4f}%"
            elif norm_method.value == "log_returns":
                dca_display = f"{current_dca_step*100:.4f}%"
            elif norm_method.value == "z-score":
                dca_display = f"{current_dca_step:.2f}σ"
            elif norm_method.value == "price_ratio":
                dca_display = f"{current_dca_step*100:.2f}%"
            else:
                dca_display = f"{current_dca_step:.6f}"
            
            log_file.write(
                f"{len(actions)-1:<6} ${price:<9.2f} "
                f"{norm_column_value:<20} "
                f"{dca_display:<12} "
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


def main():
    """Run the model(s) on last 10% of prices."""
    parser = argparse.ArgumentParser(description='Run trained RL trading model(s)')
    parser.add_argument('--prices', '-p', type=str, default=None,
                        help='Input prices file (default: from config/settings.json)')
    parser.add_argument('--split', type=float, default=None,
                        help='Training split ratio - test uses remaining (default: from config/settings.json)')
    parser.add_argument('model_files', nargs='*',
                        help='Model zip file(s) to run (default: models/model_base.zip)')
    
    args = parser.parse_args()
    
    # Determine model files to use
    if args.model_files:
        model_files = args.model_files
    else:
        # Default to models/model_base.zip
        model_files = ["models/model_base.zip"]
    
    print(f"\n{'='*70}")
    print(f"RUNNING {len(model_files)} MODEL(S)")
    print(f"{'='*70}")
    print(f"Model files: {', '.join(model_files)}")
    
    # Load default settings to get default values for prices, split
    settings_module.load_settings("config/settings.json")
    from alpharl.settings import default_prices_file, train_split_ratio
    
    prices_file = args.prices or default_prices_file
    split = args.split if args.split is not None else train_split_ratio
    
    # Run each model
    results = []
    for i, model_file in enumerate(model_files, 1):
        print(f"\n{'#'*70}")
        print(f"# Running model {i}/{len(model_files)}")
        print(f"{'#'*70}")
        try:
            config_name, html_path, log_path = run_model_for_config(
                model_file=model_file,
                prices_file=prices_file,
                split=split
            )
            results.append((config_name, html_path, log_path))
        except Exception as e:
            print(f"\n✗ ERROR running model {model_file}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("EXECUTION SUMMARY")
    print(f"{'='*70}")
    for config_name, html_path, log_path in results:
        print(f"  {config_name}:")
        print(f"    Report: {log_path}")
        print(f"    HTML:   {html_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
