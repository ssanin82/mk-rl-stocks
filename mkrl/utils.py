"""
Utility functions for trading strategy execution and metrics calculation.
"""

import numpy as np
from mkrl.settings import trading_fee_rate


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


def calculate_metrics(portfolio_values, initial_capital):
    """Calculate performance metrics from portfolio values."""
    final_capital = portfolio_values[-1]
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    peak = portfolio_values[0]
    max_dd = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        if dd > max_dd:
            max_dd = dd
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    volatility = np.std(returns) * 100
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'volatility': volatility,
        'total_pnl': final_capital - initial_capital
    }
