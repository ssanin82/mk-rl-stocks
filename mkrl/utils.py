"""
Utility functions for trading strategy execution and metrics calculation.
"""

import numpy as np


def run_strategy(env, model, log_file=None):
    """Run the trained model on the environment and log all actions."""
    obs, info = env.reset()
    actions = []
    portfolio_values = []
    trades = []  # Store trade details
    
    if log_file:
        log_file.write(f"{'Step':<6} {'Price':<10} {'Action':<8} {'Cash':<12} {'Holdings':<12} {'Portfolio':<12} {'Note':<20}\n")
        log_file.write("-" * 90 + "\n")
    
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
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record state after action
        cash_after = env.cash
        holdings_after = env.holdings
        # Use current price (step() increments current_step, so use the price at current_step if valid, otherwise last price)
        current_price_idx = min(env.current_step, len(env.prices) - 1)
        portfolio_after = env.cash + env.holdings * env.prices[current_price_idx]
        portfolio_values.append(portfolio_after)
        
        # Log action details
        if log_file:
            action_name = ["HOLD", "BUY", "SELL"][action]
            note = ""
            # Show if action was filtered (converted from invalid action to HOLD)
            if original_action != action:
                original_action_name = ["HOLD", "BUY", "SELL"][original_action]
                if original_action == 2:
                    note = f"[FILTERED: {original_action_name}->HOLD - no holdings]"
                elif original_action == 1:
                    note = f"[FILTERED: {original_action_name}->HOLD - insufficient cash]"
                # Action was filtered to HOLD, so skip detailed logging
            elif action == 1:  # Buy
                if cash_after < cash_before:  # Trade executed (cash decreased)
                    shares_bought = holdings_after - holdings_before
                    notional = shares_bought * price
                    fee = notional * env.trading_fee_rate
                    avg_entry = env.avg_entry_price if env.holdings > 0 else 0.0
                    note = f"Bought {shares_bought:.4f} @ ${price:.2f}, avg_entry=${avg_entry:.2f}"
                    trades.append({"step": len(actions)-1, "type": "BUY", "price": price, "shares": shares_bought})
                else:
                    # Invalid buy action (insufficient cash)
                    shares_needed = max(env.min_size, env.min_notional / price)
                    notional = shares_needed * price
                    fee = notional * env.trading_fee_rate
                    cost_needed = notional + fee
                    note = f"BUY invalid: need ${cost_needed:.2f}, have ${cash_before:.2f}"
            elif action == 2:  # Sell
                if holdings_after < holdings_before:  # Trade executed (holdings decreased)
                    shares_sold = holdings_before - holdings_after
                    notional = shares_sold * price
                    fee = notional * env.trading_fee_rate
                    proceeds = notional - fee
                    avg_entry = env.avg_entry_price if env.holdings > 0 else 0.0
                    note = f"Sold {shares_sold:.4f} @ ${price:.2f}, avg_entry=${avg_entry:.2f}"
                    trades.append({"step": len(actions)-1, "type": "SELL", "price": price, "shares": shares_sold})
                else:
                    # Invalid sell action (no holdings)
                    if holdings_before <= 0:
                        note = f"SELL invalid: no holdings (have {holdings_before:.4f} shares)"
                    else:
                        # Shouldn't happen, but log it
                        note = f"SELL invalid: insufficient holdings"
            
            log_file.write(f"{len(actions)-1:<6} ${price:<9.2f} {action_name:<8} ${cash_after:<11.2f} {holdings_after:<12.4f} ${portfolio_after:<11.2f} {note}\n")
    
    # Verify position is zero at the end
    final_position = env.holdings
    if final_position > 1e-6:  # Small tolerance for floating point
        if log_file:
            log_file.write(f"\nWARNING: Position not fully closed at end. Remaining position: {final_position:.6f} shares\n")
    
    return actions, portfolio_values, trades


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
