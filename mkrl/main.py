import sys
from pathlib import Path

# Add parent directory to Python path if running as a script (not as a module)
if __name__ == "__main__" and __file__:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
import webbrowser
import time

from stable_baselines3 import PPO
from mkrl.web import create_static_html

# Configuration constants
N_PRICE_POINTS = 10000  # Total number of price points to generate
INITIAL_CAPITAL = 5000  # Strategy's starting capital in dollars
MIN_NOTIONAL = 5.0  # Minimum dollar amount per trade (price * shares)
MIN_SIZE = 0.1  # Minimum number of shares per trade
TRADING_FEE_RATE = 0.0001  # Trading fee as fraction (reduced for better learning: 0.01% = 0.0001)
TRAINING_TIMESTEPS = 200000  # Number of training timesteps (more = better learning)

# Reward shaping parameters
TRADE_EXECUTION_REWARD = 0.01  # Small positive reward for executing a valid trade (encourages exploration)
MOMENTUM_REWARD_SCALE = 0.05  # Bonus for trading in the direction of price momentum

# Position management thresholds (configurable)
PROFIT_THRESHOLD = 0.10  # 10% profit threshold for partial sell
PARTIAL_SELL_RATIO = 0.20  # Sell 20% of position when profit threshold reached
DCA_THRESHOLD = 0.10  # 10% down from average entry for adding to position
DCA_RATIO = 0.10  # Add 10% to position when down threshold reached


class TradingEnv(gym.Env):
    def __init__(self, prices, initial_capital=1000, min_notional=5.0, min_size=0.1, trading_fee_rate=0.001):
        super().__init__()
        self.prices = prices
        self.initial_capital = initial_capital
        self.min_notional = min_notional
        self.min_size = min_size
        self.trading_fee_rate = trading_fee_rate
        self.current_step = 0
        self.cash = initial_capital
        self.holdings = 0
        self.avg_entry_price = 0.0  # Volume-weighted average entry price
        self.total_cost_basis = 0.0  # Total cost basis for calculating average entry price
        self.action_space = gym.spaces.Discrete(3)
        # Observation: [price_norm, price_change, relative_price, cash_ratio, holdings_ratio, entry_price_ratio, can_buy, can_sell]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.holdings = 0
        self.avg_entry_price = 0.0
        self.total_cost_basis = 0.0
        return self._get_obs(), {}
    
    def _get_obs(self):
        price = self.prices[self.current_step]
        # Add price history (last 5 prices) and price change indicators
        if self.current_step >= 1:
            price_change = (price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
        else:
            price_change = 0.0
        
        # Add relative price (normalized by initial price)
        relative_price = price / self.prices[0] if len(self.prices) > 0 else 1.0
        
        # Add cash and holdings as percentage of initial capital
        cash_ratio = self.cash / self.initial_capital
        holdings_value = self.holdings * price
        holdings_ratio = holdings_value / self.initial_capital if self.initial_capital > 0 else 0.0
        
        # Add average entry price ratio (relative to current price)
        # If no position, use 1.0 (neutral signal)
        if self.holdings > 0 and self.avg_entry_price > 0:
            entry_price_ratio = price / self.avg_entry_price
        else:
            entry_price_ratio = 1.0
        
        # Action validity flags (tell model which actions are valid)
        # Check if we can afford minimum buy
        shares_needed_for_notional = self.min_notional / price if price > 0 else 0
        min_trade_shares = max(self.min_size, shares_needed_for_notional)
        min_trade_cost = min_trade_shares * price * (1 + self.trading_fee_rate)
        can_buy = 1.0 if self.cash >= min_trade_cost else 0.0
        can_sell = 1.0 if self.holdings > 0 else 0.0
        
        return np.array([
            price / 100.0,  # Normalize price (rough normalization)
            price_change * 100,  # Price change percentage
            relative_price,  # Relative price
            cash_ratio,  # Cash as ratio
            holdings_ratio,  # Holdings value as ratio
            entry_price_ratio,  # Current price / average entry price
            can_buy,  # 1.0 if can buy, 0.0 if cannot
            can_sell,  # 1.0 if can sell, 0.0 if cannot
        ], dtype=np.float32)
    
    def step(self, action):
        price = self.prices[self.current_step]
        
        # Store previous portfolio value for incremental reward calculation
        prev_price = self.prices[self.current_step - 1] if self.current_step > 0 else price
        prev_portfolio_value = self.cash + self.holdings * prev_price
        
        # Action masking: prevent invalid actions
        # Action 0: HOLD - always valid
        # Action 1: BUY - only valid if we have enough cash
        # Action 2: SELL - only valid if we have holdings > 0
        
        action_executed = False
        
        if action == 1:  # BUY
            # Check if we can afford a minimum trade
            shares_needed_for_notional = self.min_notional / price if price > 0 else 0
            trade_shares = max(self.min_size, shares_needed_for_notional)
            notional = trade_shares * price
            fee = notional * self.trading_fee_rate
            total_cost = notional + fee
            
            # Only execute if we have enough cash
            if self.cash >= total_cost:
                # Execute buy
                self.holdings += trade_shares
                self.cash -= total_cost
                self.cash = max(0, self.cash)  # Safety check
                
                # Update volume-weighted average entry price
                # avg_entry = (old_total_cost + new_cost) / (old_shares + new_shares)
                cost_this_trade = notional  # Cost basis is price * shares (not including fee)
                self.total_cost_basis += cost_this_trade
                if self.holdings > 0:
                    self.avg_entry_price = self.total_cost_basis / self.holdings
                else:
                    self.avg_entry_price = 0.0
                
                action_executed = True
            # If not enough cash, action is ignored (effectively HOLD)
            
        elif action == 2:  # SELL
            # Only execute if we have holdings
            if self.holdings > 0:
                # Calculate trade size: at least min_size, and notional at least min_notional
                shares_needed_for_notional = self.min_notional / price if price > 0 else 0
                trade_shares = max(self.min_size, shares_needed_for_notional)
                
                # Don't sell more than we have
                trade_shares = min(trade_shares, self.holdings)
                
                # Check if trade meets minimum notional (unless we're selling all remaining holdings)
                notional = trade_shares * price
                # Allow sell if: (1) meets min_notional, OR (2) selling all holdings
                if (notional >= self.min_notional or trade_shares >= self.holdings) and trade_shares > 0:
                    fee = notional * self.trading_fee_rate
                    proceeds = notional - fee
                    
                    # Update average entry price (volume-weighted - remove sold portion)
                    # When selling, we reduce total_cost_basis proportionally
                    if self.holdings > 0:
                        cost_basis_of_sold = self.total_cost_basis * (trade_shares / self.holdings)
                        self.total_cost_basis -= cost_basis_of_sold
                        self.holdings -= trade_shares
                        if self.holdings > 0:
                            self.avg_entry_price = self.total_cost_basis / self.holdings
                        else:
                            self.avg_entry_price = 0.0
                            self.total_cost_basis = 0.0
                    else:
                        self.holdings = 0.0
                        self.avg_entry_price = 0.0
                        self.total_cost_basis = 0.0
                    
                    self.cash += proceeds
                    self.cash = max(0, self.cash)  # Safety check
                    action_executed = True
            # If no holdings, action is ignored (effectively HOLD)
            
        # Action 0 (HOLD) or invalid actions: do nothing
        
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        # Position constraint: must end with 0 position (all holdings sold)
        # Force sell everything at the end, ignoring minimum notional
        if done and self.holdings > 0:
            current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
            notional = self.holdings * current_price
            fee = notional * self.trading_fee_rate
            proceeds = notional - fee
            self.cash += proceeds
            self.holdings = 0.0
            self.avg_entry_price = 0.0
            self.total_cost_basis = 0.0
        
        terminated = done
        truncated = False
        
        # Calculate portfolio value using current price
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        portfolio_value = self.cash + self.holdings * current_price
        
        # Calculate remaining steps (how close to end)
        remaining_steps = len(self.prices) - self.current_step - 1
        total_steps = len(self.prices) - 1
        steps_remaining_ratio = remaining_steps / max(total_steps, 1)
        
        # Reward function: maximize Total P&L with intermediate rewards
        # Use incremental portfolio change for better learning signal
        portfolio_change = portfolio_value - self.initial_capital
        prev_portfolio_change = prev_portfolio_value - self.initial_capital
        
        # Intermediate reward: change in portfolio value this step
        # This gives immediate feedback at every step, making learning easier
        incremental_change = portfolio_change - prev_portfolio_change
        reward = incremental_change / self.initial_capital * 1000  # Scale for better learning
        
        # 1. Reward for executing trades (encourages exploration and learning)
        if action_executed:
            reward += TRADE_EXECUTION_REWARD
        
        # 4. Momentum-based reward shaping (bonus for trading in the direction of price movement)
        if self.current_step > 0:
            price_change_pct = (price - prev_price) / prev_price if prev_price > 0 else 0.0
            if action == 1 and action_executed and price_change_pct > 0:  # Buy when price rising
                momentum_bonus = MOMENTUM_REWARD_SCALE * abs(price_change_pct) * 100  # Scale by momentum strength
                reward += momentum_bonus
            elif action == 2 and action_executed and price_change_pct < 0:  # Sell when price falling
                momentum_bonus = MOMENTUM_REWARD_SCALE * abs(price_change_pct) * 100
                reward += momentum_bonus
            # Small penalty for trading against momentum (discourages bad timing)
            elif action == 1 and action_executed and price_change_pct < 0:  # Buy when price falling
                momentum_penalty = -MOMENTUM_REWARD_SCALE * 0.5 * abs(price_change_pct) * 100
                reward += momentum_penalty
            elif action == 2 and action_executed and price_change_pct > 0:  # Sell when price rising
                momentum_penalty = -MOMENTUM_REWARD_SCALE * 0.5 * abs(price_change_pct) * 100
                reward += momentum_penalty
        
        # Add bonus at episode end based on total P&L (shape final behavior)
        if done:
            total_pnl_reward = portfolio_change / self.initial_capital * 100
            reward += total_pnl_reward
        
        # Penalty for having position near the end (encourage closing position before end)
        # Penalty increases as we approach the end
        if steps_remaining_ratio < 0.1 and self.holdings > 0:  # Last 10% of episode
            # Calculate position value as ratio of initial capital for penalty scaling
            position_value = self.holdings * current_price
            position_ratio = position_value / self.initial_capital if self.initial_capital > 0 else 0
            position_penalty = -5.0 * (1 - steps_remaining_ratio) * position_ratio
            reward += position_penalty
        
        return self._get_obs(), reward, terminated, truncated, {}


def run_strategy(env, model, log_file=None):
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
        # portfolio_before = cash_before + holdings_before * price
        
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


def realistic_price_feed(
    S0=100,
    mu=0.0001,
    sigma0=0.01,
    alpha=0.05,     # reaction to shocks
    beta=0.9,       # volatility persistence
    jump_prob=0.01,
    jump_scale=0.05,
    n=N_PRICE_POINTS,
    seed=42
):
    np.random.seed(seed)
    prices = [S0]
    sigma = sigma0
    ret_prev = 0
    for _ in range(n):
        # GARCH-like volatility update
        sigma = np.sqrt(
            alpha * ret_prev**2 +
            beta * sigma**2 +
            (1 - alpha - beta) * sigma0**2
        )
        # Jump component
        jump = 0
        if np.random.rand() < jump_prob:
            jump = np.random.normal(0, jump_scale)

        # Return
        ret = mu + sigma * np.random.normal() + jump
        prices.append(prices[-1] * np.exp(ret))
        ret_prev = ret
    return np.array(prices)


def main():
    """Main entry point for the trading simulator."""
    # Generate synthetic price data
    all_prices = realistic_price_feed()
    
    # Split into training (90%) and testing (10%) sets
    split_idx = int(len(all_prices) * 0.9)
    train_prices = all_prices[:split_idx]
    test_prices = all_prices[split_idx:]
    
    print(f"Generated {len(all_prices)} price points")
    print(f"Training set: {len(train_prices)} price points (first 90%)")
    print(f"Testing set: {len(test_prices)} price points (last 10%)")

    # Train RL agent on training data
    print("\nTraining RL agent on training data...")
    ts = time.time()
    env_train = TradingEnv(
        train_prices,
        initial_capital=INITIAL_CAPITAL,
        min_notional=MIN_NOTIONAL,
        min_size=MIN_SIZE,
        trading_fee_rate=TRADING_FEE_RATE
    )
    # Use PPO with better exploration settings
    model = PPO(
        "MlpPolicy", 
        env_train, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,  # Higher entropy = more exploration (increased from 0.01)
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    model.learn(total_timesteps=TRAINING_TIMESTEPS)
    print(f"Training complete! Took {round(time.time() - ts, 3)} seconds")
    
    # Run strategy on test data
    print("\nRunning strategy on test data...")
    ts = time.time()
    env_test = TradingEnv(
        test_prices,
        initial_capital=INITIAL_CAPITAL,
        min_notional=MIN_NOTIONAL,
        min_size=MIN_SIZE,
        trading_fee_rate=TRADING_FEE_RATE
    )
    
    # Generate trading log
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
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_values, INITIAL_CAPITAL)
    
    print(f"Execution complete! Took {round(time.time() - ts, 3)} seconds")
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
    
    # Create static HTML file (visualize test prices only for clarity)
    print("\nGenerating static HTML report...")
    html_file = create_static_html(test_prices, actions, portfolio_values, metrics)
    html_path = Path(html_file).resolve()
    
    print(f"HTML report saved to: {html_path}")
    print("Opening in browser...")
    
    # Open in external browser
    webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
    