"""
Trading environment for reinforcement learning.
"""

import gymnasium as gym
import numpy as np
from mkrl.constants import TRADE_EXECUTION_REWARD, MOMENTUM_REWARD_SCALE


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
