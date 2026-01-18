"""
Trading environment for reinforcement learning.
"""

import gymnasium as gym
import numpy as np
from mkrl.settings import (
    trade_execution_reward, momentum_reward_scale,
    profit_threshold, partial_sell_ratio, dca_threshold, dca_ratio,
    lot_size, pnl_penalty
)


class TradingEnv(gym.Env):
    def __init__(self, prices, initial_capital=1000, min_notional=5.0, min_size=0.1, trading_fee_rate=0.001, lot_size=None):
        super().__init__()
        self.prices = prices
        self.initial_capital = initial_capital
        self.min_notional = min_notional
        self.min_size = min_size
        self.trading_fee_rate = trading_fee_rate
        self.lot_size = lot_size if lot_size is not None else 0.001  # Default to 0.001 if not provided
        self.current_step = 0
        self.cash = initial_capital
        self.holdings = 0
        self.avg_entry_price = 0.0  # Volume-weighted average entry price
        self.total_cost_basis = 0.0  # Total cost basis for calculating average entry price
        self.action_space = gym.spaces.Discrete(3)
        # Observation: [log_return, price_change, relative_price, cash_ratio, holdings_ratio, entry_price_ratio, can_buy, can_sell]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    
    def _round_to_lot_size(self, shares):
        """Round trade quantity to the nearest lot_size increment."""
        if self.lot_size <= 0:
            return shares
        return round(shares / self.lot_size) * self.lot_size
        
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
        
        # Calculate log return (log(price_today / price_yesterday))
        # This is a standard financial metric that normalizes price movements
        if self.current_step >= 1 and self.prices[self.current_step - 1] > 0:
            log_return = np.log(price / self.prices[self.current_step - 1])
        else:
            log_return = 0.0
        
        # Price change percentage (for backward compatibility and additional signal)
        if self.current_step >= 1:
            price_change = (price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
        else:
            price_change = 0.0
        
        # Add relative price (normalized by initial price)
        relative_price = price / self.prices[0] if len(self.prices) > 0 and self.prices[0] > 0 else 1.0
        
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
            log_return * 100,  # Log return (scaled by 100 for better learning signal)
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
            trade_shares = self._round_to_lot_size(trade_shares)  # Round to lot size
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
                trade_shares = self._round_to_lot_size(trade_shares)  # Round to lot size
                
                # Don't sell more than we have
                trade_shares = min(trade_shares, self.holdings)
                trade_shares = self._round_to_lot_size(trade_shares)  # Round again after capping
                
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
        
        # ========================================================================
        # CRITICAL: RULE-BASED POSITION MANAGEMENT (PRIMARY MECHANISM)
        # These rules are PRIMARY and always apply on EVERY step to manage positions.
        # They execute regardless of model action - this ensures active position management.
        # ========================================================================
        
        # Track if position management rules triggered any trades
        position_mgmt_executed = False
        position_mgmt_action = None
        
        # Rules 1 & 2: Position management when we have holdings
        # NOTE: These rules are CRITICAL for proper position management
        if self.holdings > 0 and self.avg_entry_price > 0:
            price_ratio = price / self.avg_entry_price
            profit_pct = price_ratio - 1.0  # e.g., 1.10 = 10% profit
            
            # Rule 1: If price grows profit_threshold% from average entry, sell partial_sell_ratio% of position
            if profit_pct >= profit_threshold:
                # Calculate how much to sell (20% of current holdings)
                shares_to_sell = self.holdings * partial_sell_ratio
                # Ensure it meets minimum size and notional
                shares_to_sell = max(shares_to_sell, self.min_size)
                shares_to_sell = min(shares_to_sell, self.holdings)  # Don't sell more than we have
                shares_to_sell = self._round_to_lot_size(shares_to_sell)  # Round to lot size
                shares_to_sell = min(shares_to_sell, self.holdings)  # Cap again after rounding
                
                notional = shares_to_sell * price
                if notional >= self.min_notional and shares_to_sell > 0:
                    # Execute automatic profit-taking sell
                    fee = notional * self.trading_fee_rate
                    proceeds = notional - fee
                    
                    # Update average entry price
                    cost_basis_of_sold = self.total_cost_basis * (shares_to_sell / self.holdings)
                    self.total_cost_basis -= cost_basis_of_sold
                    self.holdings -= shares_to_sell
                    if self.holdings > 0:
                        self.avg_entry_price = self.total_cost_basis / self.holdings
                    else:
                        self.avg_entry_price = 0.0
                        self.total_cost_basis = 0.0
                    
                    self.cash += proceeds
                    self.cash = max(0, self.cash)
                    # Track that position management rule executed a trade
                    position_mgmt_executed = True
                    position_mgmt_action = f"PROFIT_TAKE: Sold {shares_to_sell:.4f} @ ${price:.2f} ({partial_sell_ratio*100:.0f}% of position)"
            
            # Rule 2: If price goes down dca_threshold% from average entry, add dca_ratio% to position
            elif profit_pct <= -dca_threshold:  # e.g., -0.10 = 10% down
                # Calculate how much to add (10% of current position value, converted to shares)
                current_position_value = self.holdings * price
                target_add_value = current_position_value * dca_ratio
                shares_to_add = target_add_value / price if price > 0 else 0
                # Ensure it meets minimum size
                shares_to_add = max(shares_to_add, self.min_size)
                shares_to_add = self._round_to_lot_size(shares_to_add)  # Round to lot size
                
                notional = shares_to_add * price
                fee = notional * self.trading_fee_rate
                total_cost = notional + fee
                
                # Only add if we have enough cash and meet minimum notional
                if self.cash >= total_cost and notional >= self.min_notional:
                    # Execute automatic DCA buy
                    self.holdings += shares_to_add
                    self.cash -= total_cost
                    self.cash = max(0, self.cash)
                    
                    # Update volume-weighted average entry price
                    cost_this_trade = notional
                    self.total_cost_basis += cost_this_trade
                    if self.holdings > 0:
                        self.avg_entry_price = self.total_cost_basis / self.holdings
                    # Track that position management rule executed a trade
                    position_mgmt_executed = True
                    position_mgmt_action = f"DCA: Added {shares_to_add:.4f} @ ${price:.2f} ({dca_ratio*100:.0f}% of position value)"
        
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        # Track if force-sell happened
        force_sell_executed = False
        force_sell_shares = 0.0
        
        # Position constraint: must end with 0 position (all holdings sold)
        # Force sell everything at the end, ignoring minimum notional
        if done and self.holdings > 0:
            force_sell_shares = self.holdings
            current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
            notional = self.holdings * current_price
            fee = notional * self.trading_fee_rate
            proceeds = notional - fee
            self.cash += proceeds
            self.holdings = 0.0
            self.avg_entry_price = 0.0
            self.total_cost_basis = 0.0
            force_sell_executed = True
        
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
            reward += trade_execution_reward
        
        # 1b. Reward for position management rule trades (critical for position management)
        if position_mgmt_executed:
            reward += trade_execution_reward  # Same reward as model trades - position mgmt is equally important
        
        # 4. Momentum-based reward shaping (bonus for trading in the direction of price movement)
        if self.current_step > 0:
            price_change_pct = (price - prev_price) / prev_price if prev_price > 0 else 0.0
            if action == 1 and action_executed and price_change_pct > 0:  # Buy when price rising
                momentum_bonus = momentum_reward_scale * abs(price_change_pct) * 100  # Scale by momentum strength
                reward += momentum_bonus
            elif action == 2 and action_executed and price_change_pct < 0:  # Sell when price falling
                momentum_bonus = momentum_reward_scale * abs(price_change_pct) * 100
                reward += momentum_bonus
            # Small penalty for trading against momentum (discourages bad timing)
            elif action == 1 and action_executed and price_change_pct < 0:  # Buy when price falling
                momentum_penalty = -momentum_reward_scale * 0.5 * abs(price_change_pct) * 100
                reward += momentum_penalty
            elif action == 2 and action_executed and price_change_pct > 0:  # Sell when price rising
                momentum_penalty = -momentum_reward_scale * 0.5 * abs(price_change_pct) * 100
                reward += momentum_penalty
        
        # Add bonus at episode end based on total P&L (shape final behavior)
        if done:
            total_pnl_reward = portfolio_change / self.initial_capital * 100
            reward += total_pnl_reward
            
            # Penalty for 0 or negative P&L at the end (encourages profitability)
            if portfolio_change <= 0:
                reward += pnl_penalty
        
        # Penalty for having position near the end (encourage closing position before end)
        # Penalty increases as we approach the end
        if steps_remaining_ratio < 0.1 and self.holdings > 0:  # Last 10% of episode
            # Calculate position value as ratio of initial capital for penalty scaling
            position_value = self.holdings * current_price
            position_ratio = position_value / self.initial_capital if self.initial_capital > 0 else 0
            position_penalty = -5.0 * (1 - steps_remaining_ratio) * position_ratio
            reward += position_penalty
        
        # Include position management info in info dict for logging/tracking
        info = {}
        if position_mgmt_executed:
            info['position_mgmt_action'] = position_mgmt_action
            info['position_mgmt_triggered'] = True
        if force_sell_executed:
            # Use the price that was used for force-sell (before holdings were set to 0)
            # This is the same price as used in the force-sell calculation
            info['force_sell_executed'] = True
            info['force_sell_shares'] = force_sell_shares
            info['force_sell_price'] = self.prices[min(self.current_step, len(self.prices) - 1)]
        
        return self._get_obs(), reward, terminated, truncated, info
