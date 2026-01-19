"""
Trading environment for reinforcement learning.
"""

import sys
import gymnasium as gym
import numpy as np
from mkrl.settings import (
    profit_threshold, partial_sell_ratio, dca_threshold, dca_ratio,
    lot_size, price_history_window, normalization_method,
    incremental_reward_scale, trade_execution_reward, momentum_reward_scale,
    hold_profit_reward_scale, entry_quality_reward_scale, end_episode_reward_scale,
    sharpe_reward_scale, drawdown_penalty_scale, position_sizing_reward_scale,
    pnl_penalty_scale
)
from mkrl.utils import normalize_prices, NormalizationMethod


class TradingEnv(gym.Env):
    def __init__(self, prices, initial_capital=1000, min_notional=5.0, min_size=0.1, trading_fee_rate=0.001, lot_size=None):
        super().__init__()
        self.prices = prices  # Keep actual prices for trading calculations
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
        # Observation includes base features + price history
        # Base: [log_return, price_change, relative_price, cash_ratio, holdings_ratio, entry_price_ratio, can_buy, can_sell]
        # History: [price_history_window] log returns
        self.price_history_window = price_history_window
        obs_size = 8 + self.price_history_window
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # Validate and store normalization method
        try:
            self.normalization_method = NormalizationMethod(normalization_method)
        except ValueError:
            available = ", ".join(NormalizationMethod.values())
            print(f"ERROR: Incorrect price normalization method '{normalization_method}', available: {available}", file=sys.stderr)
            import sys
            sys.exit(1)
        
        # Pre-compute normalized prices for observation space (but keep actual prices for trading)
        self.normalized_prices = normalize_prices(prices, self.normalization_method)
        
        # Track portfolio history for reward shaping (drawdown, Sharpe ratio, etc.)
        self.portfolio_history = []  # Store recent portfolio values for volatility/drawdown calculation
        self.price_history_for_entry = []  # Store recent prices for entry quality assessment
        self.max_portfolio_value = initial_capital  # Track peak for drawdown calculation
    
    def _round_to_lot_size(self, shares):
        """Round trade quantity to the nearest lot_size increment."""
        if self.lot_size <= 0:
            return shares
        return round(shares / self.lot_size) * self.lot_size
    
    def _calculate_reward_shaping(
        self,
        portfolio_value,
        prev_portfolio_value,
        price,
        prev_price,
        action,
        action_executed,
        position_mgmt_executed,
        done,
        steps_remaining_ratio,
        current_price
    ):
        """
        Comprehensive reward shaping function to maximize P&L.
        
        This function implements 10 key reward shaping strategies:
        
        1. SCALED INCREMENTAL REWARDS:
           - Rationale: Portfolio changes need to be scaled aggressively to be meaningful
           - For BTC, small price movements (0.01%) translate to tiny rewards without scaling
           - Higher scale (5000+) makes portfolio changes the dominant signal
           - Logic: incremental_change / initial_capital * scale_factor
        
        2. CONDITIONAL TRADE EXECUTION REWARD:
           - Rationale: Fixed rewards encourage overtrading regardless of profitability
           - Only reward trades that improve portfolio value or are strategically sound
           - Reduces noise from unprofitable trading activity
           - Logic: Reward only if trade improves portfolio or aligns with momentum
        
        3. ENHANCED MOMENTUM REWARDS:
           - Rationale: Trading with price momentum is a proven profitable strategy
           - Higher scale (0.3+) makes momentum alignment more valuable
           - Exponential scaling for stronger momentum signals
           - Logic: Scale * abs(price_change_pct) * exponential_factor for strong moves
        
        4. HOLDING PROFITABLE POSITIONS:
           - Rationale: Model should be rewarded for holding winning positions, not just trading
           - Prevents premature exit from profitable trades
           - Rewards patience when position is in profit
           - Logic: Small reward proportional to profit ratio and position size
        
        5. ENTRY QUALITY REWARD:
           - Rationale: Entering at good prices (buying dips, selling peaks) improves P&L
           - Rewards mean reversion entries (buying below recent average)
           - Rewards momentum entries (buying after price drops, selling after rises)
           - Logic: Reward based on price position relative to recent average
        
        6. PROPORTIONAL END-OF-EPISODE REWARD:
           - Rationale: Large absolute rewards can overshadow intermediate learning signals
           - Use proportional scaling to keep end reward in balance
           - Prevents model from only optimizing for final state
           - Logic: portfolio_change / initial_capital * scale_factor (capped)
        
        7. SHARPE RATIO / RISK-ADJUSTED RETURN:
           - Rationale: High returns with low volatility are better than high returns with high volatility
           - Encourages consistent performance over erratic gains
           - Rewards smooth equity curves
           - Logic: (return / volatility) * scale_factor, where volatility is std of portfolio returns
        
        8. DRAWDOWN PENALTY:
           - Rationale: Drawdowns indicate poor risk management
           - Penalizes when portfolio drops below recent peak
           - Encourages capital preservation
           - Logic: -abs(drawdown_pct) * penalty_scale
        
        9. POSITION SIZING REWARD:
           - Rationale: Optimal position sizing improves risk-adjusted returns
           - Rewards appropriate position size relative to capital
           - Penalizes over-leveraging (too large positions)
           - Logic: Reward when position size is 20-80% of available capital
        
        10. PROPORTIONAL P&L PENALTY:
            - Rationale: Fixed penalties don't scale with loss magnitude
            - Larger losses should receive proportionally larger penalties
            - More nuanced feedback for learning
            - Logic: -abs(portfolio_change / initial_capital) * penalty_scale
        
        Args:
            portfolio_value: Current total portfolio value (cash + holdings * price)
            prev_portfolio_value: Previous step's portfolio value
            price: Current price
            prev_price: Previous step's price
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            action_executed: Whether the model's action was executed
            position_mgmt_executed: Whether position management rules executed a trade
            done: Whether episode is complete
            steps_remaining_ratio: Ratio of steps remaining (0.0 = end, 1.0 = start)
            current_price: Current price for calculations
        
        Returns:
            float: Total reward from all shaping components
        """
        reward = 0.0
        
        # Calculate key metrics
        portfolio_change = portfolio_value - self.initial_capital
        prev_portfolio_change = prev_portfolio_value - self.initial_capital
        incremental_change = portfolio_change - prev_portfolio_change
        
        # Update portfolio history for volatility/drawdown calculations
        self.portfolio_history.append(portfolio_value)
        if len(self.portfolio_history) > 50:  # Keep last 50 steps for calculations
            self.portfolio_history.pop(0)
        
        # Update price history for entry quality assessment
        self.price_history_for_entry.append(price)
        if len(self.price_history_for_entry) > 20:  # Keep last 20 prices
            self.price_history_for_entry.pop(0)
        
        # Update max portfolio value for drawdown calculation
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        # ====================================================================
        # 1. SCALED INCREMENTAL REWARDS
        # ====================================================================
        # Rationale: Make portfolio changes the primary learning signal
        # For BTC, 0.01% moves = $1 on $10k capital = 0.0001 incremental change
        # Without scaling, this is negligible compared to fixed trade rewards
        # Scale of 5000+ makes portfolio changes dominant
        incremental_reward = incremental_change / self.initial_capital * incremental_reward_scale
        reward += incremental_reward
        
        # ====================================================================
        # 2. CONDITIONAL TRADE EXECUTION REWARD
        # ====================================================================
        # Rationale: Only reward trades that improve portfolio or are strategically sound
        # Prevents overtrading for the sake of fixed rewards
        # Reward is conditional on trade quality, not just execution
        if action_executed:
            # Only reward if trade improved portfolio or we have momentum alignment
            trade_improved_portfolio = incremental_change > 0
            if self.current_step > 0:
                price_change_pct = (price - prev_price) / prev_price if prev_price > 0 else 0.0
                momentum_aligned = (
                    (action == 1 and price_change_pct > 0) or  # Buy on rise
                    (action == 2 and price_change_pct < 0)     # Sell on fall
                )
            else:
                momentum_aligned = False
            
            if trade_improved_portfolio or momentum_aligned:
                # Scale reward by trade notional to reward larger strategic trades
                if action == 1:  # BUY
                    trade_notional = self.holdings * price if self.holdings > 0 else 0
                else:  # SELL
                    trade_notional = self.holdings * price if self.holdings > 0 else 0
                notional_ratio = min(trade_notional / self.initial_capital, 1.0)
                reward += trade_execution_reward * (1.0 + notional_ratio)
        
        # Also reward position management trades (they're rule-based and important)
        if position_mgmt_executed:
            reward += trade_execution_reward
        
        # ====================================================================
        # 3. ENHANCED MOMENTUM REWARDS
        # ====================================================================
        # Rationale: Trading with momentum is a proven profitable strategy
        # Higher scale makes momentum alignment more valuable than before
        # Use exponential scaling for stronger momentum signals
        if self.current_step > 0 and action_executed:
            price_change_pct = (price - prev_price) / prev_price if prev_price > 0 else 0.0
            abs_price_change = abs(price_change_pct)
            
            # Exponential scaling for stronger momentum
            momentum_strength = abs_price_change * 100  # Convert to basis points
            exponential_factor = 1.0 + (momentum_strength / 10.0)  # Stronger momentum = higher factor
            
            if action == 1 and price_change_pct > 0:  # Buy when price rising
                momentum_bonus = momentum_reward_scale * momentum_strength * exponential_factor
                reward += momentum_bonus
            elif action == 2 and price_change_pct < 0:  # Sell when price falling
                momentum_bonus = momentum_reward_scale * momentum_strength * exponential_factor
                reward += momentum_bonus
            # Penalty for trading against momentum (discourages bad timing)
            elif action == 1 and price_change_pct < 0:  # Buy when price falling
                momentum_penalty = -momentum_reward_scale * 0.5 * momentum_strength
                reward += momentum_penalty
            elif action == 2 and price_change_pct > 0:  # Sell when price rising
                momentum_penalty = -momentum_reward_scale * 0.5 * momentum_strength
                reward += momentum_penalty
        
        # ====================================================================
        # 4. HOLDING PROFITABLE POSITIONS
        # ====================================================================
        # Rationale: Reward patience when holding winning positions
        # Prevents premature exit from profitable trades
        # Only applies when we have holdings and they're in profit
        if self.holdings > 0 and self.avg_entry_price > 0:
            entry_price_ratio = current_price / self.avg_entry_price
            if entry_price_ratio > 1.01:  # Position is in profit (>1%)
                holdings_ratio = (self.holdings * current_price) / self.initial_capital
                profit_ratio = entry_price_ratio - 1.0
                # Reward proportional to profit and position size
                hold_profit_reward = hold_profit_reward_scale * profit_ratio * holdings_ratio * 100
                reward += hold_profit_reward
        
        # ====================================================================
        # 5. ENTRY QUALITY REWARD
        # ====================================================================
        # Rationale: Entering at good prices improves P&L significantly
        # Rewards mean reversion entries (buying below recent average)
        # Rewards momentum entries (buying after price drops)
        if action_executed and len(self.price_history_for_entry) >= 10:
            recent_avg_price = np.mean(self.price_history_for_entry[-10:])
            price_vs_avg = (price - recent_avg_price) / recent_avg_price if recent_avg_price > 0 else 0.0
            
            if action == 1:  # BUY
                # Reward buying when price is below recent average (mean reversion)
                if price_vs_avg < -0.001:  # Price is 0.1%+ below recent average
                    entry_quality_bonus = entry_quality_reward_scale * abs(price_vs_avg) * 100
                    reward += entry_quality_bonus
                # Also reward buying after price drops (buying the dip)
                if self.current_step > 0:
                    price_drop = (prev_price - price) / prev_price if prev_price > 0 else 0.0
                    if price_drop > 0.001:  # Price dropped 0.1%+
                        dip_buy_bonus = entry_quality_reward_scale * price_drop * 50
                        reward += dip_buy_bonus
            elif action == 2:  # SELL
                # Reward selling when price is above recent average (taking profit)
                if price_vs_avg > 0.001:  # Price is 0.1%+ above recent average
                    entry_quality_bonus = entry_quality_reward_scale * abs(price_vs_avg) * 100
                    reward += entry_quality_bonus
        
        # ====================================================================
        # 6. PROPORTIONAL END-OF-EPISODE REWARD
        # ====================================================================
        # Rationale: Large absolute rewards can overshadow intermediate learning
        # Use proportional scaling to keep end reward balanced
        # Prevents model from only optimizing for final state
        if done:
            # Use tanh to cap the reward and prevent it from dominating
            pnl_ratio = portfolio_change / self.initial_capital
            capped_pnl_ratio = np.tanh(pnl_ratio * 10)  # Cap at ~1.0 for large gains
            total_pnl_reward = capped_pnl_ratio * end_episode_reward_scale
            reward += total_pnl_reward
        
        # ====================================================================
        # 7. SHARPE RATIO / RISK-ADJUSTED RETURN
        # ====================================================================
        # Rationale: High returns with low volatility are better than high returns with high volatility
        # Encourages consistent performance over erratic gains
        # Only calculate if we have enough history
        if len(self.portfolio_history) >= 10:
            portfolio_returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            if len(portfolio_returns) > 0:
                mean_return = np.mean(portfolio_returns)
                std_return = np.std(portfolio_returns)
                if std_return > 1e-8:  # Avoid division by zero
                    sharpe_ratio = mean_return / std_return
                    # Reward positive Sharpe, penalize negative
                    sharpe_reward = sharpe_reward_scale * sharpe_ratio * 100
                    reward += sharpe_reward
        
        # ====================================================================
        # 8. DRAWDOWN PENALTY
        # ====================================================================
        # Rationale: Drawdowns indicate poor risk management
        # Penalizes when portfolio drops below recent peak
        # Encourages capital preservation
        if self.max_portfolio_value > self.initial_capital:
            drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            if drawdown > 0.001:  # Only penalize if drawdown > 0.1%
                drawdown_penalty = -drawdown_penalty_scale * drawdown * 100
                reward += drawdown_penalty
        
        # ====================================================================
        # 9. POSITION SIZING REWARD
        # ====================================================================
        # Rationale: Optimal position sizing improves risk-adjusted returns
        # Rewards appropriate position size (20-80% of capital)
        # Penalizes over-leveraging (too large positions)
        if self.holdings > 0:
            position_value = self.holdings * current_price
            position_ratio = position_value / self.initial_capital if self.initial_capital > 0 else 0
            
            # Optimal range: 20-80% of capital
            if 0.2 <= position_ratio <= 0.8:
                # Reward for being in optimal range
                optimal_sizing_reward = position_sizing_reward_scale * 10
                reward += optimal_sizing_reward
            elif position_ratio > 0.9:
                # Penalize over-leveraging (too large position)
                over_leverage_penalty = -position_sizing_reward_scale * (position_ratio - 0.8) * 100
                reward += over_leverage_penalty
        
        # ====================================================================
        # 10. PROPORTIONAL P&L PENALTY
        # ====================================================================
        # Rationale: Fixed penalties don't scale with loss magnitude
        # Larger losses should receive proportionally larger penalties
        # More nuanced feedback for learning
        if done:
            if portfolio_change <= 0:
                # Proportional penalty based on loss magnitude
                loss_ratio = abs(portfolio_change) / self.initial_capital
                proportional_penalty = -pnl_penalty_scale * loss_ratio
                reward += proportional_penalty
        
        # ====================================================================
        # ADDITIONAL: Position near end penalty (existing logic, kept for consistency)
        # ====================================================================
        # Penalty for having position near the end (encourage closing position before end)
        if steps_remaining_ratio < 0.1 and self.holdings > 0:  # Last 10% of episode
            position_value = self.holdings * current_price
            position_ratio = position_value / self.initial_capital if self.initial_capital > 0 else 0
            position_penalty = -5.0 * (1 - steps_remaining_ratio) * position_ratio
            reward += position_penalty
        
        return reward
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.holdings = 0
        self.avg_entry_price = 0.0
        self.total_cost_basis = 0.0
        # Initialize portfolio history for reward shaping
        self.portfolio_history = [self.initial_capital]
        self.price_history_for_entry = [self.prices[0]] if len(self.prices) > 0 else []
        self.max_portfolio_value = self.initial_capital
        return self._get_obs(), {}
    
    def _get_obs(self):
        price = self.prices[self.current_step]  # Use actual price for trading calculations
        norm_price = self.normalized_prices[self.current_step]  # Use normalized price for observation
        
        # Use normalized price based on selected method
        if self.normalization_method == NormalizationMethod.PERCENTAGE_CHANGES:
            # Percentage changes: use the normalized value directly
            price_feature = norm_price * 100  # Scale by 100 for better learning signal
            price_change = norm_price  # Already percentage change
        elif self.normalization_method == NormalizationMethod.LOG_RETURNS:
            # Log returns: use the normalized value directly
            price_feature = norm_price * 100  # Scale by 100 for better learning signal
            # Calculate price change for backward compatibility
            if self.current_step >= 1:
                price_change = (price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
            else:
                price_change = 0.0
        elif self.normalization_method == NormalizationMethod.Z_SCORE:
            # Z-score: use the normalized value directly
            price_feature = norm_price
            # Calculate price change for backward compatibility
            if self.current_step >= 1:
                price_change = (price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
            else:
                price_change = 0.0
        elif self.normalization_method == NormalizationMethod.PRICE_RATIO:
            # Price ratio: use the normalized value directly
            price_feature = norm_price
            # Calculate price change for backward compatibility
            if self.current_step >= 1:
                price_change = (price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
            else:
                price_change = 0.0
        else:
            # Fallback to log returns
            if self.current_step >= 1 and self.prices[self.current_step - 1] > 0:
                price_feature = np.log(price / self.prices[self.current_step - 1]) * 100
                price_change = (price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
            else:
                price_feature = 0.0
                price_change = 0.0
        
        # Add relative price (normalized by initial price) - always use actual prices for this
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
        
        # Base features
        base_features = [
            price_feature,  # Normalized price feature (method-dependent)
            price_change * 100,  # Price change percentage (for backward compatibility)
            relative_price,  # Relative price (always actual price ratio)
            cash_ratio,  # Cash as ratio
            holdings_ratio,  # Holdings value as ratio
            entry_price_ratio,  # Current price / average entry price
            can_buy,  # 1.0 if can buy, 0.0 if cannot
            can_sell,  # 1.0 if can sell, 0.0 if cannot
        ]
        
        # Add price history (using normalized prices based on method)
        price_history = []
        for i in range(self.price_history_window):
            hist_idx = self.current_step - (i + 1)
            if hist_idx >= 0:
                # Use normalized price value for history
                hist_norm_value = self.normalized_prices[hist_idx]
                # Scale based on method
                if self.normalization_method in [NormalizationMethod.PERCENTAGE_CHANGES, NormalizationMethod.LOG_RETURNS]:
                    price_history.append(hist_norm_value * 100)  # Scale by 100
                else:
                    price_history.append(hist_norm_value)  # Use as-is for z-score and price_ratio
            else:
                price_history.append(0.0)  # Pad with zeros if not enough history
        
        return np.array(base_features + price_history, dtype=np.float32)
    
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
        
        # Calculate reward using comprehensive reward shaping function
        # This function implements all 10 reward shaping strategies to maximize P&L
        reward = self._calculate_reward_shaping(
            portfolio_value=portfolio_value,
            prev_portfolio_value=prev_portfolio_value,
            price=price,
            prev_price=prev_price,
            action=action,
            action_executed=action_executed,
            position_mgmt_executed=position_mgmt_executed,
            done=done,
            steps_remaining_ratio=steps_remaining_ratio,
            current_price=current_price
        )
        
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
