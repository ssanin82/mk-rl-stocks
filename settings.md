# Settings Configuration

This document describes all configuration parameters available in `settings.json`. The configuration file is organized into two main sections: `common` (applies to all strategies) and `btc` (BTC/USDT-specific settings).

## Table of Contents

- [Common Settings](#common-settings)
- [Reward Shaping Settings](#reward-shaping-settings)
- [BTC-Specific Settings](#btc-specific-settings)
- [Normalization Methods](#normalization-methods)
- [Position Management Rules](#position-management-rules)
- [Example Configuration](#example-configuration)

## Common Settings

These settings apply to all trading strategies and control data, training, and model behavior.

### `n_price_points`
- **Type**: `integer`
- **Default**: `100000`
- **Description**: Total number of price data points to generate or use for training and testing. The first `train_split_ratio × n_price_points` points are used for training, and the remaining points are used for testing/evaluation.

### `default_prices_file`
- **Type**: `string`
- **Default**: `"btc_usdt_1m_prices.txt"`
- **Description**: Filename containing historical price data. The file should contain one price per line, formatted with 6 decimal places. This file is used by `1_generate_prices.py` (for synthetic data) or `get_prices.py` (for real Binance data), and consumed by `2_train_model.py` and `3_run_model.py`.

### `training_episodes`
- **Type**: `integer`
- **Default**: `30`
- **Description**: Number of training episodes for the RL model. Each episode runs through the entire training dataset. Higher values provide more training but take longer. Typical values range from 10-100 depending on dataset size and desired training time.

### `train_split_ratio`
- **Type**: `float`
- **Default**: `0.9`
- **Description**: Ratio of data used for training (e.g., `0.9` = 90% training, 10% testing). The remaining data is reserved for model evaluation. Must be between 0.0 and 1.0. Common values are 0.8-0.9.

### `default_model_file`
- **Type**: `string`
- **Default**: `"trading_model.zip"`
- **Description**: Filename where the trained model is saved (by `2_train_model.py`) and loaded (by `3_run_model.py`). Uses Stable-Baselines3's model format.

### `price_history_window`
- **Type**: `integer`
- **Default**: `10`
- **Description**: Number of historical price points to include in the observation space. The model receives the last N log returns as additional features, providing temporal context for decision-making. Higher values give more historical context but increase observation space size. Typical values: 5-20.

### `use_lstm_policy`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Whether to use an LSTM-based policy network instead of MLP. LSTM policies can better capture temporal patterns in price sequences. Requires `sb3-contrib` package. Currently defaults to enhanced MLP policy even when set to `true` (feature prepared for future implementation).

## Reward Shaping Settings

These settings control the reward function that guides the RL agent's learning. All parameters are nested under `reward_shaping` in the `common` section. The reward function implements 10 different shaping strategies to maximize P&L.

### `incremental_reward_scale`
- **Type**: `float`
- **Default**: `5000`
- **Description**: Scale factor for incremental portfolio value changes. Portfolio changes are multiplied by this factor to make them the dominant learning signal. For BTC, small price movements (0.01%) translate to tiny rewards without scaling. Higher values (5000+) make portfolio changes more impactful relative to fixed trade rewards. Typical range: 1000-10000.

### `trade_execution_reward`
- **Type**: `float`
- **Default**: `0.01`
- **Description**: Base reward given when a trade is executed. This reward is conditional—only applied if the trade improves portfolio value or aligns with price momentum. Prevents overtrading for the sake of fixed rewards. The actual reward may be scaled by trade notional value. Typical range: 0.001-0.1.

### `momentum_reward_scale`
- **Type**: `float`
- **Default**: `0.3`
- **Description**: Scale factor for momentum-based rewards. Rewards trading in the direction of price movement (buying when price rises, selling when price falls) and penalizes trading against momentum. Uses exponential scaling for stronger momentum signals. Higher values make momentum alignment more valuable. Typical range: 0.1-0.5.

### `hold_profit_reward_scale`
- **Type**: `float`
- **Default**: `0.02`
- **Description**: Scale factor for rewards when holding profitable positions. Prevents premature exit from winning trades by providing a small reward proportional to profit ratio and position size. Only applies when position is in profit (>1% above entry). Typical range: 0.01-0.05.

### `entry_quality_reward_scale`
- **Type**: `float`
- **Default**: `0.05`
- **Description**: Scale factor for rewards when entering positions at favorable prices. Rewards mean reversion entries (buying below recent average) and momentum entries (buying after price drops, selling after price rises). Encourages strategic entry timing. Typical range: 0.02-0.1.

### `end_episode_reward_scale`
- **Type**: `float`
- **Default**: `10`
- **Description**: Scale factor for end-of-episode P&L reward. The final portfolio change is normalized, capped using tanh function, and multiplied by this scale. Prevents the end reward from overshadowing intermediate learning signals. Typical range: 5-20.

### `sharpe_reward_scale`
- **Type**: `float`
- **Default**: `0.1`
- **Description**: Scale factor for Sharpe ratio (risk-adjusted return) rewards. Rewards high returns with low volatility, encouraging consistent performance over erratic gains. Calculated using the last 50 portfolio values. Positive Sharpe ratios are rewarded, negative ones penalized. Typical range: 0.05-0.2.

### `drawdown_penalty_scale`
- **Type**: `float`
- **Default**: `2.0`
- **Description**: Scale factor for drawdown penalties. Penalizes when portfolio drops below recent peak, encouraging capital preservation and risk management. Only applies when drawdown > 0.1%. Higher values penalize drawdowns more aggressively. Typical range: 1.0-5.0.

### `position_sizing_reward_scale`
- **Type**: `float`
- **Default**: `0.01`
- **Description**: Scale factor for position sizing rewards. Rewards when position size is in the optimal range (20-80% of capital) and penalizes over-leveraging (positions > 90% of capital). Encourages appropriate risk management through position sizing. Typical range: 0.005-0.02.

### `pnl_penalty_scale`
- **Type**: `float`
- **Default**: `100`
- **Description**: Scale factor for proportional P&L penalties at episode end. Larger losses receive proportionally larger penalties, providing more nuanced feedback for learning. Only applies when final P&L is zero or negative. Typical range: 50-200.

## BTC-Specific Settings

These settings are specific to BTC/USDT trading and control capital, trading constraints, and position management.

### `initial_capital`
- **Type**: `float`
- **Default**: `10000`
- **Description**: Starting capital in USD for the trading strategy. This is the initial cash available for trading. All P&L calculations are relative to this value.

### `min_notional`
- **Type**: `float`
- **Default**: `100.0`
- **Minimum**: `0.0`
- **Description**: Minimum trade notional value in USD. The product of price × quantity must be at least this value for a trade to be valid. This simulates exchange minimum trade size requirements. For BTC, typical values are $50-$100.

### `min_size`
- **Type**: `float`
- **Default**: `0.001`
- **Minimum**: `0.0`
- **Description**: Minimum trade size in shares (BTC units). The quantity traded must be at least this value. This is the minimum quantity that can be traded in a single order. For BTC, typical values are 0.001-0.01 BTC.

### `trading_fee_rate`
- **Type**: `float`
- **Default**: `0.0001`
- **Range**: `0.0` to `1.0`
- **Description**: Trading fee rate per trade, expressed as a decimal (0.0001 = 0.01% = 1 basis point). This fee is applied to both buy and sell orders. The fee is calculated as `notional × trading_fee_rate` and deducted from proceeds (sells) or added to cost (buys). Typical values: 0.0001-0.001 (0.01%-0.1%).

### `profit_threshold`
- **Type**: `float`
- **Default**: `0.002`
- **Description**: Percentage profit threshold for automatic profit-taking, expressed as a decimal (0.002 = 0.2%). When the price grows by this percentage from the average entry price, the position management system automatically sells a portion of holdings. This implements a take-profit strategy. Typical values: 0.001-0.01 (0.1%-1%).

### `partial_sell_ratio`
- **Type**: `float`
- **Default**: `0.002`
- **Range**: `0.0` to `1.0`
- **Description**: Fraction of current holdings to sell when profit threshold is reached, expressed as a decimal (0.002 = 0.2% of holdings). After selling, the average entry price is recalculated. This allows partial profit-taking while maintaining a position. Typical values: 0.001-0.01 (0.1%-1% of position).

### `dca_threshold`
- **Type**: `float`
- **Default**: `0.002`
- **Description**: Percentage decline threshold for automatic dollar-cost averaging (DCA), expressed as a decimal (0.002 = 0.2%). When the price declines by this percentage from the average entry price, the system automatically buys additional shares to lower the average entry price. This implements a DCA strategy. Typical values: 0.001-0.01 (0.1%-1%).

### `dca_ratio`
- **Type**: `float`
- **Default**: `0.02`
- **Range**: `0.0` to `1.0`
- **Description**: Fraction of current position value to add when DCA threshold is reached, expressed as a decimal (0.02 = 2% of current position value). The system buys shares worth this percentage of the current position value. After buying, the average entry price is recalculated. Typical values: 0.01-0.05 (1%-5% of position value).

### `lot_size`
- **Type**: `float`
- **Default**: `0.001`
- **Minimum**: `0.0`
- **Description**: Minimum increment for trade quantities. All trade quantities (buy, sell, auto-trades, force-sells) are rounded to the nearest multiple of this value. This simulates exchange lot size requirements. For BTC, typical values are 0.001-0.01 BTC. Set to `0` to disable lot size rounding.

### `normalization_method`
- **Type**: `string`
- **Default**: `"log_returns"`
- **Options**: `"percentage_changes"`, `"log_returns"`, `"z-score"`, `"price_ratio"`
- **Description**: Price normalization method used for model observations and DCA triggers. The normalization method determines how prices are transformed before being fed to the model and how DCA thresholds are calculated. See [Normalization Methods](#normalization-methods) for details.

### `percentage_changes_step`
- **Type**: `float`
- **Default**: `0.001`
- **Description**: DCA step threshold for percentage changes normalization method (0.001 = 0.1%). Used only when `normalization_method` is `"percentage_changes"`. This defines the interval between DCA tiers. For each tier, the threshold is `step × tier_number`. Ignored if normalization method is different. Typical values: 0.0005-0.002 (0.05%-0.2%).

### `log_returns_step`
- **Type**: `float`
- **Default**: `0.001`
- **Description**: DCA step threshold for log returns normalization method (0.001 = 0.1%). Used only when `normalization_method` is `"log_returns"`. This defines the interval between DCA tiers. For each tier, the threshold is `step × tier_number`. Ignored if normalization method is different. Typical values: 0.0005-0.002 (0.05%-0.2%).

### `z-score_step`
- **Type**: `float`
- **Default**: `1.1`
- **Description**: DCA step threshold for z-score normalization method (1.1 = 1.1 standard deviations). Used only when `normalization_method` is `"z-score"`. This defines the interval between DCA tiers in standard deviation units. For each tier, the threshold is `step × tier_number`. Ignored if normalization method is different. Typical values: 0.5-1.5.

### `price_ratio_step`
- **Type**: `float`
- **Default**: `0.01`
- **Description**: DCA step threshold for price ratio normalization method (0.01 = 1%). Used only when `normalization_method` is `"price_ratio"`. This defines the interval between DCA tiers as a ratio difference. For each tier, the threshold is `step × tier_number`. Ignored if normalization method is different. Typical values: 0.005-0.02 (0.5%-2%).

### `dca_volatility_window`
- **Type**: `integer`
- **Default**: `20`
- **Description**: Number of recent normalized price points used for calculating rolling volatility in DCA trigger calculations. Volatility adjustment is applied to DCA thresholds for `z-score`, `percentage_changes`, and `log_returns` normalization methods. Larger windows provide more stable volatility estimates but react slower to changing market conditions. Typical values: 10-50.

### `dca_max_tiers`
- **Type**: `integer`
- **Default**: `3`
- **Description**: Maximum number of DCA tiers (trigger levels). Multi-tier DCA allows progressively larger buys as price drops further below entry. Tier 1 triggers at 1× step, tier 2 at 2× step, tier 3 at 3× step, etc. Higher tiers result in larger buy sizes (scaled by `dca_tier_multiplier`). Typical values: 2-5.

### `dca_base_ratio`
- **Type**: `float`
- **Default**: `0.02`
- **Range**: `0.0` to `1.0`
- **Description**: Base fraction of current position value for DCA tier 1 buys (0.02 = 2%). For tier N, the buy ratio is `dca_base_ratio × (dca_tier_multiplier ^ (N-1))`. This defines the smallest DCA buy size. Typical values: 0.01-0.05 (1%-5%).

### `dca_tier_multiplier`
- **Type**: `float`
- **Default**: `1.5`
- **Description**: Multiplier for scaling buy size across DCA tiers. Tier N buys `dca_base_ratio × (dca_tier_multiplier ^ (N-1))` of position value. Higher tiers (deeper price drops) trigger larger buys. For example, with base_ratio=0.02 and multiplier=1.5: tier 1=2%, tier 2=3%, tier 3=4.5%. Typical values: 1.2-2.0.

## Normalization Methods

The system supports four normalization methods for price data. The normalization method affects both model observations and DCA trigger calculations.

### `percentage_changes`
- **Calculation**: `(price_t - price_{t-1}) / price_{t-1}`
- **Scale**: Typically -0.01 to +0.01 (1% moves)
- **Use Case**: Captures relative price changes, good for momentum strategies
- **DCA Step**: Use `percentage_changes_step` (default 0.001 = 0.1%)

### `log_returns`
- **Calculation**: `log(price_t / price_{t-1})`
- **Scale**: Typically -0.01 to +0.01 (similar to percentage_changes for small moves)
- **Use Case**: Mathematically convenient, handles large moves better than percentage_changes
- **DCA Step**: Use `log_returns_step` (default 0.001 = 0.1%)

### `z-score`
- **Calculation**: `(price - mean_price) / std_price`
- **Scale**: Typically -3.0 to +3.0 (standard deviations from mean)
- **Use Case**: Accounts for market volatility, good for mean-reversion strategies
- **DCA Step**: Use `z-score_step` (default 1.1 = 1.1 standard deviations)
- **Volatility Adjustment**: Automatically adjusts thresholds based on rolling standard deviation

### `price_ratio`
- **Calculation**: `price_t / price_0` (normalized by first price)
- **Scale**: Typically 0.9 to 1.1 (relative to starting price)
- **Use Case**: Shows cumulative price movement from start
- **DCA Step**: Use `price_ratio_step` (default 0.01 = 1%)
- **Note**: Less useful for volatility adjustment

## Position Management Rules

The strategy includes automatic position management rules that execute independently of model actions. These rules help manage risk and optimize entry/exit points. DCA triggers use **normalized values** while position sizing uses **actual prices**.

### Profit-Taking (Auto-Sell)

**Trigger**: When `current_price ≥ avg_entry_price × (1 + profit_threshold)`

**Action**: 
- Sells `partial_sell_ratio × current_holdings` shares
- Recalculates `avg_entry_price` based on remaining position
- Trade is marked with "AUTO" flag in logs

**Rationale**: Locks in profits while maintaining exposure to further upside. Prevents giving back gains during price reversals.

### Dollar-Cost Averaging (Auto-Buy)

**Trigger**: When `current_price ≤ avg_entry_price × (1 - dca_threshold)`

**Action**:
- Buys shares worth `dca_ratio × (current_holdings × current_price)`
- Recalculates `avg_entry_price` based on new total position
- Trade is marked with "AUTO" flag in logs

**Rationale**: Lowers average entry price when position is underwater, improving break-even point and reducing time to profitability.

### Force-Sell at Episode End

**Trigger**: At the final price point of the episode, if `holdings > 0`

**Action**:
- Sells all remaining holdings at market price
- Ignores `min_notional` constraint for this final trade
- Trade is marked with "FORCE" flag in logs

**Rationale**: Ensures the strategy starts and ends with zero position, allowing fair comparison across episodes and preventing position carryover.

### Notes

- All position management trades are rounded to the nearest `lot_size` increment
- Position management trades are logged separately from model actions
- The model can still execute its own trades; position management runs after model actions
- Average entry price is calculated as a volume-weighted average of all buy prices

## Example Configuration

Here's a complete example `settings.json` file with all parameters:

```json
{
  "common": {
    "n_price_points": 100000,
    "default_prices_file": "btc_usdt_1m_prices.txt",
    "training_episodes": 30,
    "train_split_ratio": 0.9,
    "default_model_file": "trading_model.zip",
    "price_history_window": 10,
    "use_lstm_policy": false,
    "reward_shaping": {
      "incremental_reward_scale": 5000,
      "trade_execution_reward": 0.01,
      "momentum_reward_scale": 0.3,
      "hold_profit_reward_scale": 0.02,
      "entry_quality_reward_scale": 0.05,
      "end_episode_reward_scale": 10,
      "sharpe_reward_scale": 0.1,
      "drawdown_penalty_scale": 2.0,
      "position_sizing_reward_scale": 0.01,
      "pnl_penalty_scale": 100
    }
  },
  "btc": {
    "initial_capital": 10000,
    "min_notional": 100.0,
    "min_size": 0.001,
    "trading_fee_rate": 0.0001,
    "profit_threshold": 0.002,
    "partial_sell_ratio": 0.002,
    "dca_threshold": 0.002,
    "dca_ratio": 0.02,
    "lot_size": 0.001
  }
}
```

## Tuning Guidelines

### For More Aggressive Trading
- Increase `trade_execution_reward` (0.02-0.05)
- Increase `momentum_reward_scale` (0.4-0.6)
- Decrease `trading_fee_rate` (0.00005)
- Decrease `min_notional` (50.0)

### For More Conservative Trading
- Decrease `trade_execution_reward` (0.001-0.005)
- Decrease `momentum_reward_scale` (0.1-0.2)
- Increase `trading_fee_rate` (0.0002)
- Increase `min_notional` (200.0)

### For Better Risk Management
- Increase `drawdown_penalty_scale` (3.0-5.0)
- Increase `sharpe_reward_scale` (0.15-0.25)
- Decrease `position_sizing_reward_scale` (0.005)
- Increase `profit_threshold` (0.003-0.005)

### For Faster Learning
- Increase `incremental_reward_scale` (7000-10000)
- Increase `end_episode_reward_scale` (15-20)
- Decrease `pnl_penalty_scale` (50-75)

### For More Stable Learning
- Decrease `incremental_reward_scale` (3000-4000)
- Decrease `end_episode_reward_scale` (5-8)
- Increase `sharpe_reward_scale` (0.15-0.2)

## Notes

- All monetary values are in USD
- All percentages are expressed as decimals (0.01 = 1%)
- The configuration file is validated on load; invalid values will raise errors
- Changes to `settings.json` require restarting training/evaluation scripts
- Default values are optimized for BTC/USDT trading but can be adjusted for other assets
