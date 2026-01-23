# AlphaRL: Reinforcement Learning Trading Strategy Simulator

A reinforcement learning-based stock trading simulator that uses PPO (Proximal Policy Optimization) to train an agent to make buy/sell decisions. Features an interactive dark-themed dashboard with performance metrics and visualizations.

## Features

- 📈 **RL-powered trading** using Stable-Baselines3 PPO algorithm
- 📊 **Interactive visualizations** with Plotly and Dash
- 🎯 **Buy/sell signals** displayed on price charts
- 💰 **Performance metrics** including P&L, returns, drawdown, and volatility
- 🌙 **Dark theme** dashboard
- 🚀 **Live web server** for real-time monitoring

## Screenshots

The dashboard displays:
- Stock price chart with buy (▲) and sell (▼) signals
- Portfolio value over time
- Key performance metrics (Initial/Final Capital, P&L, Returns, Max Drawdown, Volatility)

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd alpharl
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulator:
```bash
python main.py
```

The script will:
1. Generate synthetic stock price data
2. Train a PPO agent (10,000 timesteps)
3. Run the trading strategy
4. Launch a Dash web server at `http://0.0.0.0:8050`

### Accessing the Dashboard

**Local machine:**
- Open `http://localhost:8050` in your browser

**Remote server:**
- Use SSH port forwarding: `ssh -L 8050:localhost:8050 user@server`
- Or access directly at `http://YOUR_SERVER_IP:8050` (ensure port 8050 is open)

**VS Code Remote:**
- VS Code should auto-detect and forward the port
- Check the "Ports" tab in the bottom panel

Press `Ctrl+C` to stop the server.

## Configuration

All configuration parameters are defined in `settings.json`. The configuration file is organized into two main sections: `common` (applies to all strategies) and `btc` (BTC/USDT-specific settings).

### Table of Contents

- [Common Settings](#common-settings)
  - [Data and Training](#data-and-training)
  - [Model Architecture](#model-architecture-settings)
  - [Training Optimization](#training-optimization)
  - [Curriculum Learning](#curriculum-learning)
  - [Reward Shaping Settings](#reward-shaping-settings)
- [BTC-Specific Settings](#btc-specific-settings)
  - [Capital and Constraints](#capital-and-constraints)
  - [Position Management](#position-management)
  - [Price Normalization](#price-normalization)
  - [DCA (Dollar-Cost Averaging)](#dca-dollar-cost-averaging)
- [Configuration Inheritance](#configuration-inheritance)
- [Normalization Methods](#normalization-methods)
- [Position Management Rules](#position-management-rules)
- [Example Configuration](#example-configuration)
- [Tuning Guidelines](#tuning-guidelines)

## Common Settings

These settings apply to all trading strategies and control data, training, and model behavior.

### Data and Training

#### `n_price_points`
- **Type**: `integer`
- **Default**: `100000`
- **Description**: Total number of price data points to generate or use for training and testing. The first `train_split_ratio × n_price_points` points are used for training, and the remaining points are used for testing/evaluation.

#### `default_prices_file`
- **Type**: `string`
- **Default**: `"btc_usdt_1m_prices.txt"`
- **Description**: Filename containing historical price data. The file should contain one price per line, formatted with 6 decimal places. This file is used by `1_generate_prices.py` (for synthetic data) or `get_prices.py` (for real Binance data), and consumed by `2_train_model.py` and `3_run_model.py`.

#### `training_episodes`
- **Type**: `integer`
- **Default**: `100`
- **Description**: Number of training episodes for the RL model. Each episode runs through the entire training dataset. Higher values provide more training but take longer. Typical values range from 10-100 depending on dataset size and desired training time.

#### `train_split_ratio`
- **Type**: `float`
- **Default**: `0.9`
- **Description**: Ratio of data used for training (e.g., `0.9` = 90% training, 10% testing). The remaining data is reserved for model evaluation. Must be between 0.0 and 1.0. Common values are 0.8-0.9.

#### `default_model_file`
- **Type**: `string`
- **Default**: `"trading_model.zip"`
- **Description**: Filename where the trained model is saved (by `2_train_model.py`) and loaded (by `3_run_model.py`). Uses Stable-Baselines3's model format.

#### `price_history_window`
- **Type**: `integer`
- **Default**: `10`
- **Description**: Number of historical price points to include in the observation space. The model receives the last N log returns as additional features, providing temporal context for decision-making. Higher values give more historical context but increase observation space size. Typical values: 5-20.

#### `use_lstm_policy`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Whether to use an LSTM-based policy network instead of MLP. LSTM policies can better capture temporal patterns in price sequences. Requires `sb3-contrib` package. Currently defaults to enhanced MLP policy even when set to `true` (feature prepared for future implementation).

### Model Architecture Settings

#### `ent_coef`
- **Type**: `float`
- **Default**: `0.3`
- **Description**: Entropy coefficient for PPO algorithm. Controls the balance between exploration and exploitation. Higher values (0.3-0.5) encourage more exploration and discovery of new strategies. Lower values (0.1-0.2) favor exploitation and fine-tuning of known strategies. Typical range: 0.1-0.5.

### Training Optimization

These settings control training performance and resource usage.

#### `training_optimization.n_envs`
- **Type**: `integer`
- **Default**: `4`
- **Description**: Number of parallel environments for training. Using multiple environments allows collecting more diverse experiences per training step and accelerates training through parallel data collection. Typical values: 2-8. Higher values require more memory.

#### `training_optimization.use_vecenv`
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Enable vectorized environments for parallel training. Uses `DummyVecEnv` which maintains compatibility with callbacks and Windows systems while providing efficient parallelization. Set to `false` to use a single environment.

#### `training_optimization.device`
- **Type**: `string`
- **Default**: `"auto"`
- **Options**: `"auto"`, `"cpu"`, `"cuda"`
- **Description**: Device for training. `"auto"` automatically detects and uses CUDA-enabled GPUs when available, falling back to CPU if unavailable. `"cuda"` forces GPU usage (will fall back to CPU if GPU unavailable). `"cpu"` forces CPU usage. GPU training is 5-10x faster on modern GPUs.

#### `training_optimization.optimize_batch_size`
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Automatically optimize batch size based on available memory (GPU vs CPU), network architecture size, and number of parallel environments. When enabled, the system selects optimal batch sizes for faster training.

#### `training_optimization.optimize_network_size`
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Automatically optimize network architecture size. When enabled, uses GPU-optimized networks `[128, 128]` on GPU or CPU-optimized networks `[64, 64]` on CPU. This provides faster training with minimal performance impact.

### Curriculum Learning

Curriculum learning uses a phased training approach to help the model learn more effectively.

#### `curriculum_learning.enabled`
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Enable curriculum learning with forced initial buys. Phase 1 forces initial buys to help the model learn selling behavior, while Phase 2 allows free choice to learn full entry/exit strategies.

#### `curriculum_learning.phase1_episodes`
- **Type**: `float`
- **Default**: `0.3`
- **Description**: Fraction of episodes with forced buys (0.3 = 30% of episodes). During Phase 1, the model is forced to buy after a delay if no trade occurred, helping it learn selling behavior. After Phase 1, the model has full control.

#### `curriculum_learning.forced_initial_buy_delay`
- **Type**: `integer`
- **Default**: `10`
- **Description**: Number of steps to wait before forcing the first buy if no trade occurred. This gives the model a chance to trade naturally before intervention.

#### `curriculum_learning.forced_initial_buy_size`
- **Type**: `float`
- **Default**: `0.001`
- **Description**: Size of forced initial buy in BTC units. This is the quantity that will be purchased when curriculum learning forces a buy.

### Reward Shaping Settings

These settings control the reward function that guides the RL agent's learning. All parameters are nested under `reward_shaping` in the `common` section. The reward function implements multiple shaping strategies to maximize P&L.

#### Core Rewards

#### `incremental_reward_scale`
- **Type**: `float`
- **Default**: `5000`
- **Description**: Scale factor for incremental portfolio value changes. Portfolio changes are multiplied by this factor to make them the dominant learning signal. For BTC, small price movements (0.01%) translate to tiny rewards without scaling. Higher values (5000+) make portfolio changes more impactful relative to fixed trade rewards. Typical range: 1000-10000.

#### `trade_execution_reward`
- **Type**: `float`
- **Default**: `0.5`
- **Description**: Base reward given when a trade is executed. This reward is conditional—only applied if the trade improves portfolio value or aligns with price momentum. Prevents overtrading for the sake of fixed rewards. The actual reward may be scaled by trade notional value. Typical range: 0.1-1.0.

#### `end_episode_reward_scale`
- **Type**: `float`
- **Default**: `10`
- **Description**: Scale factor for end-of-episode P&L reward. The final portfolio change is normalized, capped using tanh function, and multiplied by this scale. Prevents the end reward from overshadowing intermediate learning signals. Typical range: 5-20.

#### Strategy Rewards

#### `momentum_reward_scale`
- **Type**: `float`
- **Default**: `0.3`
- **Description**: Scale factor for momentum-based rewards. Rewards trading in the direction of price movement (buying when price rises, selling when price falls) and penalizes trading against momentum. Uses exponential scaling for stronger momentum signals. Higher values make momentum alignment more valuable. Typical range: 0.1-0.5.

#### `hold_profit_reward_scale`
- **Type**: `float`
- **Default**: `0.02`
- **Description**: Scale factor for rewards when holding profitable positions. Prevents premature exit from winning trades by providing a small reward proportional to profit ratio and position size. Only applies when position is in profit (>1% above entry). Typical range: 0.01-0.05.

#### `entry_quality_reward_scale`
- **Type**: `float`
- **Default**: `0.5`
- **Description**: Scale factor for rewards when entering positions at favorable prices. Rewards mean reversion entries (buying below recent average) and momentum entries (buying after price drops, selling after price rises). Encourages strategic entry timing. Typical range: 0.02-0.1.

#### Risk Management

#### `sharpe_reward_scale`
- **Type**: `float`
- **Default**: `0.1`
- **Description**: Scale factor for Sharpe ratio (risk-adjusted return) rewards. Rewards high returns with low volatility, encouraging consistent performance over erratic gains. Calculated using the last 50 portfolio values. Positive Sharpe ratios are rewarded, negative ones penalized. Typical range: 0.05-0.2.

#### `drawdown_penalty_scale`
- **Type**: `float`
- **Default**: `2.0`
- **Description**: Scale factor for drawdown penalties. Penalizes when portfolio drops below recent peak, encouraging capital preservation and risk management. Only applies when drawdown > 0.1%. Higher values penalize drawdowns more aggressively. Typical range: 1.0-5.0.

#### `position_sizing_reward_scale`
- **Type**: `float`
- **Default**: `0.01`
- **Description**: Scale factor for position sizing rewards. Rewards when position size is in the optimal range (20-80% of capital) and penalizes over-leveraging (positions > 90% of capital). Encourages appropriate risk management through position sizing. Typical range: 0.005-0.02.

#### `pnl_penalty_scale`
- **Type**: `float`
- **Default**: `10`
- **Description**: Scale factor for proportional P&L penalties at episode end. Larger losses receive proportionally larger penalties, providing more nuanced feedback for learning. Only applies when final P&L is negative (not zero). Typical range: 5-20.

#### Trading Incentives

These settings encourage active trading and exploration.

#### `entry_incentive_reward`
- **Type**: `float`
- **Default**: `5.0`
- **Description**: Large reward for first buy when position is 0. This helps overcome the initial exploration problem by strongly incentivizing the model to make its first trade. Multiplied by `risk_taking_multiplier`. Typical range: 2.0-10.0.

#### `pre_trade_buy_incentive`
- **Type**: `float`
- **Default**: `2.0`
- **Description**: Reward for buying before any trades occur. Encourages the model to enter the market early rather than waiting indefinitely. Typical range: 1.0-5.0.

#### `exploration_bonus`
- **Type**: `float`
- **Default**: `0.5`
- **Description**: Bonus reward for buying when holdings are 0. Encourages exploration and market entry. Applied when executing a BUY action with zero holdings. Typical range: 0.2-1.0.

#### `aggressive_trading_bonus`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Bonus for active trading. Rewards the model for making trades rather than holding indefinitely. Helps prevent the model from becoming too conservative. Typical range: 0.5-2.0.

#### `risk_taking_multiplier`
- **Type**: `float`
- **Default**: `1.5`
- **Description**: Multiplier applied to most reward/penalty components. Higher values make the model more risk-taking and aggressive. Values > 1.0 amplify rewards and penalties, encouraging more active trading. Typical range: 1.0-2.0.

#### Penalties

These settings penalize undesirable behaviors.

#### `invalid_sell_penalty`
- **Type**: `float`
- **Default**: `-1000.0`
- **Description**: Heavy penalty for attempting to sell with 0 position. This is a critical bug that should never occur, so the penalty is extremely large to strongly discourage this behavior. Typical range: -500.0 to -2000.0.

#### `no_position_hold_penalty`
- **Type**: `float`
- **Default**: `-0.5`
- **Description**: Penalty for holding without a position. Applied each step when the model holds (action 0) with zero holdings. Encourages the model to enter positions rather than staying in cash. Typical range: -0.2 to -1.0.

#### `full_cash_no_position_penalty`
- **Type**: `float`
- **Default**: `-1.0`
- **Description**: Penalty when cash is 100% and no position exists. Stronger signal than `no_position_hold_penalty` to discourage keeping all capital in cash. Applied each step when cash >= 99% of initial capital and holdings == 0. Typical range: -0.5 to -2.0.

#### `steps_since_trade_penalty_scale`
- **Type**: `float`
- **Default**: `0.05`
- **Description**: Penalty scale for inactivity. The penalty grows quadratically with the number of steps since the last trade: `penalty = -scale × (steps_since_trade ^ 1.5)`. This creates accumulating pressure to trade. Typical range: 0.01-0.1.

#### `no_trade_penalty_scale`
- **Type**: `float`
- **Default**: `0.1`
- **Description**: Penalty scale for not trading. Applied when the model has never traded or has gone many steps without trading. Creates accumulating penalty that grows over time. Typical range: 0.05-0.2.

#### `has_position_reward`
- **Type**: `float`
- **Default**: `0.02`
- **Description**: Small positive reward each step when a position exists. Encourages the model to maintain positions rather than staying in cash. Applied every step when holdings > 0. Typical range: 0.01-0.05.

#### `initial_training_fixed_reward`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Fixed reward during initial training phase. Helps bootstrap learning in early episodes when the model is still exploring. Typical range: 0.5-2.0.

## BTC-Specific Settings

These settings are specific to BTC/USDT trading and control capital, trading constraints, and position management.

### Capital and Constraints

#### `initial_capital`
- **Type**: `float`
- **Default**: `10000`
- **Description**: Starting capital in USD for the trading strategy. This is the initial cash available for trading. All P&L calculations are relative to this value.

#### `min_notional`
- **Type**: `float`
- **Default**: `50.0`
- **Minimum**: `0.0`
- **Description**: Minimum trade notional value in USD. The product of price × quantity must be at least this value for a trade to be valid. This simulates exchange minimum trade size requirements. For BTC, typical values are $50-$100.

#### `min_size`
- **Type**: `float`
- **Default**: `0.001`
- **Minimum**: `0.0`
- **Description**: Minimum trade size in shares (BTC units). The quantity traded must be at least this value. This is the minimum quantity that can be traded in a single order. For BTC, typical values are 0.001-0.01 BTC.

#### `trading_fee_rate`
- **Type**: `float`
- **Default**: `0.0001`
- **Range**: `0.0` to `1.0`
- **Description**: Trading fee rate per trade, expressed as a decimal (0.0001 = 0.01% = 1 basis point). This fee is applied to both buy and sell orders. The fee is calculated as `notional × trading_fee_rate` and deducted from proceeds (sells) or added to cost (buys). Typical values: 0.0001-0.001 (0.01%-0.1%).

#### `lot_size`
- **Type**: `float`
- **Default**: `0.001`
- **Minimum**: `0.0`
- **Description**: Minimum increment for trade quantities. All trade quantities (buy, sell, auto-trades, force-sells) are rounded to the nearest multiple of this value. This simulates exchange lot size requirements. For BTC, typical values are 0.001-0.01 BTC. Set to `0` to disable lot size rounding.

### Position Management

#### `profit_threshold`
- **Type**: `float`
- **Default**: `0.002`
- **Description**: Percentage profit threshold for automatic profit-taking, expressed as a decimal (0.002 = 0.2%). When the price grows by this percentage from the average entry price, the position management system automatically sells a portion of holdings. This implements a take-profit strategy. Typical values: 0.001-0.01 (0.1%-1%).

#### `partial_sell_ratio`
- **Type**: `float`
- **Default**: `0.002`
- **Range**: `0.0` to `1.0`
- **Description**: Fraction of current holdings to sell when profit threshold is reached, expressed as a decimal (0.002 = 0.2% of holdings). After selling, the average entry price is recalculated. This allows partial profit-taking while maintaining a position. Typical values: 0.001-0.01 (0.1%-1% of position).

#### `dca_threshold`
- **Type**: `float`
- **Default**: `0.02`
- **Description**: Price decline threshold for automatic dollar-cost averaging (DCA), expressed as a decimal (0.02 = 2%). When the price declines by this percentage from the average entry price, the system automatically buys additional shares to lower the average entry price. This implements a DCA strategy. Note: This is a legacy parameter; the multi-tier DCA system uses normalization-based thresholds instead. Typical values: 0.01-0.05 (1%-5%).

#### `dca_ratio`
- **Type**: `float`
- **Default**: `0.02`
- **Range**: `0.0` to `1.0`
- **Description**: Fraction of current position value to add when DCA threshold is reached, expressed as a decimal (0.02 = 2% of current position value). The system buys shares worth this percentage of the current position value. After buying, the average entry price is recalculated. Typical values: 0.01-0.05 (1%-5% of position value).

### Price Normalization

#### `normalization_method`
- **Type**: `string`
- **Default**: `"log_returns"`
- **Options**: `"percentage_changes"`, `"log_returns"`, `"z-score"`, `"price_ratio"`
- **Description**: Price normalization method used for model observations and DCA triggers. The normalization method determines how prices are transformed before being fed to the model and how DCA thresholds are calculated. See [Normalization Methods](#normalization-methods) for details.

#### `percentage_changes_step`
- **Type**: `float`
- **Default**: `0.001`
- **Description**: DCA step threshold for percentage changes normalization method (0.001 = 0.1%). Used only when `normalization_method` is `"percentage_changes"`. This defines the interval between DCA tiers. For each tier, the threshold is `step × tier_number`. Ignored if normalization method is different. Typical values: 0.0005-0.002 (0.05%-0.2%).

#### `log_returns_step`
- **Type**: `float`
- **Default**: `0.1`
- **Description**: DCA step threshold for log returns normalization method (0.1 = 10% in log space). Used only when `normalization_method` is `"log_returns"`. This defines the interval between DCA tiers. For each tier, the threshold is `step × tier_number`. Smaller values make DCA more sensitive (triggers more frequently). Typical values: 0.05-0.2.

#### `z-score_step`
- **Type**: `float`
- **Default**: `1.1`
- **Description**: DCA step threshold for z-score normalization method (1.1 = 1.1 standard deviations). Used only when `normalization_method` is `"z-score"`. This defines the interval between DCA tiers in standard deviation units. For each tier, the threshold is `step × tier_number`. Ignored if normalization method is different. Typical values: 0.5-1.5.

#### `price_ratio_step`
- **Type**: `float`
- **Default**: `0.01`
- **Description**: DCA step threshold for price ratio normalization method (0.01 = 1%). Used only when `normalization_method` is `"price_ratio"`. This defines the interval between DCA tiers as a ratio difference. For each tier, the threshold is `step × tier_number`. Ignored if normalization method is different. Typical values: 0.005-0.02 (0.5%-2%).

### DCA (Dollar-Cost Averaging)

#### `dca_volatility_window`
- **Type**: `integer`
- **Default**: `20`
- **Description**: Number of recent normalized price points used for calculating rolling volatility in DCA trigger calculations. Volatility adjustment is applied to DCA thresholds for `z-score`, `percentage_changes`, and `log_returns` normalization methods. Larger windows provide more stable volatility estimates but react slower to changing market conditions. Typical values: 10-50.

#### `dca_max_tiers`
- **Type**: `integer`
- **Default**: `3`
- **Description**: Maximum number of DCA tiers (trigger levels). Multi-tier DCA allows progressively larger buys as price drops further below entry. Tier 1 triggers at 1× step, tier 2 at 2× step, tier 3 at 3× step, etc. Higher tiers result in larger buy sizes (scaled by `dca_tier_multiplier`). Typical values: 2-5.

#### `dca_base_ratio`
- **Type**: `float`
- **Default**: `0.02`
- **Range**: `0.0` to `1.0`
- **Description**: Base fraction of current position value for DCA tier 1 buys (0.02 = 2%). For tier N, the buy ratio is `dca_base_ratio × (dca_tier_multiplier ^ (N-1))`. This defines the smallest DCA buy size. Typical values: 0.01-0.05 (1%-5%).

#### `dca_tier_multiplier`
- **Type**: `float`
- **Default**: `1.5`
- **Description**: Multiplier for scaling buy size across DCA tiers. Tier N buys `dca_base_ratio × (dca_tier_multiplier ^ (N-1))` of position value. Higher tiers (deeper price drops) trigger larger buys. For example, with base_ratio=0.02 and multiplier=1.5: tier 1=2%, tier 2=3%, tier 3=4.5%. Typical values: 1.2-2.0.

#### `dca_virtual_entry_enabled`
- **Type**: `boolean`
- **Default**: `true`
- **Description**: Enable virtual entry point for DCA when no position exists. When enabled, the system tracks the lowest normalized price seen in the lookback window and uses it as a virtual entry point to calculate DCA tiers. This allows DCA to trigger even when starting from zero holdings, helping bootstrap entries at favorable prices.

#### `dca_virtual_entry_lookback`
- **Type**: `integer`
- **Default**: `50`
- **Description**: Lookback period for virtual entry point calculation. The system tracks the lowest normalized price in the last N steps and uses it as the virtual entry point for DCA tier calculations when no position exists. Larger values provide more stable virtual entries but react slower to recent price drops. Typical values: 20-100.

### Configuration Inheritance

AlphaRL implements a **layered configuration system** with **multi-stage inheritance** support. This allows you to create variant configurations that inherit from base settings and selectively override specific values, eliminating duplication and making configuration management more maintainable.

#### How It Works

The configuration system uses three key concepts:

1. **Configuration Inheritance** (the pattern): Settings files can include other settings files using the `"include"` field. The included file's settings serve as the base, and the current file's settings override them.

2. **Deep Merging** (the implementation): When merging configurations, the system performs **recursive dictionary merging**. This means:
   - Nested dictionaries (e.g., `common.reward_shaping`) are merged recursively
   - Only the specific keys you override are changed; all other keys are inherited
   - Lists and primitive values are replaced entirely (not merged)

3. **Layered Configuration** (the conceptual model): Think of it as layers:
   - **Layer 1 (Base)**: `settings.json` - Contains all default settings
   - **Layer 2 (Override)**: `settings_half_aggressive.json` - Overrides specific values
   - **Layer 3 (Further Override)**: `settings_ltsm_ha.json` - Overrides values from Layer 2
   - Final result: All layers merged, with later layers taking precedence

#### Single-Stage Inheritance Example

```json
// settings.json (base)
{
  "config_name": "base",
  "common": {
    "ent_coef": 0.3,
    "reward_shaping": {
      "trade_execution_reward": 0.5,
      "entry_incentive_reward": 5.0,
      "momentum_reward_scale": 0.3
    }
  }
}

// settings_half_aggressive.json
{
  "include": "settings.json",
  "config_name": "half_aggressive",
  "common": {
    "ent_coef": 0.15,
    "reward_shaping": {
      "trade_execution_reward": 0.25,
      "entry_incentive_reward": 2.5
    }
  }
}
```

**Result for `settings_half_aggressive.json`:**
```json
{
  "config_name": "half_aggressive",
  "common": {
    "ent_coef": 0.15,  // Overridden
    "reward_shaping": {
      "trade_execution_reward": 0.25,      // Overridden
      "entry_incentive_reward": 2.5,      // Overridden
      "momentum_reward_scale": 0.3         // Inherited from base
    }
  }
}
```

#### Multi-Stage Inheritance Example

The system supports **unlimited nesting levels**. For example:

```json
// settings.json (base layer)
{
  "config_name": "base",
  "common": {
    "ent_coef": 0.3,
    "use_lstm_policy": false,
    "reward_shaping": {
      "trade_execution_reward": 0.5
    }
  }
}

// settings_half_aggressive.json (layer 2)
{
  "include": "settings.json",
  "config_name": "half_aggressive",
  "common": {
    "ent_coef": 0.15,
    "reward_shaping": {
      "trade_execution_reward": 0.25
    }
  }
}

// settings_ltsm_ha.json (layer 3)
{
  "include": "settings_half_aggressive.json",
  "config_name": "ltsm_ha",
  "common": {
    "use_lstm_policy": true
  }
}
```

**Result for `settings_ltsm_ha.json`:**
```json
{
  "config_name": "ltsm_ha",
  "common": {
    "ent_coef": 0.15,              // From layer 2
    "use_lstm_policy": true,       // Overridden in layer 3
    "reward_shaping": {
      "trade_execution_reward": 0.25  // From layer 2
    }
  }
}
```

#### Technical Details

**Deep Merge Algorithm:**
- The system recursively traverses nested dictionaries
- For each key in the override file:
  - If both base and override values are dictionaries → **recursive merge**
  - Otherwise → **override replaces base value**
- The `"include"` key is removed from the final merged result

**Path Resolution:**
- Include paths are resolved relative to the current settings file's directory
- Absolute paths are supported
- Paths are normalized to absolute paths for consistent comparison

**Circular Dependency Detection:**
- The system detects circular dependencies (e.g., A includes B, B includes A)
- Circular dependencies raise a `ValueError` with details about the dependency chain
- This prevents infinite recursion during settings loading

**Merge Order:**
1. Base file is loaded first (recursively if it also has includes)
2. Current file's values override base values
3. Deep merge ensures nested structures are merged, not replaced

#### Benefits

- **DRY Principle**: No duplication of settings across multiple files
- **Maintainability**: Change base settings once, all inheriting configs update
- **Flexibility**: Create any combination of settings by layering includes
- **Clarity**: Each variant config only shows what's different from its base

#### Best Practices

1. **Base Configuration**: Keep `settings.json` as the comprehensive base with all defaults
2. **Variant Files**: Create variant files that only define differences
3. **Naming Convention**: Use descriptive names like `settings_<feature>_<variant>.json`
4. **Documentation**: Comment complex inheritance chains in your config files
5. **Avoid Circular Dependencies**: The system detects and prevents circular includes, but avoid creating them for clarity
6. **Layer Depth**: While unlimited depth is supported, keep inheritance chains reasonable (2-4 levels) for maintainability

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
- **DCA Step**: Use `log_returns_step` (default 0.1)

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
  "config_name": "base",
  "common": {
    "n_price_points": 100000,
    "default_prices_file": "btc_usdt_1m_prices.txt",
    "training_episodes": 100,
    "train_split_ratio": 0.9,
    "default_model_file": "trading_model.zip",
    "price_history_window": 10,
    "use_lstm_policy": false,
    "ent_coef": 0.3,
    "training_optimization": {
      "n_envs": 4,
      "use_vecenv": true,
      "device": "auto",
      "optimize_batch_size": true,
      "optimize_network_size": true
    },
    "curriculum_learning": {
      "enabled": true,
      "phase1_episodes": 0.3,
      "forced_initial_buy_delay": 10,
      "forced_initial_buy_size": 0.001
    },
    "reward_shaping": {
      "incremental_reward_scale": 5000,
      "trade_execution_reward": 0.5,
      "momentum_reward_scale": 0.3,
      "hold_profit_reward_scale": 0.02,
      "entry_quality_reward_scale": 0.5,
      "end_episode_reward_scale": 10,
      "sharpe_reward_scale": 0.1,
      "drawdown_penalty_scale": 2.0,
      "position_sizing_reward_scale": 0.01,
      "pnl_penalty_scale": 10,
      "entry_incentive_reward": 5.0,
      "invalid_sell_penalty": -1000.0,
      "no_position_hold_penalty": -0.5,
      "has_position_reward": 0.02,
      "full_cash_no_position_penalty": -1.0,
      "steps_since_trade_penalty_scale": 0.05,
      "pre_trade_buy_incentive": 2.0,
      "exploration_bonus": 0.5,
      "initial_training_fixed_reward": 1.0,
      "no_trade_penalty_scale": 0.1,
      "aggressive_trading_bonus": 1.0,
      "risk_taking_multiplier": 1.5
    }
  },
  "btc": {
    "initial_capital": 10000,
    "min_notional": 50.0,
    "min_size": 0.001,
    "trading_fee_rate": 0.0001,
    "profit_threshold": 0.002,
    "partial_sell_ratio": 0.002,
    "dca_threshold": 0.02,
    "dca_ratio": 0.02,
    "lot_size": 0.001,
    "normalization_method": "log_returns",
    "percentage_changes_step": 0.001,
    "log_returns_step": 0.1,
    "z-score_step": 1.1,
    "price_ratio_step": 0.01,
    "dca_volatility_window": 20,
    "dca_max_tiers": 3,
    "dca_base_ratio": 0.02,
    "dca_tier_multiplier": 1.5,
    "dca_virtual_entry_enabled": true,
    "dca_virtual_entry_lookback": 50
  }
}
```

## Tuning Guidelines

### For More Aggressive Trading
- Increase `trade_execution_reward` (0.5-1.0)
- Increase `momentum_reward_scale` (0.4-0.6)
- Decrease `trading_fee_rate` (0.00005)
- Decrease `min_notional` (50.0)
- Increase `risk_taking_multiplier` (1.5-2.0)

### For More Conservative Trading
- Decrease `trade_execution_reward` (0.1-0.25)
- Decrease `momentum_reward_scale` (0.1-0.2)
- Increase `trading_fee_rate` (0.0002)
- Increase `min_notional` (200.0)
- Decrease `risk_taking_multiplier` (1.0-1.2)

### For Better Risk Management
- Increase `drawdown_penalty_scale` (3.0-5.0)
- Increase `sharpe_reward_scale` (0.15-0.25)
- Decrease `position_sizing_reward_scale` (0.005)
- Increase `profit_threshold` (0.003-0.005)

### For Faster Learning
- Increase `incremental_reward_scale` (7000-10000)
- Increase `end_episode_reward_scale` (15-20)
- Decrease `pnl_penalty_scale` (5-8)

### For More Stable Learning
- Decrease `incremental_reward_scale` (3000-4000)
- Decrease `end_episode_reward_scale` (5-8)
- Increase `sharpe_reward_scale` (0.15-0.2)

### Notes

- All monetary values are in USD
- All percentages are expressed as decimals (0.01 = 1%)
- The configuration file is validated on load; invalid values will raise errors
- Changes to `settings.json` require restarting training/evaluation scripts
- Default values are optimized for BTC/USDT trading but can be adjusted for other assets

## Model Architecture

### Reinforcement Learning Algorithm: PPO (Proximal Policy Optimization)

AlphaRL uses **PPO (Proximal Policy Optimization)** from Stable-Baselines3, a state-of-the-art on-policy RL algorithm. PPO is chosen for several reasons:

**Why PPO?**
- **Stability**: PPO uses clipped objective functions that prevent large policy updates, making training more stable than vanilla policy gradient methods
- **Sample Efficiency**: Better sample efficiency compared to other on-policy methods like A2C
- **Robustness**: Works well with continuous and discrete action spaces, making it suitable for trading environments
- **Proven Performance**: Widely used in finance and trading applications due to its reliability

### Policy Network Architecture

The model uses a **Multi-Layer Perceptron (MLP)** policy network with the following structure:

- **Input Layer**: Observation space includes:
  - Current normalized price
  - Cash ratio (cash / initial_capital)
  - Holdings ratio (holdings value / initial_capital)
  - Price history window (last N normalized prices for temporal context)
  - Trading flags (has_never_traded, steps_since_last_trade normalized)
  
- **Hidden Layers**: 
  - **GPU-optimized**: `[128, 128]` for both policy and value networks
  - **CPU-optimized**: `[64, 64]` for faster training on CPU
  - Uses ReLU activation functions

- **Output Layer**: 
  - **Action Space**: Discrete(3) - Hold(0), Buy(1), Sell(2)
  - **Value Function**: Estimates expected return for state value estimation

**Why MLP instead of LSTM?**
- **Temporal Context via Price History**: The `price_history_window` parameter (default: 10) provides sufficient temporal context without the complexity of LSTM
- **Faster Training**: MLP networks train significantly faster than LSTM, enabling quicker experimentation
- **Simplicity**: Easier to tune and debug than recurrent networks
- **Sufficient for Short-Term Patterns**: For minute-level trading, recent price history (10-20 points) captures most relevant patterns

### Training Optimizations

**Vectorized Environments**: Uses `DummyVecEnv` with multiple parallel environments (default: 4) to:
- Collect more diverse experiences per training step
- Accelerate training through parallel data collection
- Maintain compatibility with callbacks and Windows systems

**GPU Acceleration**: Automatically detects and uses CUDA-enabled GPUs when available:
- Significantly faster training (5-10x speedup on modern GPUs)
- Falls back to CPU if GPU unavailable
- Optimizes network size based on device (larger networks for GPU, smaller for CPU)

**Batch Size Optimization**: Automatically adjusts batch size based on:
- Available memory (GPU vs CPU)
- Network architecture size
- Number of parallel environments

**Entropy Coefficient**: Default `0.3` balances exploration vs exploitation:
- Higher values (0.3-0.5): More exploration, better for discovering new strategies
- Lower values (0.1-0.2): More exploitation, better for fine-tuning known strategies

## Dollar-Cost Averaging (DCA)

### What is DCA?

**Dollar-Cost Averaging (DCA)** is an automatic position management strategy that buys additional shares when the price drops below the average entry price. This lowers the average cost basis and improves the break-even point.

### How DCA Works in AlphaRL

**Multi-Tier System**: The DCA implementation uses a **tiered approach** that triggers progressively larger buys as price drops further:

1. **Tier 1**: Triggers when price drops 1× step threshold below entry
2. **Tier 2**: Triggers when price drops 2× step threshold below entry  
3. **Tier 3**: Triggers when price drops 3× step threshold below entry
4. And so on, up to `dca_max_tiers` (default: 3)

**Buy Size Scaling**: Each tier buys a progressively larger amount:
- **Tier 1**: `dca_base_ratio × position_value` (default: 2%)
- **Tier 2**: `dca_base_ratio × dca_tier_multiplier × position_value` (default: 3%)
- **Tier 3**: `dca_base_ratio × (dca_tier_multiplier²) × position_value` (default: 4.5%)

**Example**: With base_ratio=0.02 and tier_multiplier=1.5:
- Price drops 0.1% → Tier 1 triggers → Buy 2% of position value
- Price drops 0.2% → Tier 2 triggers → Buy 3% of position value
- Price drops 0.3% → Tier 3 triggers → Buy 4.5% of position value

### Virtual Entry Point

When **no position exists**, DCA can still trigger using a **virtual entry point**:
- Tracks the lowest normalized price seen in the last `dca_virtual_entry_lookback` steps (default: 50)
- Uses this virtual entry to calculate DCA tiers
- Allows the system to "bootstrap" entries when prices drop significantly
- Helps the model enter positions at favorable prices even when starting from zero holdings

**Why Virtual Entry?**
- **Bootstrap Problem**: Without a position, there's no "entry price" to compare against
- **Entry Optimization**: Encourages buying when prices are at recent lows
- **Momentum Capture**: Helps catch price reversals after significant drops

### Normalization Methods for DCA

DCA thresholds are calculated using **normalized prices** to account for:
- **Volatility**: Higher volatility markets need larger thresholds
- **Price Scale**: Works across different price ranges (BTC at $50k vs $100k)
- **Market Conditions**: Adapts to changing market volatility

The normalization method (`log_returns`, `percentage_changes`, `z-score`, or `price_ratio`) determines how price drops are measured:
- **Log Returns** (default): `log(price_t / price_{t-1})` - Mathematically convenient, handles large moves well
- **Percentage Changes**: `(price_t - price_{t-1}) / price_{t-1}` - Intuitive, good for momentum strategies
- **Z-Score**: `(price - mean) / std` - Accounts for volatility, good for mean reversion
- **Price Ratio**: `price_t / price_0` - Relative to starting price, less adaptive

### Why DCA is Important

1. **Risk Management**: Automatically reduces average entry price when positions are underwater
2. **Profitability**: Lowers break-even point, reducing time to profitability
3. **Automation**: Works independently of model decisions, ensuring consistent position management
4. **Market Adaptation**: Multi-tier system adapts to different market conditions and volatility levels
5. **Entry Optimization**: Virtual entry point helps identify favorable entry opportunities

### DCA vs Model Actions

- **DCA is Rule-Based**: Executes automatically based on price thresholds, independent of model decisions
- **Model Actions are Learned**: The RL agent learns when to buy/sell based on reward signals
- **Complementary**: DCA handles position management, while the model learns optimal entry/exit timing
- **Execution Order**: Model actions execute first, then DCA rules apply (if conditions are met)

## Environment

The `TradingEnv` is a custom Gymnasium environment with:
- **Action space**: Discrete(3) - Hold(0), Buy(1), Sell(2)
- **Observation space**: [price, cash, holdings]
- **Reward**: Portfolio value change from initial capital

## Performance Metrics

- **Initial/Final Capital**: Starting and ending portfolio value
- **Total P&L**: Profit and Loss in absolute terms
- **Total Return**: Percentage gain/loss for the test period
- **Annualized Return**: Total return annualized (assumes 1-minute data frequency)
- **Max Drawdown**: Maximum peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return metric (Mean Return / Volatility, risk-free rate = 0)
- **Calmar Ratio**: Risk-adjusted return metric (Annualized Return / Max Drawdown)

**Note on Time Period**: Performance metrics assume price data points are 1-minute intervals. The Calmar ratio calculation annualizes returns based on this assumption (525,600 minutes per year). If your data uses a different frequency, the annualized metrics will need adjustment.

## Project Structure

```
.
├── alpharl/            # Main package
│   ├── 1_generate_prices.py  # Generate synthetic price data
│   ├── 2_train_model.py      # Train RL model
│   ├── 3_run_model.py        # Run trained model
│   ├── env.py                # Trading environment
│   ├── utils.py              # Utility functions
│   └── web.py                # Visualization
├── settings.json        # Configuration file
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technologies Used

- **Gymnasium**: Environment framework
- **Stable-Baselines3**: RL algorithms (PPO)
- **PyTorch**: Neural network backend
- **Plotly**: Interactive charts
- **Dash**: Web dashboard framework
- **NumPy**: Numerical computations

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This is a educational project for demonstrating reinforcement learning in trading. **DO NOT use this for actual trading decisions.** Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.
