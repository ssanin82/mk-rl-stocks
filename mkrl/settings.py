"""
Settings loader for the RL trading simulator.
Loads configuration from settings.json file.
"""

import json
from pathlib import Path
from typing import Dict, Any

# Load settings from JSON file
_settings_path = Path(__file__).parent.parent / "settings.json"

if not _settings_path.exists():
    raise FileNotFoundError(f"Settings file not found: {_settings_path}")

with open(_settings_path, 'r') as f:
    _settings_data: Dict[str, Dict[str, Any]] = json.load(f)

# Extract sections
_common_settings = _settings_data.get("common", {})
_btc_settings = _settings_data.get("btc", {})

# Common settings (access as attributes for backward compatibility)
n_price_points = _common_settings.get("n_price_points")
default_prices_file = _common_settings.get("default_prices_file")
training_episodes = _common_settings.get("training_episodes")
train_split_ratio = _common_settings.get("train_split_ratio")
default_model_file = _common_settings.get("default_model_file")
price_history_window = _common_settings.get("price_history_window", 10)
use_lstm_policy = _common_settings.get("use_lstm_policy", False)
ent_coef = _common_settings.get("ent_coef", 0.2)

# Curriculum learning settings
_curriculum = _common_settings.get("curriculum_learning", {})
curriculum_enabled = _curriculum.get("enabled", True)
curriculum_phase1_episodes = _curriculum.get("phase1_episodes", 0.3)
curriculum_forced_buy_delay = _curriculum.get("forced_initial_buy_delay", 10)
curriculum_forced_buy_size = _curriculum.get("forced_initial_buy_size", 0.001)

# Training optimization settings
_training_opt = _common_settings.get("training_optimization", {})
n_envs = _training_opt.get("n_envs", 4)
use_vecenv = _training_opt.get("use_vecenv", True)
device_setting = _training_opt.get("device", "auto")
optimize_batch_size = _training_opt.get("optimize_batch_size", True)
optimize_network_size = _training_opt.get("optimize_network_size", True)

# Reward shaping settings
_reward_shaping = _common_settings.get("reward_shaping", {})
incremental_reward_scale = _reward_shaping.get("incremental_reward_scale", 5000)
trade_execution_reward = _reward_shaping.get("trade_execution_reward", 0.01)
momentum_reward_scale = _reward_shaping.get("momentum_reward_scale", 0.3)
hold_profit_reward_scale = _reward_shaping.get("hold_profit_reward_scale", 0.02)
entry_quality_reward_scale = _reward_shaping.get("entry_quality_reward_scale", 0.05)
end_episode_reward_scale = _reward_shaping.get("end_episode_reward_scale", 10)
sharpe_reward_scale = _reward_shaping.get("sharpe_reward_scale", 0.1)
drawdown_penalty_scale = _reward_shaping.get("drawdown_penalty_scale", 2.0)
position_sizing_reward_scale = _reward_shaping.get("position_sizing_reward_scale", 0.01)
pnl_penalty_scale = _reward_shaping.get("pnl_penalty_scale", 10)
entry_incentive_reward = _reward_shaping.get("entry_incentive_reward", 1.0)
invalid_sell_penalty = _reward_shaping.get("invalid_sell_penalty", -0.5)
no_position_hold_penalty = _reward_shaping.get("no_position_hold_penalty", -0.01)
has_position_reward = _reward_shaping.get("has_position_reward", 0.02)
full_cash_no_position_penalty = _reward_shaping.get("full_cash_no_position_penalty", -0.02)
steps_since_trade_penalty_scale = _reward_shaping.get("steps_since_trade_penalty_scale", 0.001)
pre_trade_buy_incentive = _reward_shaping.get("pre_trade_buy_incentive", 0.5)
exploration_bonus = _reward_shaping.get("exploration_bonus", 0.1)
initial_training_fixed_reward = _reward_shaping.get("initial_training_fixed_reward", 0.2)
no_trade_penalty_scale = _reward_shaping.get("no_trade_penalty_scale", 0.1)
aggressive_trading_bonus = _reward_shaping.get("aggressive_trading_bonus", 1.0)
risk_taking_multiplier = _reward_shaping.get("risk_taking_multiplier", 1.5)

# BTC-specific settings
initial_capital = _btc_settings.get("initial_capital")
min_notional = _btc_settings.get("min_notional")
min_size = _btc_settings.get("min_size")
trading_fee_rate = _btc_settings.get("trading_fee_rate")
profit_threshold = _btc_settings.get("profit_threshold")
partial_sell_ratio = _btc_settings.get("partial_sell_ratio")
dca_threshold = _btc_settings.get("dca_threshold")
dca_ratio = _btc_settings.get("dca_ratio")
lot_size = _btc_settings.get("lot_size")
normalization_method = _btc_settings.get("normalization_method", "log_returns")
percentage_changes_step = _btc_settings.get("percentage_changes_step", 0.001)
log_returns_step = _btc_settings.get("log_returns_step", 0.001)
z_score_step = _btc_settings.get("z-score_step", 1.1)
price_ratio_step = _btc_settings.get("price_ratio_step", 0.01)
dca_volatility_window = _btc_settings.get("dca_volatility_window", 20)
dca_max_tiers = _btc_settings.get("dca_max_tiers", 3)
dca_base_ratio = _btc_settings.get("dca_base_ratio", 0.02)
dca_tier_multiplier = _btc_settings.get("dca_tier_multiplier", 1.5)
dca_virtual_entry_enabled = _btc_settings.get("dca_virtual_entry_enabled", True)
dca_virtual_entry_lookback = _btc_settings.get("dca_virtual_entry_lookback", 50)

# Helper function to get all settings
def get_all_settings() -> Dict[str, Dict[str, Any]]:
    """Get all settings as a dictionary."""
    return _settings_data.copy()

# Helper function to get common settings
def get_common_settings() -> Dict[str, Any]:
    """Get common settings as a dictionary."""
    return _common_settings.copy()

# Helper function to get BTC settings
def get_btc_settings() -> Dict[str, Any]:
    """Get BTC settings as a dictionary."""
    return _btc_settings.copy()