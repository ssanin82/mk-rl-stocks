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
normalization_method = _common_settings.get("normalization_method", "log_returns")

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
pnl_penalty_scale = _reward_shaping.get("pnl_penalty_scale", 100)

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