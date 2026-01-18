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
trade_execution_reward = _common_settings.get("trade_execution_reward")
momentum_reward_scale = _common_settings.get("momentum_reward_scale")
train_split_ratio = _common_settings.get("train_split_ratio")
default_model_file = _common_settings.get("default_model_file")

# BTC-specific settings
initial_capital = _btc_settings.get("initial_capital")
min_notional = _btc_settings.get("min_notional")
min_size = _btc_settings.get("min_size")
trading_fee_rate = _btc_settings.get("trading_fee_rate")
profit_threshold = _btc_settings.get("profit_threshold")
partial_sell_ratio = _btc_settings.get("partial_sell_ratio")
dca_threshold = _btc_settings.get("dca_threshold")
dca_ratio = _btc_settings.get("dca_ratio")

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