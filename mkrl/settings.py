"""
Settings loader for the RL trading simulator.
Loads configuration from settings.json file.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Default settings path
_default_settings_path = Path(__file__).parent.parent / "settings.json"

# Global settings data (loaded lazily)
_settings_data: Optional[Dict[str, Dict[str, Any]]] = None
_settings_path: Optional[Path] = None
_config_name: Optional[str] = None

def load_settings(settings_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load settings from a JSON file.
    
    Args:
        settings_file: Path to settings file. If None, uses default settings.json
    
    Returns:
        Dictionary containing all settings data
    """
    global _settings_data, _settings_path, _config_name
    
    if settings_file:
        settings_path = Path(settings_file)
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
    else:
        settings_path = _default_settings_path
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
    
    with open(settings_path, 'r') as f:
        _settings_data = json.load(f)
    
    _settings_path = settings_path
    _config_name = _settings_data.get("config_name", "base")
    
    return _settings_data

def reload_settings():
    """Reload settings from the current settings file."""
    if _settings_path:
        load_settings(str(_settings_path))
    else:
        load_settings()

# Load default settings on import
try:
    load_settings()
except FileNotFoundError:
    # If default settings.json doesn't exist, that's okay - user will specify one
    pass

# Extract sections
if _settings_data:
    _common_settings = _settings_data.get("common", {})
    _btc_settings = _settings_data.get("btc", {})
else:
    _common_settings = {}
    _btc_settings = {}

def _get_settings():
    """Get current settings, loading default if not loaded."""
    global _settings_data
    if _settings_data is None:
        load_settings()
        return _settings_data
    return _settings_data

# Common settings (access as attributes for backward compatibility)
def _load_common():
    """Load common settings from current settings data."""
    data = _get_settings()
    common = data.get("common", {})
    return common.get("n_price_points"), common.get("default_prices_file"), common.get("training_episodes"), common.get("train_split_ratio"), common.get("default_model_file"), common.get("price_history_window", 10), common.get("use_lstm_policy", False), common.get("ent_coef", 0.2)

n_price_points = _common_settings.get("n_price_points") if _common_settings else None
default_prices_file = _common_settings.get("default_prices_file") if _common_settings else None
training_episodes = _common_settings.get("training_episodes") if _common_settings else None
train_split_ratio = _common_settings.get("train_split_ratio") if _common_settings else None
default_model_file = _common_settings.get("default_model_file") if _common_settings else None
price_history_window = _common_settings.get("price_history_window", 10) if _common_settings else 10
use_lstm_policy = _common_settings.get("use_lstm_policy", False) if _common_settings else False
ent_coef = _common_settings.get("ent_coef", 0.2) if _common_settings else 0.2

# Curriculum learning settings
_curriculum = _common_settings.get("curriculum_learning", {}) if _common_settings else {}
curriculum_enabled = _curriculum.get("enabled", True) if _curriculum else True
curriculum_phase1_episodes = _curriculum.get("phase1_episodes", 0.3) if _curriculum else 0.3
curriculum_forced_buy_delay = _curriculum.get("forced_initial_buy_delay", 10) if _curriculum else 10
curriculum_forced_buy_size = _curriculum.get("forced_initial_buy_size", 0.001) if _curriculum else 0.001

# Training optimization settings
_training_opt = _common_settings.get("training_optimization", {}) if _common_settings else {}
n_envs = _training_opt.get("n_envs", 4) if _training_opt else 4
use_vecenv = _training_opt.get("use_vecenv", True) if _training_opt else True
device_setting = _training_opt.get("device", "auto") if _training_opt else "auto"
optimize_batch_size = _training_opt.get("optimize_batch_size", True) if _training_opt else True
optimize_network_size = _training_opt.get("optimize_network_size", True) if _training_opt else True

# Reward shaping settings
_reward_shaping = _common_settings.get("reward_shaping", {}) if _common_settings else {}
incremental_reward_scale = _reward_shaping.get("incremental_reward_scale", 5000) if _reward_shaping else 5000
trade_execution_reward = _reward_shaping.get("trade_execution_reward", 0.01) if _reward_shaping else 0.01
momentum_reward_scale = _reward_shaping.get("momentum_reward_scale", 0.3) if _reward_shaping else 0.3
hold_profit_reward_scale = _reward_shaping.get("hold_profit_reward_scale", 0.02) if _reward_shaping else 0.02
entry_quality_reward_scale = _reward_shaping.get("entry_quality_reward_scale", 0.05) if _reward_shaping else 0.05
end_episode_reward_scale = _reward_shaping.get("end_episode_reward_scale", 10) if _reward_shaping else 10
sharpe_reward_scale = _reward_shaping.get("sharpe_reward_scale", 0.1) if _reward_shaping else 0.1
drawdown_penalty_scale = _reward_shaping.get("drawdown_penalty_scale", 2.0) if _reward_shaping else 2.0
position_sizing_reward_scale = _reward_shaping.get("position_sizing_reward_scale", 0.01) if _reward_shaping else 0.01
pnl_penalty_scale = _reward_shaping.get("pnl_penalty_scale", 10) if _reward_shaping else 10
entry_incentive_reward = _reward_shaping.get("entry_incentive_reward", 1.0) if _reward_shaping else 1.0
invalid_sell_penalty = _reward_shaping.get("invalid_sell_penalty", -0.5) if _reward_shaping else -0.5
no_position_hold_penalty = _reward_shaping.get("no_position_hold_penalty", -0.01) if _reward_shaping else -0.01
has_position_reward = _reward_shaping.get("has_position_reward", 0.02) if _reward_shaping else 0.02
full_cash_no_position_penalty = _reward_shaping.get("full_cash_no_position_penalty", -0.02) if _reward_shaping else -0.02
steps_since_trade_penalty_scale = _reward_shaping.get("steps_since_trade_penalty_scale", 0.001) if _reward_shaping else 0.001
pre_trade_buy_incentive = _reward_shaping.get("pre_trade_buy_incentive", 0.5) if _reward_shaping else 0.5
exploration_bonus = _reward_shaping.get("exploration_bonus", 0.1) if _reward_shaping else 0.1
initial_training_fixed_reward = _reward_shaping.get("initial_training_fixed_reward", 0.2) if _reward_shaping else 0.2
no_trade_penalty_scale = _reward_shaping.get("no_trade_penalty_scale", 0.1) if _reward_shaping else 0.1
aggressive_trading_bonus = _reward_shaping.get("aggressive_trading_bonus", 1.0) if _reward_shaping else 1.0
risk_taking_multiplier = _reward_shaping.get("risk_taking_multiplier", 1.5) if _reward_shaping else 1.5

# BTC-specific settings
initial_capital = _btc_settings.get("initial_capital") if _btc_settings else None
min_notional = _btc_settings.get("min_notional") if _btc_settings else None
min_size = _btc_settings.get("min_size") if _btc_settings else None
trading_fee_rate = _btc_settings.get("trading_fee_rate") if _btc_settings else None
profit_threshold = _btc_settings.get("profit_threshold") if _btc_settings else None
partial_sell_ratio = _btc_settings.get("partial_sell_ratio") if _btc_settings else None
dca_threshold = _btc_settings.get("dca_threshold") if _btc_settings else None
dca_ratio = _btc_settings.get("dca_ratio") if _btc_settings else None
lot_size = _btc_settings.get("lot_size") if _btc_settings else None
normalization_method = _btc_settings.get("normalization_method", "log_returns") if _btc_settings else "log_returns"
percentage_changes_step = _btc_settings.get("percentage_changes_step", 0.001) if _btc_settings else 0.001
log_returns_step = _btc_settings.get("log_returns_step", 0.001) if _btc_settings else 0.001
z_score_step = _btc_settings.get("z-score_step", 1.1) if _btc_settings else 1.1
price_ratio_step = _btc_settings.get("price_ratio_step", 0.01) if _btc_settings else 0.01
dca_volatility_window = _btc_settings.get("dca_volatility_window", 20) if _btc_settings else 20
dca_max_tiers = _btc_settings.get("dca_max_tiers", 3) if _btc_settings else 3
dca_base_ratio = _btc_settings.get("dca_base_ratio", 0.02) if _btc_settings else 0.02
dca_tier_multiplier = _btc_settings.get("dca_tier_multiplier", 1.5) if _btc_settings else 1.5
dca_virtual_entry_enabled = _btc_settings.get("dca_virtual_entry_enabled", True) if _btc_settings else True
dca_virtual_entry_lookback = _btc_settings.get("dca_virtual_entry_lookback", 50) if _btc_settings else 50

# Get config name
def get_config_name() -> str:
    """Get the current config name."""
    return _config_name if _config_name else "base"

# Helper function to get all settings
def get_all_settings() -> Dict[str, Dict[str, Any]]:
    """Get all settings as a dictionary."""
    data = _get_settings()
    return data.copy() if data else {}

# Helper function to get common settings
def get_common_settings() -> Dict[str, Any]:
    """Get common settings as a dictionary."""
    data = _get_settings()
    return data.get("common", {}).copy() if data else {}

# Helper function to get BTC settings
def get_btc_settings() -> Dict[str, Any]:
    """Get BTC settings as a dictionary."""
    data = _get_settings()
    return data.get("btc", {}).copy() if data else {}
