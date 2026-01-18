"""
MK-RL Stocks: Reinforcement Learning Trading Strategy Simulator

A reinforcement learning-based stock trading simulator that uses PPO 
(Proximal Policy Optimization) to train an agent to make buy/sell decisions.
"""

__version__ = "0.1.0"

from mkrl.main import (
    TradingEnv,
    run_strategy,
    calculate_metrics,
    realistic_price_feed,
    main,
)

__all__ = [
    "TradingEnv",
    "run_strategy",
    "calculate_metrics",
    "realistic_price_feed",
    "main",
]
