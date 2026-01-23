"""
AlphaRL: Reinforcement Learning Trading Strategy Simulator

A reinforcement learning-based stock trading simulator that uses PPO 
(Proximal Policy Optimization) to train an agent to make buy/sell decisions.
"""

__version__ = "0.1.0"

from alpharl.env import TradingEnv
from alpharl.utils import calculate_metrics, realistic_price_feed

__all__ = [
    "TradingEnv",
    "calculate_metrics",
    "realistic_price_feed",
]
