"""
MK-RL Stocks: Reinforcement Learning Trading Strategy Simulator

A reinforcement learning-based stock trading simulator that uses PPO 
(Proximal Policy Optimization) to train an agent to make buy/sell decisions.
"""

__version__ = "0.1.0"

from mkrl.env import TradingEnv
from mkrl.utils import run_strategy, calculate_metrics
from mkrl.settings import n_price_points

# Import realistic_price_feed from 1_generate_prices for backward compatibility
def realistic_price_feed(
    S0=100,
    mu=0.0001,
    sigma0=0.01,
    alpha=0.05,
    beta=0.9,
    jump_prob=0.01,
    jump_scale=0.05,
    n=n_price_points,
    seed=42
):
    """Generate realistic price feed using GARCH-like model."""
    import numpy as np
    np.random.seed(seed)
    prices = [S0]
    sigma = sigma0
    ret_prev = 0
    for _ in range(n):
        sigma = np.sqrt(
            alpha * ret_prev**2 +
            beta * sigma**2 +
            (1 - alpha - beta) * sigma0**2
        )
        jump = 0
        if np.random.rand() < jump_prob:
            jump = np.random.normal(0, jump_scale)
        ret = mu + sigma * np.random.normal() + jump
        prices.append(prices[-1] * np.exp(ret))
        ret_prev = ret
    return np.array(prices)


__all__ = [
    "TradingEnv",
    "run_strategy",
    "calculate_metrics",
    "realistic_price_feed",
]
