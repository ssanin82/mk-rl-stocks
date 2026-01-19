"""
Utility functions for trading strategy execution and metrics calculation.
"""

import numpy as np
import sys
from enum import Enum
from mkrl.settings import n_price_points


class NormalizationMethod(str, Enum):
    """Enumeration of available price normalization methods."""
    PERCENTAGE_CHANGES = "percentage_changes"
    LOG_RETURNS = "log_returns"
    Z_SCORE = "z-score"
    PRICE_RATIO = "price_ratio"
    
    @classmethod
    def values(cls):
        """Return list of all valid enum values."""
        return [item.value for item in cls]


def format_time(seconds):
    """Format time in seconds to a human-readable string (hours, minutes, seconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or len(parts) == 0:
        parts.append(f"{secs:.3f}s")
    
    return " ".join(parts)


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
    np.random.seed(seed)
    prices = [S0]
    sigma = sigma0
    ret_prev = 0
    for _ in range(n):
        # GARCH-like volatility update
        sigma = np.sqrt(
            alpha * ret_prev**2 +
            beta * sigma**2 +
            (1 - alpha - beta) * sigma0**2
        )
        # Jump component
        jump = 0
        if np.random.rand() < jump_prob:
            jump = np.random.normal(0, jump_scale)
        
        # Return
        ret = mu + sigma * np.random.normal() + jump
        prices.append(prices[-1] * np.exp(ret))
        ret_prev = ret
    return np.array(prices)


def normalize_prices(prices, method):
    """
    Normalize price data using the specified method.
    
    Args:
        prices: Array of price values
        method: NormalizationMethod enum value or string
    
    Returns:
        Normalized price array
    """
    if isinstance(method, str):
        try:
            method = NormalizationMethod(method)
        except ValueError:
            available = ", ".join(NormalizationMethod.values())
            print(f"ERROR: Incorrect price normalization method '{method}', available: {available}", file=sys.stderr)
            sys.exit(1)
    
    if method == NormalizationMethod.PERCENTAGE_CHANGES:
        # Percentage Changes: (price_t - price_{t-1}) / price_{t-1}
        normalized = np.zeros_like(prices)
        normalized[0] = 0.0  # First value is 0 (no previous price)
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                normalized[i] = (prices[i] - prices[i-1]) / prices[i-1]
            else:
                normalized[i] = 0.0
        return normalized
    
    elif method == NormalizationMethod.LOG_RETURNS:
        # Log Returns: log(price_t / price_{t-1})
        normalized = np.zeros_like(prices)
        normalized[0] = 0.0  # First value is 0 (no previous price)
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                normalized[i] = np.log(prices[i] / prices[i-1])
            else:
                normalized[i] = 0.0
        return normalized
    
    elif method == NormalizationMethod.Z_SCORE:
        # Z-score Normalization: (price - mean) / std
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        if std_price > 0:
            normalized = (prices - mean_price) / std_price
        else:
            normalized = np.zeros_like(prices)
        return normalized
    
    elif method == NormalizationMethod.PRICE_RATIO:
        # Price Ratio: price_t / price_0 (normalized by first price)
        if len(prices) > 0 and prices[0] > 0:
            normalized = prices / prices[0]
        else:
            normalized = np.ones_like(prices)
        return normalized
    
    else:
        available = ", ".join(NormalizationMethod.values())
        print(f"ERROR: Incorrect price normalization method '{method}', available: {available}", file=sys.stderr)
        sys.exit(1)


def calculate_metrics(portfolio_values, initial_capital):
    """Calculate performance metrics from portfolio values."""
    final_capital = portfolio_values[-1]
    total_return = ((final_capital - initial_capital) / initial_capital) * 100
    
    peak = portfolio_values[0]
    max_dd = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        if dd > max_dd:
            max_dd = dd
    
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    volatility = np.std(returns) * 100
    
    return {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'volatility': volatility,
        'total_pnl': final_capital - initial_capital
    }
