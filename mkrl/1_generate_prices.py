"""
Script to generate synthetic price data and save to prices.txt
One price per line.
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add parent directory to Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from mkrl.constants import N_PRICE_POINTS, DEFAULT_PRICES_FILE


def realistic_price_feed(
    S0=100,
    mu=0.0001,
    sigma0=0.01,
    alpha=0.05,     # reaction to shocks
    beta=0.9,       # volatility persistence
    jump_prob=0.01,
    jump_scale=0.05,
    n=N_PRICE_POINTS,
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


def main():
    """Generate prices and save to file."""
    parser = argparse.ArgumentParser(description='Generate synthetic price data')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_PRICES_FILE,
                        help=f'Output file path (default: {DEFAULT_PRICES_FILE})')
    parser.add_argument('--n-points', '-n', type=int, default=N_PRICE_POINTS,
                        help=f'Number of price points to generate (default: {N_PRICE_POINTS})')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print(f"Generating {args.n_points} price points...")
    prices = realistic_price_feed(n=args.n_points, seed=args.seed)
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for price in prices:
            f.write(f"{price:.6f}\n")
    
    print(f"✓ Saved {len(prices)} prices to {output_path.absolute()}")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")


if __name__ == "__main__":
    main()
