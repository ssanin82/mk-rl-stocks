"""
Script to generate synthetic price data and save to prices.txt
One price per line.
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from alpharl.settings import n_price_points, default_prices_file
from alpharl.utils import realistic_price_feed


def main():
    """Generate prices and save to file."""
    parser = argparse.ArgumentParser(description='Generate synthetic price data')
    parser.add_argument('--output', '-o', type=str, default=default_prices_file,
                        help=f'Output file path (default: {default_prices_file})')
    parser.add_argument('--n-points', '-n', type=int, default=n_price_points,
                        help=f'Number of price points to generate (default: {n_price_points})')
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
