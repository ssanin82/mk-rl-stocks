"""
Script to train the RL model on first 90% of prices.txt
Saves the trained model to a file for reuse.
"""

import sys
from pathlib import Path
import numpy as np
import time
import argparse

# Add parent directory to Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from mkrl.main import TradingEnv
from mkrl.constants import (
    INITIAL_CAPITAL, MIN_NOTIONAL, MIN_SIZE, TRADING_FEE_RATE,
    TRAINING_EPISODES, DEFAULT_PRICES_FILE, DEFAULT_MODEL_FILE, TRAIN_SPLIT_RATIO
)


def load_prices(prices_file):
    """Load prices from file (one price per line)."""
    prices_path = Path(prices_file)
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices file not found: {prices_path}")
    
    prices = []
    with open(prices_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    prices.append(float(line))
                except ValueError:
                    continue
    
    if len(prices) == 0:
        raise ValueError(f"No valid prices found in {prices_path}")
    
    return np.array(prices)


def main():
    """Train the model on first 90% of prices."""
    parser = argparse.ArgumentParser(description='Train RL trading model')
    parser.add_argument('--prices', '-p', type=str, default=DEFAULT_PRICES_FILE,
                        help=f'Input prices file (default: {DEFAULT_PRICES_FILE})')
    parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL_FILE,
                        help=f'Output model file (default: {DEFAULT_MODEL_FILE})')
    parser.add_argument('--episodes', '-e', type=int, default=TRAINING_EPISODES,
                        help=f'Number of training episodes (default: {TRAINING_EPISODES})')
    parser.add_argument('--split', type=float, default=TRAIN_SPLIT_RATIO,
                        help=f'Training split ratio (default: {TRAIN_SPLIT_RATIO})')
    
    args = parser.parse_args()
    
    # Load prices
    print(f"Loading prices from {args.prices}...")
    all_prices = load_prices(args.prices)
    print(f"  Loaded {len(all_prices)} price points")
    
    # Split into training set
    split_idx = int(len(all_prices) * args.split)
    train_prices = all_prices[:split_idx]
    print(f"  Using first {len(train_prices)} prices ({args.split*100:.0f}%) for training")
    
    # Calculate training timesteps: one episode = one pass through all training prices
    # So total timesteps = number of training prices * number of episodes
    training_timesteps = len(train_prices) * args.episodes
    print(f"  Training for {args.episodes} episodes = {training_timesteps} timesteps")
    
    # Create training environment
    print("\nCreating training environment...")
    env_train = TradingEnv(
        train_prices,
        initial_capital=INITIAL_CAPITAL,
        min_notional=MIN_NOTIONAL,
        min_size=MIN_SIZE,
        trading_fee_rate=TRADING_FEE_RATE
    )
    
    # Create and train model
    print(f"\nTraining PPO model for {training_timesteps} timesteps ({args.episodes} episodes)...")
    print("  (This may take several minutes...)")
    ts = time.time()
    
    model = PPO(
        "MlpPolicy", 
        env_train, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,  # Higher entropy = more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    model.learn(total_timesteps=training_timesteps)
    
    training_time = time.time() - ts
    print(f"\n✓ Training complete! Took {round(training_time, 3)} seconds")
    
    # Save model
    model_path = Path(args.model)
    model.save(str(model_path))
    print(f"✓ Model saved to {model_path.absolute()}")


if __name__ == "__main__":
    main()
