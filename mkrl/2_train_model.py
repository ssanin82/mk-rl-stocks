"""
Script to train the RL model on first 90% of prices.txt
Saves the trained model to a file for reuse.
"""

import sys
from pathlib import Path
import numpy as np
import time
import argparse
import webbrowser

# Add parent directory to Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from mkrl.env import TradingEnv
from mkrl.settings import (
    initial_capital, min_notional, min_size, trading_fee_rate,
    training_episodes, default_prices_file, default_model_file, train_split_ratio
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


class TrainingProgressCallback(BaseCallback):
    """Callback to log training progress to file and console."""
    
    def __init__(self, log_file, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
    def _on_step(self) -> bool:
        # Log every 1000 steps or at major milestones
        if self.num_timesteps % 1000 == 0 or self.num_timesteps == self.total_timesteps:
            elapsed = time.time() - self.start_time
            progress = (self.num_timesteps / self.total_timesteps) * 100
            
            # Estimate remaining time and calculate speed
            if self.num_timesteps > 0 and elapsed > 0:
                steps_per_second = self.num_timesteps / elapsed
                remaining_steps = self.total_timesteps - self.num_timesteps
                eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                speed_str = f"{steps_per_second:.1f} steps/s"
            else:
                eta_str = "calculating..."
                speed_str = "calculating..."
            
            log_line = (
                f"Step {self.num_timesteps:,}/{self.total_timesteps:,} "
                f"({progress:.1f}%) | "
                f"Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s | "
                f"ETA: {eta_str} | "
                f"Speed: {speed_str}"
            )
            
            # Write to log file
            if self.log_file:
                self.log_file.write(f"{log_line}\n")
                self.log_file.flush()
            
            # Also print to console every 5% progress
            if int(progress) % 5 == 0 or self.num_timesteps == self.total_timesteps:
                print(f"  {log_line}")
            
            self.last_log_time = time.time()
        
        return True


def main():
    """Train the model on first 90% of prices."""
    parser = argparse.ArgumentParser(description='Train RL trading model')
    parser.add_argument('--prices', '-p', type=str, default=default_prices_file,
                        help=f'Input prices file (default: {default_prices_file})')
    parser.add_argument('--model', '-m', type=str, default=default_model_file,
                        help=f'Output model file (default: {default_model_file})')
    parser.add_argument('--episodes', '-e', type=int, default=training_episodes,
                        help=f'Number of training episodes (default: {training_episodes})')
    parser.add_argument('--split', type=float, default=train_split_ratio,
                        help=f'Training split ratio (default: {train_split_ratio})')
    
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
        initial_capital=initial_capital,
        min_notional=min_notional,
        min_size=min_size,
        trading_fee_rate=trading_fee_rate
    )
    
    # Create and train model
    print(f"\nTraining PPO model for {training_timesteps} timesteps ({args.episodes} episodes)...")
    print("  (This may take several minutes...)")
    ts = time.time()
    
    # Create log file for training progress
    log_file_path = Path("training_progress.log")
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Training Progress Log\n")
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Total timesteps: {training_timesteps:,}\n")
        log_file.write(f"Episodes: {args.episodes}\n")
        log_file.write(f"Training prices: {len(train_prices)}\n")
        log_file.write(f"{'='*80}\n\n")
        
        # Create progress callback
        progress_callback = TrainingProgressCallback(
            log_file=log_file,
            total_timesteps=training_timesteps,
            verbose=1
        )
        
        # Create model (with optional TensorBoard logging if available)
        try:
            import tensorboard
            tensorboard_available = True
            tensorboard_log_dir = "./tensorboard_logs/"
        except ImportError:
            tensorboard_available = False
            tensorboard_log_dir = None
        
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": env_train,
            "verbose": 1,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,  # Higher entropy = more exploration
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
        
        if tensorboard_available:
            model_kwargs["tensorboard_log"] = tensorboard_log_dir
        
        model = PPO(**model_kwargs)
        
        # Train with callback
        model.learn(
            total_timesteps=training_timesteps,
            callback=progress_callback
        )
    
    print(f"✓ Training progress logged to {log_file_path.absolute()}")
    if tensorboard_available:
        print(f"✓ TensorBoard logs saved to ./tensorboard_logs/")
        print(f"  Run 'tensorboard --logdir ./tensorboard_logs' to view visual progress")
    
    training_time = time.time() - ts
    print(f"\n✓ Training complete! Took {round(training_time, 3)} seconds")
    
    # Save model
    model_path = Path(args.model)
    model.save(str(model_path))
    print(f"✓ Model saved to {model_path.absolute()}")
    
    # Create and open completion page in browser
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Complete</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .container {{
            text-align: center;
            padding: 40px;
            background: #2a2a2a;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            color: #4CAF50;
            margin-bottom: 20px;
        }}
        .message {{
            font-size: 24px;
            margin: 20px 0;
        }}
        .time {{
            font-size: 32px;
            color: #81C784;
            font-weight: bold;
            margin: 20px 0;
        }}
        .details {{
            margin-top: 30px;
            color: #b0b0b0;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>✓ Model Training Complete!</h1>
        <div class="message">Training time:</div>
        <div class="time">{round(training_time, 3)} seconds</div>
        <div class="details">
            <p>Model saved to: <code>{model_path.name}</code></p>
            <p>Training data: {len(train_prices)} prices</p>
            <p>Episodes: {args.episodes}</p>
            <p>Total timesteps: {training_timesteps:,}</p>
        </div>
    </div>
</body>
</html>"""
    
    # Save to temporary HTML file
    html_file = Path("training_complete.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    html_path = html_file.resolve()
    print(f"\nOpening completion page in browser...")
    webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
