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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from mkrl.env import TradingEnv
import torch
from mkrl.utils import format_time, normalize_prices, NormalizationMethod
import mkrl.settings as settings_module


def create_training_complete_html(training_time, model_path, train_prices_count, episodes, training_timesteps):
    """Create and save HTML page showing training completion summary."""
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
        <div class="time">{format_time(training_time)}</div>
        <div class="details">
            <p>Model saved to: <code>{model_path.name}</code></p>
            <p>Training data: {train_prices_count} prices</p>
            <p>Episodes: {episodes}</p>
            <p>Total timesteps: {training_timesteps:,}</p>
        </div>
    </div>
</body>
</html>"""
    
    # Save to temporary HTML file
    html_file = Path("training_complete.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_file.resolve()


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
    """Callback to log training progress to file and console with in-place updates."""
    
    def __init__(self, log_file, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.header_printed = False
        
    def _print_header(self):
        """Print the header frame once."""
        if not self.header_printed:
            print("\n" + "=" * 80)
            print("Training Progress")
            print("=" * 80)
            self.header_printed = True
    
    def _on_step(self) -> bool:
        # Log every 1000 steps or at major milestones
        if self.num_timesteps % 1000 == 0 or self.num_timesteps == self.total_timesteps:
            self._print_header()
            
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
            
            # Format elapsed time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            
            # Create progress line
            progress_line = (
                f"Step {self.num_timesteps:,}/{self.total_timesteps:,} "
                f"({progress:.1f}%) | "
                f"Elapsed: {elapsed_str} | "
                f"ETA: {eta_str} | "
                f"Speed: {speed_str}"
            )
            
            # Write to log file
            if self.log_file:
                self.log_file.write(f"{progress_line}\n")
                self.log_file.flush()
            
            # Print to console with carriage return for in-place update
            # Use \r to return to start of line and padding to clear previous content
            # Add extra spaces at the end to clear any leftover characters from previous updates
            print(f"\r{progress_line}" + " " * 20, end="", flush=True)
            
            # Print newline when training is complete
            if self.num_timesteps >= self.total_timesteps:
                print()  # Final newline
            
            self.last_log_time = time.time()
        
        return True


def train_model_for_config(settings_file: str, prices_file: str, split: float, episodes: int):
    """
    Train a single model for a given settings file.
    
    Args:
        settings_file: Path to settings JSON file
        prices_file: Path to prices file
        split: Train/test split ratio
        episodes: Number of training episodes
    
    Returns:
        Tuple of (config_name, model_path, training_time)
    """
    # Reload settings from the specified file
    settings_module.load_settings(settings_file)
    
    # Import settings after reload
    from mkrl.settings import (
        initial_capital, min_notional, min_size, trading_fee_rate, lot_size,
        training_episodes, default_prices_file, train_split_ratio,
        use_lstm_policy, normalization_method, ent_coef,
        curriculum_enabled, curriculum_phase1_episodes, curriculum_forced_buy_delay,
        curriculum_forced_buy_size, n_envs, use_vecenv, device_setting,
        optimize_batch_size, optimize_network_size, get_config_name
    )
    
    config_name = get_config_name()
    
    # Load prices
    print(f"\n{'='*70}")
    print(f"Training model: {config_name}")
    print(f"Settings file: {settings_file}")
    print(f"{'='*70}")
    print(f"Loading prices from {prices_file}...")
    all_prices = load_prices(prices_file)
    print(f"  Loaded {len(all_prices)} price points")
    
    # Validate normalization method
    try:
        norm_method = NormalizationMethod(normalization_method)
    except ValueError:
        available = ", ".join(NormalizationMethod.values())
        print(f"ERROR: Incorrect price normalization method '{normalization_method}', available: {available}", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Using normalization method: {norm_method.value}")
    
    # Split into training set (use actual prices for trading, normalization happens in observation space)
    split_idx = int(len(all_prices) * split)
    train_prices = all_prices[:split_idx]
    print(f"  Using first {len(train_prices)} prices ({split*100:.0f}%) for training")
    
    # Calculate training timesteps: one episode = one pass through all training prices
    # So total timesteps = number of training prices * number of episodes
    training_timesteps = len(train_prices) * episodes
    print(f"  Training for {episodes} episodes = {training_timesteps} timesteps")
    
    # GPU Detection and Status
    print("\n" + "="*70)
    print("GPU / DEVICE DETECTION")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {gpu_props.name}")
            print(f"    Memory: {gpu_props.total_memory / 1e9:.2f} GB")
            print(f"    Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        device = "cuda"
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\n⚠ GPU not available, using CPU")
        print("  Note: Training will be slower on CPU")
        print("  To enable GPU:")
        print("  1. Ensure you have an NVIDIA GPU")
        print("  2. Install CUDA-enabled PyTorch:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("="*70 + "\n")
    
    # Determine device based on settings
    if device_setting == "auto":
        final_device = device  # Use detected device
    elif device_setting == "cuda" and cuda_available:
        final_device = "cuda"
    else:
        final_device = "cpu"
        if device_setting == "cuda" and not cuda_available:
            print(f"⚠ Warning: device='cuda' requested but CUDA not available, using CPU")
    
    # Create training environment(s)
    print("Creating training environment...")
    if use_vecenv and n_envs > 1:
        print(f"  Using VecEnv with {n_envs} parallel environments for faster training")
        # Always use DummyVecEnv (works better with callbacks, especially on Windows)
        # DummyVecEnv is still efficient and avoids multiprocessing issues
        # SubprocVecEnv has issues with callbacks accessing environment state on Windows
        print("  Using DummyVecEnv (compatible with callbacks and Windows)")
        print("  Note: DummyVecEnv still provides efficient parallelization")
        env_train = DummyVecEnv([lambda: TradingEnv(
            train_prices,
            initial_capital=initial_capital,
            min_notional=min_notional,
            min_size=min_size,
            trading_fee_rate=trading_fee_rate,
            lot_size=lot_size
        )] * n_envs)
        print(f"  ✓ Created {n_envs} parallel environments")
    else:
        print("  Using single environment")
        env_train = TradingEnv(
            train_prices,
            initial_capital=initial_capital,
            min_notional=min_notional,
            min_size=min_size,
            trading_fee_rate=trading_fee_rate,
            lot_size=lot_size
        )
    
    # Create and train model
    print(f"\nTraining PPO model for {training_timesteps} timesteps ({episodes} episodes)...")
    print("(This may take several minutes...)\n")
    ts = time.time()
    
    # Create progress callback (console output only, no file)
    progress_callback = TrainingProgressCallback(
        log_file=None,  # No file logging
        total_timesteps=training_timesteps,
        verbose=1
    )
    
    # Create model (with optional TensorBoard logging if available)
    try:
        import tensorboard
        tensorboard_available = True
        tensorboard_log_dir = Path(f"./tensorboard_logs_{config_name}/")
    except ImportError:
        tensorboard_available = False
        tensorboard_log_dir = None
    
    # Choose policy based on settings
    # Note: For LSTM policies, you would need sb3-contrib or custom feature extractors
    # The price_history_window already provides temporal context to the MLP
    
    # Optimize network size for speed if enabled
    if optimize_network_size:
        # Smaller network = faster training (with minimal performance impact for this problem)
        # Original: [256, 256, 128] -> Optimized: [128, 128] for faster training
        if cuda_available:
            # GPU can handle larger networks, but still optimize for speed
            net_arch = dict(pi=[128, 128], vf=[128, 128])
            print("  Network: Optimized [128, 128] (GPU-optimized)")
        else:
            # CPU benefits more from smaller networks
            net_arch = dict(pi=[64, 64], vf=[64, 64])
            print("  Network: Optimized [64, 64] (CPU-optimized)")
    else:
        # Use larger network (original)
        net_arch = dict(pi=[256, 256, 128], vf=[256, 256, 128])
        print("  Network: Standard [256, 256, 128]")
    
    if use_lstm_policy:
        # For LSTM, we'll use a deeper MLP that can learn patterns from price history
        # The price history is already included in observations via price_history_window
        print("Note: Using enhanced MLP with price history (LSTM requires sb3-contrib)")
        policy_kwargs = {"net_arch": net_arch}
        policy_name = "MlpPolicy"
    else:
        # Enhanced MLP policy with price history
        policy_kwargs = {"net_arch": net_arch}
        policy_name = "MlpPolicy"
    
    # Optimize batch size for speed if enabled
    if optimize_batch_size:
        if use_vecenv and n_envs > 1:
            # With parallel environments, use larger batches
            if cuda_available:
                batch_size = 128  # GPU can handle larger batches
                n_steps = 2048 * n_envs  # Scale steps with number of envs
            else:
                batch_size = 64  # CPU: moderate batch size
                n_steps = 1024 * n_envs  # Scale steps with number of envs
        else:
            # Single environment
            if cuda_available:
                batch_size = 64
                n_steps = 2048
            else:
                batch_size = 32  # CPU: smaller batches for faster updates
                n_steps = 1024
        print(f"  Batch size: {batch_size} (optimized)")
        print(f"  N-steps: {n_steps} (optimized)")
    else:
        batch_size = 64
        n_steps = 2048
        print(f"  Batch size: {batch_size} (standard)")
        print(f"  N-steps: {n_steps} (standard)")
    
    model_kwargs = {
        "policy": policy_name,
        "policy_kwargs": policy_kwargs,
        "env": env_train,
        "verbose": 1,
        "learning_rate": 3e-4,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": ent_coef,  # Load from settings (default 0.2)
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "device": final_device,  # Explicitly set device
    }
    
    if tensorboard_available:
        model_kwargs["tensorboard_log"] = str(tensorboard_log_dir)
    
    print(f"\n  Device: {final_device}")
    print(f"  Policy: {policy_name}")
    print()
    
    model = PPO(**model_kwargs)
    
    # Curriculum learning: Phase 1 with forced initial buys, Phase 2 free choice
    phase1_timesteps = int(training_timesteps * curriculum_phase1_episodes) if curriculum_enabled else 0
    phase2_timesteps = training_timesteps - phase1_timesteps
    
    if curriculum_enabled and phase1_timesteps > 0:
        print(f"\n📚 Curriculum Learning Phase 1: {phase1_timesteps:,} timesteps (forced initial buys)")
        print(f"   Model will be forced to buy after {curriculum_forced_buy_delay} steps if no trade occurred")
        
        # Phase 1: Train with forced initial buys
        class ForcedBuyCallback(BaseCallback):
            def __init__(self, env, forced_buy_delay, forced_buy_size, verbose=0):
                super().__init__(verbose)
                self.env = env
                self.forced_buy_delay = forced_buy_delay
                self.forced_buy_size = forced_buy_size
                self.step_count = 0
                # Check if we're using VecEnv
                self.is_vecenv = isinstance(self.env, (DummyVecEnv, SubprocVecEnv))
                self.is_subproc = isinstance(self.env, SubprocVecEnv)
            
            def _force_buy_in_env(self, env_unwrapped):
                """Helper method to force a buy in a single environment."""
                if (not env_unwrapped.has_ever_traded and env_unwrapped.holdings == 0):
                    # Force a buy action
                    price = env_unwrapped.prices[env_unwrapped.current_step]
                    shares = max(self.forced_buy_size, env_unwrapped.min_size)
                    shares = env_unwrapped._round_to_lot_size(shares)
                    notional = shares * price
                    fee = notional * env_unwrapped.trading_fee_rate
                    total_cost = notional + fee
                    
                    if env_unwrapped.cash >= total_cost and notional >= env_unwrapped.min_notional:
                        env_unwrapped.holdings += shares
                        env_unwrapped.cash -= total_cost
                        env_unwrapped.cash = max(0, env_unwrapped.cash)
                        cost_this_trade = notional
                        env_unwrapped.total_cost_basis += cost_this_trade
                        if env_unwrapped.holdings > 0:
                            env_unwrapped.avg_entry_price = env_unwrapped.total_cost_basis / env_unwrapped.holdings
                            current_norm_price = env_unwrapped.normalized_prices[env_unwrapped.current_step]
                            env_unwrapped.normalized_entry_price = current_norm_price
                        env_unwrapped.has_ever_traded = True
                        env_unwrapped.last_trade_step = env_unwrapped.current_step
                        env_unwrapped.steps_since_last_trade = 0
                        return True
                return False
            
            def _on_step(self) -> bool:
                self.step_count += 1
                
                # Only force buys periodically
                if self.step_count % self.forced_buy_delay != 0:
                    return True
                
                # Check if we should force a buy
                if self.is_vecenv:
                    # For VecEnv, need to access individual environments
                    if isinstance(self.env, DummyVecEnv):
                        # DummyVecEnv: direct access to envs list
                        for env_idx in range(self.env.num_envs):
                            env_unwrapped = self.env.envs[env_idx].unwrapped if hasattr(self.env.envs[env_idx], 'unwrapped') else self.env.envs[env_idx]
                            self._force_buy_in_env(env_unwrapped)
                    else:
                        # SubprocVecEnv: Skip forced buys (not currently supported with SubprocVecEnv)
                        # Since we now always use DummyVecEnv, this branch shouldn't be reached
                        # But keeping it as a fallback
                        pass
                else:
                    # Single environment
                    env_unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                    self._force_buy_in_env(env_unwrapped)
                
                return True
        
        forced_buy_callback = ForcedBuyCallback(
            env_train, 
            curriculum_forced_buy_delay, 
            curriculum_forced_buy_size
        )
        
        phase1_callbacks = [progress_callback, forced_buy_callback]
        model.learn(
            total_timesteps=phase1_timesteps,
            callback=phase1_callbacks
        )
        
        if phase2_timesteps > 0:
            print(f"\n📚 Curriculum Learning Phase 2: {phase2_timesteps:,} timesteps (free choice)")
            # Phase 2: Continue training with free choice
            model.learn(
                total_timesteps=phase2_timesteps,
                callback=progress_callback,
                reset_num_timesteps=False  # Continue from Phase 1
            )
    else:
        # Standard training without curriculum
        model.learn(
            total_timesteps=training_timesteps,
            callback=progress_callback
        )
    
    if tensorboard_available:
        tensorboard_log_path = Path(f"./tensorboard_logs_{config_name}/")
        print(f"✓ TensorBoard logs saved to {tensorboard_log_path}")
        print(f"  Run 'tensorboard --logdir {tensorboard_log_path}' to view visual progress")
    
    training_time = time.time() - ts
    print(f"✓ Training complete for {config_name}! Took {format_time(training_time)}")
    
    # Save model with config name
    model_path = Path(f"model_{config_name}.zip")
    model.save(str(model_path))
    print(f"✓ Model saved to {model_path.absolute()}")
    
    return config_name, model_path, training_time


def main():
    """Train the model(s) on first 90% of prices."""
    parser = argparse.ArgumentParser(description='Train RL trading model(s)')
    parser.add_argument('--prices', '-p', type=str, default=None,
                        help='Input prices file (default: from settings.json)')
    parser.add_argument('--episodes', '-e', type=int, default=None,
                        help='Number of training episodes (default: from settings.json)')
    parser.add_argument('--split', type=float, default=None,
                        help='Training split ratio (default: from settings.json)')
    parser.add_argument('settings_files', nargs='*', 
                        help='Settings JSON file(s) to use (default: settings.json)')
    
    args = parser.parse_args()
    
    # Determine settings files to use
    if args.settings_files:
        settings_files = args.settings_files
    else:
        # Default to settings.json
        settings_files = ["settings.json"]
    
    print(f"\n{'='*70}")
    print(f"TRAINING {len(settings_files)} MODEL(S)")
    print(f"{'='*70}")
    print(f"Settings files: {', '.join(settings_files)}")
    
    # Load default settings to get default values for prices, episodes, split
    settings_module.load_settings(settings_files[0])
    from mkrl.settings import default_prices_file, training_episodes, train_split_ratio
    
    prices_file = args.prices or default_prices_file
    episodes = args.episodes or training_episodes
    split = args.split if args.split is not None else train_split_ratio
    
    # Train models for each settings file
    overall_start_time = time.time()
    results = []
    for i, settings_file in enumerate(settings_files, 1):
        print(f"\n{'#'*70}")
        print(f"# Training model {i}/{len(settings_files)}")
        print(f"{'#'*70}")
        try:
            config_name, model_path, training_time = train_model_for_config(
                settings_file=settings_file,
                prices_file=prices_file,
                split=split,
                episodes=episodes
            )
            results.append((config_name, model_path, training_time))
        except Exception as e:
            print(f"\n✗ ERROR training model with {settings_file}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
    
    # Overall summary (only report once after all models)
    overall_time = time.time() - overall_start_time
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE - SUMMARY")
    print(f"{'='*70}")
    print(f"Total models trained: {len(results)}")
    print(f"Overall time: {format_time(overall_time)}")
    print(f"\nModels:")
    for config_name, model_path, training_time in results:
        print(f"  • {config_name}: {model_path.name} ({format_time(training_time)})")
    print(f"{'='*70}\n")
    
    # Create and open single completion page in browser (only if models were trained)
    if results:
        # Use the first model's details for the HTML page (or create a summary page)
        first_config_name, first_model_path, _ = results[0]
        html_path = create_training_complete_html(
            training_time=overall_time,
            model_path=first_model_path,
            train_prices_count=0,  # Not applicable for multi-model summary
            episodes=episodes,
            training_timesteps=0  # Not applicable for multi-model summary
        )
        print(f"Opening completion page in browser...")
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
