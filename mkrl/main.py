import sys
from pathlib import Path

# Add parent directory to Python path if running as a script (not as a module)
if __name__ == "__main__" and __file__:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
import webbrowser
import time

from stable_baselines3 import PPO
from mkrl.web import create_static_html


class TradingEnv(gym.Env):
    def __init__(self, prices, initial_capital=1000):
        super().__init__()
        self.prices = prices
        self.initial_capital = initial_capital
        self.current_step = 0
        self.cash = initial_capital
        self.holdings = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.holdings = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        price = self.prices[self.current_step]
        return np.array([price, self.cash, self.holdings], dtype=np.float32)
    
    def step(self, action):
        price = self.prices[self.current_step]
        
        if action == 1 and self.cash > 0:
            self.holdings += self.cash / price
            self.cash = 0
        elif action == 2 and self.holdings > 0:
            self.cash += self.holdings * price
            self.holdings = 0
            
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        terminated = done
        truncated = False
        
        portfolio_value = self.cash + self.holdings * self.prices[self.current_step] if not done else self.cash + self.holdings * price
        reward = portfolio_value - self.initial_capital
        
        return self._get_obs(), reward, terminated, truncated, {}


def run_strategy(env, model):
    obs, info = env.reset()
    actions = []
    portfolio_values = []
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(env.cash + env.holdings * env.prices[env.current_step])
    
    return actions, portfolio_values


def calculate_metrics(portfolio_values, initial_capital):
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



def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')


def realistic_price_feed(
    S0=100,
    mu=0.0001,
    sigma0=0.01,
    alpha=0.05,     # reaction to shocks
    beta=0.9,       # volatility persistence
    jump_prob=0.01,
    jump_scale=0.05,
    n=10000,
    seed=42
):
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
    """Main entry point for the trading simulator."""
    # Generate synthetic price data
    prices = realistic_price_feed()

    # Train RL agent
    print("Training RL agent...")
    ts = time.time()
    initial_capital = 1000
    env = TradingEnv(prices, initial_capital=initial_capital)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    print(f"Training complete! Took {round(time.time() - ts, 3)} seconds")
    
    # Run strategy
    print("Running strategy...")
    ts = time.time()
    env_test = TradingEnv(prices, initial_capital=initial_capital)
    actions, portfolio_values = run_strategy(env_test, model)
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_values, initial_capital)
    
    print(f"Execution complete! Took {round(time.time() - ts, 3)} seconds")
    print(f"\nResults:")
    print(f"Initial Capital: ${metrics['initial_capital']:.2f}")
    print(f"Final Capital: ${metrics['final_capital']:.2f}")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    
    # Create static HTML file
    print("\nGenerating static HTML report...")
    html_file = create_static_html(prices, actions, portfolio_values, metrics)
    html_path = Path(html_file).resolve()
    
    print(f"HTML report saved to: {html_path}")
    print("Opening in browser...")
    
    # Open in external browser
    webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
    