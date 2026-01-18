import gymnasium as gym
import numpy as np
import plotly.graph_objects as go
import webbrowser
import time

from stable_baselines3 import PPO
from plotly.subplots import make_subplots
from dash import Dash, html, dcc


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

def create_figure(prices, actions, portfolio_values, metrics):
    # Find buy and sell points
    buy_indices = [i for i, a in enumerate(actions) if a == 1]
    sell_indices = [i for i, a in enumerate(actions) if a == 2]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "indicator"}, {"type": "indicator"}]
        ],
        subplot_titles=("Stock Price with Buy/Sell Signals", "Portfolio Value Over Time"),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.4, 0.2]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=list(range(len(prices))), y=prices, name="Stock Price",
                  line=dict(color='#2196F3', width=2), fill='tozeroy',
                  fillcolor='rgba(33, 150, 243, 0.1)'),
        row=1, col=1
    )
    
    # Buy signals
    fig.add_trace(
        go.Scatter(x=buy_indices, y=[prices[i] for i in buy_indices],
                  mode='markers', name='Buy Signal',
                  marker=dict(color='#4CAF50', size=12, symbol='triangle-up')),
        row=1, col=1
    )
    
    # Sell signals
    fig.add_trace(
        go.Scatter(x=sell_indices, y=[prices[i] for i in sell_indices],
                  mode='markers', name='Sell Signal',
                  marker=dict(color='#f44336', size=12, symbol='triangle-down')),
        row=1, col=1
    )
    
    # Portfolio value chart
    fig.add_trace(
        go.Scatter(x=list(range(len(portfolio_values))), y=portfolio_values,
                  name="Portfolio Value", line=dict(color='#4CAF50', width=2),
                  fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.1)'),
        row=2, col=1
    )
    
    # Initial capital line
    fig.add_trace(
        go.Scatter(x=list(range(len(portfolio_values))),
                  y=[metrics['initial_capital']] * len(portfolio_values),
                  name="Initial Capital", line=dict(color='#FF9800', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Metrics indicators
    pnl_color = '#4CAF50' if metrics['total_pnl'] >= 0 else '#f44336'
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=metrics['final_capital'],
            title={"text": "Final Capital"},
            delta={'reference': metrics['initial_capital'], 'relative': False},
            number={'prefix': "$", 'valueformat': ".2f"},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=metrics['total_return'],
            title={"text": "Total Return"},
            number={'suffix': "%", 'valueformat': ".2f", 'font': {'color': pnl_color}},
            domain={'x': [0, 1], 'y': [0, 1]}
        ),
        row=3, col=2
    )
    
    # Update layout with dark theme
    fig.update_layout(
        template='plotly_dark',
        height=1000,
        showlegend=True,
        title={
            'text': '📈 Reinforcement Learning Trading Strategy Results',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#4CAF50'}
        },
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#2a2a2a',
        font={'color': '#e0e0e0'}
    )
    
    return fig


def create_dash_app(prices, actions, portfolio_values, metrics):
    app = Dash(__name__)
    pnl_color = '#4CAF50' if metrics['total_pnl'] >= 0 else '#f44336'
    app.layout = html.Div(
        style={'backgroundColor': '#1a1a1a', 'padding': '20px', 'fontFamily': 'Segoe UI, sans-serif'},
        children=[
            html.Div(
                style={'maxWidth': '1400px', 'margin': '0 auto'},
                children=[
                    # Metrics grid
                    html.Div(
                        style={
                            'background': '#2a2a2a',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'margin': '20px 0'
                        },
                        children=[
                            html.H2(
                                'Performance Metrics',
                                style={'color': '#4CAF50', 'textAlign': 'center'}
                            ),
                            html.Div(
                                style={
                                    'display': 'grid',
                                    'gridTemplateColumns': 'repeat(3, 1fr)',
                                    'gap': '20px',
                                    'marginTop': '20px'
                                },
                                children=[
                                    html.Div([
                                        html.Div('Initial Capital', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"${metrics['initial_capital']:.2f}", style={'color': '#e0e0e0', 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Final Capital', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"${metrics['final_capital']:.2f}", style={'color': pnl_color, 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Total P&L', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"${metrics['total_pnl']:.2f}", style={'color': pnl_color, 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Max Drawdown', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"{metrics['max_drawdown']:.2f}%", style={'color': '#f44336', 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Volatility', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"{metrics['volatility']:.2f}%", style={'color': '#e0e0e0', 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ]),
                                    html.Div([
                                        html.Div('Total Return', style={'color': '#888', 'fontSize': '14px', 'textAlign': 'center'}),
                                        html.Div(f"{metrics['total_return']:.2f}%", style={'color': pnl_color, 'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
                                    ])
                                ]
                            )
                        ]
                    ),
                    # Chart
                    dcc.Graph(
                        figure=create_figure(prices, actions, portfolio_values, metrics),
                        config={'displayModeBar': True}
                    )
                ]
            )
        ]
    )
    
    return app

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


if __name__ == "__main__":
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
    
    # Create and run Dash app
    app = create_dash_app(prices, actions, portfolio_values, metrics)
    
    print("\nStarting dashboard server...")
    print("Dashboard running at http://0.0.0.0:8050/")
    print("Access it at http://YOUR_SERVER_IP:8050/ from your browser")
    print("Press Ctrl+C to stop the server")
    
    # Run server on all interfaces
    app.run(debug=False, host='0.0.0.0', port=8050)
    