# RL Trading Strategy Simulator

A reinforcement learning-based stock trading simulator that uses PPO (Proximal Policy Optimization) to train an agent to make buy/sell decisions. Features an interactive dark-themed dashboard with performance metrics and visualizations.

## Features

- 📈 **RL-powered trading** using Stable-Baselines3 PPO algorithm
- 📊 **Interactive visualizations** with Plotly and Dash
- 🎯 **Buy/sell signals** displayed on price charts
- 💰 **Performance metrics** including P&L, returns, drawdown, and volatility
- 🌙 **Dark theme** dashboard
- 🚀 **Live web server** for real-time monitoring

## Screenshots

The dashboard displays:
- Stock price chart with buy (▲) and sell (▼) signals
- Portfolio value over time
- Key performance metrics (Initial/Final Capital, P&L, Returns, Max Drawdown, Volatility)

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd mk-rl-stocks
```

2. Create a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulator:
```bash
python main.py
```

The script will:
1. Generate synthetic stock price data
2. Train a PPO agent (10,000 timesteps)
3. Run the trading strategy
4. Launch a Dash web server at `http://0.0.0.0:8050`

### Accessing the Dashboard

**Local machine:**
- Open `http://localhost:8050` in your browser

**Remote server:**
- Use SSH port forwarding: `ssh -L 8050:localhost:8050 user@server`
- Or access directly at `http://YOUR_SERVER_IP:8050` (ensure port 8050 is open)

**VS Code Remote:**
- VS Code should auto-detect and forward the port
- Check the "Ports" tab in the bottom panel

Press `Ctrl+C` to stop the server.

## Configuration

All configuration parameters are defined in `settings.json`. The file is organized into two sections:

### Common Settings

These settings apply to all trading strategies:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_price_points` | integer | 20000 | Total number of price data points to generate/use for training and testing |
| `default_prices_file` | string | `"btc_usdt_1m_prices.txt"` | Filename containing price data (one price per line, 6 decimal places) |
| `training_episodes` | integer | 20 | Number of training episodes for the RL model |
| `trade_execution_reward` | float | 0.01 | Small positive reward given when a trade is executed (encourages trading activity) |
| `momentum_reward_scale` | float | 0.05 | Scale factor for momentum-based rewards (rewards trading with price momentum, penalizes against it) |
| `train_split_ratio` | float | 0.9 | Ratio of data used for training (e.g., 0.9 = 90% training, 10% testing) |
| `default_model_file` | string | `"trading_model.zip"` | Filename where the trained model is saved/loaded |

### BTC-Specific Settings

These settings are specific to BTC/USDT trading:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 10000 | Starting capital in USD for the trading strategy |
| `min_notional` | float | 100.0 | Minimum trade notional value in USD (price × quantity must be ≥ this value) |
| `min_size` | float | 0.001 | Minimum trade size in shares (minimum quantity that can be traded) |
| `trading_fee_rate` | float | 0.0001 | Trading fee rate per trade (0.0001 = 0.01% = 1 basis point) |
| `profit_threshold` | float | 0.002 | Percentage profit threshold for automatic profit-taking (0.002 = 0.2%) |
| `partial_sell_ratio` | float | 0.002 | Fraction of position to sell when profit threshold is reached (0.002 = 0.2% of holdings) |
| `dca_threshold` | float | 0.002 | Percentage decline threshold for automatic dollar-cost averaging (0.002 = 0.2% below entry) |
| `dca_ratio` | float | 0.02 | Fraction of position value to add when DCA threshold is reached (0.02 = 2% of current position value) |
| `lot_size` | float | 0.001 | Minimum increment for trade quantities (all trades must be multiples of this value) |

### Example Configuration

```json
{
  "common": {
    "n_price_points": 20000,
    "default_prices_file": "btc_usdt_1m_prices.txt",
    "training_episodes": 20,
    "trade_execution_reward": 0.01,
    "momentum_reward_scale": 0.05,
    "train_split_ratio": 0.9,
    "default_model_file": "trading_model.zip"
  },
  "btc": {
    "initial_capital": 10000,
    "min_notional": 100.0,
    "min_size": 0.001,
    "trading_fee_rate": 0.0001,
    "profit_threshold": 0.002,
    "partial_sell_ratio": 0.002,
    "dca_threshold": 0.002,
    "dca_ratio": 0.02,
    "lot_size": 0.001
  }
}
```

### Position Management Rules

The strategy includes automatic position management rules that execute independently of model actions:

1. **Profit-Taking (Auto-Sell)**: When the price grows by `profit_threshold`% from the average entry price, automatically sells `partial_sell_ratio`% of the current holdings and adjusts the average entry price.

2. **Dollar-Cost Averaging (Auto-Buy)**: When the price declines by `dca_threshold`% from the average entry price, automatically buys shares worth `dca_ratio`% of the current position value to lower the average entry price.

All trades are rounded to the nearest `lot_size` increment to comply with exchange requirements.

## Environment

The `TradingEnv` is a custom Gymnasium environment with:
- **Action space**: Discrete(3) - Hold(0), Buy(1), Sell(2)
- **Observation space**: [price, cash, holdings]
- **Reward**: Portfolio value change from initial capital

## Performance Metrics

- **Initial/Final Capital**: Starting and ending portfolio value
- **Total P&L**: Profit and Loss in absolute terms
- **Total Return**: Percentage gain/loss
- **Max Drawdown**: Maximum peak-to-trough decline
- **Volatility**: Standard deviation of returns

## Project Structure

```
.
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Technologies Used

- **Gymnasium**: Environment framework
- **Stable-Baselines3**: RL algorithms (PPO)
- **PyTorch**: Neural network backend
- **Plotly**: Interactive charts
- **Dash**: Web dashboard framework
- **NumPy**: Numerical computations

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This is a educational project for demonstrating reinforcement learning in trading. **DO NOT use this for actual trading decisions.** Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.
