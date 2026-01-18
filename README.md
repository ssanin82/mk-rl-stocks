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

You can modify the following parameters in `main.py`:

```python
initial_capital = 1000  # Starting capital
total_timesteps = 10000  # Training timesteps
```

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
