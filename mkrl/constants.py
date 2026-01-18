"""
Shared constants for the RL trading simulator.
"""

# Data configuration
N_PRICE_POINTS = 20000  # Total number of price points to generate
# DEFAULT_PRICES_FILE = "prices.txt"
DEFAULT_PRICES_FILE = "btc_usdt_1m_prices.txt"

# Trading environment configuration
INITIAL_CAPITAL = 5000  # Strategy's starting capital in dollars
MIN_NOTIONAL = 5.0  # Minimum dollar amount per trade (price * shares)
MIN_SIZE = 0.1  # Minimum number of shares per trade
TRADING_FEE_RATE = 0.0001  # Trading fee as fraction (0.01% = 0.0001)

# Training configuration
# TRAINING_EPISODES = 50  # Number of training episodes (one episode = one pass through training data)
TRAINING_EPISODES = 10
# Note: TRAINING_TIMESTEPS will be calculated as (number of training prices * TRAINING_EPISODES)

# Reward shaping parameters
TRADE_EXECUTION_REWARD = 0.01  # Small positive reward for executing a valid trade (encourages exploration)
MOMENTUM_REWARD_SCALE = 0.05  # Bonus for trading in the direction of price momentum

# Position management thresholds (configurable)
PROFIT_THRESHOLD = 0.10  # 10% profit threshold for partial sell
PARTIAL_SELL_RATIO = 0.20  # Sell 20% of position when profit threshold reached
DCA_THRESHOLD = 0.10  # 10% down from average entry for adding to position
DCA_RATIO = 0.10  # Add 10% to position when down threshold reached

# Data split configuration
TRAIN_SPLIT_RATIO = 0.9  # Use first 90% for training, last 10% for testing

# Model files
DEFAULT_MODEL_FILE = "trading_model.zip"
