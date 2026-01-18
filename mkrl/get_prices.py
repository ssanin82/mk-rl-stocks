import requests
import pandas as pd
import time

BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000
TOTAL_KLINES = 20_000
OUTPUT_FILE = "btc_usdt_1m_prices.txt"


def fetch_klines(symbol, interval, limit, start_time=None):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = start_time

    response = requests.get(f"{BASE_URL}/fapi/v1/klines", params=params)
    response.raise_for_status()
    return response.json()


all_klines = []
start_time = None

while len(all_klines) < TOTAL_KLINES:
    print(f"Fetching {len(all_klines)}/{TOTAL_KLINES} klines")
    klines = fetch_klines(SYMBOL, INTERVAL, LIMIT, start_time)

    if not klines:
        print("No klines found")
        break

    all_klines.extend(klines)

    # next batch starts after last candle
    start_time = klines[-1][0] + 1

    time.sleep(0.05)  # be nice to Binance API

# trim to exactly 20,000
all_klines = all_klines[:TOTAL_KLINES]

# convert to DataFrame
df = pd.DataFrame(
    all_klines,
    columns=[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_base_vol",
        "taker_quote_vol",
        "ignore",
    ],
)

# convert types
df["close"] = df["close"].astype(float)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

# final price series
prices = df["close"].values

print(f"Fetched {len(prices)} prices")

with open(OUTPUT_FILE, "w") as f:
    for price in prices:
        f.write(f"{price}\n")