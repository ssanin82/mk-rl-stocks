"""
Script to fetch BTCUSDT price data from Binance perpetuals futures
and save to btc_usdt_1m_prices.txt (one price per line, same format as 1_generate_prices.py)
"""

import sys
from pathlib import Path
import requests
import time

# Add parent directory to Python path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from mkrl.settings import n_price_points

BASE_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000  # Binance API limit per request
TOTAL_KLINES = n_price_points
OUTPUT_FILE = "btc_usdt_1m_prices.txt"


def fetch_klines(symbol, interval, limit, start_time=None):
    """Fetch klines from Binance API."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = start_time

    response = requests.get(f"{BASE_URL}/fapi/v1/klines", params=params)
    response.raise_for_status()
    # pprint.pprint(response.json())
    return response.json()


def main():
    """Fetch prices from Binance and save to file."""
    import datetime
    
    all_klines = []
    
    # Start from 20000 minutes (TOTAL_KLINES minutes) ago and move forward
    # 1 minute = 60 seconds = 60,000 milliseconds
    minutes_in_past = TOTAL_KLINES
    milliseconds_per_minute = 60 * 1000
    start_time = int((datetime.datetime.now() - datetime.timedelta(minutes=minutes_in_past)).timestamp() * 1000)
    
    # Round down to the nearest minute boundary
    # Binance klines start at minute boundaries (e.g., 00:00:00, 00:01:00, etc.)
    start_time = (start_time // milliseconds_per_minute) * milliseconds_per_minute

    print(f"Fetching {TOTAL_KLINES} {SYMBOL} {INTERVAL} klines from Binance...")
    print(f"  Starting from {datetime.datetime.fromtimestamp(start_time/1000)} ({minutes_in_past} minutes ago)")
    print(f"  Moving forward in time...")
    
    while len(all_klines) < TOTAL_KLINES:
        try:
            klines = fetch_klines(SYMBOL, INTERVAL, LIMIT, start_time)
        except Exception as e:
            print(f"  Error fetching klines: {e}")
            break

        if not klines:
            print("  No more klines available")
            break

        # For the first batch, use all klines starting from start_time
        # For subsequent batches, filter to avoid duplicates
        if len(all_klines) == 0:
            # First batch: use klines starting from our start_time or later
            filtered = [k for k in klines if k[0] >= start_time]
            if filtered:
                all_klines.extend(filtered)
            else:
                print("  No klines found at start time")
                break
        else:
            # Subsequent batches: skip candles that overlap with previous batch
            # Start from after the last candle's close_time
            last_close = all_klines[-1][6]
            filtered = [k for k in klines if k[0] > last_close]
            if filtered:
                all_klines.extend(filtered)
            else:
                # If no new candles, we might have reached current time
                print("  Reached current time, no more historical data")
                break

        # Next batch starts after the close_time of the last candle + 1ms
        last_close_time = klines[-1][6]
        start_time = last_close_time + 1

        print(f"  Progress: {len(all_klines)}/{TOTAL_KLINES} klines")

        time.sleep(0.05)  # Be nice to Binance API
        
        # If we got fewer than LIMIT candles, we might be near current time
        if len(klines) < LIMIT:
            # Check if we have enough data
            if len(all_klines) >= TOTAL_KLINES:
                break
            print(f"  Warning: Got only {len(klines)} candles (less than {LIMIT}), may be near current time")

    # Trim to exactly the requested number
    all_klines = all_klines[:TOTAL_KLINES]
    
    # Extract close prices (index 4 in kline array)
    prices = [float(kline[4]) for kline in all_klines]

    print(f"\n✓ Fetched {len(prices)} prices")
    if prices:
        print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")

    # Save to file in same format as 1_generate_prices.py (one price per line, 6 decimals)
    output_path = Path(OUTPUT_FILE)
    with open(output_path, 'w') as f:
        for price in prices:
            f.write(f"{price:.6f}\n")
    
    print(f"✓ Saved to {output_path.absolute()}")


if __name__ == "__main__":
    main()