"""data/fetch_historical.py
Member 1 script — Pull 3 months of BTC/USDT 1-minute kline data from Binance.

No API key required. Public endpoint.

Output: data/btcusdt_1m_raw.csv

Usage:
    python data/fetch_historical.py
"""

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
INTERVAL = "1m"
LIMIT = 1000          # max rows per request Binance allows
MONTHS_BACK = 3
BASE_URL = "https://api.binance.com/api/v3/klines"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _months_ago_ms(months: int) -> int:
    """Approximate: 30 days per month."""
    return _now_ms() - months * 30 * 24 * 60 * 60 * 1000


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Fetch all 1-minute klines between start_ms and end_ms, paginated."""
    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        all_rows.extend(rows)
        # Last row's close time + 1ms = next start
        current_start = rows[-1][6] + 1

        print(
            f"  Fetched {len(all_rows):,} candles so far "
            f"| last: {datetime.fromtimestamp(rows[-1][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        )
        time.sleep(0.2)  # be polite to Binance

    return all_rows


def save_csv(rows: list[list], path: Path) -> None:
    """Save raw kline rows to CSV with readable column headers."""
    headers = [
        "open_time_ms",
        "open", "high", "low", "close",
        "volume",
        "close_time_ms",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nSaved {len(rows):,} rows → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOLUSDT", help="Symbol to fetch")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    output_file = Path(__file__).parent / f"{symbol.lower()}_1m_raw.csv"

    start_ms = _months_ago_ms(MONTHS_BACK)
    end_ms = _now_ms()

    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"Pulling {symbol} {INTERVAL} klines from {start_dt} to {end_dt} ...")

    rows = fetch_klines(symbol, start_ms, end_ms)
    save_csv(rows, output_file)
    print(f"Done. Hand off {output_file.name} to Member 2.")
