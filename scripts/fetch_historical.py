"""scripts/fetch_historical.py
Fetches 3 months of SOLUSDT 1-minute OHLCV bars from Binance public REST API,
resamples to 10-second equivalent, and saves as data/solusdt_historical.npz.

Feature columns match VARIABLE_NAMES from core/feature_bar.py exactly (19 vars).
Columns that cannot be computed from 1m OHLCV are approximated or zeroed.
This is logged explicitly so Phase A training has clear provenance.

Approximations applied (logged at INFO level):
  - trade_intensity:         bar volume / rolling mean volume (proxy for intensity)
  - order_book_imbalance:    0.0 (L2 not available in historical data)
  - algo_health_p_normal:    1.0 (cold-start default — no ML1 in historical)
  - algo_health_p_stressed:  0.0
  - algo_health_p_degraded:  0.0
  - session_sin/cos:         computed from bar open timestamp (real)
  - btc_return/eth_return:   fetched from Binance BTCUSDT/ETHUSDT 1m bars (real)
  - regime_volatile/trending: 0.0 (no regime classifier in historical mode)

Usage:
    python scripts/fetch_historical.py [--symbol SOLUSDT] [--days 90] [--out data/solusdt_historical.npz]
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BINANCE_KLINE_URL = "https://api.binance.com/api/v3/klines"
KLINE_INTERVAL = "1m"          # smallest publicly available interval
BAR_WIDTH_S = 10               # target bar width (for naming only — we use 1m bars directly)
MAX_BARS_PER_REQUEST = 1000    # Binance limit per request
REQUEST_DELAY_S = 0.25         # stay well within rate limits

# Must match core/feature_bar.py VARIABLE_NAMES exactly
VARIABLE_NAMES = [
    "price_return",          # 0
    "volume",                # 1
    "spread",                # 2  — approximated as 0 (no bid/ask in 1m OHLCV)
    "volatility",            # 3  — (high - low) / open as proxy
    "rsi",                   # 4  — Wilder RSI-14 computed from close prices
    "trade_intensity",       # 5  — taker_buy_volume / volume (aggression ratio)
    "order_book_imbalance",  # 6  — 0.0 (L2 not available)
    "ema_spread",            # 7  — fast EMA - slow EMA
    "bollinger_width",       # 8  — upper - lower band
    "price_in_band",         # 9  — (close - lower) / (upper - lower)
    "regime_volatile",       # 10 — 0.0 (no classifier)
    "regime_trending",       # 11 — 0.0 (no classifier)
    "btc_return",            # 12 — fetched from BTCUSDT
    "eth_return",            # 13 — fetched from ETHUSDT
    "algo_health_p_normal",  # 14 — 1.0 (cold-start)
    "algo_health_p_stressed",# 15 — 0.0
    "algo_health_p_degraded",# 16 — 0.0
    "session_sin",           # 17 — computed from timestamp
    "session_cos",           # 18 — computed from timestamp
]
N_VARS = len(VARIABLE_NAMES)

# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers (same implementations as core/feature_bar.py)
# ─────────────────────────────────────────────────────────────────────────────

def _wilder_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(0.0,  c) for c in changes]
    losses = [max(0.0, -c) for c in changes]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0.0:
        return 100.0
    return round(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)), 2)


def _ema(prices: list[float], period: int) -> float:
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema


# ─────────────────────────────────────────────────────────────────────────────
# Binance fetch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Fetch all 1m klines for a symbol between start_ms and end_ms (epoch ms)."""
    all_klines: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": KLINE_INTERVAL,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_BARS_PER_REQUEST,
        }
        try:
            resp = requests.get(BINANCE_KLINE_URL, params=params, timeout=15)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            logger.error("Binance fetch error (symbol=%s): %s — retrying in 5s", symbol, exc)
            time.sleep(5)
            continue

        if not batch:
            break

        all_klines.extend(batch)
        cursor = int(batch[-1][6]) + 1  # close_time of last bar + 1ms
        time.sleep(REQUEST_DELAY_S)

        pct = min(100, (cursor - start_ms) / (end_ms - start_ms) * 100)
        logger.info("  Fetched %d bars | %.1f%% complete", len(all_klines), pct)

    return all_klines


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_matrix(
    sol_klines: list[list],
    btc_klines: list[list],
    eth_klines: list[list],
) -> np.ndarray:
    """Build (N, 19) feature matrix from raw Binance klines.

    Approximations applied (explicitly logged):
      - spread: 0.0 (bid/ask not in 1m OHLCV)
      - order_book_imbalance: 0.0 (L2 not available)
      - algo_health: [1.0, 0.0, 0.0] cold-start defaults
      - regime_volatile/trending: 0.0 (no classifier)
    """
    logger.info("=== Approximation provenance for Phase A training ===")
    logger.info("  spread:                0.0  (bid/ask not in 1m OHLCV)")
    logger.info("  order_book_imbalance:  0.0  (L2 not available in historical data)")
    logger.info("  algo_health:           [1.0, 0.0, 0.0]  (cold-start defaults — no ML1)")
    logger.info("  regime_volatile/trending: 0.0  (no classifier run on historical)")
    logger.info("  trade_intensity:       taker_buy_volume / total_volume  (real, from Binance)")
    logger.info("  rsi / ema / bollinger: computed from close prices  (real)")
    logger.info("  btc_return / eth_return: from BTCUSDT/ETHUSDT 1m bars  (real)")
    logger.info("=====================================================")

    # Index BTC/ETH by open timestamp for fast lookup
    btc_by_ts: dict[int, list] = {int(k[0]): k for k in btc_klines}
    eth_by_ts: dict[int, list] = {int(k[0]): k for k in eth_klines}

    rows: list[np.ndarray] = []
    close_history: list[float] = []
    prev_close: float | None = None
    prev_btc: float | None = None
    prev_eth: float | None = None
    vol_history: list[float] = []

    for kline in sol_klines:
        open_ms   = int(kline[0])
        open_p    = float(kline[1])
        high_p    = float(kline[2])
        low_p     = float(kline[3])
        close_p   = float(kline[4])
        volume    = float(kline[5])
        taker_buy = float(kline[9])   # taker buy base asset volume

        close_history.append(close_p)
        vol_history.append(volume)
        rolling_closes = close_history[-200:]
        rolling_vols   = vol_history[-200:]

        # price_return
        price_return = 0.0
        if prev_close is not None and prev_close > 0:
            price_return = (close_p - prev_close) / prev_close
        prev_close = close_p

        # volatility proxy: (high - low) / open
        volatility = (high_p - low_p) / open_p if open_p > 0 else 0.0

        # RSI
        rsi = _wilder_rsi(rolling_closes, 14)

        # trade_intensity: taker buy ratio (real aggression signal)
        trade_intensity = taker_buy / volume if volume > 0 else 0.5

        # EMA spread
        fast_ema = _ema(rolling_closes, 9)
        slow_ema = _ema(rolling_closes, 21)
        ema_spread_val = fast_ema - slow_ema

        # Bollinger bands
        bollinger_width = 0.0
        price_in_band = 0.5
        if len(rolling_closes) >= 20:
            window = rolling_closes[-20:]
            mid = float(np.mean(window))
            std = float(np.std(window))
            upper = mid + 2.0 * std
            lower = mid - 2.0 * std
            bollinger_width = upper - lower
            rng = upper - lower
            price_in_band = (close_p - lower) / rng if rng > 0 else 0.5

        # volume (normalised relative to rolling mean)
        mean_vol = float(np.mean(rolling_vols)) if rolling_vols else 1.0
        norm_volume = volume / mean_vol if mean_vol > 0 else 1.0

        # BTC / ETH returns
        btc_return = 0.0
        if open_ms in btc_by_ts:
            btc_close = float(btc_by_ts[open_ms][4])
            if prev_btc is not None and prev_btc > 0:
                btc_return = (btc_close - prev_btc) / prev_btc
            prev_btc = btc_close

        eth_return = 0.0
        if open_ms in eth_by_ts:
            eth_close = float(eth_by_ts[open_ms][4])
            if prev_eth is not None and prev_eth > 0:
                eth_return = (eth_close - prev_eth) / prev_eth
            prev_eth = eth_close

        # Session encoding from open timestamp
        ts_s = open_ms / 1000.0
        dt_bar = datetime.fromtimestamp(ts_s, tz=timezone.utc)
        hour_frac = dt_bar.hour + dt_bar.minute / 60.0
        session_sin = math.sin(2 * math.pi * hour_frac / 24.0)
        session_cos = math.cos(2 * math.pi * hour_frac / 24.0)

        row = np.array([
            price_return,
            norm_volume,
            0.0,              # spread — approximated
            volatility,
            rsi,
            trade_intensity,
            0.0,              # order_book_imbalance — approximated
            ema_spread_val,
            bollinger_width,
            price_in_band,
            0.0,              # regime_volatile — approximated
            0.0,              # regime_trending — approximated
            btc_return,
            eth_return,
            1.0,              # algo_health_p_normal — cold-start
            0.0,              # algo_health_p_stressed — cold-start
            0.0,              # algo_health_p_degraded — cold-start
            session_sin,
            session_cos,
        ], dtype=np.float64)

        rows.append(row)

    matrix = np.stack(rows, axis=0)
    logger.info("Built feature matrix | shape=%s | vars=%d", matrix.shape, N_VARS)
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SOLUSDT historical bars from Binance")
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--days", type=int, default=90, help="Number of days to fetch (default 90)")
    parser.add_argument("--out", default="data/solusdt_historical.npz")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(now.timestamp() * 1000)

    logger.info(
        "Fetching %d days of %s 1m bars (%s → %s)",
        args.days, args.symbol,
        start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"),
    )

    logger.info("Fetching SOL klines...")
    sol_klines = _fetch_klines(args.symbol, start_ms, end_ms)
    logger.info("Fetched %d SOL bars", len(sol_klines))

    logger.info("Fetching BTC klines (for cross-returns)...")
    btc_klines = _fetch_klines("BTCUSDT", start_ms, end_ms)
    logger.info("Fetched %d BTC bars", len(btc_klines))

    logger.info("Fetching ETH klines (for cross-returns)...")
    eth_klines = _fetch_klines("ETHUSDT", start_ms, end_ms)
    logger.info("Fetched %d ETH bars", len(eth_klines))

    matrix = _build_feature_matrix(sol_klines, btc_klines, eth_klines)

    # Validation split: last 14 days = validation, rest = training
    bars_per_day = 1440  # 1440 1m bars per day
    val_bars = bars_per_day * 14
    train_bars = len(matrix) - val_bars

    train_matrix = matrix[:train_bars]
    val_matrix   = matrix[train_bars:]

    import os
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        args.out,
        train=train_matrix,
        val=val_matrix,
        variable_names=np.array(VARIABLE_NAMES),
    )
    logger.info(
        "Saved to %s | train=%d bars | val=%d bars",
        args.out, len(train_matrix), len(val_matrix),
    )

    # Sanity checks
    nan_count = np.isnan(matrix).sum()
    inf_count = np.isinf(matrix).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning(
            "Data quality: %d NaN values and %d Inf values found — "
            "they will be replaced with 0.0 during training",
            nan_count, inf_count,
        )
    else:
        logger.info("Data quality: no NaN or Inf values found")


if __name__ == "__main__":
    main()
