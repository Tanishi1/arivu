"""core/feature_bar.py  — Task 2

Buckets the live tick stream into fixed-width 10-second bars and computes
all 19 causal discovery input variables per bar.

EXCLUDED_FROM_GRAPH (hard constraint — self-referential leakage prevention):
    position_size, capital_deployed, active_strategy, last_tick_timestamp, price (raw)

The feature matrix is consumed by ml/causal_discovery.py on every PCMCI run.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.constants import THRESHOLDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BAR_WIDTH_S: int = 10          # seconds per bar
WINDOW_BARS: int = 200         # rolling history for PCMCI (200 × 10s = ~33 min)
EMA_FAST_PERIOD: int = 9
EMA_SLOW_PERIOD: int = 21
BOLLINGER_PERIOD: int = 20
BOLLINGER_STD: float = 2.0
RSI_PERIOD: int = 14
MIN_BARS_FOR_DISCOVERY: int = 60   # tau_max * 2 + 20 safety margin

# Variables that are NEVER included in the causal graph input.
# See implementation_plan.md §Self-Referential Leakage for full rationale.
EXCLUDED_FROM_GRAPH: frozenset[str] = frozenset({
    "position_size",
    "capital_deployed",
    "active_strategy",
    "last_tick_timestamp",
    "price",          # non-stationary — use price_return instead
})

# Variable name index (column order in feature matrix) — must stay stable
VARIABLE_NAMES: list[str] = [
    "price_return",          # 0
    "volume",                # 1
    "spread",                # 2
    "volatility",            # 3
    "rsi",                   # 4
    "trade_intensity",       # 5
    "order_book_imbalance",  # 6
    "ema_spread",            # 7
    "bollinger_width",       # 8
    "price_in_band",         # 9
    "regime_volatile",       # 10
    "regime_trending",       # 11
    "btc_return",            # 12
    "eth_return",            # 13
    "algo_health_p_normal",  # 14
    "algo_health_p_stressed",# 15
    "algo_health_p_degraded",# 16
    "session_sin",           # 17
    "session_cos",           # 18
]
N_VARS = len(VARIABLE_NAMES)

_COLD_START_ALGO_HEALTH = [1.0, 0.0, 0.0]  # default before ML1 warms up


@dataclass
class _BarAccumulator:
    """Accumulates ticks within a single 10-second bar."""
    open_time: datetime
    prices: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    spreads: list[float] = field(default_factory=list)
    trade_count: int = 0
    buyer_trades: int = 0
    total_trades: int = 0
    btc_prices: list[float] = field(default_factory=list)
    eth_prices: list[float] = field(default_factory=list)
    # L2 imbalance — filled if L2 feed is available, else derived from aggression
    imbalance_values: list[float] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.prices) == 0


class FeatureBarBuilder:
    """Converts live ticks into fixed-width 10-second bars of 19 variables.

    Thread-safe: tick ingestion (update_*) and snapshot reads (get_feature_matrix)
    are protected by a threading.Lock so this can be called from asyncio threads.

    Usage:
        builder = FeatureBarBuilder(regime_classifier)
        # on every SOL tick:
        builder.update_sol(price, volume, spread)
        # on every BTC/ETH macro tick:
        builder.update_macro("BTC", price)
        # on every trade:
        builder.update_trade(is_buyer_maker)
        # on algo_health update:
        builder.update_algo_health([p_normal, p_stressed, p_degraded])
        # on L2 depth update:
        builder.update_imbalance(value)
        # when causal discovery runs:
        matrix, names = builder.get_feature_matrix()
    """

    def __init__(self, regime_classifier=None, on_bar_closed=None) -> None:
        self._lock = threading.Lock()
        self._regime_classifier = regime_classifier
        # Optional async callback: on_bar_closed(features: dict[str, float]) -> None
        # Fired on the calling asyncio loop via asyncio.ensure_future when a bar closes.
        self._on_bar_closed = on_bar_closed

        # Current bar accumulator
        self._current_bar: _BarAccumulator = _BarAccumulator(
            open_time=datetime.now(timezone.utc)
        )
        self._bar_start_ts: float = _monotonic_bar_start()

        # Rolling bar history — completed bars only
        self._bars: deque[np.ndarray] = deque(maxlen=WINDOW_BARS)

        # Rolling price history for indicator computation (across bars)
        self._close_prices: deque[float] = deque(maxlen=WINDOW_BARS)
        self._prev_close: Optional[float] = None

        # Macro prices
        self._prev_btc: Optional[float] = None
        self._prev_eth: Optional[float] = None

        # Latest algo health from ML1
        self._algo_health: list[float] = list(_COLD_START_ALGO_HEALTH)

        # Track how many bars have cold-start algo health
        self._cold_start_bar_count: int = 0

        # Latest closed bar as a named dict — read by get_latest_features()
        self._last_bar_features: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Tick ingestion — called per WebSocket message
    # ------------------------------------------------------------------

    def update_sol(self, price: float, volume: float, spread: float) -> None:
        with self._lock:
            self._maybe_close_bar()
            self._current_bar.prices.append(price)
            self._current_bar.volumes.append(volume)
            self._current_bar.spreads.append(spread)

    def update_macro(self, symbol: str, price: float) -> None:
        with self._lock:
            sym = symbol.upper()
            if sym in ("BTC", "BTCUSDT"):
                self._current_bar.btc_prices.append(price)
            elif sym in ("ETH", "ETHUSDT"):
                self._current_bar.eth_prices.append(price)

    def update_trade(self, is_buyer_maker: bool) -> None:
        """Record aggression side of a single trade (from aggTrade stream)."""
        with self._lock:
            self._current_bar.total_trades += 1
            self._current_bar.trade_count += 1
            if not is_buyer_maker:   # buyer is aggressor when m=False
                self._current_bar.buyer_trades += 1

    def update_imbalance(self, value: float) -> None:
        """Record one L2 order book imbalance reading (if L2 feed available)."""
        with self._lock:
            self._current_bar.imbalance_values.append(value)

    def update_algo_health(self, vector: list[float]) -> None:
        with self._lock:
            self._algo_health = list(vector)

    def update_regime(self, regime_classifier) -> None:
        with self._lock:
            self._regime_classifier = regime_classifier

    # ------------------------------------------------------------------
    # Feature matrix for PCMCI
    # ------------------------------------------------------------------

    def get_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return (matrix, variable_names) for PCMCI input.

        Returns:
            matrix: shape (n_bars, N_VARS), dtype float64.
            names:  VARIABLE_NAMES list — column labels.

        Raises:
            ValueError: if fewer than MIN_BARS_FOR_DISCOVERY bars are ready.
        """
        with self._lock:
            bars = list(self._bars)

        n = len(bars)
        if n < MIN_BARS_FOR_DISCOVERY:
            raise ValueError(
                f"FeatureBarBuilder: only {n}/{MIN_BARS_FOR_DISCOVERY} bars ready."
            )

        matrix = np.stack(bars, axis=0)   # shape (n, N_VARS)
        return matrix, list(VARIABLE_NAMES)

    def get_latest_features(self) -> dict[str, float]:
        """Return the latest closed bar as a {variable_name: value} dict.

        Returns an empty dict if no bar has closed yet.
        Useful for warm-starting CausalState on process restart.
        """
        with self._lock:
            return dict(self._last_bar_features)

    def n_bars_ready(self) -> int:
        with self._lock:
            return len(self._bars)

    def is_ready(self) -> bool:
        return self.n_bars_ready() >= MIN_BARS_FOR_DISCOVERY

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_close_bar(self) -> None:
        """Close the current bar if BAR_WIDTH_S seconds have elapsed."""
        import asyncio
        import time
        now = time.monotonic()
        elapsed = now - self._bar_start_ts
        if elapsed < BAR_WIDTH_S:
            return

        bar = self._current_bar
        if not bar.is_empty():
            feature_row = self._compute_features(bar)
            if feature_row is not None:
                self._bars.append(feature_row)
                # Build named dict for CausalState update
                self._last_bar_features = dict(zip(VARIABLE_NAMES, feature_row.tolist()))
                # Fire async callback if registered (runs on the event loop)
                if self._on_bar_closed is not None:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                self._on_bar_closed(self._last_bar_features)
                            )
                    except RuntimeError:
                        pass  # no event loop yet — skip, first bar before loop starts
                if self._cold_start_bar_count > 0 and self._algo_health != _COLD_START_ALGO_HEALTH:
                    logger.info(
                        "FeatureBarBuilder: ML1 now warm — algo_health cold-start bars excluded: %d",
                        self._cold_start_bar_count,
                    )
                    self._cold_start_bar_count = 0

        self._current_bar = _BarAccumulator(open_time=datetime.now(timezone.utc))

        # Snap to the next clean multiple of BAR_WIDTH_S, advancing until we catch up
        while self._bar_start_ts + BAR_WIDTH_S <= now:
            self._bar_start_ts += BAR_WIDTH_S

    def _compute_features(self, bar: _BarAccumulator) -> Optional[np.ndarray]:
        """Compute one row of 19 features from a closed bar."""
        if not bar.prices:
            return None

        close = bar.prices[-1]
        mean_vol = float(np.mean(bar.volumes)) if bar.volumes else 0.0
        mean_spread = float(np.mean(bar.spreads)) if bar.spreads else 0.0

        # --- Price return (stationary) ---
        price_return = 0.0
        if self._prev_close is not None and self._prev_close > 0:
            price_return = (close - self._prev_close) / self._prev_close
        self._prev_close = close
        self._close_prices.append(close)

        closes = list(self._close_prices)

        # --- Volatility (rolling std of returns) ---
        volatility = 0.0
        if len(closes) >= 5:
            rets = [
                (closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]
            volatility = float(np.std(rets)) if len(rets) > 1 else 0.0

        # --- RSI (Wilder's method) ---
        rsi = _compute_wilder_rsi(closes, RSI_PERIOD)

        # --- EMA fast/slow ---
        fast_ema = _compute_ema(closes, EMA_FAST_PERIOD)
        slow_ema = _compute_ema(closes, EMA_SLOW_PERIOD)
        ema_spread_val = fast_ema - slow_ema

        # --- Bollinger bands ---
        bollinger_width = 0.0
        price_in_band = 0.5
        if len(closes) >= BOLLINGER_PERIOD:
            window = closes[-BOLLINGER_PERIOD:]
            mid = float(np.mean(window))
            std = float(np.std(window))
            upper = mid + BOLLINGER_STD * std
            lower = mid - BOLLINGER_STD * std
            bollinger_width = upper - lower
            band_range = upper - lower
            price_in_band = (close - lower) / band_range if band_range > 0 else 0.5

        # --- Trade intensity ---
        trade_intensity = float(bar.trade_count)

        # --- Order book imbalance (L2 if available, else aggression proxy) ---
        if bar.imbalance_values:
            imbalance = float(np.mean(bar.imbalance_values))
        elif bar.total_trades > 0:
            imbalance = bar.buyer_trades / bar.total_trades - 0.5  # centre at 0
        else:
            imbalance = 0.0

        # --- Market regime (encoded volatile and trending) ---
        regime_volatile = 0.0
        regime_trending = 0.0
        if self._regime_classifier is not None:
            try:
                label = self._regime_classifier.classify(
                    volatility, mean_spread, 0.0, mean_vol, trend_slope=0.0
                )
                if label == "volatile":
                    regime_volatile = 1.0
                elif label == "trending":
                    regime_trending = 1.0
            except Exception:
                pass

        # --- BTC / ETH returns ---
        btc_return = 0.0
        if bar.btc_prices:
            btc_close = bar.btc_prices[-1]
            if self._prev_btc is not None and self._prev_btc > 0:
                btc_return = (btc_close - self._prev_btc) / self._prev_btc
            self._prev_btc = btc_close

        eth_return = 0.0
        if bar.eth_prices:
            eth_close = bar.eth_prices[-1]
            if self._prev_eth is not None and self._prev_eth > 0:
                eth_return = (eth_close - self._prev_eth) / self._prev_eth
            self._prev_eth = eth_close

        # --- Algo health (ML1 vector) ---
        algo = self._algo_health
        is_cold = (algo == _COLD_START_ALGO_HEALTH)
        if is_cold:
            self._cold_start_bar_count += 1
            if self._cold_start_bar_count <= 50 and self._cold_start_bar_count % 10 == 1:
                logger.warning(
                    "FeatureBarBuilder: ML1 still at cold-start defaults "
                    "(%d bars included in PCMCI window, waiting for ML1 to warm up)",
                    self._cold_start_bar_count,
                )
            # S-10 FIX: We no longer return None here. The causal agent requires
            # time (10 mins) to initialize, not market conditions (100 trades).
            # The constant cold-start vector is ignored by PCMCI safely.

        p_normal, p_stressed, p_degraded = algo[0], algo[1], algo[2]

        # --- Time of day (cyclical encoding) ---
        now_utc = datetime.now(timezone.utc)
        hour_frac = now_utc.hour + now_utc.minute / 60.0
        session_sin = math.sin(2 * math.pi * hour_frac / 24.0)
        session_cos = math.cos(2 * math.pi * hour_frac / 24.0)

        return np.array([
            price_return,
            mean_vol,
            mean_spread,
            volatility,
            rsi,
            trade_intensity,
            imbalance,
            ema_spread_val,
            bollinger_width,
            price_in_band,
            regime_volatile,
            regime_trending,
            btc_return,
            eth_return,
            p_normal,
            p_stressed,
            p_degraded,
            session_sin,
            session_cos,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Indicator helpers (pure functions — no external deps)
# ---------------------------------------------------------------------------

def _compute_ema(prices: list[float], period: int) -> float:
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for p in prices[period:]:
        ema = p * k + ema * (1 - k)
    return ema


def _compute_wilder_rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
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


def _monotonic_bar_start() -> float:
    import time
    return time.monotonic()
