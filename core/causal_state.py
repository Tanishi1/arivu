"""core/causal_state.py
CausalState singleton + 4 propagation rules.

The causal model is the heart of the digital twin. It tracks WHY variables
change, not just THAT they changed. A correlation model detects that
volatility is high and P&L is declining; the causal model explains the path:

  volatility spike
    → market makers widen quotes
      → spread increases
        → strategy spread assumption approaching violation
          → expected P&L degrading along specific predicted path

Only market_feed_loop may write to this. See DECISIONS.md ADR-003.
All writes must acquire the asyncio.Lock before modifying state.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

import numpy as np

from core.constants import (
    VOLATILITY_WINDOW, TREND_WINDOW, VOLUME_WINDOW,
    SLOPE_NORMALISER, SPREAD_PROPAGATION_THRESHOLD,
    SPREAD_PROPAGATION_FACTOR, THRESHOLDS,
)
from core.schemas import CausalState, MarketTick

logger = logging.getLogger(__name__)

# Trigger thresholds — from per-instrument config in core.constants
_SOL_THRESHOLDS = THRESHOLDS["SOLUSDT"]
VOLATILITY_THRESHOLD = _SOL_THRESHOLDS["volatility_limit"]
SPREAD_THRESHOLD = _SOL_THRESHOLDS["spread_limit"]
TREND_SLOPE_FLIP_TOLERANCE = 0.001  # minimum slope magnitude to qualify as a real reversal
                                    # 0.0 fired on ANY sign change, including ±ε numerical noise
                                    # 0.001 = 10% of SLOPE_NORMALISER — requires a meaningful trend
STALE_THRESHOLD_S = 10            # seconds before feed is considered stale

# Warmup period: rolling windows (volatility, trend, volume) all use VOLATILITY_WINDOW
# ticks minimum. Until all windows are warm, computed values jump from 0.0 to their
# true value in a single tick, firing spurious threshold crossings. Suppress triggers
# until the rolling windows are fully saturated.
_WARMUP_TICKS = max(VOLATILITY_WINDOW, TREND_WINDOW, VOLUME_WINDOW)


class CausalStateManager:
    """Manages the single shared CausalState instance.

    Usage:
        manager = CausalStateManager()
        await manager.update(tick)     # called by market_feed_loop
        state = manager.snapshot()     # called by any coroutine (read-only)
    """

    def __init__(self, decision_queue: asyncio.Queue) -> None:
        self._state = CausalState()
        self._lock = asyncio.Lock()
        self._decision_queue = decision_queue

        # Rolling price buffer — maxlen=50 gives trend slope wider context
        # while volatility/volume slice to their respective window sizes
        self._prices: deque[float] = deque(maxlen=50)
        self._volumes: deque[float] = deque(maxlen=50)
        self._prev_trend_slope: float = 0.0

        # Edge detection — only fire triggers on threshold CROSSING, not sustained exceedance
        self._prev_above_vol: bool = False
        self._prev_above_spread: bool = False

        # Warmup counter — rolling windows need _WARMUP_TICKS ticks before their
        # outputs are meaningful. Triggers are suppressed during this period to
        # prevent the cold-start artifact (all windows go 0.0 → real value together).
        self._tick_count: int = 0

    async def update(self, tick: MarketTick) -> None:
        """Process an incoming MarketTick: update state + propagate causally.

        Called by market_feed_loop on every WebSocket message.
        Acquires the asyncio.Lock for the duration of the write.
        """
        # Capture trigger values inside the lock, then release BEFORE firing
        # queue events. Holding the lock during queue.put() causes deadlock
        # when the decision cycle calls update_algo_health() while the feed
        # loop is suspended waiting for queue space.
        async with self._lock:
            self._prices.append(tick.price)
            self._volumes.append(tick.volume)

            volatility = self._compute_rolling_volatility()
            trend_slope = self._compute_trend_slope()
            trend_strength = min(1.0, abs(trend_slope) / SLOPE_NORMALISER)
            volume = float(np.mean(self._volumes)) if self._volumes else 0.0

            # --- Propagation Rule 1: volatility → spread -----------------
            propagated_spread = tick.spread * (
                1 + max(0.0, volatility - SPREAD_PROPAGATION_THRESHOLD)
                * SPREAD_PROPAGATION_FACTOR
            )

            # --- Propagation Rule 2: spread → assumption proximity --------
            # Handled downstream in monitor.py when checking each assumption.

            # --- Propagation Rule 3: trend reversal → trigger queue ------
            trend_reversed = (
                self._prev_trend_slope > TREND_SLOPE_FLIP_TOLERANCE
                and trend_slope < -TREND_SLOPE_FLIP_TOLERANCE
            ) or (
                self._prev_trend_slope < -TREND_SLOPE_FLIP_TOLERANCE
                and trend_slope > TREND_SLOPE_FLIP_TOLERANCE
            )

            # --- Propagation Rule 4: ML1 vector → algo health ------------
            # algo_health_vector is updated separately via update_algo_health().

            rsi = self._compute_rsi()
            divergence_span = self._compute_divergence_span()

            self._state = self._state.model_copy(
                update={
                    "timestamp": tick.timestamp,
                    "price": tick.price,
                    "volatility": volatility,
                    "spread": propagated_spread,
                    "trend_slope": trend_slope,
                    "trend_strength": trend_strength,
                    "volume": volume,
                    "rsi_current": rsi,
                    "divergence_candle_span": divergence_span,
                    "last_tick_timestamp": datetime.now(timezone.utc),
                    "price_history": list(self._prices),
                }
            )

            logger.debug(
                "CausalState updated | price=%.6f vol=%.6f spread=%.6f trend=%.6f",
                tick.price, volatility, propagated_spread, trend_slope,
            )

            # Capture for use outside lock — do NOT await inside the lock
            self._prev_trend_slope = trend_slope
            self._tick_count += 1
            _vol = volatility
            _spread = propagated_spread
            _trend_reversed = trend_reversed
            _warm = self._tick_count >= _WARMUP_TICKS

        # K1 FIX: lock is now released before any queue interaction
        if _warm:
            await self._check_thresholds(_vol, _spread, _trend_reversed)
        else:
            logger.debug(
                "Warmup | tick %d/%d — threshold triggers suppressed",
                self._tick_count, _WARMUP_TICKS,
            )

    async def update_algo_health(self, vector: list[float]) -> None:
        """Update the algo_health_vector from the latest ML1 predict_proba().

        Called by market_feed_loop after executor emits ExecutionTelemetry.
        ML1 runs continuously in the background; this is NOT a one-time snapshot.
        The latest vector is captured at the moment of each ledger commit.
        """
        async with self._lock:
            self._state = self._state.model_copy(
                update={"algo_health_vector": vector}
            )

    async def update_graph_features(self, features: dict[str, float]) -> None:
        """Push the latest 10-second bar features from FeatureBarBuilder into CausalState.

        Called once per bar (every ~10 seconds) after _compute_features() closes a bar.
        The features dict uses the same VARIABLE_NAMES keys from core/feature_bar.py.
        This makes every causal graph edge variable available via getattr(state, name)
        so the CausalAgent can build Assumption objects without silent drop-outs.
        """
        async with self._lock:
            self._state = self._state.model_copy(update=features)
        logger.debug(
            "CausalState: graph features updated | btc_ret=%.6f eth_ret=%.6f "
            "ema_spread=%.6f bollinger_w=%.6f price_ret=%.6f",
            features.get("btc_return", 0.0),
            features.get("eth_return", 0.0),
            features.get("ema_spread", 0.0),
            features.get("bollinger_width", 0.0),
            features.get("price_return", 0.0),
        )

    async def update_position(self, qty: float, capital: float) -> None:
        """Update live position exposure after an execution."""
        async with self._lock:
            self._state = self._state.model_copy(
                update={"position_size": qty, "capital_deployed": capital}
            )

    def snapshot(self) -> CausalState:
        """Return an immutable snapshot of the current state (no lock needed for reads).

        CausalState is frozen=True — no deep copy needed. model_copy() creates a
        new wrapper but shares the underlying field values, which is safe since
        the frozen constraint prevents reassignment and all updates use model_copy(update=...).
        """
        return self._state.model_copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rolling_volatility(self) -> float:
        """Rolling standard deviation of log returns over VOLATILITY_WINDOW ticks."""
        if len(self._prices) < VOLATILITY_WINDOW:
            return 0.0
        prices = np.array(list(self._prices)[-VOLATILITY_WINDOW:])
        log_returns = np.diff(np.log(prices + 1e-10))
        return float(np.std(log_returns))

    def _compute_trend_slope(self) -> float:
        """Least-squares slope of price over TREND_WINDOW ticks."""
        if len(self._prices) < TREND_WINDOW:
            return 0.0
        prices = np.array(list(self._prices)[-TREND_WINDOW:])
        x = np.arange(TREND_WINDOW)
        slope = float(np.polyfit(x, prices, 1)[0])
        return slope

    def _compute_rsi(self, period: int = 14) -> float:
        """Compute RSI using Wilder's exponential smoothing method.

        CRIT-2 FIX: Replaces the previous SMA-RSI (simple-average) implementation
        which was incorrectly documented as "Wilder's method". Wilder's method uses
        exponential smoothing: avg_gain = (prev_avg * (period-1) + current) / period.
        SMA-RSI and Wilder RSI produce different values from the same price series.

        RSIStrategy.generate_signal() uses ta.rsi() which is Wilder's method. Without
        this fix, the monitor checked a different RSI than the strategy used to fire
        BUY/SELL signals, making every rsi_not_extreme breach record scientifically
        incorrect.

        Returns 50.0 (neutral) when fewer than period+1 prices are available.
        Range: [0, 100]. Above 80 = overbought (assumption breach threshold).
        """
        prices = list(self._prices)
        if len(prices) < period + 1:
            return 50.0

        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains  = [max(0.0,  c) for c in changes]
        losses = [max(0.0, -c) for c in changes]

        # Wilder initial average over first `period` changes
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Wilder smoothing for all subsequent periods
        for i in range(period, len(changes)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            return 100.0
        return round(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)), 2)

    def _compute_divergence_span(self) -> float:
        """Compute candles-since-most-recent-local-extreme (low OR high).

        HIGH-5 FIX: Previously only scanned local LOWS. RSI bearish divergence is
        detected from local HIGHS in generate_signal(). When a SELL signal fired,
        the span measured "bars since last LOW" which could be very old while the
        actual bearish pattern spanned only 2-3 recent highs. The assumption was
        incorrectly passing for all bearish signals.

        Now scans BOTH local lows and local highs. Returns the MINIMUM span (most
        recent price extreme). This ensures:
          - Bullish divergence (uses lows): low_span is directly relevant.
          - Bearish divergence (uses highs): high_span is directly relevant.
          - minimum = "how recently did ANY significant swing occur"
            if the most recent swing was < 3 bars ago, the pattern is noisy
            for EITHER signal direction.

        Returns 0.0 if fewer than 5 prices available or no extreme found.
        """
        prices = list(self._prices)
        if len(prices) < 5:
            # B2 FIX: Return the same safe default as CausalState schema (3.5),
            # not 0.0. On crash-restart with a recovered ACTIVE DO, the prices
            # deque is empty. The monitor fires within 5s before enough ticks
            # arrive. 0.0 < RSI threshold (3.0) → false breach → recovered DO
            # immediately interrupted. 3.5 > 3.0 keeps the assumption safe until
            # real price history accumulates.
            return 3.5

        last = len(prices) - 1
        low_span: float | None = None
        high_span: float | None = None

        # Most recent local minimum
        for i in range(last - 1, 0, -1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                low_span = float(last - i)
                break

        # Most recent local maximum
        for i in range(last - 1, 0, -1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                high_span = float(last - i)
                break

        spans = [s for s in (low_span, high_span) if s is not None]
        return min(spans) if spans else 3.5

    async def _check_thresholds(
        self,
        volatility: float,
        spread: float,
        trend_reversed: bool,
    ) -> None:
        """Fire decision_queue event if any market threshold is CROSSED.

        Uses edge detection: only fires on threshold crossing, not sustained
        exceedance. Uses put_nowait() so a full queue never suspends this
        coroutine — stale triggers are discarded with a warning (K2 fix).
        """
        above_vol = volatility > VOLATILITY_THRESHOLD
        if above_vol and not self._prev_above_vol:
            try:
                self._decision_queue.put_nowait({
                    "type": "threshold_crossed",
                    "reason": "volatility",
                    "value": volatility,
                })
                logger.info("Trigger fired | reason=volatility value=%.6f", volatility)
            except asyncio.QueueFull:
                logger.warning("Decision queue full — volatility trigger discarded")
        self._prev_above_vol = above_vol

        above_spread = spread > SPREAD_THRESHOLD
        if above_spread and not self._prev_above_spread:
            try:
                self._decision_queue.put_nowait({
                    "type": "threshold_crossed",
                    "reason": "spread",
                    "value": spread,
                })
                logger.info("Trigger fired | reason=spread value=%.6f", spread)
            except asyncio.QueueFull:
                logger.warning("Decision queue full — spread trigger discarded")
        self._prev_above_spread = above_spread

        if trend_reversed:
            try:
                self._decision_queue.put_nowait({
                    "type": "threshold_crossed",
                    "reason": "trend_reversal",
                    "value": volatility,
                })
                logger.info("Trigger fired | reason=trend_reversal")
            except asyncio.QueueFull:
                logger.warning("Decision queue full — trend_reversal trigger discarded")
