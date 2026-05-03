"""shared/causal_state.py
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
from typing import Optional

import numpy as np

from core.schemas import CausalState, MarketTick

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rolling window sizes
# ---------------------------------------------------------------------------
VOLATILITY_WINDOW = 20   # ticks
TREND_WINDOW = 20        # ticks
VOLUME_WINDOW = 20       # ticks

# ---------------------------------------------------------------------------
# Trigger thresholds — crossing these fires the decision_queue
# ---------------------------------------------------------------------------
VOLATILITY_THRESHOLD = 0.04
SPREAD_THRESHOLD = 0.002
TREND_SLOPE_FLIP_TOLERANCE = 0.0  # sign change triggers
STALE_THRESHOLD_S = 10            # seconds before feed is considered stale


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

        # Rolling price buffer for rolling calculations
        self._prices: deque[float] = deque(maxlen=VOLATILITY_WINDOW)
        self._volumes: deque[float] = deque(maxlen=VOLUME_WINDOW)
        self._prev_trend_slope: float = 0.0

    async def update(self, tick: MarketTick) -> None:
        """Process an incoming MarketTick: update state + propagate causally.

        Called by market_feed_loop on every WebSocket message.
        Acquires the asyncio.Lock for the duration of the write.
        """
        async with self._lock:
            self._prices.append(tick.price)
            self._volumes.append(tick.volume)

            volatility = self._compute_rolling_volatility()
            trend_slope = self._compute_trend_slope()
            trend_strength = min(1.0, abs(trend_slope) / 0.01)  # normalise
            volume = float(np.mean(self._volumes)) if self._volumes else 0.0

            # --- Propagation Rule 1: volatility → spread -----------------
            # High volatility causes market makers to widen quotes.
            # The spread in the causal model is not just the raw bid-ask;
            # it reflects the volatility-adjusted pressure on spreads.
            propagated_spread = tick.spread * (1 + max(0.0, volatility - 0.02) * 5)

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
            # That value is already in self._state.

            self._state = CausalState(
                timestamp=tick.timestamp,
                price=tick.price,
                volatility=volatility,
                spread=propagated_spread,
                trend_slope=trend_slope,
                trend_strength=trend_strength,
                volume=volume,
                algo_health_vector=self._state.algo_health_vector,
                active_strategy=self._state.active_strategy,
                position_size=self._state.position_size,
                capital_deployed=self._state.capital_deployed,
                last_tick_timestamp=datetime.now(timezone.utc),
            )

            logger.debug(
                "CausalState updated | price=%.4f vol=%.4f spread=%.4f trend=%.4f",
                tick.price, volatility, propagated_spread, trend_slope,
            )

            # Threshold check — fires trigger if any boundary is crossed
            self._prev_trend_slope = trend_slope
            await self._check_thresholds(volatility, propagated_spread, trend_reversed)

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

    def snapshot(self) -> CausalState:
        """Return an immutable snapshot of the current state (no lock needed for reads)."""
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rolling_volatility(self) -> float:
        """Rolling standard deviation of log returns over VOLATILITY_WINDOW ticks."""
        if len(self._prices) < 2:
            return 0.0
        prices = np.array(self._prices)
        log_returns = np.diff(np.log(prices + 1e-10))
        return float(np.std(log_returns))

    def _compute_trend_slope(self) -> float:
        """Least-squares slope of price over TREND_WINDOW ticks."""
        if len(self._prices) < 2:
            return 0.0
        prices = np.array(self._prices)
        x = np.arange(len(prices))
        slope = float(np.polyfit(x, prices, 1)[0])
        return slope

    async def _check_thresholds(
        self,
        volatility: float,
        spread: float,
        trend_reversed: bool,
    ) -> None:
        """Fire decision_queue event if any market threshold is crossed."""
        if volatility > VOLATILITY_THRESHOLD:
            await self._decision_queue.put({
                "type": "threshold_crossed",
                "reason": "volatility",
                "value": volatility,
            })
            logger.info("Trigger fired | reason=volatility value=%.4f", volatility)

        if spread > SPREAD_THRESHOLD:
            await self._decision_queue.put({
                "type": "threshold_crossed",
                "reason": "spread",
                "value": spread,
            })
            logger.info("Trigger fired | reason=spread value=%.4f", spread)

        if trend_reversed:
            await self._decision_queue.put({
                "type": "threshold_crossed",
                "reason": "trend_reversal",
                "value": volatility,
            })
            logger.info("Trigger fired | reason=trend_reversal")
