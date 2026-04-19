"""strategies/rsi_divergence.py  —  Person 3
RSI Momentum Divergence Strategy.

Signal logic:
  Bullish divergence: last 2 price LOWS show lower price but HIGHER RSI
    → price_low2 < price_low1 AND rsi_at_low2 > rsi_at_low1 + divergence_threshold → BUY
  Bearish divergence: last 2 price HIGHS show higher price but LOWER RSI
    → price_high2 > price_high1 AND rsi_at_high2 < rsi_at_high1 - divergence_threshold → SELL
  Otherwise → HOLD

Three causal assumptions:
  1. divergence_span     — divergence_candle_span > 3   (pattern must have enough candles)
  2. volatility_acceptable — volatility < 0.05          (too volatile = unreliable RSI)
  3. rsi_not_extreme     — RSI between 20 and 80        (avoid overbought/oversold extremes)

Evaluate scoring:
  if HOLD: return 0.0
  divergence_confidence = min(1.0, divergence_span / 5)
  score = 0.008 × (1 - volatility/0.05) × divergence_confidence
  proximity > 0.8: multiply score × 0.3  (strictest of the three strategies)

Assigned to: Person 3
Done when: Simulator selects RSI in an uncertain CausalState and evaluate()
           returns a conservative base_return.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401

from core.schemas import Assumption, CausalState
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_PRICE_HISTORY: list[float] = []
MAX_HISTORY = 60


def _find_local_lows(prices: list[float]) -> list[tuple[int, float]]:
    """Return (index, price) for local minima in prices."""
    lows = []
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            lows.append((i, prices[i]))
    return lows


def _find_local_highs(prices: list[float]) -> list[tuple[int, float]]:
    """Return (index, price) for local maxima in prices."""
    highs = []
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            highs.append((i, prices[i]))
    return highs


class RSIStrategy(BaseStrategy):
    """RSI Momentum Divergence strategy — conservative, uncertainty regime."""

    def generate_signal(self, state: CausalState, params: dict) -> str:
        _PRICE_HISTORY.append(state.price)
        if len(_PRICE_HISTORY) > MAX_HISTORY:
            _PRICE_HISTORY.pop(0)

        period = params.get("rsi_lookback_period", 14)
        if len(_PRICE_HISTORY) < period + 5:
            return "HOLD"

        closes = pd.Series(_PRICE_HISTORY)
        rsi = ta.rsi(closes, length=period)
        if rsi is None:
            return "HOLD"

        prices = _PRICE_HISTORY
        threshold = params.get("divergence_threshold", 5)

        lows = _find_local_lows(prices)
        if len(lows) >= 2:
            i1, p1 = lows[-2]
            i2, p2 = lows[-1]
            rsi1 = rsi.iloc[i1] if i1 < len(rsi) else None
            rsi2 = rsi.iloc[i2] if i2 < len(rsi) else None
            if rsi1 is not None and rsi2 is not None:
                if p2 < p1 and rsi2 > rsi1 + threshold:
                    return "BUY"  # bullish divergence

        highs = _find_local_highs(prices)
        if len(highs) >= 2:
            i1, p1 = highs[-2]
            i2, p2 = highs[-1]
            rsi1 = rsi.iloc[i1] if i1 < len(rsi) else None
            rsi2 = rsi.iloc[i2] if i2 < len(rsi) else None
            if rsi1 is not None and rsi2 is not None:
                if p2 > p1 and rsi2 < rsi1 - threshold:
                    return "SELL"  # bearish divergence

        return "HOLD"

    def get_assumptions(self, state: CausalState, params: dict) -> list[Assumption]:
        """Return exactly 3 assumptions for the RSI Divergence strategy."""
        return [
            Assumption(
                name="divergence_span",
                variable="trend_strength",  # proxy variable
                operator="gt",
                threshold=0.05,  # divergence_candle_span > 3 (normalised proxy)
            ),
            Assumption(
                name="volatility_acceptable",
                variable="volatility",
                operator="lt",
                threshold=0.05,
            ),
            Assumption(
                name="rsi_not_extreme",
                variable="volatility",   # proxy: low vol correlates with stable RSI
                operator="lt",
                threshold=0.08,
            ),
        ]

    def get_default_params(self) -> dict:
        return {
            "rsi_lookback_period": 14,
            "divergence_threshold": 5,
            "position_fraction": 0.05,  # most conservative of three strategies
        }

    def get_parameter_bounds(self) -> dict:
        return {
            "rsi_lookback_period":  {"min": 7,    "max": 21,  "step": 1},
            "divergence_threshold": {"min": 3,    "max": 10,  "step": 1},
            "position_fraction":    {"min": 0.02, "max": 0.10, "step": 0.01},
        }

    def evaluate(self, state: CausalState, params: dict) -> float:
        """Score conservatively. Strictest proximity penalty (×0.3)."""
        if self.generate_signal(state, params) == "HOLD":
            return 0.0

        # Divergence confidence based on span (approximated from price history length)
        divergence_span = max(1, len(_PRICE_HISTORY) // 10)
        divergence_confidence = min(1.0, divergence_span / 5)

        score = (
            0.008
            * (1 - min(1.0, state.volatility / 0.05))
            * divergence_confidence
        )

        # Strictest proximity penalty of the three strategies
        assumptions = self.get_assumptions(state, params)
        for assumption in assumptions:
            current = getattr(state, assumption.variable, 0.0)
            if assumption.threshold != 0:
                proximity = current / assumption.threshold
                if proximity > 0.8:
                    score *= 0.3

        return score
