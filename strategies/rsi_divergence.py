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
  if insufficient history: return 0.0
  divergence_confidence = min(1.0, divergence_span / 5)
  score = 0.008 × (1 - volatility/0.05) × divergence_confidence
  proximity > 0.8: multiply score × 0.3  (strictest of the three strategies)

  IMPORTANT: evaluate() is a PURE function — it does NOT call generate_signal().
  generate_signal() appends to self._price_history as a side effect. Calling it
  inside evaluate() during hill-climbing would corrupt the history with 100+ duplicate
  prices, destroying divergence pattern detection permanently.

Assigned to: Person 3
Done when: Simulator selects RSI in an uncertain CausalState and evaluate()
           returns a conservative base_return.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401

from core.schemas import Assumption, CausalState
from strategies.base import Strategy

logger = logging.getLogger(__name__)

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


class RSIStrategy(Strategy):
    """RSI Momentum Divergence strategy — conservative, uncertainty regime."""

    def __init__(self) -> None:
        # Instance-level price history — NOT module-level.
        # Module-level lists are corrupted by evaluate() during hill-climbing.
        self._price_history: list[float] = []

    def generate_signal(self, state: CausalState, params: dict) -> str:
        self._price_history.append(state.price)
        if len(self._price_history) > MAX_HISTORY:
            self._price_history.pop(0)

        period = params.get("rsi_lookback_period", 14)
        if len(self._price_history) < period + 5:
            return "HOLD"

        closes = pd.Series(self._price_history)
        rsi = ta.rsi(closes, length=period)
        if rsi is None:
            return "HOLD"

        prices = self._price_history
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
        """Return exactly 3 assumptions for the RSI Divergence strategy.

        S3-3: current_value and proximity are intentionally 0.0 (Pydantic defaults).
        core/decision_utils.annotate_proximity() is the single source of truth.
        """
        return [
            Assumption(
                name="divergence_span",
                variable="divergence_candle_span",
                operator="gt",
                threshold=3.0,
            ),
            Assumption(
                name="volatility_acceptable",
                variable="volatility",
                operator="lt",
                threshold=0.05,
            ),
            Assumption(
                name="rsi_not_extreme",
                variable="rsi_current",
                operator="lt",
                threshold=80.0,
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
        """Score conservatively. Strictest proximity penalty (×0.3).

        PURE FUNCTION — does NOT call generate_signal() and does NOT modify
        self._price_history. Score computed from CausalState fields (read-only).

        HIGH-1 FIX: Uses state.divergence_candle_span (the real observable from
        causal_state._compute_divergence_span()) instead of len(_price_history)//10.
        Previously hill-climbing evaluated against a proxy that disagreed with what
        the divergence_span assumption actually checks.
        """
        # RSI divergence needs some price history to detect patterns
        if len(self._price_history) < 10:
            return 0.0

        # HIGH-1 FIX: use the real divergence span from CausalState
        div_span = getattr(state, "divergence_candle_span", 3.5)
        divergence_confidence = min(1.0, div_span / 5.0)

        score = (
            0.008
            * (1 - min(1.0, state.volatility / 0.05))
            * divergence_confidence
        )

        # B12 FIX: use operator-aware proximity in evaluate(), matching decision_utils.py.
        # Previously used current/threshold for ALL operators. For divergence_span
        # (operator='gt', threshold=3.0), a safe value of 5.0 gave proximity=1.67
        # which always triggered the ×0.3 penalty — RSI was permanently underscored.
        assumptions = self.get_assumptions(state, params)
        for assumption in assumptions:
            current = getattr(state, assumption.variable, 0.0)
            if assumption.threshold == 0:
                continue  # zero-threshold handled elsewhere (EMA slope special case)
            if assumption.operator == "gt":
                proximity = (assumption.threshold / current) if current > 0 else 1.0
            else:
                proximity = current / assumption.threshold
            if proximity > 0.8:
                score *= 0.3

        return score
