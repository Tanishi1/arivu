"""strategies/bollinger.py  —  Person 2
Volatility-Banded Mean Reversion Strategy (Bollinger Bands).

Signal logic:
  BUY  if close[-1] < lower_band[-1]  (price below lower band — oversold)
  SELL if close[-1] > middle[-1] AND close[-2] <= middle[-2]  (crosses back through middle)
  HOLD otherwise

Three causal assumptions:
  1. no_active_trend       — trend_strength < 0.4  (mean reversion fails in strong trends)
  2. volatility_normal     — volatility < 0.035    (too much volatility widens bands unreliably)
  3. reversion_within_window — time-based: candles since BUY < 6

Evaluate scoring:
  if trend_strength > 0.5: return -0.01  (strong penalty — strategy should not run)
  if HOLD: return 0.0
  score = (1 - trend_strength) × penalty_for_vol_deviation × 0.012
  proximity > 0.8: multiply score × 0.4  (strictest proximity penalty)

Assigned to: Person 2
Done when: Simulator selects Bollinger in a ranging CausalState and penalties
           apply correctly in a trending state.
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
_candles_since_buy: int = 0


class BollingerStrategy(BaseStrategy):
    """Volatility-Banded Mean Reversion using Bollinger Bands."""

    def generate_signal(self, state: CausalState, params: dict) -> str:
        global _candles_since_buy

        _PRICE_HISTORY.append(state.price)
        if len(_PRICE_HISTORY) > MAX_HISTORY:
            _PRICE_HISTORY.pop(0)

        period = params.get("baseline_period", 20)
        if len(_PRICE_HISTORY) < period + 1:
            return "HOLD"

        closes = pd.Series(_PRICE_HISTORY)
        middle = ta.sma(closes, length=period)
        std = closes.rolling(period).std()
        mult = params.get("std_dev_multiplier", 2.0)

        if middle is None or std is None:
            return "HOLD"

        lower = middle - mult * std
        _candles_since_buy += 1

        if closes.iloc[-1] < lower.iloc[-1]:
            _candles_since_buy = 0
            return "BUY"

        if (
            closes.iloc[-1] > middle.iloc[-1]
            and closes.iloc[-2] <= middle.iloc[-2]
        ):
            return "SELL"

        return "HOLD"

    def get_assumptions(self, state: CausalState, params: dict) -> list[Assumption]:
        """Return exactly 3 assumptions for the Bollinger strategy."""
        return [
            Assumption(
                name="no_active_trend",
                variable="trend_strength",
                operator="lt",
                threshold=0.4,
            ),
            Assumption(
                name="volatility_normal",
                variable="volatility",
                operator="lt",
                threshold=0.035,
            ),
            Assumption(
                # Time-based assumption — tracked via candle count
                name="reversion_within_window",
                variable="trend_strength",  # proxy: use trend as the check variable
                operator="lt",
                threshold=0.6,             # overridden by monitor logic for time check
            ),
        ]

    def get_default_params(self) -> dict:
        return {
            "baseline_period": 20,
            "std_dev_multiplier": 2.0,
            "position_fraction": 0.08,
        }

    def get_parameter_bounds(self) -> dict:
        return {
            "baseline_period":     {"min": 10,   "max": 40,   "step": 2},
            "std_dev_multiplier":  {"min": 1.5,  "max": 3.0,  "step": 0.1},
            "position_fraction":   {"min": 0.04, "max": 0.15, "step": 0.01},
        }

    def evaluate(self, state: CausalState, params: dict) -> float:
        """Score this configuration. Heavy penalty in trending markets."""
        # Hard penalty: mean reversion in a strong trend is a bad trade
        if state.trend_strength > 0.5:
            return -0.01

        if self.generate_signal(state, params) == "HOLD":
            return 0.0

        # Optimal volatility for Bollinger is moderate (~0.02)
        vol_deviation = abs(state.volatility - 0.02) / 0.02
        score = (1 - state.trend_strength) * (1 - min(1.0, vol_deviation)) * 0.012

        # Proximity penalty
        assumptions = self.get_assumptions(state, params)
        for assumption in assumptions:
            current = getattr(state, assumption.variable, 0.0)
            if assumption.threshold != 0:
                proximity = current / assumption.threshold
                if proximity > 0.8:
                    score *= 0.4

        return score
