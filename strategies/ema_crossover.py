"""strategies/ema_crossover.py  —  Person 1
Dynamic EMA Crossover Strategy (Momentum).

Signal logic:
  BUY  if fast_ema[-1] > slow_ema[-1] AND fast_ema[-2] <= slow_ema[-2]  (crossover up)
  SELL if fast_ema[-1] < slow_ema[-1] AND fast_ema[-2] >= slow_ema[-2]  (crossover down)
  HOLD otherwise

Three causal assumptions:
  1. trend_persistence  — trend_slope > 0.0    (trend must continue)
  2. volatility_control — volatility < 0.04   (not too volatile)
  3. spread_constraint  — spread < 0.002      (spread must be tight)

Evaluate scoring (Hill-climbing objective):
  score = trend_strength × (1 - spread/0.002) × (1 - volatility/0.1) × 0.015
  For each assumption where proximity > 0.8: multiply score × 0.5

Assigned to: Person 1
Done when: Simulator selects EMA in a trending CausalState and Hill-climbing
           returns parameters different from defaults.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401 — used via pd.Series.ta

from core.schemas import Assumption, CausalState
from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Rolling price buffer length — strategies need a history to compute indicators
_PRICE_HISTORY: list[float] = []
MAX_HISTORY = 60  # keep last 60 prices


class EMAStrategy(BaseStrategy):
    """Dynamic EMA Crossover momentum strategy."""

    def generate_signal(self, state: CausalState, params: dict) -> str:
        """Compute EMA crossover signal from live price history."""
        _PRICE_HISTORY.append(state.price)
        if len(_PRICE_HISTORY) > MAX_HISTORY:
            _PRICE_HISTORY.pop(0)

        if len(_PRICE_HISTORY) < params.get("slow_period", 21) + 1:
            return "HOLD"

        closes = pd.Series(_PRICE_HISTORY)
        fast = ta.ema(closes, length=params["fast_period"])
        slow = ta.ema(closes, length=params["slow_period"])

        if fast is None or slow is None or len(fast) < 2 or len(slow) < 2:
            return "HOLD"

        # Crossover up
        if fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]:
            return "BUY"
        # Crossover down
        if fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]:
            return "SELL"
        return "HOLD"

    def get_assumptions(self, state: CausalState, params: dict) -> list[Assumption]:
        """Return exactly 3 assumptions for the EMA Crossover strategy."""
        return [
            Assumption(
                name="trend_persistence",
                variable="trend_slope",
                operator="gt",
                threshold=0.0,
            ),
            Assumption(
                name="volatility_control",
                variable="volatility",
                operator="lt",
                threshold=0.04,
            ),
            Assumption(
                name="spread_constraint",
                variable="spread",
                operator="lt",
                threshold=0.002,
            ),
        ]

    def get_default_params(self) -> dict:
        return {
            "fast_period": 9,
            "slow_period": 21,
            "position_fraction": 0.10,
        }

    def get_parameter_bounds(self) -> dict:
        return {
            "fast_period":       {"min": 5,    "max": 15,   "step": 1},
            "slow_period":       {"min": 15,   "max": 50,   "step": 1},
            "position_fraction": {"min": 0.05, "max": 0.20, "step": 0.01},
        }

    def evaluate(self, state: CausalState, params: dict) -> float:
        """Score this parameter configuration for hill-climbing."""
        if self.generate_signal(state, params) == "HOLD":
            return 0.0

        score = (
            state.trend_strength
            * (1 - min(1.0, state.spread / 0.002))
            * (1 - min(1.0, state.volatility / 0.1))
            * 0.015
        )

        # Proximity penalty: each near-breach assumption halves the score
        assumptions = self.get_assumptions(state, params)
        for assumption in assumptions:
            current = getattr(state, assumption.variable, 0.0)
            if assumption.threshold != 0:
                proximity = current / assumption.threshold
                if proximity > 0.8:
                    score *= 0.5

        return score
