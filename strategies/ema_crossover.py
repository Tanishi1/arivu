"""strategies/ema_crossover.py  —  Person 1
Dynamic EMA Crossover Strategy (Momentum).

Signal logic (§10.2 HOLD-trap fix — state-based, not event-based):
  BUY  if fast_ema > slow_ema  (fast is currently above slow — sustained uptrend)
  SELL if fast_ema < slow_ema  (fast is currently below slow — sustained downtrend)
  HOLD if insufficient data

FORMER BUG: checked for crossover on exactly this tick
  (fast[-1] > slow[-1] AND fast[-2] <= slow[-2]).
  Because the decision cycle samples the market every ~10–30 seconds, catching
  a crossover on the exact second it occurs is statistically very rare — causing
  permanent HOLD even when the trend was clearly established. State-based check
  fires whenever fast_ema is above slow_ema, which is the correct condition
  for a momentum strategy.
  The legacy strategies are kept as a comparison arm — this fix makes that
  comparison meaningful (three strategies that can actually trade vs. one agent).

Three causal assumptions:
  1. trend_persistence  — trend_slope > 0.0    (trend must continue)
  2. volatility_control — volatility < 0.04   (not too volatile)
  3. spread_constraint  — spread < 0.002      (spread must be tight)

Evaluate scoring (Hill-climbing objective):
  score = trend_strength × (1 - spread/0.002) × (1 - volatility/0.1) × 0.015
  For each assumption where proximity > 0.8: multiply score × 0.5

  IMPORTANT: evaluate() is a PURE function — it does NOT call generate_signal().
  generate_signal() appends to self._price_history as a side effect. Calling it
  inside evaluate() during hill-climbing would append the same price 100+ times,
  filling the buffer with duplicates and making all subsequent EMA signals HOLD.

Assigned to: Person 1
Done when: Simulator selects EMA in a trending CausalState and Hill-climbing
           returns parameters different from defaults.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401 — used via pd.Series.ta

from core.schemas import Assumption, CausalState
from core.constants import THRESHOLDS
from strategies.base import Strategy

logger = logging.getLogger(__name__)

MAX_HISTORY = 60  # keep last 60 prices
# SOL is the only active instrument — all thresholds are SOLUSDT.
_T = THRESHOLDS["SOLUSDT"]


class EMAStrategy(Strategy):
    """Dynamic EMA Crossover momentum strategy."""

    def __init__(self) -> None:
        pass

    def generate_signal(self, state: CausalState, params: dict) -> str:
        """Compute EMA state-based signal from live price history.

        §10.2 HOLD-trap fix: checks current state, not exact-instant crossover.
        BUY when fast EMA is currently above slow EMA (trend is up right now).
        SELL when fast EMA is currently below slow EMA (trend is down right now).
        """
        slow_period = params.get("slow_period", 21)
        if len(state.price_history) < slow_period + 1:
            return "HOLD"

        closes = pd.Series(state.price_history)
        fast = ta.ema(closes, length=params["fast_period"])
        slow = ta.ema(closes, length=params["slow_period"])

        if fast is None or slow is None or len(fast) < 1 or len(slow) < 1:
            return "HOLD"

        fast_now = fast.iloc[-1]
        slow_now = slow.iloc[-1]

        if pd.isna(fast_now) or pd.isna(slow_now):
            return "HOLD"

        # State-based: is the trend currently up or down?
        if fast_now > slow_now:
            return "BUY"
        if fast_now < slow_now:
            return "CLOSE"
        return "HOLD"

    def get_assumptions(self, state: CausalState, params: dict) -> list[Assumption]:
        """Return exactly 3 assumptions for the EMA Crossover strategy.

        S3-3: current_value and proximity are intentionally 0.0 (Pydantic defaults).
        core/decision_utils.annotate_proximity() is the single source of truth for
        proximity computation. This ensures the H4 SLOPE_NORMALISER special case
        (threshold=0.0 with operator='gt') lives in one place and cannot diverge
        between what is committed to the ledger and what ML2 trains on.
        """
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
                threshold=_T["volatility_limit"],
            ),
            Assumption(
                name="spread_constraint",
                variable="spread",
                operator="lt",
                threshold=_T["spread_limit"],
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
        """Score this parameter configuration for hill-climbing.

        PURE FUNCTION — does NOT call generate_signal() and does NOT modify
        self._price_history. Hill-climbing calls evaluate() up to 100+ times
        per decision cycle; calling generate_signal() here would append the
        same price repeatedly, destroying the history buffer.

        All thresholds use SOLUSDT constants — SOL is the only active instrument.

        S3-4: The vol denominator uses volatility_limit * 1.5 (=0.105) intentionally.
        EMA can tolerate slightly elevated volatility — it scores 0.0 at 10.5%, not 7%.
        Assumption breach still fires at the hard limit (7%). The extra tolerance
        prevents hill-climbing from zeroing out EMA in mildly elevated vol conditions
        where EMA signals are still reliable.
        """
        if state.trend_strength < 0.05:
            return 0.0

        score = (
            state.trend_strength
            * (1 - min(1.0, state.spread / _T["spread_limit"]))
            * (1 - min(1.0, state.volatility / (_T["volatility_limit"] * 1.5)))
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
