"""strategies/bollinger.py  —  Person 2
Volatility-Banded Mean Reversion Strategy (Bollinger Bands).

Signal logic:
  BUY  if close[-1] < lower_band[-1]  (price below lower band — oversold)
  SELL if close[-1] > middle[-1] AND close[-2] <= middle[-2]  (crosses back through middle)
  HOLD otherwise

Three causal assumptions:
  1. no_active_trend       — trend_strength < 0.4  (mean reversion fails in strong trends)
  2. volatility_normal     — volatility < 0.035    (too much volatility widens bands unreliably)
  3. spread_within_tolerance — spread < 0.003      (tight spread needed for profitable reversion)

  NOTE: Assumption 3 was previously "reversion_within_window" tracking _candles_since_buy
  via trend_strength as a proxy — this was scientifically invalid (trend_strength does not
  measure time since entry). Replaced with a spread assumption that genuinely constrains
  execution quality, consistent with the other two strategies' causal model.

Evaluate scoring:
  if trend_strength > 0.5: return -0.01  (strong penalty — strategy should not run)
  if trend_strength > 0.3: return 0.0    (mild trend — signal would be HOLD)
  score = (1 - trend_strength) × penalty_for_vol_deviation × 0.012
  proximity > 0.8: multiply score × 0.4  (strictest proximity penalty)

  IMPORTANT: evaluate() is a PURE function — it does NOT call generate_signal().
  generate_signal() has side effects (_price_history, _candles_since_buy). Calling it
  inside evaluate() during hill-climbing corrupts these counters across 100+ calls.

Assigned to: Person 2
Done when: Simulator selects Bollinger in a ranging CausalState and penalties
           apply correctly in a trending state.
"""

from __future__ import annotations

import logging

import pandas as pd
import pandas_ta as ta  # noqa: F401

from core.schemas import Assumption, CausalState
from core.constants import THRESHOLDS
from strategies.base import Strategy

logger = logging.getLogger(__name__)

MAX_HISTORY = 60
# SOL is the only active instrument — all thresholds are SOLUSDT.
_T = THRESHOLDS["SOLUSDT"]


class BollingerStrategy(Strategy):
    """Volatility-Banded Mean Reversion using Bollinger Bands."""

    def __init__(self) -> None:
        self._candles_since_buy: int = 0

    def generate_signal(self, state: CausalState, params: dict) -> str:
        """Bollinger mean-reversion signal — state-based (§10.2 HOLD-trap fix).

        BUY:  price is currently below the lower band (oversold state).
        SELL: price is currently above the middle band (reversion complete state).

        FORMER BUG (SELL): checked for exact moment of crossing middle band
          (closes[-1] > middle[-1] AND closes[-2] <= middle[-2]).
          This is event-based — misses the signal unless sampled at that exact tick.
        """
        period = params.get("baseline_period", 20)
        if len(state.price_history) < period + 1:
            return "HOLD"

        closes = pd.Series(state.price_history)
        middle = ta.sma(closes, length=period)
        std = closes.rolling(period).std()
        mult = params.get("std_dev_multiplier", 2.0)

        if middle is None or std is None:
            return "HOLD"

        last_close = closes.iloc[-1]
        last_middle = middle.iloc[-1]
        last_lower = (middle - mult * std).iloc[-1]

        if pd.isna(last_middle) or pd.isna(last_lower):
            return "HOLD"

        # BUY: currently below lower band (already state-based — preserved)
        if last_close < last_lower:
            return "BUY"

        # SELL: currently above middle band (state-based fix)
        if last_close > last_middle:
            return "CLOSE"

        return "HOLD"


    def get_assumptions(self, state: CausalState, params: dict) -> list[Assumption]:
        """Return exactly 3 assumptions for the Bollinger strategy.

        S3-3: current_value and proximity are intentionally 0.0 (Pydantic defaults).
        core/decision_utils.annotate_proximity() is the single source of truth.
        """
        vol_thresh = _T["volatility_limit"] * 0.875   # 87.5% of vol limit (tighter than EMA)
        return [
            Assumption(
                name="no_active_trend",
                variable="trend_strength",
                operator="lt",
                threshold=_T["trend_strength_max"],
            ),
            Assumption(
                name="volatility_normal",
                variable="volatility",
                operator="lt",
                threshold=vol_thresh,
            ),
            Assumption(
                # Spread constraint: tight spread needed for profitable mean reversion.
                name="spread_within_tolerance",
                variable="spread",
                operator="lt",
                threshold=_T["spread_limit"],
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
        """Score this configuration. Heavy penalty in trending markets.

        PURE FUNCTION — does NOT call generate_signal() and does NOT modify
        self._price_history or self._candles_since_buy.
        All thresholds use SOLUSDT constants.
        """
        if state.trend_strength > _T["trend_strength_max"]:
            return -0.01

        # Mild trend — Bollinger signal would likely be HOLD
        if state.trend_strength > _T["trend_strength_max"] * 0.5:
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
