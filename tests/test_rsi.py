"""tests/test_rsi.py  —  Person 3
Unit tests for strategies/rsi_divergence.py

Five test cases per impl plan E48:
  1. test_generate_signal_hold_no_history
  2. test_generate_signal_hold_neutral
  3. test_get_assumptions_count
  4. test_evaluate_returns_float
  5. test_evaluate_penalises_breach
"""

from __future__ import annotations

import pytest

from core.schemas import CausalState
from strategies.rsi_divergence import RSIStrategy


# NOTE: No module-level _PRICE_HISTORY fixture needed.
# RSIStrategy.__init__() creates a fresh self._price_history per instance.

@pytest.fixture
def strategy():
    return RSIStrategy()


def _uncertain_state() -> CausalState:
    """Uncertain/ranging market — RSI divergence strategy's home regime."""
    return CausalState(
        price=50000.0, volatility=0.025, spread=0.0012,
        trend_slope=0.001, trend_strength=0.3, volume=25000,
        algo_health_vector=[0.85, 0.12, 0.03],
    )


def _high_vol_state() -> CausalState:
    return CausalState(
        price=50000.0, volatility=0.06, spread=0.003,
        trend_slope=0.002, trend_strength=0.4, volume=20000,
        algo_health_vector=[0.7, 0.2, 0.1],
    )


# ---------------------------------------------------------------------------

def test_generate_signal_hold_no_history(strategy):
    """With no price history, generate_signal must return HOLD."""
    params = strategy.get_default_params()
    signal = strategy.generate_signal(_uncertain_state(), params)
    assert signal == "HOLD", f"Expected HOLD with no history, got {signal}"


def test_generate_signal_hold_neutral(strategy):
    """Neutral price action should return HOLD even with some history."""
    params = strategy.get_default_params()
    # Feed a flat sequence (no divergence pattern)
    flat_prices = [50000.0] * 30
    for p in flat_prices:
        state = CausalState(price=p, volatility=0.02, spread=0.001,
                            trend_slope=0.0, trend_strength=0.2, volume=30000,
                            algo_health_vector=[0.9, 0.08, 0.02])
        signal = strategy.generate_signal(state, params)

    assert signal == "HOLD", "Flat prices should not generate a divergence signal"


def test_get_assumptions_count(strategy):
    """get_assumptions() must return exactly 3 assumptions."""
    assumptions = strategy.get_assumptions(_uncertain_state(), strategy.get_default_params())
    assert len(assumptions) == 3


def test_evaluate_returns_float(strategy):
    """evaluate() must return a float for any valid state."""
    result = strategy.evaluate(_uncertain_state(), strategy.get_default_params())
    assert isinstance(result, float)


def test_evaluate_penalises_breach(strategy):
    """With high volatility (proximity > 0.8 for volatility_acceptable), score should be reduced."""
    params = strategy.get_default_params()

    # Safe: volatility=0.02, threshold=0.05 → proximity=0.4 (no penalty)
    safe_state = CausalState(
        price=50000.0, volatility=0.02, spread=0.001,
        trend_slope=0.001, trend_strength=0.3, volume=25000,
        algo_health_vector=[0.9, 0.08, 0.02],
    )
    # Near breach: volatility=0.048, threshold=0.05 → proximity=0.96 (penalty ×0.3 applies)
    breach_state = CausalState(
        price=50000.0, volatility=0.048, spread=0.002,
        trend_slope=0.001, trend_strength=0.4, volume=20000,
        algo_health_vector=[0.7, 0.2, 0.1],
    )

    score_safe = strategy.evaluate(safe_state, params)
    score_breach = strategy.evaluate(breach_state, params)

    assert score_safe >= score_breach, (
        f"Safe score ({score_safe:.6f}) should be >= near-breach score ({score_breach:.6f})"
    )
