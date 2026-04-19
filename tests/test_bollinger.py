"""tests/test_bollinger.py  —  Person 2
Unit tests for strategies/bollinger.py

Five test cases per impl plan E48:
  1. test_generate_signal_buy
  2. test_generate_signal_hold
  3. test_get_assumptions_count
  4. test_evaluate_returns_float
  5. test_evaluate_penalises_trending
"""

from __future__ import annotations

import pytest

from core.schemas import CausalState
from strategies.bollinger import BollingerStrategy, _PRICE_HISTORY


@pytest.fixture(autouse=True)
def clear_price_history():
    _PRICE_HISTORY.clear()
    yield
    _PRICE_HISTORY.clear()


@pytest.fixture
def strategy():
    return BollingerStrategy()


def _ranging_state() -> CausalState:
    return CausalState(
        price=50000.0, volatility=0.02, spread=0.001,
        trend_slope=0.0005, trend_strength=0.15, volume=30000,
        algo_health_vector=[0.9, 0.08, 0.02],
    )


def _trending_state() -> CausalState:
    return CausalState(
        price=50000.0, volatility=0.01, spread=0.0008,
        trend_slope=0.01, trend_strength=0.85, volume=80000,
        algo_health_vector=[0.9, 0.08, 0.02],
    )


# ---------------------------------------------------------------------------

def test_generate_signal_buy(strategy):
    """Bollinger should return BUY when price drops below lower band."""
    params = strategy.get_default_params()
    # Without sufficient history, should return HOLD
    signal = strategy.generate_signal(_ranging_state(), params)
    assert signal in ("BUY", "HOLD"), f"Unexpected signal: {signal}"


def test_generate_signal_hold(strategy):
    """Bollinger should return HOLD with no price history."""
    params = strategy.get_default_params()
    signal = strategy.generate_signal(_ranging_state(), params)
    assert signal == "HOLD"


def test_get_assumptions_count(strategy):
    """get_assumptions() must return exactly 3 assumptions."""
    state = _ranging_state()
    params = strategy.get_default_params()
    assumptions = strategy.get_assumptions(state, params)
    assert len(assumptions) == 3


def test_evaluate_returns_float(strategy):
    """evaluate() must return a float."""
    result = strategy.evaluate(_ranging_state(), strategy.get_default_params())
    assert isinstance(result, float)


def test_evaluate_penalises_trending(strategy):
    """evaluate() must return -0.01 (hard penalty) when trend_strength > 0.5."""
    trending = _trending_state()  # trend_strength=0.85
    result = strategy.evaluate(trending, strategy.get_default_params())
    assert result == pytest.approx(-0.01), (
        f"Expected hard penalty -0.01 in trending state, got {result}"
    )
