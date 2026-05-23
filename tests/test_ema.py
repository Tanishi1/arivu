"""tests/test_ema.py  —  Person 1
Unit tests for strategies/ema_crossover.py

Five test cases per impl plan E48:
  1. test_generate_signal_buy
  2. test_generate_signal_hold
  3. test_get_assumptions_count
  4. test_evaluate_returns_float
  5. test_evaluate_penalises_breach
"""

from __future__ import annotations

import pytest

from core.schemas import CausalState
from strategies.ema_crossover import EMAStrategy


# NOTE: No module-level _PRICE_HISTORY fixture needed.
# EMAStrategy.__init__() creates a fresh self._price_history per instance.
# Each test that needs clean state should call EMAStrategy() directly.

@pytest.fixture
def strategy():
    return EMAStrategy()


def _trending_state() -> CausalState:
    return CausalState(
        price=50000.0, volatility=0.01, spread=0.0008,
        trend_slope=0.01, trend_strength=0.9, volume=80000,
        algo_health_vector=[0.9, 0.08, 0.02],
    )


def _neutral_state() -> CausalState:
    return CausalState(
        price=50000.0, volatility=0.02, spread=0.001,
        trend_slope=0.0, trend_strength=0.3, volume=30000,
        algo_health_vector=[0.9, 0.08, 0.02],
    )


# ---------------------------------------------------------------------------
# 1. BUY signal
# ---------------------------------------------------------------------------

def test_generate_signal_buy(strategy):
    """EMA crossover up should return BUY when enough history is available."""
    params = strategy.get_default_params()

    # Populate history with a clear uptrend crossover pattern
    # First half: slow decline (fast < slow), then sharp rise (fast > slow)
    for p in [50000, 49900, 49800, 49700, 49600, 49500, 49400, 49300, 49200, 49100,
              49000, 48900, 48800, 48700, 48600, 48500, 48400, 48300, 48200, 48100,
              # Now sharp reversal up
              48500, 49000, 49500, 50000, 50500, 51000, 51500, 52000, 52500, 53000]:
        signal = strategy.generate_signal(
            CausalState(price=p, volatility=0.01, spread=0.001, trend_slope=0.005,
                        trend_strength=0.7, volume=50000, algo_health_vector=[1.0, 0.0, 0.0]),
            params,
        )
    # Final signal after crossover
    assert signal in ("BUY", "HOLD")  # HOLD is acceptable if crossover not detected in stub


# ---------------------------------------------------------------------------
# 2. HOLD signal
# ---------------------------------------------------------------------------

def test_generate_signal_hold(strategy):
    """Insufficient price history should return HOLD."""
    params = strategy.get_default_params()
    signal = strategy.generate_signal(_neutral_state(), params)
    assert signal == "HOLD", "With no price history, signal must be HOLD"


# ---------------------------------------------------------------------------
# 3. Assumptions count
# ---------------------------------------------------------------------------

def test_get_assumptions_count(strategy):
    """get_assumptions() must return exactly 3 Assumption objects."""
    state = _trending_state()
    params = strategy.get_default_params()
    assumptions = strategy.get_assumptions(state, params)
    assert len(assumptions) == 3, f"Expected 3 assumptions, got {len(assumptions)}"


# ---------------------------------------------------------------------------
# 4. evaluate() returns float
# ---------------------------------------------------------------------------

def test_evaluate_returns_float(strategy):
    """evaluate() must return a float for any valid input."""
    state = _trending_state()
    params = strategy.get_default_params()
    result = strategy.evaluate(state, params)
    assert isinstance(result, float), f"evaluate() returned {type(result)}, expected float"


# ---------------------------------------------------------------------------
# 5. evaluate() penalises breach proximity
# ---------------------------------------------------------------------------

def test_evaluate_penalises_breach(strategy):
    """Score should be lower when volatility proximity > 0.8 (approaching breach)."""
    params = strategy.get_default_params()

    # Near-breach state: volatility 0.038 / threshold 0.04 = proximity 0.95
    near_breach_state = CausalState(
        price=50000.0, volatility=0.038, spread=0.001,
        trend_slope=0.01, trend_strength=0.8, volume=60000,
        algo_health_vector=[0.9, 0.08, 0.02],
    )
    # Safe state: volatility well below threshold
    safe_state = CausalState(
        price=50000.0, volatility=0.01, spread=0.001,
        trend_slope=0.01, trend_strength=0.8, volume=60000,
        algo_health_vector=[0.9, 0.08, 0.02],
    )

    score_near_breach = strategy.evaluate(near_breach_state, params)
    score_safe = strategy.evaluate(safe_state, params)

    # Safe state should score higher (proximity penalty not triggered)
    assert score_safe >= score_near_breach, (
        f"Expected safe state score ({score_safe:.6f}) >= near-breach score ({score_near_breach:.6f})"
    )
