"""tests/test_week4.py
Week 4 test suite: strategies + simulator.

All tests use manually constructed CausalState objects.
No live data. No network calls. No database access.

Key design rules:
  - CausalState is frozen=True — all fields set via constructor, not attribute assignment.
  - Only SOLUSDT is active — thresholds use SOLUSDT constants.
  - Simulator uses class names: "EMAStrategy", "BollingerStrategy", "RSIStrategy".
"""

from __future__ import annotations

import pytest

from core.schemas import CausalState
from core.constants import THRESHOLDS, SLOPE_NORMALISER
from core.decision_utils import annotate_proximity
from strategies.ema_crossover import EMAStrategy
from strategies.bollinger import BollingerStrategy
from strategies.rsi_divergence import RSIStrategy

_T = THRESHOLDS["SOLUSDT"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(
    price: float = 150.0,
    volatility: float = 0.02,
    spread: float = 0.002,
    trend_slope: float = 0.005,
    trend_strength: float = 0.7,
    volume: float = 1000.0,
    rsi_current: float = 50.0,
    divergence_candle_span: float = 3.5,
) -> CausalState:
    """Build a CausalState for testing.

    CausalState is frozen=True — cannot set attributes after creation.
    All values default to a "trending SOLUSDT market" profile.
    """
    return CausalState(
        price=price,
        volatility=volatility,
        spread=spread,
        trend_slope=trend_slope,
        trend_strength=trend_strength,
        volume=volume,
        rsi_current=rsi_current,
        divergence_candle_span=divergence_candle_span,
    )


def make_simulator(regime_classifier=None):
    """Create a Simulator with all 3 SOL strategies and no regime classifier."""
    from core.simulator import Simulator
    strategies = [EMAStrategy(), BollingerStrategy(), RSIStrategy()]
    return Simulator(strategies=strategies, regime_classifier=regime_classifier)


# ---------------------------------------------------------------------------
# Test 1 — EMAStrategy: get_assumptions returns exactly 3 named assumptions
# ---------------------------------------------------------------------------

def test_ema_get_assumptions_returns_three():
    strategy = EMAStrategy()
    state = make_state()
    assumptions = strategy.get_assumptions(state, strategy.get_default_params())

    assert len(assumptions) == 3, f"Expected 3 assumptions, got {len(assumptions)}"
    names = [a.name for a in assumptions]
    assert "trend_persistence" in names
    assert "volatility_control" in names
    assert "spread_constraint" in names


# ---------------------------------------------------------------------------
# Test 2 — EMA: trend_persistence proximity is NOT always 1.0 (H4 fix verify)
# ---------------------------------------------------------------------------

def test_ema_trend_persistence_proximity_not_always_one():
    """Strong trend → low proximity (safe). Weak trend → high proximity (breach risk).

    S3-3 DESIGN: Proximity is computed by annotate_proximity() in core/decision_utils,
    NOT by strategy.get_assumptions(). This test imports and calls annotate_proximity()
    directly to verify the H4 SLOPE_NORMALISER special case lives in one place.
    """
    strategy = EMAStrategy()
    strong_state = make_state(trend_slope=0.01)   # slope = SLOPE_NORMALISER → proximity ≈ 0.0
    weak_state   = make_state(trend_slope=0.001)  # slope = 10% of SLOPE_NORMALISER → proximity ≈ 0.9

    # annotate_proximity() is the single source of truth — strategies return proximity=0.0
    strong_assumptions = annotate_proximity(
        strategy.get_assumptions(strong_state, strategy.get_default_params()),
        strong_state,
    )
    weak_assumptions = annotate_proximity(
        strategy.get_assumptions(weak_state, strategy.get_default_params()),
        weak_state,
    )

    a_strong = next(a for a in strong_assumptions if a.name == "trend_persistence")
    a_weak   = next(a for a in weak_assumptions   if a.name == "trend_persistence")

    assert a_strong.proximity < a_weak.proximity, (
        f"FAIL: strong trend should have lower proximity than weak trend. "
        f"Got strong={a_strong.proximity:.4f}, weak={a_weak.proximity:.4f}"
    )
    assert a_strong.proximity < 0.5, (
        f"FAIL: strong trend proximity should be below 0.5, got {a_strong.proximity:.4f}"
    )
    assert a_weak.proximity > 0.5, (
        f"FAIL: weak trend proximity should be above 0.5, got {a_weak.proximity:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — BollingerStrategy: get_assumptions returns exactly 3
# ---------------------------------------------------------------------------

def test_bollinger_get_assumptions_returns_three():
    strategy = BollingerStrategy()
    state = make_state(trend_strength=0.1, volatility=0.02)
    assumptions = strategy.get_assumptions(state, strategy.get_default_params())

    assert len(assumptions) == 3, f"Expected 3 assumptions, got {len(assumptions)}"
    names = [a.name for a in assumptions]
    assert "no_active_trend" in names
    assert "volatility_normal" in names
    assert "spread_within_tolerance" in names


# ---------------------------------------------------------------------------
# Test 4 — RSIStrategy: get_assumptions returns exactly 3
# ---------------------------------------------------------------------------

def test_rsi_get_assumptions_returns_three():
    strategy = RSIStrategy()
    state = make_state(rsi_current=50.0, divergence_candle_span=5.0)
    assumptions = strategy.get_assumptions(state, strategy.get_default_params())

    assert len(assumptions) == 3, f"Expected 3 assumptions, got {len(assumptions)}"
    names = [a.name for a in assumptions]
    assert "divergence_span" in names
    assert "volatility_acceptable" in names
    assert "rsi_not_extreme" in names


# ---------------------------------------------------------------------------
# Test 5 — C_NEW_1 fix: divergence_candle_span default does NOT cause breach
# ---------------------------------------------------------------------------

def test_rsi_divergence_span_default_above_threshold():
    """CausalState default divergence_candle_span (3.5) must be > threshold (3.0).

    Before C_NEW_1 fix: default was 0.0 → 0.0 <= 3.0 → immediate breach
    on first monitor cycle → every RSI cycle closes within 5s → corrupt data.

    S3-3: After removing proximity from get_assumptions(), current_value is 0.0.
    This test now verifies C_NEW_1 by checking CausalState's actual default field
    value directly (the ground truth), and separately verifies the assumption
    threshold is correctly set to 3.0.
    """
    strategy = RSIStrategy()
    # Use default divergence_candle_span (3.5 from CausalState schema default)
    state = CausalState(price=150.0, volume=1000.0)

    # C_NEW_1: CausalState's actual default must be > 3.0 to avoid startup breach
    assert state.divergence_candle_span > 3.0, (
        f"FAIL: CausalState.divergence_candle_span default ({state.divergence_candle_span}) "
        f"is not > 3.0. C_NEW_1 fix not applied — every RSI cycle will breach within 5s."
    )
    assert state.divergence_candle_span == 3.5, (
        f"FAIL: expected default 3.5, got {state.divergence_candle_span}"
    )

    # Verify the assumption threshold is 3.0 (the breach level)
    div_assumption = next(
        a for a in strategy.get_assumptions(state, strategy.get_default_params())
        if a.name == "divergence_span"
    )
    assert div_assumption.threshold == 3.0, (
        f"FAIL: divergence_span threshold should be 3.0, got {div_assumption.threshold}"
    )
    assert div_assumption.operator == "gt", (
        f"FAIL: divergence_span operator should be 'gt', got {div_assumption.operator}"
    )


# ---------------------------------------------------------------------------
# Test 6 — evaluate() is a pure function: no mutation, deterministic
# ---------------------------------------------------------------------------

def test_evaluate_is_pure_no_mutation():
    """Hill-climbing calls evaluate() up to 100× per cycle.

    If evaluate() mutates _price_history, duplicate prices destroy the
    EMA crossover signal permanently after the first cycle.
    """
    strategy = EMAStrategy()
    # Seed price history with realistic values
    for price in range(100, 160):
        strategy._price_history.append(float(price))

    history_before = list(strategy._price_history)
    state = make_state(price=160.0, trend_strength=0.7)
    params = strategy.get_default_params()

    # Call evaluate() 10 times, as Hill-climbing would
    scores = [strategy.evaluate(state, params) for _ in range(10)]

    history_after = list(strategy._price_history)

    assert history_before == history_after, (
        f"FAIL: evaluate() mutated _price_history. "
        f"Before length: {len(history_before)}, after: {len(history_after)}"
    )
    assert len(set(scores)) == 1, (
        f"FAIL: evaluate() is not deterministic. Got scores: {scores}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Simulator Stage 1: selects EMA in trending market
# ---------------------------------------------------------------------------

def test_simulator_stage1_selects_ema_in_trending_market():
    """Strong trend (0.85), low volatility → EMA Crossover should win Stage 1."""
    sim = make_simulator()
    trending_state = make_state(
        trend_strength=0.85,
        trend_slope=0.008,
        volatility=0.01,
        spread=0.001,
    )
    result = sim.run(trending_state)
    assert result.strategy_name == "EMAStrategy", (
        f"FAIL: expected EMAStrategy in trending market, got {result.strategy_name}"
    )


# ---------------------------------------------------------------------------
# Test 8 — Simulator Stage 1: selects RSI in volatile market
# ---------------------------------------------------------------------------

def test_simulator_stage1_selects_rsi_in_ambiguous_market():
    """Moderate trend (0.45) + high volatility (0.09) → RSIStrategy should win Stage 1.

    RSI is the 'uncertainty hedge': it wins when the market is neither clearly
    trending (EMA's domain) nor clearly calm/ranging (Bollinger's domain).
    With trend=0.45 and vol above the SOLUSDT limit:
      EMA-comfort = 0.45*0.5 + (1-1.0)*0.3 = 0.225    (penalised by high vol)
      Bol-comfort = 0.55*0.5 + 1.0*0.3    = 0.575    (penalised by moderate trend)
      RSI uncertainty = 1 - (0.225+0.575)/2 = 0.6     → RSI ≈ 0.80 (wins)
    """
    sim = make_simulator()
    ambiguous_state = make_state(
        volatility=0.09,       # above SOLUSDT limit 0.07 → high vol
        trend_strength=0.45,   # moderate: not clearly trending or ranging
        trend_slope=0.003,
        spread=0.004,
    )
    result = sim.run(ambiguous_state)
    assert result.strategy_name == "RSIStrategy", (
        f"FAIL: expected RSIStrategy in ambiguous market, got {result.strategy_name}"
    )



# ---------------------------------------------------------------------------
# Test 9 — Hill-climbing: all tuned parameters within declared bounds
# ---------------------------------------------------------------------------

def test_hillclimbing_respects_parameter_bounds():
    sim = make_simulator()
    state = make_state(trend_strength=0.7)
    result = sim.run(state)

    strategy = sim._strategies[result.strategy_name]
    bounds = strategy.get_parameter_bounds()

    for param_name, bound in bounds.items():
        val = result.tuned_params[param_name]
        assert bound["min"] <= val <= bound["max"], (
            f"FAIL: {param_name}={val} out of bounds "
            f"[{bound['min']}, {bound['max']}] for {result.strategy_name}"
        )


# ---------------------------------------------------------------------------
# Test 10 — Hill-climbing: logs at least 1 iteration (Research Metric 3)
# ---------------------------------------------------------------------------

def test_hillclimbing_logs_iterations():
    sim = make_simulator()
    state = make_state()
    result = sim.run(state)

    assert result.hill_climb_iterations >= 1, (
        f"FAIL: hill_climb_iterations must be >= 1, got {result.hill_climb_iterations}"
    )
    assert isinstance(result.hill_climb_iterations, int)


# ---------------------------------------------------------------------------
# Test 11 — SimulatorResult has exactly 3 assumptions
# ---------------------------------------------------------------------------

def test_simulator_result_has_three_assumptions():
    sim = make_simulator()
    state = make_state()
    result = sim.run(state)

    assert len(result.assumptions) == 3, (
        f"FAIL: expected 3 assumptions in result, got {len(result.assumptions)}"
    )


# ---------------------------------------------------------------------------
# Test 12 — All strategy parameter bounds are valid (step > 0, min < max)
# ---------------------------------------------------------------------------

def test_parameter_bounds_have_valid_steps():
    for StratClass in [EMAStrategy, BollingerStrategy, RSIStrategy]:
        s = StratClass()
        for param_name, bound in s.get_parameter_bounds().items():
            assert bound.get("step", 0) > 0, (
                f"FAIL: {StratClass.__name__}.{param_name} has step <= 0"
            )
            assert bound["min"] < bound["max"], (
                f"FAIL: {StratClass.__name__}.{param_name} has min >= max"
            )
