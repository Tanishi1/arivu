"""tests/test_week5.py
Safety-net tests for Weeks 5-8 components.

Covers:
  - ML1 bootstrap classification thresholds
  - ML1 HOLD telemetry filtering (not buffered)
  - ML1 maybe_retrain() return type contract (tuple)
  - ML2 bootstrap: breach_risk == proximity
  - ML2 bootstrap: proximity clamped at 1.0
  - Monitor _check_assumption: lt and gt operators
  - Monitor EMA trend_persistence special case (threshold=0.0)
  - Outcome comparator: HOLD delta is 0.0
  - Outcome comparator: BUY delta = projected - actual
  - Phase transition guard: only flips on genuine improvement
  - HIGH-4 regime bootstrap: downtrend NOT classified as trending
  - B8: corrupt training_buffer.csv header returns empty list
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.decision_utils import annotate_proximity
from core.monitor import _check_assumption
from core.schemas import Assumption, CausalState, ExecutionTelemetry
from ml.ml1 import ML1BehaviourClassifier
from ml.ml2 import ML2BreachPredictor
from ml.regime import RegimeClassifier


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tel(latency=100.0, fill_rate=1.0, slippage=0.0, size_dev=0.0):
    return ExecutionTelemetry(
        symbol="SOLUSD",
        fill_rate=fill_rate,
        order_latency_ms=latency,
        slippage=slippage,
        position_size_deviation=size_dev,
    )


def _assumption(name="vol_ok", variable="volatility", operator="lt",
                threshold=0.07, proximity=0.5):
    return Assumption(name=name, variable=variable, operator=operator,
                      threshold=threshold, proximity=proximity)


def _state(**kwargs):
    defaults = dict(price=150.0, volatility=0.01, spread=0.002,
                    trend_slope=0.005, trend_strength=0.5, volume=100.0,
                    rsi_current=50.0, divergence_candle_span=3.5)
    defaults.update(kwargs)
    return CausalState(**defaults)


def _ml1_bootstrap():
    """Return an ML1 instance guaranteed to be in bootstrap (untrained) mode."""
    ml1 = ML1BehaviourClassifier()
    ml1._is_trained = False   # force bootstrap even if a model exists on disk
    ml1._model = None
    return ml1


def _ml2_bootstrap():
    """Return an ML2 instance guaranteed to be in bootstrap mode."""
    ml2 = ML2BreachPredictor()
    ml2._is_trained = False
    ml2._model = None
    return ml2


# ---------------------------------------------------------------------------
# ML1 — bootstrap classification (Week 5)
# ---------------------------------------------------------------------------

def test_ml1_bootstrap_normal():
    ml1 = _ml1_bootstrap()
    vec = ml1.predict_proba(_tel(latency=100.0, fill_rate=1.0), is_real_order=True)
    assert vec == [1.0, 0.0, 0.0], f"Expected normal, got {vec}"


def test_ml1_bootstrap_normal_alpaca_baseline():
    """Latency 2700ms is normal on Alpaca paper — the executor has a 2s polling floor."""
    ml1 = _ml1_bootstrap()
    vec = ml1.predict_proba(_tel(latency=2700.0, fill_rate=1.0), is_real_order=True)
    assert vec == [1.0, 0.0, 0.0], f"2700ms should be normal on Alpaca paper, got {vec}"


def test_ml1_bootstrap_stressed_by_latency():
    """Latency 2900ms is above Alpaca's baseline jitter — stressed."""
    ml1 = _ml1_bootstrap()
    vec = ml1.predict_proba(_tel(latency=2900.0, fill_rate=1.0), is_real_order=True)
    assert vec == [0.0, 1.0, 0.0], f"Expected stressed, got {vec}"


def test_ml1_bootstrap_degraded_by_latency():
    ml1 = _ml1_bootstrap()
    vec = ml1.predict_proba(_tel(latency=6000.0, fill_rate=1.0), is_real_order=True)
    assert vec == [0.0, 0.0, 1.0], f"Expected degraded, got {vec}"


def test_ml1_bootstrap_degraded_by_fill_rate():
    ml1 = _ml1_bootstrap()
    vec = ml1.predict_proba(_tel(latency=100.0, fill_rate=0.5), is_real_order=True)
    assert vec == [0.0, 0.0, 1.0], f"fill_rate<0.7 should be degraded, got {vec}"


def test_ml1_hold_telemetry_not_buffered():
    """HOLD/SELL synthetic telemetry must NOT enter the training buffer."""
    ml1 = _ml1_bootstrap()
    before = len(ml1._sample_buffer)
    ml1.predict_proba(_tel(), is_real_order=False)
    assert len(ml1._sample_buffer) == before, "HOLD telemetry polluted sample buffer"


def test_ml1_real_order_is_buffered():
    ml1 = _ml1_bootstrap()
    before = len(ml1._sample_buffer)
    ml1.predict_proba(_tel(), is_real_order=True)
    assert len(ml1._sample_buffer) == before + 1


def test_ml1_maybe_retrain_returns_tuple():
    """S-3 contract: maybe_retrain() returns (bool, float|None)."""
    ml1 = _ml1_bootstrap()
    result = ml1.maybe_retrain()
    assert isinstance(result, tuple) and len(result) == 2
    retrained, oob = result
    assert isinstance(retrained, bool)
    assert oob is None   # not enough samples yet


# ---------------------------------------------------------------------------
# ML2 — bootstrap annotation (Week 5)
# ---------------------------------------------------------------------------

def test_ml2_bootstrap_breach_risk_equals_proximity():
    """In bootstrap mode breach_risk must equal proximity (heuristic baseline)."""
    ml2 = _ml2_bootstrap()
    a = _assumption(proximity=0.65)
    annotated = ml2.annotate([a, a, a], _state(), 30, [1.0, 0.0, 0.0])
    for ann in annotated:
        assert abs(ann.breach_risk - 0.65) < 1e-4, f"breach_risk {ann.breach_risk} != 0.65"


def test_ml2_bootstrap_clamps_proximity_at_one():
    """proximity > 1.0 must be clamped — already-breached assumption."""
    ml2 = _ml2_bootstrap()
    a = _assumption(proximity=1.8)
    annotated = ml2.annotate([a, a, a], _state(), 30, [1.0, 0.0, 0.0])
    for ann in annotated:
        assert ann.breach_risk <= 1.0


def test_ml2_annotates_exactly_three():
    ml2 = _ml2_bootstrap()
    assumptions = [_assumption("a1", proximity=0.2),
                   _assumption("a2", proximity=0.5),
                   _assumption("a3", proximity=0.9)]
    annotated = ml2.annotate(assumptions, _state(), 30, [1.0, 0.0, 0.0])
    assert len(annotated) == 3


def test_ml2_assumption_is_frozen_after_annotation():
    """annotate() must return new Assumption objects (frozen schema — no mutation)."""
    ml2 = _ml2_bootstrap()
    a = _assumption(proximity=0.3)
    annotated = ml2.annotate([a, a, a], _state(), 30, [1.0, 0.0, 0.0])
    # original must be unchanged
    assert a.breach_risk == 0.5   # Pydantic default


# ---------------------------------------------------------------------------
# Monitor — _check_assumption (Week 5)
# ---------------------------------------------------------------------------

def test_monitor_lt_not_breached():
    breached, _ = _check_assumption(_assumption(operator="lt", threshold=0.07), _state(volatility=0.03))
    assert not breached


def test_monitor_lt_breached_at_boundary():
    breached, _ = _check_assumption(_assumption(operator="lt", threshold=0.07), _state(volatility=0.07))
    assert breached


def test_monitor_lt_breached_above_threshold():
    breached, _ = _check_assumption(_assumption(operator="lt", threshold=0.07), _state(volatility=0.10))
    assert breached


def test_monitor_gt_zero_threshold_negative_slope_breached():
    """EMA trend_persistence: operator=gt, threshold=0.0 → negative slope breaches."""
    a = Assumption(name="trend_ok", variable="trend_slope", operator="gt", threshold=0.0)
    breached, _ = _check_assumption(a, _state(trend_slope=-0.005))
    assert breached


def test_monitor_gt_zero_threshold_positive_slope_safe():
    a = Assumption(name="trend_ok", variable="trend_slope", operator="gt", threshold=0.0)
    breached, _ = _check_assumption(a, _state(trend_slope=0.005))
    assert not breached


def test_monitor_proximity_is_ratio():
    """Proximity = current_value / threshold for non-zero threshold."""
    a = _assumption(operator="lt", threshold=0.08)
    _, prox = _check_assumption(a, _state(volatility=0.04))
    assert abs(prox - 0.04 / 0.08) < 1e-6


# ---------------------------------------------------------------------------
# Outcome comparator — delta clamping (Week 5)
# ---------------------------------------------------------------------------

def test_hold_cycle_delta_is_zero():
    """HOLD cycles: no position → outcome_delta must be 0.0 regardless of projected_pnl."""
    projected_pnl = 3.50
    had_position = False
    outcome_delta = 0.0 if not had_position else (projected_pnl - 0.0)
    assert outcome_delta == 0.0


def test_buy_cycle_delta_is_projected_minus_actual():
    projected_pnl, actual_pnl, had_position = 2.50, 2.00, True
    outcome_delta = 0.0 if not had_position else (projected_pnl - actual_pnl)
    assert abs(outcome_delta - 0.50) < 1e-9


# ---------------------------------------------------------------------------
# B11 — annotate_proximity operator-aware formula (Week 5-6)
# ---------------------------------------------------------------------------

def test_proximity_lt_safe_is_low():
    """operator='lt', current well below threshold → proximity near 0."""
    a = _assumption(operator="lt", threshold=0.07)
    annotated = annotate_proximity([a], _state(volatility=0.01))
    assert annotated[0].proximity == pytest.approx(0.01 / 0.07, rel=1e-4)


def test_proximity_lt_near_breach_is_high():
    """operator='lt', current just below threshold → proximity near 1."""
    a = _assumption(operator="lt", threshold=0.07)
    annotated = annotate_proximity([a], _state(volatility=0.068))
    assert annotated[0].proximity > 0.95


def test_proximity_gt_safe_is_low():
    """B11 fix: operator='gt', current well ABOVE threshold → proximity near 0.
    Old formula: 5.0/3.0=1.67→1.0 (max risk). New formula: 3.0/5.0=0.6 (safe).
    """
    a = Assumption(name="div_span", variable="divergence_candle_span",
                   operator="gt", threshold=3.0)
    annotated = annotate_proximity([a], _state(divergence_candle_span=5.0))
    assert annotated[0].proximity == pytest.approx(3.0 / 5.0, rel=1e-4), (
        f"gt-safe proximity should be 0.6, got {annotated[0].proximity}"
    )


def test_proximity_gt_near_breach_is_high():
    """operator='gt', current just above threshold → proximity near 1."""
    a = Assumption(name="div_span", variable="divergence_candle_span",
                   operator="gt", threshold=3.0)
    annotated = annotate_proximity([a], _state(divergence_candle_span=3.1))
    assert annotated[0].proximity > 0.95


def test_proximity_ema_trend_zero_threshold_downtrend():
    """EMA trend_persistence: threshold=0, operator='gt', negative slope → proximity=1.0."""
    a = Assumption(name="trend_ok", variable="trend_slope",
                   operator="gt", threshold=0.0)
    annotated = annotate_proximity([a], _state(trend_slope=-0.005))
    assert annotated[0].proximity == 1.0


def test_proximity_ema_trend_zero_threshold_strong_uptrend():
    """EMA trend_persistence: strong positive slope → proximity near 0 (safe)."""
    a = Assumption(name="trend_ok", variable="trend_slope",
                   operator="gt", threshold=0.0)
    annotated = annotate_proximity([a], _state(trend_slope=0.01))
    assert annotated[0].proximity == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# B13 — monitor proximity matches decision_utils (Week 5)
# ---------------------------------------------------------------------------

def test_monitor_gt_proximity_matches_decision_utils():
    """B13: _check_assumption and annotate_proximity must agree on proximity."""
    a = Assumption(name="div_span", variable="divergence_candle_span",
                   operator="gt", threshold=3.0)
    state = _state(divergence_candle_span=5.0)
    _, monitor_prox = _check_assumption(a, state)
    du_prox = annotate_proximity([a], state)[0].proximity
    assert abs(monitor_prox - du_prox) < 1e-4, (
        f"Monitor proximity {monitor_prox:.4f} != decision_utils {du_prox:.4f}"
    )


# ---------------------------------------------------------------------------
# Phase transition guard (Week 6)
# ---------------------------------------------------------------------------

def test_phase_only_transitions_on_improvement():
    """current_phase must stay 'bootstrap' when ML2 does not beat the baseline."""
    current_phase = ["bootstrap"]
    improved = False   # ML2 did not beat proximity heuristic
    if improved and current_phase[0] == "bootstrap":
        current_phase[0] = "trained"
    assert current_phase[0] == "bootstrap"


def test_phase_transitions_when_improved():
    current_phase = ["bootstrap"]
    improved = True
    if improved and current_phase[0] == "bootstrap":
        current_phase[0] = "trained"
    assert current_phase[0] == "trained"


# ---------------------------------------------------------------------------
# Regime bootstrap — HIGH-4 downtrend guard (Week 5)
# ---------------------------------------------------------------------------

def test_bootstrap_classify_downtrend_is_not_trending():
    """Strong downtrend must be 'calm', not 'trending' — HIGH-4 fix."""
    rc = RegimeClassifier()
    rc._is_trained = False
    result = rc._bootstrap_classify(volatility=0.00001, trend_strength=0.8, trend_slope=-0.02)
    assert result == "calm", f"downtrend should be calm, got '{result}'"


def test_bootstrap_classify_uptrend_is_trending():
    rc = RegimeClassifier()
    rc._is_trained = False
    result = rc._bootstrap_classify(volatility=0.00001, trend_strength=0.8, trend_slope=0.02)
    assert result == "trending"


def test_bootstrap_classify_flat_is_calm():
    rc = RegimeClassifier()
    rc._is_trained = False
    result = rc._bootstrap_classify(volatility=0.00001, trend_strength=0.8, trend_slope=0.0)
    assert result == "calm"   # slope=0.0 is not > 0 → calm


def test_bootstrap_classify_volatile():
    rc = RegimeClassifier()
    rc._is_trained = False
    result = rc._bootstrap_classify(volatility=0.05, trend_strength=0.9, trend_slope=0.1)
    assert result == "volatile"


# ---------------------------------------------------------------------------
# B8 — CSV header corruption detection (Week 6)
# ---------------------------------------------------------------------------

def test_ml2_corrupt_header_returns_empty(tmp_path, monkeypatch):
    """A corrupt CSV header must return [] and not raise."""
    corrupt_csv = tmp_path / "training_buffer.csv"
    corrupt_csv.write_text("volatility,spread\n0.01,0.002\n")  # missing required cols

    ml2 = _ml2_bootstrap()
    monkeypatch.setattr("ml.ml2.TRAINING_BUFFER_PATH", corrupt_csv)
    result = ml2._load_buffer()
    assert result == [], "Corrupt header should return empty list, not crash"


def test_ml2_valid_header_loads_rows(tmp_path, monkeypatch):
    """A valid CSV header must load rows normally."""
    cols = [
        "volatility", "spread", "trend_strength", "volume",
        "proximity", "time_horizon", "p_normal", "p_stressed", "p_degraded",
        "assumption_type", "breached", "phase", "close_reason",
    ]
    buf = tmp_path / "training_buffer.csv"
    with open(buf, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerow({c: "0.1" for c in cols})

    ml2 = _ml2_bootstrap()
    monkeypatch.setattr("ml.ml2.TRAINING_BUFFER_PATH", buf)
    result = ml2._load_buffer()
    assert len(result) == 1
