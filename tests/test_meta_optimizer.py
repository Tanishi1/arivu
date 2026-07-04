"""tests/test_meta_optimizer.py
Unit tests for ml/meta_optimizer.py
"""

from __future__ import annotations

import os
import sqlite3
import pytest
from ml.meta_optimizer import (
    MetaParameterOptimizer, MetaParams, TradeOutcome, RegimeType,
    MIN_STEP_K, MIN_STEP_THRESH, STAGNATION_LIMIT, IMMUNITY_CYCLES,
)

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "arivu.db"
    return str(db_path)


def _dummy_outcome(cand, regime="calm"):
    return TradeOutcome(
        meta_params_used=cand.to_dict(),
        regime=regime,
        predicted_return=0.01,
        actual_return=0.01,
        pnl_usd=10.0,
        causal_chain_held=True,
        regime_was_stable=True,
    )


def test_meta_optimizer_initialization(temp_db):
    opt = MetaParameterOptimizer(db_path=temp_db)
    for regime in RegimeType:
        assert regime.value in opt._state
        assert len(opt._state[regime.value]["population"]) == 3
        assert opt._state[regime.value]["history"] == []


def test_meta_optimizer_get_params(temp_db):
    opt = MetaParameterOptimizer(db_path=temp_db)
    params = opt.get_params("calm")
    assert isinstance(params, MetaParams)
    assert params.regime == "calm"


def test_meta_optimizer_stagnation_replacement(temp_db):
    opt = MetaParameterOptimizer(db_path=temp_db)
    regime = "calm"

    bucket = opt._state[regime]
    population = bucket["population"]
    assert len(population) == 3

    # Force best candidate to minimum step sizes so stagnation increments
    best_cand = population[0]
    best_cand.step_k = MIN_STEP_K
    best_cand.step_thresh = MIN_STEP_THRESH

    # Force worst candidate to non-minimum step sizes (realistic scenario)
    worst_cand = population[2]
    worst_cand.step_k = 10
    worst_cand.step_thresh = 0.08

    for _ in range(STAGNATION_LIMIT):
        opt.update(_dummy_outcome(best_cand, regime))

    new_population = opt._state[regime]["population"]
    assert len(new_population) == 3
    # Stagnation counter resets to 0 after replacement fires
    assert opt._stagnation_counts.get(regime, 0) == 0


def test_stagnation_grants_immunity_after_replacement(temp_db):
    """After replacement fires, immunity must be set so the new random candidate
    cannot be immediately evicted on the very next stagnation cycle."""
    opt = MetaParameterOptimizer(db_path=temp_db)
    regime = "calm"

    population = opt._state[regime]["population"]
    best_cand = population[0]
    best_cand.step_k = MIN_STEP_K
    best_cand.step_thresh = MIN_STEP_THRESH

    # Run enough updates to trigger replacement
    for _ in range(STAGNATION_LIMIT):
        opt.update(_dummy_outcome(best_cand, regime))

    # The immunity decrement runs at the TOP of update() before replacement fires.
    # On the call that triggers replacement, the decrement ran on the previous value
    # (which was 0 → stays 0), then replacement sets immunity = IMMUNITY_CYCLES.
    # So the value immediately after replacement is exactly IMMUNITY_CYCLES.
    immunity_after = opt._candidate_immunity.get(regime, 0)
    assert immunity_after == IMMUNITY_CYCLES, (
        f"Expected immunity={IMMUNITY_CYCLES} immediately after replacement, "
        f"got {immunity_after}"
    )


def test_immunity_suppresses_second_replacement(temp_db):
    """If stagnation fires again while immunity > 0, the worst candidate must
    NOT be replaced — population stays intact until immunity expires."""
    opt = MetaParameterOptimizer(db_path=temp_db)
    regime = "calm"

    population = opt._state[regime]["population"]
    best_cand = population[0]
    best_cand.step_k = MIN_STEP_K
    best_cand.step_thresh = MIN_STEP_THRESH

    # Trigger first replacement
    for _ in range(STAGNATION_LIMIT):
        opt.update(_dummy_outcome(best_cand, regime))

    assert opt._stagnation_counts.get(regime, 0) == 0  # reset after replacement
    assert opt._candidate_immunity.get(regime, 0) > 0  # immunity still active

    # Force the new best candidate to also stagnate immediately
    new_pop = opt._state[regime]["population"]
    new_best = new_pop[0]
    new_best.step_k = MIN_STEP_K
    new_best.step_thresh = MIN_STEP_THRESH

    # Run another STAGNATION_LIMIT updates while immunity is still active
    for _ in range(STAGNATION_LIMIT):
        opt.update(_dummy_outcome(new_best, regime))

    # Population must still be intact — no second replacement while immune
    assert len(opt._state[regime]["population"]) == 3
