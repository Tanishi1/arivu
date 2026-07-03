"""tests/test_meta_optimizer.py
Unit tests for ml/meta_optimizer.py
"""

from __future__ import annotations

import os
import sqlite3
import pytest
from ml.meta_optimizer import MetaParameterOptimizer, MetaParams, TradeOutcome, RegimeType, MIN_STEP_K, MIN_STEP_THRESH, STAGNATION_LIMIT

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "arivu.db"
    return str(db_path)

def test_meta_optimizer_initialization(temp_db):
    opt = MetaParameterOptimizer(db_path=temp_db)
    # Check that all regimes are initialized with a population of 3 candidates
    for regime in RegimeType:
        assert regime.value in opt._state
        assert len(opt._state[regime.value]["population"]) == 3
        assert opt._state[regime.value]["history"] == []

def test_meta_optimizer_get_params(temp_db):
    opt = MetaParameterOptimizer(db_path=temp_db)
    # Get parameters for calm regime
    params = opt.get_params("calm")
    assert isinstance(params, MetaParams)
    assert params.regime == "calm"

def test_meta_optimizer_stagnation_replacement(temp_db):
    opt = MetaParameterOptimizer(db_path=temp_db)
    regime = "calm"
    
    # Let's inspect the initial population
    bucket = opt._state[regime]
    population = bucket["population"]
    assert len(population) == 3
    
    # Force the best candidate to have minimum step sizes so stagnation count increments
    best_cand = population[0]
    best_cand.step_k = MIN_STEP_K
    best_cand.step_thresh = MIN_STEP_THRESH
    
    # Also force the worst candidate to NOT have minimum step sizes (like in the user's issue)
    worst_cand = population[2]
    worst_cand.step_k = 10
    worst_cand.step_thresh = 0.08
    
    # We will simulate STAGNATION_LIMIT updates with no improvements to trigger replacement of the worst candidate
    # A score of 0.5 for a dummy trade outcome using the best candidate's parameters
    for _ in range(STAGNATION_LIMIT):
        outcome = TradeOutcome(
            meta_params_used=best_cand.to_dict(),
            regime=regime,
            predicted_return=0.01,
            actual_return=0.01,
            pnl_usd=10.0,
            causal_chain_held=True,
            regime_was_stable=True,
        )
        opt.update(outcome)
        
    # The worst candidate should have been replaced by a fresh randomly initialized candidate
    new_population = opt._state[regime]["population"]
    assert len(new_population) == 3
    
    # Verify that the stagnation count has been reset to 0
    assert opt._stagnation_counts.get(regime, 0) == 0
