"""tests/test_ledger.py
Unit tests for ledger/writer.py

Five test cases per impl plan E48:
  1. test_commit_success
  2. test_immutability_guard
  3. test_close_creates_outcome
  4. test_only_one_active
  5. test_indexes_exist
"""

from __future__ import annotations

import json
import pytest
from uuid import uuid4
from datetime import datetime, timezone

from ledger.models import init_db, SessionLocal, DecisionObjectRow
from ledger.writer import LedgerWriter, LedgerWriteError
from core.schemas import Assumption, DecisionObject, OutcomeRecord


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Point SQLite at a temp file for each test."""
    import ledger.models as lm
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("SQLITE_PATH", str(db_file))
    # Re-create engine pointing to temp file
    lm.engine = lm.create_engine(f"sqlite:///{db_file}", echo=False)
    from sqlalchemy.orm import sessionmaker
    lm.SessionLocal = sessionmaker(bind=lm.engine, autoflush=False, autocommit=False)
    lm.Base.metadata.create_all(lm.engine)
    
    # Crucial: patch writer's imported SessionLocal to point to the test db
    import ledger.writer as lw
    lw.SessionLocal = lm.SessionLocal
    yield


def _make_decision_object(phase="bootstrap") -> DecisionObject:
    return DecisionObject(
        strategy_name="EMAStrategy",
        tuned_params={"fast_period": 9, "slow_period": 21, "position_fraction": 0.10},
        market_state_snapshot={"volatility": 0.02, "spread": 0.001, "trend_strength": 0.7},
        algo_health_vector=[0.8, 0.15, 0.05],
        assumptions=[
            Assumption(name="trend_persistence", variable="trend_slope", operator="gt", threshold=0.0),
            Assumption(name="volatility_control", variable="volatility", operator="lt", threshold=0.04),
            Assumption(name="spread_constraint", variable="spread", operator="lt", threshold=0.002),
        ],
        projected_pnl=0.014,
        confidence=0.81,
        hill_climb_iterations=7,
        phase=phase,
    )


# ---------------------------------------------------------------------------
# 1. Commit success
# ---------------------------------------------------------------------------

def test_commit_success():
    """commit() should write all fields to SQLite with status=COMMITTED."""
    writer = LedgerWriter()
    do = _make_decision_object()
    committed_id = writer.commit(do)

    import ledger.models as lm
    with lm.SessionLocal() as session:
        row = session.get(DecisionObjectRow, committed_id)
        assert row is not None
        assert row.strategy_name == "EMAStrategy"
        assert row.status == "COMMITTED"
        assert row.phase == "bootstrap"
        assert row.projected_pnl == pytest.approx(0.014)
        # Assumptions are stored as JSON and parseable
        assumptions = json.loads(row.assumptions)
        assert len(assumptions) == 3


# ---------------------------------------------------------------------------
# 2. Immutability guard
# ---------------------------------------------------------------------------

def test_immutability_guard():
    """Calling guard_immutable() on any immutable field should raise ValueError."""
    writer = LedgerWriter()
    with pytest.raises(ValueError, match="Immutability violation"):
        writer.guard_immutable("projected_pnl")

    with pytest.raises(ValueError, match="Immutability violation"):
        writer.guard_immutable("strategy_name")

    # 'status' is the only field that should NOT raise
    try:
        writer.guard_immutable("status")
        # 'status' is NOT in _IMMUTABLE_FIELDS — this should not raise
    except ValueError:
        pytest.fail("guard_immutable('status') should not raise ValueError")


# ---------------------------------------------------------------------------
# 3. Close creates OutcomeRecord
# ---------------------------------------------------------------------------

def test_close_creates_outcome():
    """Closing a DecisionObject should create an OutcomeRecord with correct delta."""
    from ledger.models import OutcomeRecordRow

    writer = LedgerWriter()
    do = _make_decision_object()
    writer.commit(do)

    record = OutcomeRecord(
        decision_object_id=do.id,
        actual_pnl=0.009,
        outcome_delta=do.projected_pnl - 0.009,
        assumptions_held=["trend_persistence", "volatility_control"],
        assumptions_breached=["spread_constraint"],
        breach_timestamps={"spread_constraint": datetime.now(timezone.utc).isoformat()},
        hill_climb_iterations=7,
        phase="bootstrap",
        close_reason="test_close",
    )
    writer.close(do.id, record)

    import ledger.models as lm
    with lm.SessionLocal() as session:
        outcome = session.query(OutcomeRecordRow).filter_by(
            decision_object_id=str(do.id)
        ).first()
        assert outcome is not None
        assert outcome.actual_pnl == pytest.approx(0.009)
        assert outcome.outcome_delta == pytest.approx(0.014 - 0.009, abs=1e-6)
        assert outcome.hill_climb_iterations == 7

        do_row = session.get(DecisionObjectRow, str(do.id))
        assert do_row.status == "CLOSED"


# ---------------------------------------------------------------------------
# 4. Only one active
# ---------------------------------------------------------------------------

def test_only_one_active():
    """get_active() should return the committed DecisionObject."""
    writer = LedgerWriter()
    do = _make_decision_object()
    writer.commit(do)

    active = writer.get_active()
    assert active is not None
    assert str(active.id) == str(do.id)


# ---------------------------------------------------------------------------
# 5. Indexes exist
# ---------------------------------------------------------------------------

def test_indexes_exist():
    """All four SQL indexes should be present after schema creation."""
    from sqlalchemy import inspect as sa_inspect
    from ledger.models import engine

    inspector = sa_inspect(engine)

    do_indexes = {idx["name"] for idx in inspector.get_indexes("decision_objects")}
    or_indexes = {idx["name"] for idx in inspector.get_indexes("outcome_records")}
    cp_indexes = {idx["name"] for idx in inspector.get_indexes("model_checkpoints")}

    assert "idx_do_status" in do_indexes, "Missing idx_do_status"
    assert "idx_do_strategy" in do_indexes, "Missing idx_do_strategy"
    assert "idx_or_doid" in or_indexes, "Missing idx_or_doid"
    assert "idx_mc_phase" in cp_indexes, "Missing idx_mc_phase"


# ---------------------------------------------------------------------------
# 6. Model Checkpoints (Week 1 requirement: all 3 tables tested)
# ---------------------------------------------------------------------------

def test_save_checkpoint():
    """save_checkpoint() should create a ModelCheckpointRow and we can read it back."""
    from ledger.models import ModelCheckpointRow

    writer = LedgerWriter()
    writer.save_checkpoint(
        checkpoint_number=1,
        phase="bootstrap",
        ml2_brier_score=0.15,
        training_sample_count=100,
        ml1_macro_f1=0.92,
    )

    import ledger.models as lm
    with lm.SessionLocal() as session:
        row = session.query(ModelCheckpointRow).filter_by(checkpoint_number=1).first()
        assert row is not None
        assert row.phase == "bootstrap"
        assert row.ml2_brier_score == 0.15
        assert row.training_sample_count == 100
        assert row.ml1_macro_f1 == 0.92
