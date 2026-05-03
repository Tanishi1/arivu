import pytest
from pathlib import Path

def test_constants_complete():
    from core.constants import SQLITE_PATH, THRESHOLDS, INSTRUMENTS, STRATEGY_HORIZON_MINUTES, K_CANDIDATES
    assert isinstance(SQLITE_PATH, Path)
    assert isinstance(THRESHOLDS, dict)
    assert 'BTCUSDT' in THRESHOLDS
    assert 'ETHUSDT' in THRESHOLDS
    assert 'SOLUSDT' in THRESHOLDS
    assert len(INSTRUMENTS) == 3
    assert STRATEGY_HORIZON_MINUTES == 30
    assert K_CANDIDATES == [2, 3, 4]

def test_schemas_import():
    from core.schemas import MarketTick, ExecutionTelemetry, Assumption, SimulatorResult, DecisionObject, OutcomeRecord
    
    assert 'id' in DecisionObject.model_fields
    assert 'phase' in DecisionObject.model_fields
    assert 'schema_version' in DecisionObject.model_fields
    assert 'status' in DecisionObject.model_fields
    
    assert 'close_reason' in OutcomeRecord.model_fields

def test_decision_object_is_frozen():
    from core.schemas import DecisionObject
    do = DecisionObject(
        strategy_name='EMAStrategy',
        tuned_params={},
        market_state_snapshot={},
        algo_health_vector=[1.0, 0.0, 0.0],
        assumptions=[],
        projected_pnl=0.0,
        confidence=0.0,
        hill_climb_iterations=1,
    )
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        do.projected_pnl = 999.0

def test_database_creates_all_tables():
    from ledger.models import init_db, engine
    from sqlalchemy import inspect
    init_db()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert 'decision_objects' in tables
    assert 'outcome_records' in tables
    assert 'model_checkpoints' in tables

def test_strategy_base_is_abstract():
    from strategies.base import Strategy
    with pytest.raises(TypeError):
        Strategy()

def test_outcome_record_has_phase():
    from core.schemas import OutcomeRecord
    import uuid
    from datetime import datetime, timezone
    record = OutcomeRecord(
        decision_object_id=uuid.uuid4(),
        timestamp_closed=datetime.now(timezone.utc),
        actual_pnl=0.0,
        outcome_delta=0.0,
        assumptions_held=[],
        assumptions_breached=[],
        breach_timestamps={},
        hill_climb_iterations=1,
        close_reason='manual',
        phase='bootstrap',
        schema_version=1,
    )
    assert record.phase == 'bootstrap'

