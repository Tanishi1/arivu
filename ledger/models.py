"""ledger/models.py
SQLAlchemy ORM models for the Arivu ledger database.

Three tables:
  1. decision_objects    — immutable records committed before actuation
  2. outcome_records     — actual results after each strategy window closes
  3. model_checkpoints   — ML2 Brier score at each retraining checkpoint

All indexes defined here. Schema must match core/schemas.py exactly.
See DECISIONS.md ADR-001 for why SQLite was chosen.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import (
    Column, Float, ForeignKey, Index, Integer, String, Text, create_engine, event
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from core.constants import SQLITE_PATH  # N4 FIX: single source of truth

load_dotenv()


def _get_engine():
    db_path = Path(SQLITE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 5, "check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def set_wal_mode(dbapi_connection, connection_record):
        dbapi_connection.execute("PRAGMA journal_mode=WAL")

    return engine


engine = _get_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Table 1 — decision_objects
# ---------------------------------------------------------------------------

class DecisionObjectRow(Base):
    """Immutable record committed before every actuation.

    phase column must be present from cycle 1. Do not add it later.
    status is the ONLY field that may change after initial commit.
    """
    __tablename__ = "decision_objects"

    id = Column(String, primary_key=True)
    timestamp_committed = Column(String, nullable=False)
    strategy_name = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    tuned_params = Column(Text, nullable=False)          # JSON
    market_state_snapshot = Column(Text, nullable=False) # JSON
    algo_health_vector = Column(Text, nullable=False)    # JSON [p_normal, p_stressed, p_degraded]
    assumptions = Column(Text, nullable=False)           # JSON list of Assumption objects
    projected_pnl = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(String, nullable=False, default="COMMITTED")
    phase = Column(String, nullable=False, default="bootstrap")
    hill_climb_iterations = Column(Integer, nullable=False, default=0)
    meta_params = Column(Text, nullable=True)                           # JSON dict (causal agent only)
    causal_chain_snapshot = Column(Text, nullable=True)                 # JSON (causal agent only)
    schema_version = Column(Integer, nullable=False, default=1)

    # Breach log — JSON dict {assumption_name: iso_timestamp}
    breach_log = Column(Text, nullable=True, default="{}")


# ---------------------------------------------------------------------------
# Table 2 — outcome_records
# ---------------------------------------------------------------------------

class OutcomeRecordRow(Base):
    """Actual result after each strategy window closes. One per DecisionObject."""
    __tablename__ = "outcome_records"

    id = Column(String, primary_key=True)
    decision_object_id = Column(String, ForeignKey("decision_objects.id"), nullable=False)
    timestamp_closed = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    actual_pnl = Column(Float, nullable=False)
    outcome_delta = Column(Float, nullable=False)
    assumptions_held = Column(Text, nullable=False)    # JSON list
    assumptions_breached = Column(Text, nullable=False) # JSON list
    breach_timestamps = Column(Text, nullable=True)    # JSON dict
    hill_climb_iterations = Column(Integer, nullable=False)
    phase = Column(String, nullable=False, default="bootstrap")  # mirrors DecisionObject.phase
    close_reason = Column(String, nullable=False, default="incomplete")  # sentinel: must be overwritten by writer.close() — if still 'incomplete' in DB, the write crashed
    schema_version = Column(Integer, nullable=False, default=1)


# ---------------------------------------------------------------------------
# Table 3 — model_checkpoints  (Research Metric 2 data source)
# ---------------------------------------------------------------------------

class ModelCheckpointRow(Base):
    """ML2 Brier score at each retraining checkpoint.

    The x-axis for Analysis 3 (Brier score progression plot).
    Log both bootstrap and trained phase checkpoints.
    """
    __tablename__ = "model_checkpoints"

    id = Column(String, primary_key=True)
    timestamp = Column(String, nullable=False)
    checkpoint_number = Column(Integer, nullable=False)
    phase = Column(String, nullable=False)             # 'bootstrap' or 'trained'
    ml2_brier_score = Column(Float, nullable=False)
    training_sample_count = Column(Integer, nullable=False)
    ml1_macro_f1 = Column(Float, nullable=True)        # optional, if ML1 retrained


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

Index("idx_do_status", DecisionObjectRow.status)
Index("idx_do_strategy", DecisionObjectRow.strategy_name)
Index("idx_do_symbol", DecisionObjectRow.symbol)
Index("idx_do_phase", DecisionObjectRow.phase)
Index("idx_or_doid", OutcomeRecordRow.decision_object_id)
Index("idx_or_phase", OutcomeRecordRow.phase)
Index("idx_mc_phase", ModelCheckpointRow.phase)


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables and indexes if they do not exist."""
    Base.metadata.create_all(engine)
    _migrate_db()


def _migrate_db() -> None:
    """Add new columns to existing tables (safe no-op if already present).

    SQLAlchemy create_all() only creates missing TABLES, not missing COLUMNS.
    Any column added after the initial schema creation must be manually migrated.
    This function runs ALTER TABLE ... ADD COLUMN with a try/except so it is
    safe to call on every startup — it silently skips columns that already exist.
    """
    from sqlalchemy import text
    migrations = [
        "ALTER TABLE decision_objects ADD COLUMN meta_params TEXT",
        "ALTER TABLE decision_objects ADD COLUMN causal_chain_snapshot TEXT",
    ]
    with engine.connect() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass  # Column already exists — ignore
