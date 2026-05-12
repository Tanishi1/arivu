"""ledger/writer.py
Ledger writer — commit, update, close, and immutability guard.

The most important module in Arivu.
Read this before touching any other file.

Core invariant:
  A DecisionObject is committed to the ledger BEFORE any actuation signal
  is sent to Alpaca. Once committed, only the 'status' field may change.
  All other fields are write-once. This is enforced by the immutability guard
  which raises ValueError if any attempt is made to modify an immutable field.

Why commit before actuation?
  If the decision object is created AFTER the outcome is known, it is a
  rationalisation, not a record. Scientific integrity requires assumptions
  to be logged before the outcome is observable — the same principle as
  pre-registering a hypothesis before running an experiment.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session  # noqa: F401 — kept for type hinting by callers

from ledger.models import DecisionObjectRow, ModelCheckpointRow, OutcomeRecordRow, SessionLocal
from core.schemas import DecisionObject, OutcomeRecord

logger = logging.getLogger(__name__)

# Fields that are IMMUTABLE after initial commit — only 'status' may change
_IMMUTABLE_FIELDS = {
    "id", "timestamp_committed", "strategy_name", "symbol", "tuned_params",
    "market_state_snapshot", "algo_health_vector", "assumptions",
    "projected_pnl", "confidence", "phase", "schema_version",
}


class LedgerWriteError(Exception):
    """Raised when a ledger write fails. Actuation must not proceed."""


class LedgerWriter:
    """Manages the full DecisionObject lifecycle: commit → active → close."""

    def __init__(self) -> None:
        self._active_id: str | None = None
        self._breach_logs: dict[str, dict] = {}  # {decision_object_id: {assumption: ts}}
        # N1 FIX: threading.Lock guards _active_id and _breach_logs.
        # asyncio.to_thread() runs these methods in a thread pool, so multiple
        # threads can now access shared state simultaneously. Always acquire
        # this lock before touching _active_id or _breach_logs.
        self._state_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Commit — THE most critical operation
    # ------------------------------------------------------------------

    def commit(self, do: DecisionObject) -> str:
        """Write a DecisionObject to the ledger. Must succeed before actuation.

        Do NOT call executor.place_order() if this raises LedgerWriteError.

        Args:
            do: the DecisionObject produced by the simulator + ML2

        Returns:
            The string ID of the committed record.

        Raises:
            LedgerWriteError: if the database write fails for any reason.
        """
        try:
            row = DecisionObjectRow(
                id=str(do.id),
                timestamp_committed=do.timestamp_committed.isoformat(),
                strategy_name=do.strategy_name,
                symbol=do.symbol,
                tuned_params=json.dumps(do.tuned_params),
                market_state_snapshot=json.dumps(do.market_state_snapshot),
                algo_health_vector=json.dumps(do.algo_health_vector),
                assumptions=json.dumps([a.model_dump() for a in do.assumptions]),
                projected_pnl=do.projected_pnl,
                confidence=do.confidence,
                status="COMMITTED",
                phase=do.phase,
                hill_climb_iterations=do.hill_climb_iterations,
                schema_version=do.schema_version,
                breach_log="{}",
            )

            with SessionLocal() as session:
                session.add(row)
                session.commit()

            # N1 FIX: acquire state lock before modifying shared in-memory state
            with self._state_lock:
                self._active_id = str(do.id)
                self._breach_logs[str(do.id)] = {}

            logger.info(
                "Ledger committed | id=%s strategy=%s phase=%s projected_pnl=%.4f",
                do.id, do.strategy_name, do.phase, do.projected_pnl,
            )
            return str(do.id)

        except Exception as exc:
            raise LedgerWriteError(f"Ledger commit failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Immutability guard
    # ------------------------------------------------------------------

    def update_status(self, decision_object_id: str, new_status: str) -> None:
        """Update the status field — the ONLY field allowed to change post-commit.

        Raises:
            ValueError: if caller attempts to pass any non-status field.
        """
        with SessionLocal() as session:
            row = session.get(DecisionObjectRow, decision_object_id)
            if row is None:
                raise LedgerWriteError(f"DecisionObject {decision_object_id} not found")
            row.status = new_status
            session.commit()

        # N17 FIX: Prevent infinite memory leak by popping from cache on terminal status
        if new_status in ("EXECUTION_FAILED", "INTERRUPTED", "CLOSED"):
            with self._state_lock:
                self._breach_logs.pop(decision_object_id, None)

        logger.info("Ledger status updated | id=%s status=%s", decision_object_id, new_status)

    def guard_immutable(self, field_name: str) -> None:
        """Raise ValueError if caller attempts to modify an immutable field."""
        if field_name in _IMMUTABLE_FIELDS:
            raise ValueError(
                f"Immutability violation: field '{field_name}' cannot be modified "
                f"after a DecisionObject is committed. Only 'status' may change."
            )

    # ------------------------------------------------------------------
    # Breach logging (called by monitor.py)
    # ------------------------------------------------------------------

    def log_breach(
        self,
        decision_object_id: str,
        assumption_name: str,
        breach_timestamp: str,
    ) -> None:
        """Record a breach event. Called by the assumption monitor."""
        # N1 FIX: acquire lock before touching _breach_logs — log_breach runs
        # in a thread via asyncio.to_thread and can race with commit/close.
        with self._state_lock:
            self._breach_logs.setdefault(decision_object_id, {})[assumption_name] = breach_timestamp

        # DB write happens outside the lock — no need to hold it during I/O
        with SessionLocal() as session:
            row = session.get(DecisionObjectRow, decision_object_id)
            if row:
                log = json.loads(row.breach_log or "{}")
                log[assumption_name] = breach_timestamp
                row.breach_log = json.dumps(log)
                session.commit()

    def get_breach_log(self, decision_object_id: str) -> dict:
        """Return the current breach log for a decision object.

        Checks in-memory cache first, falls back to database.
        This ensures breaches survive process crashes (N28 Fix).
        """
        with self._state_lock:
            cached = self._breach_logs.get(decision_object_id, {})
            if cached:
                return dict(cached)  # return copy — caller must not mutate

        # N28 FIX: Fallback to database (e.g. after mid-cycle crash recovery)
        with SessionLocal() as session:
            row = session.get(DecisionObjectRow, decision_object_id)
            if row and row.breach_log:
                log = json.loads(row.breach_log)
                if log:
                    with self._state_lock:
                        self._breach_logs[decision_object_id] = log
                    return dict(log)

        return {}

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_active(self) -> DecisionObject | None:
        """Return the currently active DecisionObject, or None."""
        # N1 FIX: snapshot _active_id under lock, then do DB read without holding lock.
        # The two-step (None check → use value) must be atomic to prevent a race
        # where close() sets _active_id=None between the check and the DB lookup.
        with self._state_lock:
            active_id = self._active_id
        if active_id is None:
            return None
        with SessionLocal() as session:
            row = session.get(DecisionObjectRow, active_id)
            if row is None or row.status in ("CLOSED", "INTERRUPTED"):
                return None
            return self._row_to_schema(row)

    def get_decision_object(self, decision_object_id: str) -> DecisionObject | None:
        """Return a DecisionObject by ID, regardless of status."""
        with SessionLocal() as session:
            row = session.get(DecisionObjectRow, decision_object_id)
            if row is None:
                return None
            return self._row_to_schema(row)

    def get_max_checkpoint(self) -> int:
        """Return the highest checkpoint_number from the model_checkpoints table."""
        from sqlalchemy.sql import func
        with SessionLocal() as session:
            max_cp = session.query(func.max(ModelCheckpointRow.checkpoint_number)).scalar()
            return max_cp if max_cp is not None else 0

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self, decision_object_id, record: OutcomeRecord) -> None:
        """Write the OutcomeRecord and mark the DecisionObject as CLOSED.

        phase is pulled from the existing DecisionObjectRow so the OutcomeRecord
        always mirrors its parent — no JOIN needed in ML2 training queries.
        """
        try:
            with SessionLocal() as session:
                do_row = session.get(DecisionObjectRow, str(decision_object_id))
                if do_row is None:
                    raise LedgerWriteError(
                        f"DecisionObject {decision_object_id} not found during close"
                    )
                phase = do_row.phase

                outcome_row = OutcomeRecordRow(
                    id=str(record.id),
                    decision_object_id=str(decision_object_id),
                    timestamp_closed=record.timestamp_closed.isoformat(),
                    symbol=record.symbol,
                    actual_pnl=record.actual_pnl,
                    outcome_delta=record.outcome_delta,
                    assumptions_held=json.dumps(record.assumptions_held),
                    assumptions_breached=json.dumps(record.assumptions_breached),
                    breach_timestamps=json.dumps(record.breach_timestamps),
                    hill_climb_iterations=record.hill_climb_iterations,
                    phase=phase,
                    close_reason=record.close_reason,
                    schema_version=record.schema_version,
                )
                session.add(outcome_row)
                do_row.status = "CLOSED"
                session.commit()

            # N1 FIX: acquire state lock for compound clear + evict operation
            with self._state_lock:
                if self._active_id == str(decision_object_id):
                    self._active_id = None
                # K4 FIX: evict from in-memory cache after close
                self._breach_logs.pop(str(decision_object_id), None)

            logger.info(
                "Ledger closed | id=%s delta=%.4f phase=%s",
                decision_object_id, record.outcome_delta, phase,
            )

        except LedgerWriteError:
            raise
        except Exception as exc:
            raise LedgerWriteError(f"Ledger close failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Model checkpoints (Research Metric 2)
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        checkpoint_number: int,
        phase: str,
        ml2_brier_score: float,
        training_sample_count: int,
        ml1_macro_f1: float | None = None,
    ) -> None:
        """Log a model checkpoint after ML2 retraining."""
        row = ModelCheckpointRow(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            checkpoint_number=checkpoint_number,
            phase=phase,
            ml2_brier_score=ml2_brier_score,
            training_sample_count=training_sample_count,
            ml1_macro_f1=ml1_macro_f1,
        )
        with SessionLocal() as session:
            session.add(row)
            session.commit()

        logger.info(
            "Model checkpoint saved | #%d phase=%s brier=%.4f samples=%d",
            checkpoint_number, phase, ml2_brier_score, training_sample_count,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_schema(self, row: DecisionObjectRow) -> DecisionObject:
        """Convert an ORM row back to a pydantic DecisionObject."""
        from core.schemas import Assumption
        assumptions = [
            Assumption(**a) for a in json.loads(row.assumptions)
        ]
        return DecisionObject(
            id=row.id,
            timestamp_committed=datetime.fromisoformat(row.timestamp_committed),
            strategy_name=row.strategy_name,
            symbol=row.symbol,
            tuned_params=json.loads(row.tuned_params),
            market_state_snapshot=json.loads(row.market_state_snapshot),
            algo_health_vector=json.loads(row.algo_health_vector),
            assumptions=assumptions,
            projected_pnl=row.projected_pnl,
            confidence=row.confidence,
            hill_climb_iterations=row.hill_climb_iterations,
            status=row.status,
            phase=row.phase,
            schema_version=row.schema_version,
        )
