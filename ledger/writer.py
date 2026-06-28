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

        # C4 FIX: clear _active_id for ALL terminal statuses, not just CLOSED.
        # Previously only close() cleared _active_id. If EXECUTION_FAILED was set
        # and the process continued, get_active() returned the dead object and
        # comparator tried to close it — writing a garbage OutcomeRecord to the
        # training buffer.
        if new_status in ("EXECUTION_FAILED", "INTERRUPTED", "CLOSED"):
            with self._state_lock:
                if self._active_id == decision_object_id:
                    self._active_id = None
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
        # H2 FIX: Only record the FIRST breach timestamp for each assumption.
        # The monitor runs every 5s; a sustained breach previously overwrote the
        # timestamp every cycle. OutcomeRecord.breach_timestamps showed the LAST
        # detection, not the first. First breach time is the scientifically correct
        # record for lifecycle plots and breach proximity analysis.
        with self._state_lock:
            inner = self._breach_logs.setdefault(decision_object_id, {})
            inner.setdefault(assumption_name, breach_timestamp)  # only writes if not already present

        # DB write: only update if this assumption has not been logged before
        with SessionLocal() as session:
            row = session.get(DecisionObjectRow, decision_object_id)
            if row:
                log = json.loads(row.breach_log or "{}")
                if assumption_name not in log:  # preserve first breach timestamp
                    log[assumption_name] = breach_timestamp
                    row.breach_log = json.dumps(log)
                    session.commit()

    def get_breach_log(self, decision_object_id: str) -> dict:
        """Return the current breach log for a decision object.

        Checks in-memory cache first, falls back to database.
        This ensures breaches survive process crashes (N28 Fix).

        CRIT-3 FIX: Use 'in' check, not truthiness check.
        An empty dict {} (zero breaches) was previously treated as "not in cache"
        because `if cached:` is False for {}. This caused every zero-breach close
        cycle to fall through to a DB read — adding latency and a subtle race with
        the monitor thread's concurrent log_breach() write.
        """
        with self._state_lock:
            if decision_object_id in self._breach_logs:
                return dict(self._breach_logs[decision_object_id])  # copy — caller must not mutate

        # N28 FIX: Fallback to database (e.g. after mid-cycle crash recovery)
        with SessionLocal() as session:
            row = session.get(DecisionObjectRow, decision_object_id)
            if row and row.breach_log:
                log = json.loads(row.breach_log)
                # Only cache non-empty logs from DB — empty DB log means no breaches
                # recorded yet (monitor may still be running). Do not overwrite an
                # in-memory {} that was explicitly initialised by commit().
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
            # C3 FIX: exclude EXECUTION_FAILED in addition to CLOSED and INTERRUPTED.
            # Previously EXECUTION_FAILED objects stayed in _active_id (update_status
            # did not clear it). get_active() would return the dead object, and the
            # next decision cycle would try to close it — writing a garbage OutcomeRecord
            # with actual_pnl=0.0 to the training buffer.
            if row is None or row.status in (
                "CLOSED", "INTERRUPTED", "EXECUTION_FAILED"
            ):
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
        """Return the highest checkpoint_number for ML2 from model_checkpoints.

        B3 FIX: Filter to rows where ml1_macro_f1 IS NULL (pure ML2 checkpoints).
        ML1 retraining now also writes to model_checkpoints (with ml1_macro_f1 set).
        Without this filter, ML2's _checkpoint_count initialises to ML1's highest
        checkpoint number, causing ML2's first checkpoint to appear as #2 or #3
        instead of #1 — breaking the Brier score progression chart for Week 9.
        """
        from sqlalchemy.sql import func
        with SessionLocal() as session:
            max_cp = session.query(
                func.max(ModelCheckpointRow.checkpoint_number)
            ).filter(
                ModelCheckpointRow.ml1_macro_f1.is_(None)  # ML2 rows only
            ).scalar()
            return max_cp if max_cp is not None else 0

    def count_closed(self) -> int:
        """Return the total number of CLOSED DecisionObjects.

        Used by the K-Means re-run trigger in main.py to decide when to retrain
        the regime classifier on live data (threshold: 40+, every 10 entries).
        """
        from sqlalchemy.sql import func
        with SessionLocal() as session:
            result = session.query(
                func.count(DecisionObjectRow.id)
            ).filter(
                DecisionObjectRow.status == "CLOSED"
            ).scalar()
            return result or 0

    def get_closed_entries_for_regime(self) -> list[dict]:
        """Return closed DecisionObject market snapshots for K-Means re-training.

        Format matches what ml/regime.py's run_kmeans_and_retrain() expects:
        each dict has keys: volatility, spread, trend_strength, volume,
        strategy_name, symbol.

        Called by decision_cycle_loop after count_closed() meets the threshold.

        S-6 FIX: Limit to the 200 most recent closed entries (ordered by commit time
        descending). At Week 8 with 160+ entries a full table scan loads all rows and
        their JSON snapshot columns into memory simultaneously — unnecessary pressure.
        The 200 most recent entries are more representative of current market conditions
        than entries from 5 weeks ago, so this also improves regime quality.
        """
        with SessionLocal() as session:
            rows = session.query(DecisionObjectRow).filter(
                DecisionObjectRow.status == "CLOSED"
            ).order_by(
                DecisionObjectRow.timestamp_committed.desc()
            ).limit(200).all()

        entries = []
        for row in rows:
            try:
                snapshot = json.loads(row.market_state_snapshot)
                entries.append({
                    "volatility":     float(snapshot.get("volatility", 0.0)),
                    "spread":         float(snapshot.get("spread", 0.0)),
                    "trend_strength": float(snapshot.get("trend_strength", 0.0)),
                    "volume":         float(snapshot.get("volume", 0.0)),
                    "strategy_name":  row.strategy_name,
                    "symbol":         row.symbol,
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed snapshot for %s: %s", row.id[:8], exc)
                continue
        return entries


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
