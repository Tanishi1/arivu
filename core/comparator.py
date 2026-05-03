"""shared/comparator.py
Outcome comparison + ledger close.

Checks periodically whether the active strategy's window has closed.
When it has:
  1. Reads actual P&L from Alpaca
  2. Computes outcome_delta = projected_pnl - actual_pnl  (Research Metric 1)
  3. Identifies which assumptions held and which breached
  4. Calls ledger.close() to create the OutcomeRecord
  5. Appends to training_buffer.csv (one row per assumption)

The outcome_delta shrinking over cycles is the primary research finding.
Both assumptions_held and assumptions_breached MUST be fully populated
for every closed entry — missing labels break ML2 training.
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from core.schemas import DecisionObject, OutcomeRecord

logger = logging.getLogger(__name__)

TRAINING_BUFFER_PATH = Path("data/training_buffer.csv")

# CSV columns for ML training buffer (E10)
BUFFER_COLUMNS = [
    "volatility", "spread", "trend_strength", "volume",
    "proximity", "time_horizon",
    "p_normal", "p_stressed", "p_degraded",
    "assumption_type", "breached",
    "phase", "close_reason",
]


class OutcomeComparator:
    """Reads Alpaca P&L, closes the ledger entry, appends to training buffer."""

    def __init__(self, ledger_writer, alpaca_client) -> None:
        self._ledger = ledger_writer
        self._alpaca = alpaca_client
        self._ensure_buffer_exists()

    def close_cycle(
        self,
        decision_object: DecisionObject,
        hill_climb_iterations: int,
        close_reason: str,
    ) -> OutcomeRecord | None:
        """Close the decision cycle.

        1. Read actual P&L from Alpaca.
        2. Determine which assumptions held / breached.
        3. Commit OutcomeRecord to ledger.
        4. Append rows to training_buffer.csv.

        Returns the OutcomeRecord, or None if Alpaca read fails or cycle is already closed.
        """
        existing = self._ledger.get_decision_object(str(decision_object.id))
        if existing and existing.status == "CLOSED":
            logger.info("Comparator | Decision %s already CLOSED. Skipping.", decision_object.id)
            return None
        actual_pnl = self._read_actual_pnl()
        if actual_pnl is None:
            logger.error("Comparator | could not read P&L from Alpaca — cycle not closed")
            return None

        outcome_delta = decision_object.projected_pnl - actual_pnl

        # Determine which assumptions held vs breached
        # The ledger holds breach records committed by the monitor
        breach_log = self._ledger.get_breach_log(str(decision_object.id))
        breached_names = list(breach_log.keys())
        held_names = [
            a.name for a in decision_object.assumptions
            if a.name not in breached_names
        ]

        record = OutcomeRecord(
            decision_object_id=decision_object.id,
            actual_pnl=actual_pnl,
            outcome_delta=outcome_delta,
            assumptions_held=held_names,
            assumptions_breached=breached_names,
            breach_timestamps=breach_log,
            hill_climb_iterations=hill_climb_iterations,
            phase=decision_object.phase,
            close_reason=close_reason,
        )

        self._ledger.close(decision_object.id, record)

        logger.info(
            "Ledger closed | id=%s delta=%.4f held=%s breached=%s",
            decision_object.id, outcome_delta, held_names, breached_names,
        )

        self._append_to_buffer(decision_object, record)
        return record

    def _read_actual_pnl(self) -> float | None:
        """Read cumulative P&L from Alpaca paper account."""
        try:
            account = self._alpaca.get_account()
            return float(account.equity) - float(account.last_equity)
        except Exception as exc:  # noqa: BLE001
            logger.error("Alpaca P&L read failed | %s", exc)
            return None

    def _append_to_buffer(
        self,
        do: DecisionObject,
        record: OutcomeRecord,
    ) -> None:
        """Append one row per assumption to training_buffer.csv."""
        market: dict = do.market_state_snapshot
        [p_normal, p_stressed, p_degraded] = do.algo_health_vector

        rows = []
        for assumption in do.assumptions:
            breached_flag = 1 if assumption.name in record.assumptions_breached else 0
            rows.append({
                "volatility": market.get("volatility", 0.0),
                "spread": market.get("spread", 0.0),
                "trend_strength": market.get("trend_strength", 0.0),
                "volume": market.get("volume", 0.0),
                "proximity": assumption.proximity,
                "time_horizon": 240,  # default strategy window in minutes
                "p_normal": p_normal,
                "p_stressed": p_stressed,
                "p_degraded": p_degraded,
                "assumption_type": assumption.name,
                "breached": breached_flag,
                "phase": do.phase,
                "close_reason": record.close_reason,
            })

        with open(TRAINING_BUFFER_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=BUFFER_COLUMNS)
            writer.writerows(rows)

        logger.info(
            "Training buffer appended | rows=%d phase=%s", len(rows), do.phase
        )

    def _ensure_buffer_exists(self) -> None:
        """Create training_buffer.csv with header if it does not exist."""
        TRAINING_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not TRAINING_BUFFER_PATH.exists():
            with open(TRAINING_BUFFER_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=BUFFER_COLUMNS)
                writer.writeheader()
            logger.info("Training buffer created | path=%s", TRAINING_BUFFER_PATH)
