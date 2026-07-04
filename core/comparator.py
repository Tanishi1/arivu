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

import concurrent.futures
import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from core.schemas import DecisionObject, OutcomeRecord
from core.constants import TRAINING_BUFFER_PATH, STRATEGY_HORIZON_MINUTES

logger = logging.getLogger(__name__)

# How long to wait for Alpaca REST calls before giving up (seconds).
# An unresponsive Alpaca API must not stall the entire decision loop.
_ALPACA_TIMEOUT_S = 15

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
        close_reason: str,
        current_graph = None,
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
        actual_pnl, had_position = self._read_actual_pnl()
        if actual_pnl is None:
            logger.error("Comparator | could not read P&L from Alpaca — cycle not closed")
            return None

        # HOLD cycles: no Alpaca position was opened so actual P&L is structurally
        # zero. Storing projected_pnl - 0.0 as outcome_delta would teach ML2 that
        # every HOLD is a bad prediction. Clamp delta to 0.0 when there was no position.
        outcome_delta = 0.0 if not had_position else (decision_object.projected_pnl - actual_pnl)

        # Determine which assumptions held vs breached
        # The ledger holds breach records committed by the monitor
        breach_log = self._ledger.get_breach_log(str(decision_object.id))
        breached_names = list(breach_log.keys())

        # If CausalAgent and graph is supplied, perform causal edge structural breach audit
        if decision_object.strategy_name == "CausalAgent" and current_graph is not None:
            for assumption in decision_object.assumptions:
                if assumption.name.startswith("causal_edge|"):
                    parts = assumption.name.split("|")
                    if len(parts) == 4:
                        source = parts[1]
                        target = parts[2]
                        lag = int(parts[3])

                        # Check if this edge exists in the latest live graph snapshot
                        matching_edge = next((
                            e for e in current_graph.edges
                            if e.source == source and e.target == target and e.lag == lag
                        ), None)

                        is_edge_breached = False
                        if matching_edge is None:
                            is_edge_breached = True
                        else:
                            # Check sign/direction of the coefficient
                            hyp_sign = 1 if assumption.current_value >= 0 else -1
                            act_sign = 1 if matching_edge.coeff >= 0 else -1
                            if hyp_sign != act_sign:
                                is_edge_breached = True

                        if is_edge_breached:
                            if assumption.name not in breached_names:
                                breached_names.append(assumption.name)
                                breach_log[assumption.name] = datetime.now(timezone.utc).isoformat()

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
            hill_climb_iterations=decision_object.hill_climb_iterations,
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

    def _read_actual_pnl(self) -> tuple[float | None, bool]:
        """Close the Alpaca position and return (pnl, had_open_position).

        Returns:
            (pnl, had_position):
              - pnl is None if the Alpaca call failed or timed out.
              - had_position is False when no position existed (HOLD cycle).
                Callers must set outcome_delta = 0.0 when had_position is False.

        A 15-second timeout is enforced so that an unresponsive Alpaca API
        does not stall the decision loop indefinitely via asyncio.to_thread().
        """
        def _do_read() -> tuple[float, bool]:
            import time as _time
            MAX_SINGLE_NOTIONAL = 180_000.0  # Alpaca caps at $200k; use $180k for safety margin
            try:
                position = self._alpaca.get_position("SOLUSD")
                unrealized_pl = float(position.unrealized_pl)
                qty_held = float(position.qty)
                current_price = float(position.current_price)
                notional = qty_held * current_price

                if notional > MAX_SINGLE_NOTIONAL:
                    # Position too large for a single close_position() call.
                    # Chunk into multiple market sell orders of ≤$180k each.
                    chunk_qty = round(MAX_SINGLE_NOTIONAL / current_price, 6)
                    remaining = qty_held
                    chunk_count = 0
                    logger.warning(
                        "Comparator: large position %.0f SOL ($%.0f) — chunked close @ %.0f SOL/order",
                        qty_held, notional, chunk_qty,
                    )
                    while remaining > 0.001:
                        sell_qty = round(min(chunk_qty, remaining), 6)
                        try:
                            self._alpaca.submit_order(
                                symbol="SOLUSD",
                                qty=sell_qty,
                                side="sell",
                                type="market",
                                time_in_force="gtc",
                            )
                            chunk_count += 1
                            logger.info(
                                "Comparator: chunk close %d | sold %.4f SOL | remaining %.4f SOL",
                                chunk_count, sell_qty, remaining - sell_qty,
                            )
                        except Exception as chunk_exc:
                            logger.error("Comparator: chunk close failed | %s", chunk_exc)
                            break
                        remaining -= sell_qty
                        if remaining > 0.001:
                            _time.sleep(2)  # brief pause between chunks
                else:
                    self._alpaca.close_position("SOLUSD")

                return unrealized_pl, True
            except Exception as exc:  # noqa: BLE001
                if "position does not exist" in str(exc).lower():
                    logger.info("Comparator | No open position for SOLUSD (HOLD cycle)")
                    return 0.0, False
                raise  # re-raise so the outer except captures it


        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_do_read)
                return future.result(timeout=_ALPACA_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            logger.error(
                "Alpaca P&L read timed out after %ds — cycle not closed", _ALPACA_TIMEOUT_S
            )
            return None, False
        except Exception as exc:  # noqa: BLE001
            logger.error("Alpaca P&L read/close failed | %s", exc)
            return None, False

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
                "time_horizon": STRATEGY_HORIZON_MINUTES,
                "p_normal": p_normal,
                "p_stressed": p_stressed,
                "p_degraded": p_degraded,
                "assumption_type": assumption.name,
                "breached": breached_flag,
                "phase": do.phase,
                "close_reason": record.close_reason,
            })

        with open(TRAINING_BUFFER_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BUFFER_COLUMNS)
            writer.writerows(rows)

        logger.info(
            "Training buffer appended | rows=%d phase=%s", len(rows), do.phase
        )

    def _ensure_buffer_exists(self) -> None:
        """Create training_buffer.csv with header if it does not exist."""
        TRAINING_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not TRAINING_BUFFER_PATH.exists():
            with open(TRAINING_BUFFER_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=BUFFER_COLUMNS)
                writer.writeheader()
            logger.info("Training buffer created | path=%s", TRAINING_BUFFER_PATH)
