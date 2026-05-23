"""core/monitor.py
60-second assumption monitoring loop.

Runs as an asyncio coroutine alongside market_feed_loop and decision_cycle_loop.
Every 60 seconds it reads the active DecisionObject from the ledger and checks
each assumption against the live CausalState.

If a breach is detected:
  1. Logs breach to the ledger entry (timestamp + which assumption)
  2. Puts an event on the decision_queue to trigger the simulator

Safety: pauses if the market feed is stale (last_tick_timestamp > 10s ago).
This prevents false breach alerts from stale data.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from core.causal_state import CausalStateManager, STALE_THRESHOLD_S
from core.constants import SLOPE_NORMALISER
from core.schemas import Assumption, DecisionObject

logger = logging.getLogger(__name__)

MONITOR_INTERVAL_S = 5  # N21 FIX: Reduced from 60s to prevent missing breaches right before horizon expiration


def _is_feed_stale(manager: CausalStateManager) -> bool:
    """Return True if the market feed has not sent a tick in > STALE_THRESHOLD_S."""
    state = manager.snapshot()
    if state.last_tick_timestamp is None:
        return True
    now = datetime.now(timezone.utc)
    age = (now - state.last_tick_timestamp).total_seconds()
    return age > STALE_THRESHOLD_S


def _check_assumption(assumption: Assumption, state) -> tuple[bool, float]:
    """Check if an assumption is breached given current state.

    Returns:
        (breached: bool, proximity: float) — proximity is current_value / threshold.

    NOTE: Assumption is frozen (pydantic). Do NOT mutate it.
    """
    current_value: float = getattr(state, assumption.variable, 0.0)

    # Compute proximity: how close current_value is to breaching the threshold.
    if assumption.threshold != 0:
        proximity = current_value / assumption.threshold
    elif assumption.operator == "gt":
        # HIGH-4 FIX: threshold=0.0 + operator='gt' (EMA trend_persistence).
        # proximity = how close the slope is to 0 (the breach point).
        # SLOPE_NORMALISER scales so strong slope→0.0 (safe), weak slope→1.0 (at risk).
        # Previously returned 1.0 unconditionally — always appeared maximum breach risk.
        proximity = max(0.0, 1.0 - min(1.0, abs(current_value) / SLOPE_NORMALISER))
    else:
        proximity = 0.0

    if assumption.operator == "lt":
        breached = current_value >= assumption.threshold
    elif assumption.operator == "gt":
        breached = current_value <= assumption.threshold
    else:
        breached = False

    return breached, proximity


async def assumption_monitor_loop(
    state_manager: CausalStateManager,
    ledger_writer,                # ledger.writer.LedgerWriter instance
    decision_queue: asyncio.Queue,
) -> None:
    """Main monitoring coroutine. Runs every 60 seconds indefinitely.

    Pauses safely when the feed is stale.
    """
    logger.info("Assumption monitor started | interval=%ds", MONITOR_INTERVAL_S)

    while True:
        await asyncio.sleep(MONITOR_INTERVAL_S)

        if _is_feed_stale(state_manager):
            logger.warning("Monitor paused | feed stale > %ds", STALE_THRESHOLD_S)
            continue

        active: DecisionObject | None = await asyncio.to_thread(ledger_writer.get_active)
        if active is None:
            logger.debug("Monitor | no active DecisionObject")
            continue

        # N22 FIX: Prevent monitoring before Alpaca limit order finishes filling
        if active.status == "COMMITTED":
            logger.debug("Monitor | object still COMMITTED, waiting for execution")
            continue

        state = state_manager.snapshot()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Collect ALL breaches in a single pass, then fire ONE consolidated event.
        # Firing one event per assumption would cause N redundant decision cycles
        # when N assumptions breach simultaneously (common during volatility spikes).
        breached_this_cycle: list[str] = []

        for assumption in active.assumptions:
            breached, proximity = _check_assumption(assumption, state)
            if breached:
                logger.warning(
                    "Assumption breached | name=%s variable=%s threshold=%.4f actual=%.4f",
                    assumption.name,
                    assumption.variable,
                    assumption.threshold,
                    getattr(state, assumption.variable, 0.0),
                )
                # thread so the async monitor loop is not blocked during disk I/O.
                try:
                    await asyncio.to_thread(
                        ledger_writer.log_breach,
                        decision_object_id=str(active.id),
                        assumption_name=assumption.name,
                        breach_timestamp=now_iso,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Monitor failed to log breach to ledger | %s", exc)
                breached_this_cycle.append(assumption.name)
            else:
                logger.debug(
                    "Assumption holds | name=%s proximity=%.3f",
                    assumption.name, proximity,
                )

        if breached_this_cycle:
            try:
                decision_queue.put_nowait({
                    "type": "assumption_breach",
                    "assumptions": breached_this_cycle,
                    "decision_object_id": str(active.id),
                })
            except asyncio.QueueFull:
                logger.warning(
                    "Decision queue full — assumption_breach trigger discarded "
                    "(will retry next monitor cycle)"
                )
