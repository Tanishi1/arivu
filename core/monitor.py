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
        (breached: bool, proximity: float)

    B13 FIX: proximity is now operator-aware, matching decision_utils.annotate_proximity().
    Previously used current_value/threshold for ALL operators. For RSI's divergence_span
    (gt, threshold=3.0), safe value=5.0 gave proximity=1.67 in debug logs while
    training_buffer.csv correctly stored 0.6 — making logs actively misleading.

    SIGN-FLIP FIX: Causal agent now sets threshold=0.0 for sign-flip assumptions
    (operator=gt meaning "breach when var ≤ 0", or operator=lt meaning "breach when
    var ≥ 0"). The old `elif threshold == 0: proximity = 0.0` branch always returned
    zero proximity for these, disabling the dynamic poll interval entirely.
    Now proximity for sign-flip assumptions is how close |current_value| is to zero
    relative to the commit-time absolute value, so the monitor speeds up as the
    driver approaches sign reversal.

    NOTE: Assumption is frozen (pydantic). Do NOT mutate it.
    """
    current_value: float = getattr(state, assumption.variable, 0.0)

    if assumption.threshold == 0 and assumption.operator == "gt":
        # Two sub-cases share this branch:
        # 1. EMA trend_persistence (legacy): threshold=0.0, operator='gt' — SLOPE_NORMALISER path
        # 2. Causal sign-flip (new): positive-coeff edge, breach when current_value ≤ 0
        #
        # For both, proximity measures "how close is the value to flipping negative?"
        # If assumption.current_value (commit-time val) is available and nonzero,
        # use it to normalise. Otherwise fall back to SLOPE_NORMALISER for legacy path.
        commit_val = abs(assumption.current_value) if assumption.current_value != 0 else SLOPE_NORMALISER
        proximity = max(0.0, 1.0 - min(1.0, abs(current_value) / commit_val))

    elif assumption.threshold == 0 and assumption.operator == "lt":
        # Causal sign-flip: negative-coeff edge, breach when current_value ≥ 0.
        # Proximity = how close is the variable to flipping positive?
        # commit_val is negative (the driver was pointing down at commit time).
        commit_val = abs(assumption.current_value) if assumption.current_value != 0 else SLOPE_NORMALISER
        proximity = max(0.0, 1.0 - min(1.0, abs(current_value) / commit_val))

    elif assumption.operator == "gt":
        # Breach when current_value <= threshold (threshold > 0).
        # Safe = current_value >> threshold → proximity near 0.
        # At-risk = current_value ≈ threshold → proximity near 1.
        proximity = (assumption.threshold / current_value) if current_value > 0 else 1.0
    else:
        # operator='lt': breach when current_value >= threshold (threshold > 0)
        proximity = current_value / assumption.threshold

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
    stop_event: asyncio.Event | None = None,
) -> None:
    """Main monitoring coroutine. Runs every MONITOR_INTERVAL_S seconds.

    Pauses safely when the feed is stale.

    B1 FIX: Accepts stop_event (the global _shutdown asyncio.Event from main.py).
    Previously ran `while True:` with no exit condition. On Ctrl+C, asyncio.gather
    blocked forever waiting for this coroutine — _graceful_shutdown never fired,
    so open Alpaca positions were never closed.
    """
    logger.info("Assumption monitor started | interval=%ds", MONITOR_INTERVAL_S)

    # Track DOs that already have a pending breach event in the queue.
    # Cleared when the DO closes (get_active returns a different ID or None).
    _pending_breach_do_ids: set[str] = set()
    _last_active_do_id: str | None = None

    while stop_event is None or not stop_event.is_set():
        if _is_feed_stale(state_manager):
            logger.warning("Monitor paused | feed stale > %ds", STALE_THRESHOLD_S)
            await asyncio.sleep(MONITOR_INTERVAL_S)
            continue

        active = await asyncio.to_thread(ledger_writer.get_active)

        if active is None:
            _pending_breach_do_ids.clear()
            _last_active_do_id = None
            logger.debug("Monitor | no active DecisionObject")
            await asyncio.sleep(MONITOR_INTERVAL_S)
            continue

        # Clear pending set when a new DO becomes active
        current_do_id = str(active.id)
        if current_do_id != _last_active_do_id:
            _pending_breach_do_ids.discard(_last_active_do_id)
            _last_active_do_id = current_do_id

        # Compute dynamic interval — continuous linear interpolation over [2, 10] seconds.
        if active.assumptions:
            max_breach_risk = max(
                a.breach_risk for a in active.assumptions
            )
            interval = max(2.0, min(10.0, round(10.0 - (max_breach_risk * 8.0), 1)))
        else:
            interval = float(MONITOR_INTERVAL_S)
            max_breach_risk = -1.0

        logger.debug(
            "Monitor: dynamic cadence | max_breach_risk=%.3f | interval=%.1fs",
            max_breach_risk, interval,
        )

        await asyncio.sleep(interval)

        if active.status == "COMMITTED":
            logger.debug("Monitor | object still COMMITTED, waiting for execution")
            continue

        if active.causal_chain_snapshot is None:
            continue  # legacy trade, no causal assumptions to monitor

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
            do_id = str(active.id)
            if do_id in _pending_breach_do_ids:
                logger.debug(
                    "Monitor | breach event already queued for DO %s — skipping duplicate",
                    do_id[:8],
                )
            else:
                try:
                    decision_queue.put_nowait({
                        "type": "assumption_breach",
                        "assumptions": breached_this_cycle,
                        "decision_object_id": do_id,
                    })
                    _pending_breach_do_ids.add(do_id)
                except asyncio.QueueFull:
                    logger.warning(
                        "Decision queue full \u2014 assumption_breach trigger discarded "
                        "(will retry next monitor cycle)"
                    )
