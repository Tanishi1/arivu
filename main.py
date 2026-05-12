"""main.py — Arivu entry point

Three asyncio coroutines running concurrently in ONE Python process:
  1. market_feed_loop       — processes every Binance WebSocket tick
  2. assumption_monitor_loop — checks active assumptions every 60 seconds
  3. decision_cycle_loop    — reads decision_queue; runs Simulator → ML2 → Ledger → Alpaca

Start: python main.py
Stop:  Ctrl+C  →  graceful shutdown (WebSocket closed, active entry marked INTERRUPTED)

Read before running:
  - DECISIONS.md for architecture decisions (especially ADR-003 and ADR-004)
  - core/schemas.py for all six data contracts
  - ledger/writer.py for the immutability invariant

Lock discipline (DO NOT VIOLATE):
  Only market_feed_loop WRITES to CausalState.
  decision_cycle_loop and assumption_monitor_loop only READ it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_PATH = Path("logs/arivu.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component imports (after logging is configured)
# ---------------------------------------------------------------------------
from core.causal_state import CausalStateManager
from core.constants import STRATEGY_HORIZON_MINUTES
from core.stream import BinanceFeed
from core.monitor import assumption_monitor_loop
from core.simulator import Simulator
from execution.executor import Executor
from ledger.models import init_db
from ledger.writer import LedgerWriter, LedgerWriteError
from ml.ml1 import ML1BehaviourClassifier
from ml.ml2 import ML2BreachPredictor
from ml.regime import RegimeClassifier
from strategies.ema_crossover import EMAStrategy
from strategies.bollinger import BollingerStrategy
from strategies.rsi_divergence import RSIStrategy
from core.comparator import OutcomeComparator


# ---------------------------------------------------------------------------
# Global shutdown event
# ---------------------------------------------------------------------------
_shutdown = asyncio.Event()

# Maps queue event types to valid OutcomeRecord.close_reason Literal values
_CLOSE_REASON_MAP = {
    "threshold_crossed": "assumption_breach",
    "assumption_breach": "assumption_breach",
    "horizon_expired": "horizon_expired",
    "system_shutdown": "system_shutdown",
}


def _handle_sigint(*_):
    logger.info("Shutdown signal received")
    _shutdown.set()


# ---------------------------------------------------------------------------
# Time Horizon Watcher
# ---------------------------------------------------------------------------

async def time_horizon_watcher(decision_id: str, horizon_minutes: int, queue: asyncio.Queue) -> None:
    """Sleeps for horizon_minutes, then triggers a horizon_expired event."""
    await asyncio.sleep(horizon_minutes * 60)
    event = {
        "type": "horizon_expired",
        "decision_object_id": decision_id,
    }
    while True:
        try:
            queue.put_nowait(event)
            break
        except asyncio.QueueFull:
            await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# Coroutine 1 — Market feed loop
# ---------------------------------------------------------------------------

async def market_feed_loop(
    feed: BinanceFeed,
    ml1: ML1BehaviourClassifier,
    state_manager: CausalStateManager,
) -> None:
    """Stream Binance WebSocket ticks into CausalState.
    Also runs ML1 inference after each ExecutionTelemetry update.
    """
    await feed.run()


# ---------------------------------------------------------------------------
# Coroutine 2 — Decision cycle loop
# ---------------------------------------------------------------------------

async def decision_cycle_loop(
    decision_queue: asyncio.Queue,
    state_manager: CausalStateManager,
    simulator: Simulator,
    ml1: ML1BehaviourClassifier,
    ml2: ML2BreachPredictor,
    ledger: LedgerWriter,
    executor: Executor,
    comparator: OutcomeComparator,
    current_phase: list[str],  # mutable container: ['bootstrap'] or ['trained']
) -> None:
    """Read events from decision_queue, run the full simulator → ledger → Alpaca cycle.

    Phase is controlled by the outer loop after checking ML2 training results.
    """
    active_watcher_task: asyncio.Task | None = None

    # K5 FIX: build strategy_map ONCE before the loop, not on every cycle.
    # Instantiating strategy objects per-cycle wastes allocations and
    # silently fell back to EMA for unknown strategy names.
    strategy_map = {
        "EMAStrategy": EMAStrategy(),
        "BollingerStrategy": BollingerStrategy(),
        "RSIStrategy": RSIStrategy(),
    }

    # N29 FIX: Recover state on startup
    active = ledger.get_active()
    if active:
        if active.status == "COMMITTED":
            # Crashed during execution, cannot be salvaged safely.
            logger.warning("Recovered COMMITTED object on startup. Marking INTERRUPTED to prevent deadlock.")
            await asyncio.to_thread(ledger.update_status, str(active.id), "INTERRUPTED")
        elif active.status == "ACTIVE":
            # Recover the time horizon watcher
            from datetime import datetime, timezone
            try:
                commit_time = datetime.fromisoformat(active.timestamp_committed)
                elapsed_s = (datetime.now(timezone.utc) - commit_time).total_seconds()
                remaining_m = max(0.0, STRATEGY_HORIZON_MINUTES - (elapsed_s / 60.0))
                logger.info("Recovered ACTIVE object. Spawning watcher for remaining %.1f mins.", remaining_m)
                active_watcher_task = asyncio.create_task(
                    time_horizon_watcher(str(active.id), remaining_m, decision_queue)
                )
            except Exception as e:
                logger.error("Failed to parse timestamp, marking INTERRUPTED: %s", e)
                await asyncio.to_thread(ledger.update_status, str(active.id), "INTERRUPTED")

    while not _shutdown.is_set():
        try:
            event = await asyncio.wait_for(decision_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            continue

        # N32 FIX: Ignore events that belong to a different (closed) decision cycle
        active = ledger.get_active()
        event_do_id = event.get("decision_object_id")
        if event_do_id and (not active or event_do_id != str(active.id)):
            logger.debug("Discarding stale event for closed cycle %s", event_do_id)
            continue

        logger.info(
            "Decision cycle started | trigger=%s phase=%s",
            event.get("type"), current_phase[0],
        )

        # 1. Close any active DecisionObject before starting a new one
        if active:
            if active_watcher_task and not active_watcher_task.done():
                active_watcher_task.cancel()
                # K8 FIX: await the cancelled task so CancelledError is consumed
                # cleanly. Without this, Python logs "Task exception was never
                # retrieved" hours later at GC time.
                try:
                    await active_watcher_task
                except asyncio.CancelledError:
                    pass

            close_reason = _CLOSE_REASON_MAP.get(event.get("type", ""), "manual")
            try:
                # K3 FIX: run synchronous SQLite commit in a thread so the event
                # loop is not blocked during disk I/O (prevents WebSocket drops).
                # N30 FIX: Do not pass volatile last_hill_climb_iterations loop variable
                record = await asyncio.to_thread(
                    comparator.close_cycle, active, close_reason
                )

                if record is None:
                    # N24 FIX: If close_cycle returns None (e.g. Alpaca API failed), mark DO as INTERRUPTED
                    logger.warning("close_cycle failed to close %s, marking INTERRUPTED to prevent zombies", active.id)
                    await asyncio.to_thread(ledger.update_status, str(active.id), "INTERRUPTED")
                else:
                    # N7 FIX: ML2 retrains on outcomes, so check after closing the cycle
                    await asyncio.to_thread(ml2.maybe_retrain, ledger)
            except Exception as exc:  # noqa: BLE001
                # N24 FIX: Prevent zombie DOs on exception
                logger.error("Decision cycle crash prevented | close failed | %s", exc)
                await asyncio.to_thread(ledger.update_status, str(active.id), "INTERRUPTED")

        state = state_manager.snapshot()

        # Stage 1 + Stage 2
        sim_result = simulator.run(state)

        # ML2 annotates assumptions with breach probabilities
        ml1_vector = state.algo_health_vector
        annotated_assumptions = ml2.annotate(
            sim_result.assumptions,
            state,
            time_horizon=STRATEGY_HORIZON_MINUTES,
            ml1_vector=ml1_vector,
        )
        # SimulatorResult is frozen — use model_copy to update
        sim_result = sim_result.model_copy(update={"assumptions": annotated_assumptions})

        # Populate current_value and proximity on each assumption from live state.
        # Assumption is frozen — use model_copy. proximity is the primary ML2 feature;
        # if left at 0.0 (the default), ML2 training data is blind to threshold proximity.
        proximity_annotated = []
        for a in sim_result.assumptions:
            current_val = getattr(state, a.variable, 0.0)
            prox = current_val / a.threshold if a.threshold != 0 else 1.0
            proximity_annotated.append(a.model_copy(update={
                "current_value": current_val,
                "proximity": prox,
            }))
        sim_result = sim_result.model_copy(update={"assumptions": proximity_annotated})

        # Build and COMMIT the DecisionObject — BEFORE any actuation
        from core.schemas import DecisionObject
        do = DecisionObject(
            strategy_name=sim_result.strategy_name,
            tuned_params=sim_result.tuned_params,
            market_state_snapshot=state.model_dump(mode="json", exclude={"algo_health_vector", "last_tick_timestamp"}),
            algo_health_vector=ml1_vector,
            assumptions=sim_result.assumptions,
            projected_pnl=sim_result.projected_pnl,
            confidence=sim_result.confidence,
            hill_climb_iterations=sim_result.hill_climb_iterations,
            phase=current_phase[0],
        )

        try:
            # K3 FIX: commit is synchronous SQLite I/O — run in thread
            await asyncio.to_thread(ledger.commit, do)
        except LedgerWriteError as exc:
            logger.error("Ledger commit failed — actuation aborted | %s", exc)
            continue

        # Actuate — ONLY after successful ledger commit
        strategy = strategy_map.get(sim_result.strategy_name)
        if strategy is None:
            logger.error(
                "Unknown strategy '%s' — skipping actuation", sim_result.strategy_name
            )
            continue
        signal = strategy.generate_signal(state, sim_result.tuned_params)

        try:
            telemetry = await executor.execute(
                signal=signal,
                params=sim_result.tuned_params,
                state=state,
                decision_object_id=str(do.id),
            )
        except Exception as exc:  # noqa: BLE001
            # N5 FIX: executor failure after a successful commit leaves the DO in
            # COMMITTED status forever, blocking every future decision cycle.
            # Mark it EXECUTION_FAILED so get_active() returns None next cycle.
            logger.error(
                "Executor failed — marking DO as EXECUTION_FAILED | id=%s | %s",
                do.id, exc,
            )
            await asyncio.to_thread(
                ledger.update_status, str(do.id), "EXECUTION_FAILED"
            )
            continue

        # N19 FIX: Compute and record actual exposure back into CausalState
        intended_qty = await executor._compute_quantity(sim_result.tuned_params, state)
        actual_qty = intended_qty * (1 + telemetry.position_size_deviation)
        await state_manager.update_position(actual_qty, actual_qty * state.price)

        # Update ML1 behavioural vector with fresh telemetry
        new_vector = ml1.predict_proba(telemetry)
        await state_manager.update_algo_health(new_vector)

        # N22 FIX: Mark the object as ACTIVE only after the Alpaca order has successfully filled
        await asyncio.to_thread(ledger.update_status, str(do.id), "ACTIVE")

        # Check if ML1 should retrain
        if await asyncio.to_thread(ml1.maybe_retrain):
            logger.info("ML1 retrained successfully")

        # Spawn time horizon watcher for this new decision
        active_watcher_task = asyncio.create_task(
            time_horizon_watcher(str(do.id), STRATEGY_HORIZON_MINUTES, decision_queue)
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    logger.info("Arivu starting up...")

    # Initialise database
    init_db()
    logger.info("Database initialised | path=%s", os.getenv("SQLITE_PATH", "data/arivu.db"))

    # Initialise components
    decision_queue: asyncio.Queue = asyncio.Queue(maxsize=10)  # K2: bounded — full queue discards triggers, not decisions
    state_manager = CausalStateManager(decision_queue=decision_queue)
    ledger = LedgerWriter()
    executor = Executor()

    ml1 = ML1BehaviourClassifier()
    ml2 = ML2BreachPredictor()
    regime = RegimeClassifier()
    comparator = OutcomeComparator(ledger, executor._api)

    strategies = [EMAStrategy(), BollingerStrategy(), RSIStrategy()]
    simulator = Simulator(strategies=strategies, regime_classifier=regime)  # N25 FIX: pass wrapper, not raw model

    feed = BinanceFeed(state_manager=state_manager)
    current_phase: list[str] = ["bootstrap"]

    logger.info("Causal state initialised | phase=%s", current_phase[0])

    # Graceful shutdown handler
    signal.signal(signal.SIGINT, _handle_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigint)

    # K7 FIX: return_exceptions=True so one dying coroutine does not cancel
    # all others (which would leave open Alpaca positions unmonitored).
    try:
        results = await asyncio.gather(
            market_feed_loop(feed, ml1, state_manager),
            assumption_monitor_loop(state_manager, ledger, decision_queue),
            decision_cycle_loop(
                decision_queue, state_manager,
                simulator, ml1, ml2, ledger, executor, comparator, current_phase,
            ),
            _shutdown_watcher(feed),
            return_exceptions=True,
        )
        coroutine_names = [
            "market_feed_loop", "assumption_monitor_loop",
            "decision_cycle_loop", "_shutdown_watcher",
        ]
        for name, result in zip(coroutine_names, results):
            if isinstance(result, Exception):
                logger.critical(
                    "Coroutine '%s' died unexpectedly | %s",
                    name, result, exc_info=result,
                )
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error in main: %s", exc, exc_info=True)
    finally:
        await _graceful_shutdown(feed, ledger, executor)


async def _shutdown_watcher(feed: BinanceFeed) -> None:
    """Wait for shutdown signal then stop the feed."""
    await _shutdown.wait()
    await feed.stop()


async def _graceful_shutdown(feed: BinanceFeed, ledger: LedgerWriter, executor: Executor) -> None:
    """Mark any active DecisionObject as INTERRUPTED on exit and close Alpaca position."""
    active = ledger.get_active()
    if active:
        ledger.update_status(str(active.id), "INTERRUPTED")
        logger.info("Active entry marked INTERRUPTED | id=%s", active.id)

        # N18 FIX: Close Alpaca position to prevent orphaned real-money exposure
        try:
            await asyncio.to_thread(executor._api.close_position, "SOLUSD")
            logger.info("Alpaca position closed on shutdown")
        except Exception as exc:  # noqa: BLE001
            if "position does not exist" not in str(exc).lower():
                logger.error("Failed to close Alpaca position on shutdown | %s", exc)

    logger.info("Graceful shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
