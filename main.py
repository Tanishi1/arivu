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


def _handle_sigint(*_):
    logger.info("Shutdown signal received")
    _shutdown.set()


# ---------------------------------------------------------------------------
# Time Horizon Watcher
# ---------------------------------------------------------------------------

async def time_horizon_watcher(decision_id: str, horizon_minutes: int, queue: asyncio.Queue) -> None:
    """Sleeps for horizon_minutes, then triggers a horizon_expired event."""
    await asyncio.sleep(horizon_minutes * 60)
    await queue.put({
        "type": "horizon_expired",
        "decision_object_id": decision_id,
    })


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
    last_hill_climb_iterations = 1
    active_watcher_task: asyncio.Task | None = None

    while not _shutdown.is_set():
        try:
            event = await asyncio.wait_for(decision_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            continue

        logger.info(
            "Decision cycle started | trigger=%s phase=%s",
            event.get("type"), current_phase[0],
        )

        # 1. Close any active DecisionObject before starting a new one
        active = ledger.get_active()
        if active:
            if active_watcher_task and not active_watcher_task.done():
                active_watcher_task.cancel()
            
            close_reason = event.get("type", "unknown")
            comparator.close_cycle(active, last_hill_climb_iterations, close_reason)

        state = state_manager.snapshot()

        # Stage 1 + Stage 2
        sim_result = simulator.run(state)
        last_hill_climb_iterations = sim_result.hill_climb_iterations

        # ML2 annotates assumptions with breach probabilities
        ml1_vector = state.algo_health_vector
        sim_result.assumptions = ml2.annotate(
            sim_result.assumptions,
            state,
            time_horizon=240,   # default 4-hour window
            ml1_vector=ml1_vector,
        )

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
            ledger.commit(do)
        except LedgerWriteError as exc:
            logger.error("Ledger commit failed — actuation aborted | %s", exc)
            continue

        # Actuate — ONLY after successful ledger commit
        strategy_map = {
            "EMAStrategy": EMAStrategy(),
            "BollingerStrategy": BollingerStrategy(),
            "RSIStrategy": RSIStrategy(),
        }
        strategy = strategy_map.get(sim_result.strategy_name, EMAStrategy())
        signal = strategy.generate_signal(state, sim_result.tuned_params)

        telemetry = await executor.execute(
            signal=signal,
            params=sim_result.tuned_params,
            state=state,
            decision_object_id=str(do.id),
        )

        # Update ML1 behavioural vector with fresh telemetry
        new_vector = ml1.predict_proba(telemetry)
        await state_manager.update_algo_health(new_vector)

        # Check if ML1 should retrain
        if ml1.maybe_retrain():
            logger.info("ML1 retrained successfully")

        # Spawn time horizon watcher for this new decision
        active_watcher_task = asyncio.create_task(
            time_horizon_watcher(str(do.id), 240, decision_queue)
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
    decision_queue: asyncio.Queue = asyncio.Queue()
    state_manager = CausalStateManager(decision_queue=decision_queue)
    ledger = LedgerWriter()
    executor = Executor()

    ml1 = ML1BehaviourClassifier()
    ml2 = ML2BreachPredictor()
    regime = RegimeClassifier()
    comparator = OutcomeComparator(ledger, executor._api)

    strategies = [EMAStrategy(), BollingerStrategy(), RSIStrategy()]
    simulator = Simulator(strategies=strategies, regime_classifier=regime._dt)

    feed = BinanceFeed(state_manager=state_manager)
    current_phase: list[str] = ["bootstrap"]

    logger.info("Causal state initialised | phase=%s", current_phase[0])

    # Graceful shutdown handler
    signal.signal(signal.SIGINT, _handle_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigint)

    # Run all three coroutines concurrently
    try:
        await asyncio.gather(
            market_feed_loop(feed, ml1, state_manager),
            assumption_monitor_loop(state_manager, ledger, decision_queue),
            decision_cycle_loop(
                decision_queue, state_manager,
                simulator, ml1, ml2, ledger, executor, comparator, current_phase,
            ),
            _shutdown_watcher(feed),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error in main: %s", exc, exc_info=True)
    finally:
        await _graceful_shutdown(feed, ledger)


async def _shutdown_watcher(feed: BinanceFeed) -> None:
    """Wait for shutdown signal then stop the feed."""
    await _shutdown.wait()
    await feed.stop()


async def _graceful_shutdown(feed: BinanceFeed, ledger: LedgerWriter) -> None:
    """Mark any active DecisionObject as INTERRUPTED on exit."""
    active = ledger.get_active()
    if active:
        ledger.update_status(str(active.id), "INTERRUPTED")
        logger.info("Active entry marked INTERRUPTED | id=%s", active.id)
    logger.info("Graceful shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
