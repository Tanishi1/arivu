"""rl_main.py — RL comparison arms entry point (separate process from main.py)

Runs Arm 2 (Standard PPO) and Arm 3 (Causal RL) as async coroutines.
Shares the same Alpaca account, ledger DB, and Executor as main.py, but
runs as a SEPARATE PROCESS to isolate crashes from the Arivu causal agent.

A cross-process file lock (data/alpaca.lock) prevents concurrent Alpaca
order placement between this process and main.py. The Executor constructor
accepts the lock path — both processes must pass the same path.

Usage:
    python rl_main.py [--dry-run]

Run AFTER main.py is already running (or alongside it). The causal data
provider reads Arivu's live graph from the shared SQLite DB.

Arm 3 Causal Features:
    During the first run after fetch_historical.py + train_ppo.py have
    been completed, Arm 3 starts with zeroed causal dims. Once Arivu's
    Layer 1 tracker has validated edges in the DB (typically ~30 minutes
    into the first run), real causal features become available and Phase B
    fine-tuning begins automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = Path("logs/rl_arm.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | rl | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared lock path — MUST match the path used in main.py's Executor
# ---------------------------------------------------------------------------
ALPACA_LOCK_PATH = "data/alpaca.lock"

# Decision cadence — match main.py's causal agent cadence
DECISION_INTERVAL_S: float = 10.0

_shutdown = asyncio.Event()


def _handle_sigint(*_):
    logger.info("RL arms: shutdown signal received")
    _shutdown.set()


# ---------------------------------------------------------------------------
# Causal data provider for Arm 3
# ---------------------------------------------------------------------------

def _make_causal_data_provider(db_path: str):
    """Return a callable that reads live causal state from the shared DB.

    The provider reads the latest GraphSnapshot and Layer1 state from
    Arivu's SQLite DB. This avoids any shared memory between processes —
    everything goes through the DB.
    """
    from ml.layer1_tracker import Layer1Tracker
    from ml.causal_discovery import CausalDiscoveryEngine
    from ml.regime import RegimeClassifier

    l1 = Layer1Tracker(db_path=db_path)
    discovery = CausalDiscoveryEngine(db_path=db_path)
    regime_clf = RegimeClassifier()

    def provider():
        graph = discovery.get_latest()
        validated = l1.get_validated_edges() if graph else []
        regime = "unknown"
        return graph, l1, None, regime   # hypotheses not needed here (zeroed in obs.py)

    return provider


# ---------------------------------------------------------------------------
# Main RL decision loop
# ---------------------------------------------------------------------------

async def rl_decision_loop(
    arm_runner,
    state_manager,
    arm_name: str,
) -> None:
    """Run one RL arm's decision loop."""
    logger.info("RL decision loop started | arm=%s", arm_name)

    while not _shutdown.is_set():
        await asyncio.sleep(DECISION_INTERVAL_S)
        try:
            state = state_manager.snapshot()
            if state.price == 0.0:
                logger.debug("RL [%s]: price=0.0, skipping step", arm_name)
                continue
            await arm_runner.run_step(state, _shutdown)
        except Exception as exc:
            logger.error("RL [%s] loop error | %s", arm_name, exc, exc_info=True)

    logger.info("RL decision loop stopped | arm=%s", arm_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(dry_run: bool = False) -> None:
    logger.info("RL arms starting up | dry_run=%s", dry_run)

    db_path = os.getenv("SQLITE_PATH", "data/arivu.db")

    # Initialise shared components
    from ledger.models import init_db
    from ledger.writer import LedgerWriter
    from execution.executor import Executor
    from core.causal_state import CausalStateManager
    from core.stream import BinanceFeed
    from core.feature_bar import FeatureBarBuilder

    init_db()

    ledger = LedgerWriter()
    executor = Executor(cross_process_lock_path=ALPACA_LOCK_PATH)

    state_manager = CausalStateManager(decision_queue=asyncio.Queue(maxsize=10))
    feature_bar = FeatureBarBuilder()
    feed = BinanceFeed(state_manager=state_manager, feature_bar_builder=feature_bar)

    # Startup cleanup
    await asyncio.to_thread(ledger.cleanup_stale_trades)

    # --- Arm 2: Standard PPO ---
    from rl.live_runner import RLLiveRunner
    arm2 = RLLiveRunner(
        arm="standard",
        model_path="data/models/ppo_standard",
        ledger=ledger,
        executor=executor,
        causal_data_provider=None,
        dry_run=dry_run,
    )

    # --- Arm 3: Causal RL ---
    causal_provider = _make_causal_data_provider(db_path)
    arm3 = RLLiveRunner(
        arm="causal",
        model_path="data/models/ppo_causal",
        ledger=ledger,
        executor=executor,
        causal_data_provider=causal_provider,
        dry_run=dry_run,
    )

    logger.info("RL arms initialised | standard=ppo_standard | causal=ppo_causal")

    signal.signal(signal.SIGINT, _handle_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigint)

    async def _stop_feed():
        await _shutdown.wait()
        await feed.stop()

    results = await asyncio.gather(
        feed.run(),
        rl_decision_loop(arm2, state_manager, "ppo_standard"),
        rl_decision_loop(arm3, state_manager, "ppo_causal"),
        _stop_feed(),
        return_exceptions=True,
    )

    task_names = ["feed", "rl_standard", "rl_causal", "stop_feed"]
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.critical("Task '%s' died | %s", name, result, exc_info=result)

    logger.info("RL arms shut down cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL comparison arms live runner")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without placing real Alpaca orders (acceptance test mode)"
    )
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
