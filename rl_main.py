"""rl_main.py — RL comparison arms entry point (separate process from main.py)

Runs Arm 2 (Standard PPO) and Arm 3 (Causal RL) as async coroutines.
Each arm runs on its own separate Alpaca paper account, writing to the same
shared ledger DB as main.py, completely isolated and lock-free.

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
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — dedicated rl_arm.log plus console
# ---------------------------------------------------------------------------
LOG_PATH = Path("logs/rl_arm.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | rl | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# Decision cadence — match main.py's causal agent cadence
DECISION_INTERVAL_S: float = 10.0

# Heartbeat: log a summary every N completed steps per arm
HEARTBEAT_STEPS: int = 100
# Obs-vector stats: log detailed obs diagnostics every N steps
OBS_STATS_STEPS: int = 500

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

    l1 = Layer1Tracker(db_path=db_path)
    discovery = CausalDiscoveryEngine(db_path=db_path)

    def provider():
        graph = discovery.get_latest()
        return graph, l1, None, "unknown"

    return provider


# ---------------------------------------------------------------------------
# RL decision loop with rich logging
# ---------------------------------------------------------------------------

async def rl_decision_loop(
    arm_runner,
    state_manager,
    arm_name: str,
) -> None:
    """Run one RL arm's decision loop with detailed per-step logging.

    Logs emitted:
      - INFO  on startup
      - INFO  every HEARTBEAT_STEPS steps: step count, price, action distribution,
              position status, elapsed time
      - INFO  every OBS_STATS_STEPS steps: obs vector min/max/mean, feature health
      - INFO  on every BUY / SELL signal
      - ERROR on loop exceptions (with full traceback)
      - INFO  on clean shutdown
    """
    logger.info(
        "RL decision loop started | arm=%s | heartbeat_every=%d steps | "
        "obs_stats_every=%d steps | interval=%.0fs",
        arm_name, HEARTBEAT_STEPS, OBS_STATS_STEPS, DECISION_INTERVAL_S,
    )

    step_idx = 0
    action_counts: Counter = Counter({"HOLD": 0, "BUY": 0, "SELL": 0})
    loop_start = time.monotonic()

    while not _shutdown.is_set():
        await asyncio.sleep(DECISION_INTERVAL_S)
        try:
            state = state_manager.snapshot()

            if state.price == 0.0:
                logger.debug("RL [%s]: price=0.0, skipping step %d", arm_name, step_idx)
                continue

            # --- Build obs and predict ---
            import numpy as np
            obs = arm_runner._build_obs(state)
            model_action, _ = await asyncio.to_thread(
                arm_runner._model.predict, obs, deterministic=True
            )
            action_label = ["HOLD", "BUY", "SELL"][int(model_action)]
            action_counts[action_label] += 1

            # --- Log obs stats periodically ---
            if step_idx > 0 and step_idx % OBS_STATS_STEPS == 0:
                logger.info(
                    "RL [%s] OBS STATS | step=%d | obs_len=%d | "
                    "min=%.4f max=%.4f mean=%.4f std=%.4f | "
                    "nan_count=%d inf_count=%d",
                    arm_name, step_idx, len(obs),
                    float(np.nanmin(obs)), float(np.nanmax(obs)),
                    float(np.nanmean(obs)), float(np.nanstd(obs)),
                    int(np.isnan(obs).sum()), int(np.isinf(obs).sum()),
                )

            # Action masking — prevent degenerate actions
            # SELL is only valid when in_position; BUY only when not in_position.
            # The policy can get stuck outputting SELL forever when not in position
            # (common PPO failure mode with sparse rewards). Masking breaks this.
            if action_label == "SELL" and not arm_runner._in_position:
                action_label = "HOLD"  # nothing to sell — suppress
            elif action_label == "BUY" and arm_runner._in_position:
                action_label = "HOLD"  # already in position — suppress

            # --- Execute if not HOLD ---
            if action_label != "HOLD":
                logger.info(
                    "RL [%s] SIGNAL | step=%d | action=%s | price=%.4f | "
                    "in_position=%s | obs_mean=%.4f",
                    arm_name, step_idx, action_label, state.price,
                    arm_runner._in_position,
                    float(np.nanmean(obs)),
                )
                await arm_runner._execute_signal(action_label, state)


            # --- Phase B fine-tune ---
            if (arm_runner._phase_b_active and
                    step_idx > 0 and
                    step_idx % getattr(arm_runner, 'EPISODE_BARS', 240) == 0):
                await arm_runner._maybe_phase_b_finetune()

            arm_runner._step_count = step_idx + 1
            step_idx += 1

            # --- Heartbeat log ---
            if step_idx % HEARTBEAT_STEPS == 0:
                elapsed_min = (time.monotonic() - loop_start) / 60.0
                total_actions = sum(action_counts.values()) or 1
                logger.info(
                    "RL [%s] HEARTBEAT | step=%d | elapsed=%.1f min | "
                    "price=%.4f | in_position=%s | "
                    "actions HOLD=%d(%.0f%%) BUY=%d(%.0f%%) SELL=%d(%.0f%%)",
                    arm_name, step_idx, elapsed_min,
                    state.price,
                    arm_runner._in_position,
                    action_counts["HOLD"],
                    100 * action_counts["HOLD"] / total_actions,
                    action_counts["BUY"],
                    100 * action_counts["BUY"] / total_actions,
                    action_counts["SELL"],
                    100 * action_counts["SELL"] / total_actions,
                )

        except Exception as exc:
            logger.error(
                "RL [%s] loop error | step=%d | %s",
                arm_name, step_idx, exc, exc_info=True,
            )

    elapsed_total = (time.monotonic() - loop_start) / 60.0
    logger.info(
        "RL decision loop stopped | arm=%s | total_steps=%d | elapsed=%.1f min | "
        "final action dist: HOLD=%d BUY=%d SELL=%d",
        arm_name, step_idx, elapsed_total,
        action_counts["HOLD"], action_counts["BUY"], action_counts["SELL"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(dry_run: bool = False) -> None:
    logger.info("RL arms starting up | dry_run=%s", dry_run)

    db_path = os.getenv("SQLITE_PATH", "data/arivu.db")

    from ledger.models import init_db
    from ledger.writer import LedgerWriter
    from execution.executor import Executor
    from core.causal_state import CausalStateManager
    from core.stream import BinanceFeed
    from core.feature_bar import FeatureBarBuilder

    init_db()

    ledger = LedgerWriter()
    executor_standard = Executor(
        api_key=os.getenv("ALPACA_KEY_PPO_STANDARD"),
        api_secret=os.getenv("ALPACA_SECRET_PPO_STANDARD"),
    )
    executor_causal = Executor(
        api_key=os.getenv("ALPACA_KEY_PPO_CAUSAL"),
        api_secret=os.getenv("ALPACA_SECRET_PPO_CAUSAL"),
    )

    state_manager = CausalStateManager(decision_queue=asyncio.Queue(maxsize=10))
    feature_bar = FeatureBarBuilder(
        on_bar_closed=state_manager.update_graph_features,
    )
    feed = BinanceFeed(state_manager=state_manager, feature_bar_builder=feature_bar)

    # Startup cleanup
    await asyncio.to_thread(ledger.cleanup_stale_trades)

    # --- Arm 2: Standard PPO ---
    from rl.live_runner import RLLiveRunner
    arm2 = RLLiveRunner(
        arm="standard",
        model_path="data/models/ppo_standard",
        ledger=ledger,
        executor=executor_standard,
        causal_data_provider=None,
        dry_run=dry_run,
    )

    # --- Arm 3: Causal RL ---
    causal_provider = _make_causal_data_provider(db_path)
    arm3 = RLLiveRunner(
        arm="causal",
        model_path="data/models/ppo_causal",
        ledger=ledger,
        executor=executor_causal,
        causal_data_provider=causal_provider,
        dry_run=dry_run,
    )

    logger.info(
        "RL arms initialised | standard=ppo_standard | causal=ppo_causal_feature | "
        "phase_b_duration=48h | decision_interval=%.0fs",
        DECISION_INTERVAL_S,
    )
    logger.info(
        "RL logging: heartbeat every %d steps (~%.0f min) | "
        "obs stats every %d steps (~%.0f min)",
        HEARTBEAT_STEPS, HEARTBEAT_STEPS * DECISION_INTERVAL_S / 60,
        OBS_STATS_STEPS, OBS_STATS_STEPS * DECISION_INTERVAL_S / 60,
    )

    signal.signal(signal.SIGINT, _handle_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigint)

    async def _stop_feed():
        await _shutdown.wait()
        await feed.stop()

    results = await asyncio.gather(
        feed.run(),
        rl_decision_loop(arm2, state_manager, "ppo_standard"),
        rl_decision_loop(arm3, state_manager, "ppo_causal_feature"),
        _stop_feed(),
        return_exceptions=True,
    )

    task_names = ["feed", "rl_standard", "rl_causal", "stop_feed"]
    for name, result in zip(task_names, results):
        if isinstance(result, Exception):
            logger.critical("Task '%s' died | %s", name, result, exc_info=result)

    # Graceful shutdown: close Alpaca positions and mark active DOs INTERRUPTED
    await _graceful_shutdown_rl(ledger, executor_standard, executor_causal)

    logger.info("RL arms shut down cleanly")


async def _graceful_shutdown_rl(
    ledger,
    executor_standard,
    executor_causal,
) -> None:
    """Mark any stale RL DecisionObjects INTERRUPTED and close Alpaca positions.

    Called once at the end of rl_main.main() regardless of shutdown reason.
    Mirrors the _graceful_shutdown() function in main.py.
    """
    from ledger.models import SessionLocal, DecisionObjectRow

    # Mark any ACTIVE or COMMITTED RL trades as INTERRUPTED in the shared ledger
    try:
        with SessionLocal() as session:
            stale = session.query(DecisionObjectRow).filter(
                DecisionObjectRow.strategy_name.in_(["ppo_standard", "ppo_causal_feature"]),
                DecisionObjectRow.status.in_(["ACTIVE", "COMMITTED"]),
            ).all()
            for row in stale:
                row.status = "INTERRUPTED"
                logger.info(
                    "RL shutdown: marked INTERRUPTED | id=%s strategy=%s",
                    row.id[:8], row.strategy_name,
                )
            session.commit()
    except Exception as exc:
        logger.error("RL shutdown: failed to mark stale trades | %s", exc)

    # Close any open Alpaca positions on both paper accounts
    for arm_name, executor in [("ppo_standard", executor_standard), ("ppo_causal", executor_causal)]:
        try:
            await asyncio.to_thread(executor._api.close_position, "SOLUSD")
            logger.info("RL shutdown: closed open position on %s account", arm_name)
        except Exception as exc:
            if "position does not exist" not in str(exc).lower():
                logger.warning("RL shutdown: could not close position on %s | %s", arm_name, exc)

        # Also cancel any pending limit orders
        try:
            open_orders = await asyncio.to_thread(executor._api.list_orders, status="open")
            for o in open_orders:
                try:
                    await asyncio.to_thread(executor._api.cancel_order, o.id)
                except Exception:
                    pass
            if open_orders:
                logger.info("RL shutdown: cancelled %d open orders on %s", len(open_orders), arm_name)
        except Exception as exc:
            logger.warning("RL shutdown: could not cancel open orders on %s | %s", arm_name, exc)

    logger.info("RL graceful shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL comparison arms live runner")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without placing real Alpaca orders (acceptance test mode)",
    )
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
