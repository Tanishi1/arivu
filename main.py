"""main.py — Arivu entry point

Four asyncio coroutines running concurrently in ONE Python process:
  1. market_feed_loop       — processes every Binance WebSocket tick
  2. assumption_monitor_loop — checks active assumptions every 60 seconds
  3. decision_cycle_loop    — legacy arm: EMA/Bollinger/RSI strategies (comparison arm)
  4. causal_agent_loop      — causal agent: discovers rules, generates hypotheses, trades

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
from datetime import datetime, timezone  # B4 FIX: was re-imported inside event loop body
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
from core.decision_utils import annotate_proximity
from core.stream import BinanceFeed
from core.monitor import assumption_monitor_loop
from core.simulator import Simulator
from core.schemas import DecisionObject
from core.feature_bar import FeatureBarBuilder, VARIABLE_NAMES
from core.twin_simulator import TwinSimulator
from execution.executor import Executor
from ledger.models import init_db
from ledger.writer import LedgerWriter, LedgerWriteError
from ml.ml1 import ML1BehaviourClassifier
from ml.ml2 import ML2BreachPredictor
from ml.regime import RegimeClassifier
from ml.causal_discovery import CausalDiscoveryEngine
from ml.layer1_tracker import Layer1Tracker
from ml.hypothesis_generator import HypothesisGenerator, MIN_TRADE_SCORE
from ml.trust_updater import Layer2TrustUpdater
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

# ---------------------------------------------------------------------------
# Causal agent configuration
# ---------------------------------------------------------------------------

GRAPH_REFRESH_S: int = 300          # 5 minutes between PCMCI runs
CAPITAL_CAUSAL_AGENT: float = 100_000  # $100k allocated to causal agent
CAPITAL_LEGACY_ARM: float = 100_000.0   # $100k for legacy strategies (run synthetically)


def _handle_sigint(*_):
    logger.info("Shutdown signal received")
    _shutdown.set()


# ---------------------------------------------------------------------------
# Time Horizon Watcher
# ---------------------------------------------------------------------------

async def time_horizon_watcher(
    decision_id: str,
    horizon_minutes: float | int,  # B5 FIX: crash-recovery passes a float (remaining_m)
    queue: asyncio.Queue,
) -> None:
    """Sleeps for horizon_minutes, then triggers a horizon_expired event."""
    await asyncio.sleep(horizon_minutes * 60)
    event = {
        "type": "horizon_expired",
        "decision_object_id": decision_id,
    }
    # Retry until the queue has space — but stop cooperatively if shutdown is
    # requested. Without this check the loop runs forever if the decision
    # cycle is stuck, making shutdown non-cooperative.
    while not _shutdown.is_set():
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            await asyncio.sleep(1.0)
    # Best-effort final attempt after shutdown so the event is not silently lost.
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.warning(
            "time_horizon_watcher: queue full at shutdown — horizon_expired dropped | id=%s",
            decision_id,
        )


# ---------------------------------------------------------------------------
# Coroutine 1 — Market feed loop
# ---------------------------------------------------------------------------

async def market_feed_loop(
    feed: BinanceFeed,
    state_manager: CausalStateManager,
) -> None:
    """Stream Binance WebSocket ticks into CausalState.

    HIGH-1 FIX: ml1 parameter removed. ML1 inference runs in decision_cycle_loop
    (line ~378), not here. The parameter was dead — market_feed_loop only calls
    feed.run() and never invoked ml1. Keeping it implied ML1 was called here,
    misleading any reader expecting health-vector updates from the feed loop.
    """
    await feed.run()


# ---------------------------------------------------------------------------
# Coroutine 2 — Decision cycle loop
# ---------------------------------------------------------------------------

async def legacy_strategy_loop(
    state_manager: CausalStateManager,
    ledger: LedgerWriter,
    executor: Executor,
    comparator: OutcomeComparator,
    ml1: ML1BehaviourClassifier,
    ml2: ML2BreachPredictor,
    layer1: Layer1Tracker,
    regime_classifier: RegimeClassifier,
    current_phase: list[str],
) -> None:
    """
    Runs EMA, Bollinger, RSI strategies independently on every decision cycle.
    Completely isolated from the causal agent.
    Tracks its own active trade locally — never calls ledger.get_active().
    """
    from ml.meta_optimizer import TradeOutcome

    ema_strategy = EMAStrategy()
    bollinger_strategy = BollingerStrategy()
    rsi_strategy = RSIStrategy()

    # --- Check if legacy strategies have been permanently stopped in a previous run ---
    if layer1.is_legacy_permanently_stopped():
        logger.info(
            "legacy_strategy_loop: legacy_stopped flag found in DB — "
            "causal graph was previously validated. Legacy strategies will NOT run. "
            "This is expected on all runs after the first."
        )
        return

    # Recover any active legacy trade from DB on startup
    # (handles case where process restarted mid-trade)
    _legacy_active_do: DecisionObject | None = None

    while not _shutdown.is_set():
        try:
            state = state_manager.snapshot()
            current_regime = regime_classifier.classify(
                volatility=state.volatility,
                spread=state.spread,
                trend_strength=state.trend_strength,
                volume=state.volume,
                trend_slope=state.trend_slope,
            )

            # --- Check if active legacy trade needs closing ---
            if _legacy_active_do is not None:
                if isinstance(_legacy_active_do.timestamp_committed, str):
                    commit_time = datetime.fromisoformat(
                        _legacy_active_do.timestamp_committed
                    )
                else:
                    commit_time = _legacy_active_do.timestamp_committed

                elapsed_s = (
                    datetime.now(timezone.utc) - commit_time
                ).total_seconds()

                if elapsed_s >= STRATEGY_HORIZON_MINUTES * 60:
                    logger.info(
                        "legacy_strategy_loop: horizon expired | id=%s",
                        str(_legacy_active_do.id)[:8],
                    )
                    try:
                        record = await asyncio.to_thread(
                            comparator.close_cycle,
                            _legacy_active_do,
                            "horizon_expired",
                        )
                        if record:
                            # Feed optimizer with outcome from this legacy trade
                            # using current optimizer params for this regime
                            current_params = layer1._optimizer.get_params(
                                current_regime
                            )
                            trade_outcome = TradeOutcome(
                                meta_params_used=current_params.to_dict(),
                                regime=current_regime,
                                predicted_return=0.0,
                                actual_return=(
                                    record.actual_pnl / 100000.0
                                ),
                                pnl_usd=record.actual_pnl,
                                causal_chain_held=True,
                                regime_was_stable=True,
                            )
                            await asyncio.to_thread(
                                layer1._optimizer.update, trade_outcome
                            )
                            logger.info(
                                "legacy_strategy_loop: MetaOptimizer updated | "
                                "regime=%s pnl=%.2f",
                                current_regime, record.actual_pnl,
                            )
                            # Check if ML2 should retrain
                            ml2_retrained, ml2_brier = await asyncio.to_thread(
                                ml2.maybe_retrain, ledger, current_phase[0]
                            )
                            if ml2_retrained:
                                logger.info("ML2 retrained | brier_score=%.4f", ml2_brier or 0.0)
                                if current_phase[0] == "bootstrap":
                                    current_phase[0] = "trained"
                                    logger.info("ML2 transitioned to trained mode")

                            # Check if Regime Classifier should retrain (every 10 closed trades, >= 40)
                            try:
                                closed_count = await asyncio.to_thread(ledger.count_closed)
                                if closed_count >= 40 and closed_count % 10 == 0:
                                    entries = await asyncio.to_thread(ledger.get_closed_entries_for_regime)
                                    regime_retrained = await asyncio.to_thread(regime_classifier.run_kmeans_and_retrain, entries)
                                    if regime_retrained:
                                        logger.info(
                                            "Regime Classifier K-Means & Decision Tree retrained | "
                                            "closed_count=%d", closed_count
                                        )
                            except Exception as regime_exc:
                                logger.error("Failed to retrain Regime Classifier | %s", regime_exc)
                    except Exception as exc:
                        logger.error(
                            "legacy_strategy_loop: failed to close | %s", exc
                        )
                        await asyncio.to_thread(
                            ledger.update_status,
                            str(_legacy_active_do.id),
                            "INTERRUPTED",
                        )
                    finally:
                        _legacy_active_do = None

                await asyncio.sleep(10)
                continue

            # --- No active trade — evaluate strategies ---
            # Check first whether the causal graph has now been validated:
            # if so, stop legacy strategies permanently and set the DB flag.
            if layer1.has_stable_edges(current_regime):
                logger.info(
                    "legacy_strategy_loop: causal graph has stable validated edges — "
                    "legacy strategies stopping permanently. "
                    "Causal agent will trade independently from now on."
                )
                await asyncio.to_thread(layer1.set_legacy_permanently_stopped)
                return

            for strategy in [ema_strategy, bollinger_strategy, rsi_strategy]:
                strategy_name = strategy.__class__.__name__
                tuned_params = strategy.get_default_params()
                signal = strategy.generate_signal(state, tuned_params)

                if signal not in ("BUY", "CLOSE"):
                    continue

                logger.info(
                    "legacy_strategy_loop | %s signal=%s",
                    strategy_name, signal,
                )

                assumptions = strategy.get_assumptions(state, tuned_params)
                assumptions = annotate_proximity(assumptions, state)
                # Annotate with ML2 breach risk predictor
                ml1_vector = state.algo_health_vector
                assumptions = ml2.annotate(
                    assumptions,
                    state,
                    time_horizon=STRATEGY_HORIZON_MINUTES,
                    ml1_vector=ml1_vector,
                )

                decision = DecisionObject(
                    strategy_name=strategy_name,
                    tuned_params={
                        "decision_source": f"{strategy_name}_legacy",
                        **tuned_params,
                    },
                    market_state_snapshot=state.model_dump(
                        mode="json",
                        exclude={
                            "algo_health_vector", "last_tick_timestamp",
                            "active_strategy", "position_size",
                            "capital_deployed", "timestamp",
                        },
                    ),
                    algo_health_vector=state.algo_health_vector,
                    assumptions=assumptions,
                    projected_pnl=strategy.evaluate(state, tuned_params)
                    * 100000,
                    confidence=strategy.evaluate(state, tuned_params),
                    phase=current_phase[0],
                    hill_climb_iterations=0,
                    meta_params=None,
                    causal_chain_snapshot=None,
                )

                try:
                    await asyncio.to_thread(ledger.commit, decision)
                except LedgerWriteError as exc:
                    logger.error(
                        "legacy_strategy_loop: commit failed | %s", exc
                    )
                    continue

                try:
                    telemetry = await executor.execute(
                        signal=signal,
                        params=tuned_params,
                        state=state,
                        decision_object_id=str(decision.id),
                    )
                    await asyncio.to_thread(
                        ledger.update_status, str(decision.id), "ACTIVE"
                    )
                    # Feed ML1 with execution telemetry
                    new_vector = ml1.predict_proba(
                        telemetry, is_real_order=(signal == "BUY")
                    )
                    await state_manager.update_algo_health(new_vector)
                    # Track this as the active legacy trade
                    _legacy_active_do = decision

                except Exception as exc:
                    logger.error(
                        "legacy_strategy_loop: execution failed | %s", exc
                    )
                    await asyncio.to_thread(
                        ledger.update_status,
                        str(decision.id),
                        "EXECUTION_FAILED",
                    )

                # Only one strategy fires per cycle
                break

        except Exception as exc:
            logger.error(
                "legacy_strategy_loop error: %s", exc, exc_info=True
            )

        await asyncio.sleep(10)
# ---------------------------------------------------------------------------
# Coroutine 4 — Causal Agent Loop (Tasks 10, 11, 13)
# ---------------------------------------------------------------------------

async def causal_agent_loop(
    state_manager: CausalStateManager,
    feature_bar: "FeatureBarBuilder",
    discovery_engine: "CausalDiscoveryEngine",
    layer1: "Layer1Tracker",
    hyp_gen: "HypothesisGenerator",
    twin_sim: "TwinSimulator",
    trust_updater: "Layer2TrustUpdater",
    ledger: LedgerWriter,
    executor: Executor,
    comparator: OutcomeComparator,
    ml1: "ML1BehaviourClassifier",
    ml2: "ML2BreachPredictor",
    regime_classifier: "RegimeClassifier",
    current_phase: list[str],
) -> None:
    """Autonomous causal discovery → hypothesis generation → trade execution loop.

    Runs independently from decision_cycle_loop (the legacy arm). These two loops:
      - Read from the same CausalState (read-only, no lock needed).
      - Write to the same ledger but tagged decision_source="causal_agent".
      - Share the executor but have isolated capital pools.
      - Do NOT share causal graph, hypotheses, or strategy signals.

    Every cycle this loop:
      1. Checks if FeatureBarBuilder has enough data (60+ bars = ~10 min).
      2. Refreshes the PCMCI graph every GRAPH_REFRESH_S seconds.
      3. Records the new graph in Layer 1 stability tracker.
      4. Generates ranked hypotheses from validated causal chains.
      5. If best hypothesis score > MIN_TRADE_SCORE: simulate, commit, execute.
      6. Writes observability JSONL (Task 11).
      7. Monitors active causal trades for trajectory breach.
      8. On close: updates Layer 2 trust scores.
    """
    import json
    import time
    from ml.meta_optimizer import TradeOutcome

    # Observability log (Task 11)
    obs_log_path = Path("logs/causal_agent.jsonl")
    obs_log_path.parent.mkdir(parents=True, exist_ok=True)

    _last_graph_refresh: float = 0.0
    _current_graph = None
    _active_hyp = None        # CausalHypothesis for current open trade
    _active_traj = None       # SimulatedTrajectory for current open trade
    _active_entry_price: float = 0.0
    _active_do_id: str | None = None
    _active_entry_time: float = 0.0
    _prev_regime: str | None = None

    BAR_POLL_S = 10   # poll every bar width

    logger.info("Causal agent loop started | graph_refresh=%ds", GRAPH_REFRESH_S)

    while not _shutdown.is_set():
        await asyncio.sleep(BAR_POLL_S)

        try:
            state = state_manager.snapshot()
            current_regime = regime_classifier.classify(
                volatility=state.volatility,
                spread=state.spread,
                trend_strength=state.trend_strength,
                volume=state.volume,
                trend_slope=state.trend_slope,
            )

            if current_regime != _prev_regime:
                logger.info(
                    "Market Regime Change detected | %s -> %s | Volatility=%.4f, Spread=%.4f, Trend=%.4f",
                    _prev_regime or "INITIAL", current_regime, state.volatility, state.spread, state.trend_strength
                )
                _prev_regime = current_regime

            # --- Step 1: Refresh causal graph every GRAPH_REFRESH_S ---
            now = time.monotonic()
            if (now - _last_graph_refresh) >= GRAPH_REFRESH_S:
                if feature_bar.is_ready():
                    try:
                        matrix, var_names = feature_bar.get_feature_matrix()
                        logger.info(
                            "CausalAgent: running PCMCI | bars=%d vars=%d",
                            matrix.shape[0], matrix.shape[1],
                        )
                        _current_graph = await asyncio.to_thread(
                            discovery_engine.run, matrix, var_names
                        )
                        await asyncio.to_thread(layer1.record_run, _current_graph, current_regime)
                        _last_graph_refresh = now
                        logger.info(
                            "CausalAgent: graph refreshed | %s", _current_graph.summary()
                        )
                    except ValueError as exc:
                        logger.debug("CausalAgent: not enough bars yet — %s", exc)
                else:
                    n = feature_bar.n_bars_ready()
                    logger.info(
                        "[ML1 Feature Store] Warm-up progress: %d/60 bars (Granger ready: %d/40) | Features: vol=%.4f, spread=%.4f, trend=%.4f",
                        n, n, state.volatility, state.spread, state.trend_strength
                    )

            # --- Step 2: Monitor active causal trade for trajectory breach ---
            if _active_hyp and _active_traj and _active_do_id:
                elapsed = int(time.monotonic() - _active_entry_time)
                if state.price > 0 and _active_entry_price > 0:
                    actual_return = (state.price - _active_entry_price) / _active_entry_price
                    is_breach = twin_sim.check_breach(_active_traj, actual_return, elapsed)
                    if is_breach:
                        logger.warning(
                            "CausalAgent: trajectory breach detected | id=%s | elapsed=%ds",
                            _active_do_id[:8], elapsed,
                        )
                        # Close the trade — record outcome and update Layer 2
                        try:
                            active_do = await asyncio.to_thread(
                                ledger.get_decision_object, _active_do_id
                            )
                            if active_do:
                                record = await asyncio.to_thread(
                                    comparator.close_cycle, active_do, "assumption_breach", _current_graph
                                )
                                if record and _active_hyp:
                                    outcome_correct = (
                                        (_active_hyp.predicted_direction == "up" and actual_return > 0)
                                        or (_active_hyp.predicted_direction == "down" and actual_return < 0)
                                    )
                                    await asyncio.to_thread(
                                        trust_updater.update,
                                        _active_hyp,
                                        outcome_correct,
                                        actual_return,
                                        _active_traj.predicted_price_return,
                                    )
                                    # Retrieve stored breach risk from the DO's meta_params
                                    avg_breach_risk = (active_do.meta_params or {}).get("avg_breach_risk", 0.5)

                                    # Compute ML2 calibration: how accurate were the pre-trade predictions?
                                    breach_log = record.assumptions_breached
                                    calibration_scores = []
                                    for assumption in active_do.assumptions:
                                        actually_breached = 1.0 if assumption.name in breach_log else 0.0
                                        error = abs(assumption.breach_risk - actually_breached)
                                        calibration_scores.append(1.0 - error)

                                    ml2_calibration = (
                                        sum(calibration_scores) / len(calibration_scores)
                                        if calibration_scores else 0.5
                                    )

                                    logger.info(
                                        "CausalAgent: ML2 calibration | avg_breach_risk=%.3f | "
                                        "ml2_calibration=%.3f | assumptions_breached=%s",
                                        avg_breach_risk, ml2_calibration, breach_log,
                                    )

                                    # Update MetaParameterOptimizer
                                    trade_outcome = TradeOutcome(
                                        meta_params_used=active_do.meta_params or {},
                                        regime=current_regime,
                                        predicted_return=_active_traj.predicted_price_return,
                                        actual_return=actual_return,
                                        pnl_usd=record.actual_pnl,
                                        causal_chain_held=False,
                                        regime_was_stable=True,
                                        avg_breach_risk=avg_breach_risk,
                                        ml2_calibration=ml2_calibration,
                                    )
                                    await asyncio.to_thread(layer1._optimizer.update, trade_outcome)
                                    # Check if ML2 should retrain
                                    ml2_retrained, ml2_brier = await asyncio.to_thread(
                                        ml2.maybe_retrain, ledger, current_phase[0]
                                    )
                                    if ml2_retrained:
                                        logger.info("ML2 retrained | brier_score=%.4f", ml2_brier or 0.0)
                                        if current_phase[0] == "bootstrap":
                                            current_phase[0] = "trained"
                                            logger.info("ML2 transitioned to trained mode")

                                    # Check if Regime Classifier should retrain (every 10 closed trades, >= 40)
                                    try:
                                        closed_count = await asyncio.to_thread(ledger.count_closed)
                                        if closed_count >= 40 and closed_count % 10 == 0:
                                            entries = await asyncio.to_thread(ledger.get_closed_entries_for_regime)
                                            regime_retrained = await asyncio.to_thread(regime_classifier.run_kmeans_and_retrain, entries)
                                            if regime_retrained:
                                                logger.info(
                                                    "Regime Classifier K-Means & Decision Tree retrained | "
                                                    "closed_count=%d", closed_count
                                                )
                                    except Exception as regime_exc:
                                        logger.error("Failed to retrain Regime Classifier | %s", regime_exc)
                        except Exception as exc:
                            logger.error("CausalAgent: failed to close breach trade | %s", exc)
                        finally:
                            _active_hyp = None
                            _active_traj = None
                            _active_do_id = None
                            _active_entry_price = 0.0
                            _active_entry_time = 0.0

                # Check mathematical horizon expiry for causal agent trades
                if _active_traj and elapsed >= _active_traj.time_horizon_s:
                    logger.info(
                        "CausalAgent: horizon expired | id=%s | elapsed=%ds | target=%ds",
                        _active_do_id[:8] if _active_do_id else "?", elapsed, _active_traj.time_horizon_s,
                    )
                    try:
                        active_do = await asyncio.to_thread(
                            ledger.get_decision_object, _active_do_id
                        )
                        if active_do:
                            actual_return = (
                                (state.price - _active_entry_price) / _active_entry_price
                                if _active_entry_price > 0 else 0.0
                            )
                            record = await asyncio.to_thread(
                                comparator.close_cycle, active_do, "horizon_expired", _current_graph
                            )
                            if record and _active_hyp:
                                outcome_correct = (
                                    (_active_hyp.predicted_direction == "up" and actual_return > 0)
                                    or (_active_hyp.predicted_direction == "down" and actual_return < 0)
                                )
                                await asyncio.to_thread(
                                    trust_updater.update,
                                    _active_hyp,
                                    outcome_correct,
                                    actual_return,
                                    _active_traj.predicted_price_return if _active_traj else 0.0,
                                )
                                # Retrieve stored breach risk from the DO's meta_params
                                avg_breach_risk = (active_do.meta_params or {}).get("avg_breach_risk", 0.5)

                                # Compute ML2 calibration: how accurate were the pre-trade predictions?
                                breach_log = record.assumptions_breached
                                calibration_scores = []
                                for assumption in active_do.assumptions:
                                    actually_breached = 1.0 if assumption.name in breach_log else 0.0
                                    error = abs(assumption.breach_risk - actually_breached)
                                    calibration_scores.append(1.0 - error)

                                ml2_calibration = (
                                    sum(calibration_scores) / len(calibration_scores)
                                    if calibration_scores else 0.5
                                )

                                logger.info(
                                    "CausalAgent: ML2 calibration | avg_breach_risk=%.3f | "
                                    "ml2_calibration=%.3f | assumptions_breached=%s",
                                    avg_breach_risk, ml2_calibration, breach_log,
                                )

                                # Update MetaParameterOptimizer
                                trade_outcome = TradeOutcome(
                                    meta_params_used=active_do.meta_params or {},
                                    regime=current_regime,
                                    predicted_return=_active_traj.predicted_price_return if _active_traj else 0.0,
                                    actual_return=actual_return,
                                    pnl_usd=record.actual_pnl,
                                    causal_chain_held=True,
                                    regime_was_stable=True,
                                    avg_breach_risk=avg_breach_risk,
                                    ml2_calibration=ml2_calibration,
                                )
                                await asyncio.to_thread(layer1._optimizer.update, trade_outcome)
                                # Check if ML2 should retrain
                                ml2_retrained, ml2_brier = await asyncio.to_thread(
                                    ml2.maybe_retrain, ledger, current_phase[0]
                                )
                                if ml2_retrained:
                                    logger.info("ML2 retrained | brier_score=%.4f", ml2_brier or 0.0)
                                    if current_phase[0] == "bootstrap":
                                        current_phase[0] = "trained"
                                        logger.info("ML2 transitioned to trained mode")

                                # Check if Regime Classifier should retrain (every 10 closed trades, >= 40)
                                try:
                                    closed_count = await asyncio.to_thread(ledger.count_closed)
                                    if closed_count >= 40 and closed_count % 10 == 0:
                                        entries = await asyncio.to_thread(ledger.get_closed_entries_for_regime)
                                        regime_retrained = await asyncio.to_thread(regime_classifier.run_kmeans_and_retrain, entries)
                                        if regime_retrained:
                                            logger.info(
                                                "Regime Classifier K-Means & Decision Tree retrained | "
                                                "closed_count=%d", closed_count
                                            )
                                except Exception as regime_exc:
                                    logger.error("Failed to retrain Regime Classifier | %s", regime_exc)
                    except Exception as exc:
                        logger.error("CausalAgent: failed to close horizon trade | %s", exc)
                    finally:
                        _active_hyp = None
                        _active_traj = None
                        _active_do_id = None
                        _active_entry_price = 0.0
                        _active_entry_time = 0.0

            # --- Step 3: Skip new trade if already in position ---
            if _active_do_id:
                continue

            # --- Step 4: Generate hypotheses from current graph ---
            if _current_graph is None:
                _obs_write(obs_log_path, state, None, [], None, "no_graph_yet")
                continue

            params = layer1._optimizer.get_params(current_regime)
            logger.info(
                "Causal Agent evaluating hypotheses | regime=%s | Active Params: k_runs=%d, min_runs=%d, threshold=%.2f",
                current_regime, params.k_runs, params.min_runs, params.threshold
            )

            hypotheses = await asyncio.to_thread(hyp_gen.generate, _current_graph, current_regime)
            
            if not hypotheses:
                continue

            from core.schemas import Assumption
            for hyp in hypotheses:
                if not hyp.chain:
                    hyp.avg_breach_risk = 0.5
                    continue
            
                proxy_assumptions = []
                for edge in hyp.chain[:3]:  # cap at 3 edges to avoid ML2 overload
                    # Use the actual current value of the edge's source variable from
                    # CausalState — not a sentinel. This gives ML2 real feature values
                    # to score against its training distribution.
                    source_var = edge.source
                    current_val = getattr(state, source_var, None)
            
                    if current_val is None:
                        # Variable exists in graph but not in CausalState — skip rather
                        # than feeding ML2 a fabricated value.
                        continue
            
                    # Proximity: how far is the current value from triggering a breach?
                    # Use Layer 1 score as a proxy for how stable this edge has been.
                    proximity = 1.0 - hyp.layer1_score
            
                    proxy_assumptions.append(Assumption(
                        name=f"edge_{edge.source}_{edge.target}_lag{edge.lag}",
                        variable=source_var,       # real variable, not a sentinel
                        operator="lt",
                        threshold=abs(current_val) * 1.5 if current_val != 0 else 1.0,
                        current_value=current_val,    # real current value from CausalState
                        proximity=proximity,
                    ))
            
                if not proxy_assumptions:
                    hyp.avg_breach_risk = 0.5
                    continue
            
                annotated_proxy = ml2.annotate(
                    proxy_assumptions,
                    state,
                    hyp.time_horizon_seconds / 60.0,
                    state.algo_health_vector,
                )
                hyp.avg_breach_risk = (
                    sum(a.breach_risk for a in annotated_proxy) / len(annotated_proxy)
                    if annotated_proxy else 0.5
                )

            best_hyp = hypotheses[0]

            if best_hyp is None or (best_hyp.composite_score < MIN_TRADE_SCORE and not best_hyp.is_escape_valve):
                score_str = f"{best_hyp.composite_score:.4f}" if best_hyp else "none"
                _obs_write(obs_log_path, state, _current_graph, hypotheses, None,
                           f"score_too_low:{score_str}")
                continue

            # --- Step 5: Simulate — pick best hypothesis ---
            # Thread-safe: use get_feature_matrix() which acquires the lock,
            # rather than directly accessing feature_bar._bars (race with tick ingestion).
            try:
                matrix, var_names = feature_bar.get_feature_matrix()
                latest_bar = matrix[-1].tolist()
                current_state_dict = dict(zip(var_names, latest_bar))
                logger.debug(
                    "CausalAgent: current_state_dict keys=%s",
                    list(current_state_dict.keys()),
                )
            except ValueError:
                # Not enough bars ready yet — skip this cycle
                continue
            selected_hyp, trajectory = twin_sim.select_best(
                hypotheses, _current_graph, current_state_dict
            )

            if selected_hyp is None or trajectory is None:
                _obs_write(obs_log_path, state, _current_graph, hypotheses, selected_hyp,
                           "zero_expected_value")
                continue

            # --- Step 6: Commit Decision Object ---
            trade_signal = "BUY" if selected_hyp.predicted_direction == "up" else "CLOSE"

            if trade_signal == "CLOSE":
                # Check if we actually have a position to close
                try:
                    pos = await asyncio.to_thread(executor._api.get_position, "SOLUSD")
                    if not pos or float(pos.qty) <= 0:
                        logger.debug("CausalAgent: CLOSE signal skipped — no open position")
                        continue
                except Exception:
                    logger.debug("CausalAgent: CLOSE signal skipped — no position exists")
                    continue

            # Represent the causal chain as the "assumptions" for ledger compatibility
            from core.schemas import Assumption
            from pydantic import ValidationError
            causal_assumptions = []
            for edge in selected_hyp.chain[:3]:
                source_var = edge.source
                current_val = getattr(state, source_var, None)
                if current_val is None:
                    continue
                proximity = 1.0 - selected_hyp.layer1_score
                try:
                    causal_assumptions.append(Assumption(
                        name=f"causal_edge|{edge.source}|{edge.target}|{edge.lag}",
                        variable=source_var,
                        operator="lt",
                        threshold=abs(current_val) * 1.5 if current_val != 0 else 1.0,
                        current_value=current_val,
                        proximity=proximity,
                        breach_risk=1.0 - selected_hyp.layer2_score,
                    ))
                except ValidationError:
                    continue

            # Annotate with ML2 breach risk predictor
            ml1_vector = state.algo_health_vector
            causal_assumptions = ml2.annotate(
                causal_assumptions,
                state,
                time_horizon=selected_hyp.time_horizon_seconds / 60.0,
                ml1_vector=ml1_vector,
            )

            # After ml2.annotate() returns causal_assumptions:
            if causal_assumptions:
                avg_breach_risk = sum(
                    a.breach_risk for a in causal_assumptions
                ) / len(causal_assumptions)
            else:
                avg_breach_risk = 0.5

            risk_multiplier = max(0.10, 1.0 - avg_breach_risk)

            original_fraction = 0.01 if selected_hyp.is_escape_valve else 0.10
            adjusted_params = {
                "position_fraction": round(original_fraction * risk_multiplier, 4)
            }

            logger.info(
                "CausalAgent: ML2 position sizing | avg_breach_risk=%.3f | "
                "multiplier=%.3f | fraction %.4f → %.4f",
                avg_breach_risk, risk_multiplier,
                original_fraction, adjusted_params["position_fraction"],
            )

            do = DecisionObject(
                strategy_name="CausalAgent",
                tuned_params={
                    "hypothesis_id": selected_hyp.id,
                    "graph_version_id": selected_hyp.graph_version_id,
                    "chain_summary": selected_hyp.chain_summary(),
                    "predicted_direction": selected_hyp.predicted_direction,
                    "predicted_magnitude": selected_hyp.predicted_magnitude,
                    "time_horizon_seconds": selected_hyp.time_horizon_seconds,
                    "layer1_score": selected_hyp.layer1_score,
                    "layer2_score": selected_hyp.layer2_score,
                    "composite_score": selected_hyp.composite_score,
                    "decision_source": "causal_agent",
                    "capital_allocated": CAPITAL_CAUSAL_AGENT,
                },
                market_state_snapshot=state.model_dump(
                    mode="json",
                    exclude={
                        "algo_health_vector", "last_tick_timestamp",
                        "active_strategy", "position_size",
                        "capital_deployed", "timestamp",
                    },
                ),
                algo_health_vector=state.algo_health_vector,
                assumptions=causal_assumptions,
                projected_pnl=float(trajectory.predicted_price_return * CAPITAL_CAUSAL_AGENT),
                confidence=selected_hyp.composite_score,
                hill_climb_iterations=0,  # causal agent doesn't hill-climb
                phase=current_phase[0],  # will upgrade later when Layer 2 matures
                meta_params={
                    **layer1._optimizer.get_params(current_regime).to_dict(),
                    "is_escape_valve": selected_hyp.is_escape_valve,
                    "edge_stability": selected_hyp.layer1_score,
                    "avg_breach_risk": avg_breach_risk,
                    "position_fraction_adjusted": adjusted_params["position_fraction"],
                },
                causal_chain_snapshot={"chain": [e.to_dict() for e in selected_hyp.chain]},
            )

            try:
                await asyncio.to_thread(ledger.commit, do)
                logger.info(
                    "CausalAgent Trade Intent: %s | expected_return=%+.4f%% | chain: %s | capital=$%d",
                    trade_signal,
                    trajectory.predicted_price_return * 100,
                    selected_hyp.chain_summary(),
                    CAPITAL_CAUSAL_AGENT,
                )
            except LedgerWriteError as exc:
                logger.error("CausalAgent: ledger commit failed | %s", exc)
                _obs_write(obs_log_path, state, _current_graph, hypotheses, selected_hyp,
                           f"ledger_commit_failed:{exc}")
                continue

            # --- Step 7: Execute ---
            try:
                # Detect escape valve trade
                is_escape_valve = selected_hyp.is_escape_valve

                if is_escape_valve:
                    logger.warning(
                        "CausalAgent: escape valve trade | "
                        "chain=%s | position capped at 1%% of normal",
                        selected_hyp.chain_summary(),
                    )

                # Use adjusted_params when calling executor
                telemetry = await executor.execute(
                    signal=trade_signal,
                    params=adjusted_params,
                    state=state,
                    decision_object_id=str(do.id),
                )
                _active_hyp = selected_hyp
                _active_traj = trajectory
                _active_do_id = str(do.id)
                _active_entry_price = state.price
                _active_entry_time = time.monotonic()
                # Update ML1 with fresh telemetry — only real orders carry meaningful signal.
                new_vector = ml1.predict_proba(telemetry, is_real_order=(trade_signal == "BUY"))
                await state_manager.update_algo_health(new_vector)

                await asyncio.to_thread(ledger.update_status, str(do.id), "ACTIVE")
                logger.info(
                    "CausalAgent: trade ACTIVE | id=%s signal=%s price=%.4f",
                    str(do.id)[:8], trade_signal, state.price,
                )

                # Check if ML1 should retrain
                ml1_retrained, ml1_oob = await asyncio.to_thread(ml1.maybe_retrain)
                if ml1_retrained:
                    logger.info("ML1 retrained | oob_accuracy=%.4f", ml1_oob or 0.0)
                    await asyncio.to_thread(
                        ledger.save_checkpoint,
                        checkpoint_number=ml1.checkpoint_count,
                        phase="bootstrap",
                        ml2_brier_score=0.0,
                        training_sample_count=ml1.last_sample_count,
                        ml1_macro_f1=ml1_oob,
                    )
            except Exception as exc:
                logger.error("CausalAgent: execution failed | %s | marking EXECUTION_FAILED", exc)
                await asyncio.to_thread(ledger.update_status, str(do.id), "EXECUTION_FAILED")

            # Observability log
            _obs_write(obs_log_path, state, _current_graph, hypotheses, selected_hyp, "traded")

        except Exception as exc:  # noqa: BLE001
            logger.error("CausalAgent: unhandled error in loop | %s", exc, exc_info=True)

    logger.info("Causal agent loop exited cleanly")


def _obs_write(
    path: Path,
    state,
    graph,
    hypotheses: list,
    selected,
    hold_reason: str,
) -> None:
    """Append one JSONL record to the causal agent observability log (Task 11)."""
    import json
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "price": state.price,
            "volatility": round(state.volatility, 6),
            "spread": round(state.spread, 6),
            "rsi": round(state.rsi_current, 2),
            "hold_reason": hold_reason,
            "graph_edges": len(graph.edges) if graph else 0,
            "graph_version": graph.version_id[:8] if graph else None,
            "n_hypotheses": len(hypotheses),
            "best_score": round(hypotheses[0].composite_score, 6) if hypotheses else 0.0,
            "selected_chain": selected.chain_summary() if selected else None,
            "predicted_direction": selected.predicted_direction if selected else None,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # observability log failure must never crash the agent


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
    executor = Executor(cross_process_lock_path="data/alpaca.lock")

    ml1 = ML1BehaviourClassifier()
    ml2 = ML2BreachPredictor()
    regime = RegimeClassifier()
    comparator = OutcomeComparator(ledger, executor._api)

    strategies = [EMAStrategy(), BollingerStrategy(), RSIStrategy()]
    simulator = Simulator(strategies=strategies, regime_classifier=regime)  # N25 FIX: pass wrapper, not raw model

    # --- Causal agent components (Task 10/13) ---
    db_path = os.getenv("SQLITE_PATH", "data/arivu.db")
    feature_bar = FeatureBarBuilder(regime_classifier=regime)
    discovery_engine = CausalDiscoveryEngine(db_path=db_path)
    layer1 = Layer1Tracker(db_path=db_path)
    trust_updater = Layer2TrustUpdater(db_path=db_path)
    twin_sim = TwinSimulator()
    hyp_gen = HypothesisGenerator(layer1=layer1, layer2_store=trust_updater)
    logger.info(
        "Causal agent initialised | graph_refresh=%ds | capital=$%.0f",
        GRAPH_REFRESH_S, CAPITAL_CAUSAL_AGENT,
    )

    feed = BinanceFeed(state_manager=state_manager, feature_bar_builder=feature_bar)
    current_phase: list[str] = ["bootstrap"]


    logger.info("Causal state initialised | phase=%s", current_phase[0])

    # --- Startup cleanup ---
    # Mark any leftover ACTIVE or COMMITTED trades as INTERRUPTED
    await asyncio.to_thread(ledger.cleanup_stale_trades)
    
    # Cancel all open Alpaca orders
    try:
        open_orders = await asyncio.to_thread(executor._api.list_orders, status="open")
        for o in open_orders:
            try:
                await asyncio.to_thread(executor._api.cancel_order, o.id)
            except Exception:
                pass
        logger.info("Startup: cancelled %d open orders", len(open_orders))
    except Exception as exc:
        logger.warning("Startup: could not cancel open orders | %s", exc)
    
    # Close any open Alpaca position
    try:
        await asyncio.to_thread(executor._api.close_position, "SOLUSD")
        logger.info("Startup: closed open Alpaca position")
    except Exception as exc:
        if "position does not exist" not in str(exc).lower():
            logger.warning("Startup: could not close position | %s", exc)

    # Graceful shutdown handler
    signal.signal(signal.SIGINT, _handle_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigint)

    # K7 FIX: return_exceptions=True so one dying coroutine does not cancel
    # all others (which would leave open Alpaca positions unmonitored).
    try:
        results = await asyncio.gather(
            market_feed_loop(feed, state_manager),
            assumption_monitor_loop(state_manager, ledger, decision_queue, _shutdown),
            legacy_strategy_loop(
                state_manager, ledger, executor, comparator, ml1, ml2, layer1, regime, current_phase
            ),
            causal_agent_loop(
                state_manager, feature_bar, discovery_engine,
                layer1, hyp_gen, twin_sim, trust_updater,
                ledger, executor, comparator, ml1, ml2, regime, current_phase,
            ),
            _shutdown_watcher(feed),
            return_exceptions=True,
        )
        coroutine_names = [
            "market_feed_loop", "assumption_monitor_loop",
            "legacy_strategy_loop", "causal_agent_loop", "_shutdown_watcher",
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
    # K3 consistency: use asyncio.to_thread for SQLite calls even during shutdown
    # so the event loop is not blocked while other coroutines finalize.
    active = await asyncio.to_thread(ledger.get_active)
    if active:
        await asyncio.to_thread(ledger.update_status, str(active.id), "INTERRUPTED")
        logger.info("Active entry marked INTERRUPTED | id=%s", active.id)

        # N18 FIX: Close Alpaca position to prevent orphaned real-money exposure
        try:
            await asyncio.to_thread(executor._api.close_position, "SOLUSD")
            logger.info("Alpaca position closed on shutdown")
        except Exception as exc:  # noqa: BLE001
            if "position does not exist" not in str(exc).lower():
                logger.error("Failed to close Alpaca position on shutdown | %s", exc)

    # CRIT-4 FIX: Cancel all open limit orders on shutdown.
    # If the system shuts down while a limit order is pending (DO is COMMITTED, not yet
    # ACTIVE), close_position("SOLUSD") above fails silently — the position doesn't exist
    # yet. The open limit order stays live on Alpaca, eventually fills, and creates a
    # position nobody monitors. Cancelling open orders prevents orphaned Alpaca exposure.
    try:
        open_orders = await asyncio.to_thread(executor._api.list_orders, status="open")
        cancelled = 0
        for o in open_orders:
            try:
                await asyncio.to_thread(executor._api.cancel_order, o.id)
                cancelled += 1
            except Exception:  # noqa: BLE001
                pass
        if cancelled:
            logger.info("Cancelled %d open orders on shutdown", cancelled)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to cancel open orders on shutdown | %s", exc)

    logger.info("Graceful shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
