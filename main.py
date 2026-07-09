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
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
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
def diagnose_trade(active_do, record, _active_hyp, _active_traj, actual_return, outcome_correct, _current_graph, layer1, current_regime, db_path):
    if not _active_hyp or not active_do:
        return
    is_escape_valve = (active_do.meta_params or {}).get("is_escape_valve", False) or getattr(_active_hyp, "is_escape_valve", False)
    if is_escape_valve:
        return

    try:
        edge = _active_hyp.chain[0] if _active_hyp.chain else None
        if not edge:
            return

        edge_key = f"{edge.source}|{edge.target}|{edge.lag}"

        first_seen_run, total_runs = -1, 0
        try:
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                all_runs = [r[0] for r in conn.execute(
                    "SELECT DISTINCT graph_version_id FROM layer1_runs ORDER BY id ASC"
                ).fetchall()]
                present_runs = [r[0] for r in conn.execute(
                    "SELECT DISTINCT graph_version_id FROM layer1_runs WHERE edge_key = ? AND present = 1",
                    (edge_key,)
                ).fetchall()]

                if present_runs:
                    if present_runs[0] in all_runs:
                        first_seen_run = all_runs.index(present_runs[0]) + 1
                    else:
                        first_seen_run = -1
                    total_runs = len(present_runs)
        except Exception:
            pass

        edge_stability = layer1.get_stability_score(edge.source, edge.target, edge.lag, current_regime)
        is_new_edge = total_runs < 10

        in_feedback_loop = False
        if _current_graph:
            in_feedback_loop = any(
                e.source == edge.target and e.target == edge.source
                for e in _current_graph.edges
            )
            
        is_tautological_edge = edge.source in ("price_in_band", "bollinger_width")

        actual_dir = "up" if actual_return > 0 else "down" if actual_return < 0 else "flat"
        pnl_usd = record.actual_pnl if record else 0.0
        pred_ret = _active_traj.predicted_price_return if _active_traj else 0.0

        logger.info(
            "NaturalTrade diagnosis | trade_id=%s | chain=%s | lag=%d | "
            "predicted_direction=%s | actual_direction=%s | correct=%s | "
            "edge_first_seen_run=%d | edge_total_runs=%d | edge_stability=%.6f | "
            "is_new_edge=%s | in_feedback_loop=%s | is_tautological_edge=%s | "
            "predicted_return=%.6f | actual_return=%.6f | pnl_usd=%.4f",
            active_do.id, _active_hyp.chain_key, edge.lag,
            _active_hyp.predicted_direction, actual_dir, outcome_correct,
            first_seen_run, total_runs, edge_stability,
            is_new_edge, in_feedback_loop, is_tautological_edge,
            pred_ret, actual_return, pnl_usd
        )
    except Exception as e:
        logger.error("Failed to diagnose natural trade: %s", e)


_shutdown = asyncio.Event()



# ---------------------------------------------------------------------------
# Causal agent configuration
# ---------------------------------------------------------------------------

GRAPH_REFRESH_S: int = 150          # 2.5-minute base interval between PCMCI runs
# Reduced from 300s: at 150s we get 24 runs/hour → Layer1 reaches 15 validation
# runs in 37.5 min (vs 75 min at 300s). CPU overhead ~2% (3s PCMCI / 150s cycle).

ADAPTIVE_VOL_WINDOW: int = 20       # rolling window for volatility-based refresh adaptation
CAPITAL_CAUSAL_AGENT: float = 1_000_000.0   # paper account size (cosmetic label — actual sizing uses live equity)

CAPITAL_LEGACY_ARM: float = 1_000_000.0   # $1M for legacy strategies (run synthetically)
# Whether the exchange account supports short-selling (crypto paper = no margin shorts by default).
# When False, DOWN hypotheses are filtered out before trade execution — they would
# become SELL signals with no existing position, silently converting to HOLD with pnl=0.
SUPPORTS_SHORT: bool = False

# Minimum expected return before a trade is committed.
# Round-trip cost for SOL is ~0.10% (maker fee each way).
# Anything below 0.15% is a guaranteed loser on fees alone.
# Uses abs() so both BUY (positive) and CLOSE (negative) signals are gated.
MIN_EXPECTED_RETURN: float = 0.0


# Variables that are always >= 0 by construction — sign-flip assumptions
# (threshold=0, operator=gt) can never fire for these because they cannot go
# negative. Used in causal_agent_loop when building DO assumptions.
# See assumption-building block for full rationale.
_ALWAYS_NON_NEGATIVE: frozenset[str] = frozenset({
    "trade_intensity",       # trade count per bar, always >= 0
    "bollinger_width",       # Bollinger band width, always >= 0
    "volatility",            # std dev of returns, always >= 0
    "volume",                # mean tick volume, always >= 0
    "price_in_band",         # fraction [0, 1]
    "regime_volatile",       # binary 0 or 1
    "regime_trending",       # binary 0 or 1
    "rsi",                   # bar-level RSI [0, 100]
    "algo_health_p_normal",  # probability [0, 1]
    "algo_health_p_stressed",
    "algo_health_p_degraded",
})


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
                                logger.info("ML2 retrained | brier_score=%.6f", ml2_brier or 0.0)
                                if current_phase[0] == "bootstrap":
                                    current_phase[0] = "trained"
                                    logger.info("ML2 transitioned to trained mode")

                            # Check if Regime Classifier should retrain (every 10 closed trades, >= 40)
                            try:
                                closed_count = await asyncio.to_thread(ledger.count_closed)
                                if closed_count >= 40 and (not regime_classifier._is_trained or (closed_count - regime_classifier.last_retrain_count) >= 20):
                                    entries = await asyncio.to_thread(ledger.get_closed_entries_for_regime)
                                    regime_retrained = await asyncio.to_thread(regime_classifier.run_kmeans_and_retrain, entries)
                                    if regime_retrained:
                                        regime_classifier.last_retrain_count = closed_count
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
                    regime=current_regime,
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
    decision_queue: asyncio.Queue | None = None,
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
    _active_traj = None       # SimulatorResult for current open trade
    _active_do_id: str | None = None      # DecisionObject ID for current open trade
    _active_entry_price: float = 0.0
    _active_entry_time: float = 0.0
    _last_breach_graph_id: str | None = None  # graph version_id that produced last breach
                                              # blocks immediate re-entry on same stale graph
    _prev_regime: str | None = None
    _vol_history: list[float] = []  # rolling window for adaptive refresh
    effective_refresh: int = GRAPH_REFRESH_S   # updated each iteration from adaptive logic
    adaptive_fast: bool = False                # True when vol > rolling median

    BAR_POLL_S = 10   # poll every bar width

    logger.info("Causal agent loop started | graph_refresh=%ds", GRAPH_REFRESH_S)

    while not _shutdown.is_set():
        # Wait up to BAR_POLL_S for a queue event (assumption_breach or threshold_crossed).
        # Wakes immediately when breach arrives for the active DO, giving sub-10s response.
        # Consuming threshold_crossed events keeps the queue open for breach events
        # (otherwise causal_state.py fills the queue to maxsize=10 and monitor drops breaches).
        _queue_breach_event: dict | None = None
        if decision_queue is not None:
            try:
                _ev = await asyncio.wait_for(decision_queue.get(), timeout=BAR_POLL_S)
                if _ev.get("type") == "assumption_breach":
                    if _ev.get("decision_object_id") == _active_do_id:
                        _queue_breach_event = _ev
                        logger.info(
                            "CausalAgent: queue breach event received | id=%s | assumptions=%s",
                            (_active_do_id or "")[:8], _ev.get("assumptions", []),
                        )
                    else:
                        logger.debug(
                            "CausalAgent: queue breach for stale DO %s — discarding",
                            str(_ev.get("decision_object_id", ""))[:8],
                        )
                elif _ev.get("type") == "threshold_crossed":
                    logger.debug(
                        "CausalAgent: queue threshold_crossed | reason=%s value=%.6f",
                        _ev.get("reason", "?"), _ev.get("value", 0.0),
                    )
                # Any other event type: consume silently to keep queue drained
            except asyncio.TimeoutError:
                pass   # No event within BAR_POLL_S — run normal cycle
        else:
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
                    "Market Regime Change detected | %s -> %s | Volatility=%.6f, Spread=%.6f, Trend=%.6f",
                    _prev_regime or "INITIAL", current_regime, state.volatility, state.spread, state.trend_strength
                )
                _prev_regime = current_regime

            # --- Step 1: Refresh causal graph every GRAPH_REFRESH_S ---
            now = time.monotonic()

            # Adaptive refresh: track rolling volatility.
            # When market is moving faster than its own median, halve the interval.
            # No hard threshold — purely relative to own rolling history.
            _vol_history.append(state.volatility)
            if len(_vol_history) > ADAPTIVE_VOL_WINDOW:
                _vol_history.pop(0)
            if len(_vol_history) >= 10:
                sorted_vol = sorted(_vol_history)
                median_vol = sorted_vol[len(sorted_vol) // 2]
                adaptive_fast = state.volatility > median_vol
                effective_refresh = (
                    GRAPH_REFRESH_S // 2
                    if adaptive_fast
                    else GRAPH_REFRESH_S
                )
            else:
                adaptive_fast = False
                effective_refresh = GRAPH_REFRESH_S

            if (now - _last_graph_refresh) >= effective_refresh:
                if feature_bar.is_ready():
                    try:
                        matrix, var_names = feature_bar.get_feature_matrix()
                        # Pull current MetaParams so discovery uses the optimizer's
                        # current tau_max and pcmci_alpha, not hardcoded constants.
                        _meta = layer1._optimizer.get_params(current_regime)
                        logger.info(
                            "CausalAgent: running PCMCI | bars=%d vars=%d "
                            "tau_max=%d pcmci_alpha=%.3f effective_refresh=%ds",
                            matrix.shape[0], matrix.shape[1],
                            _meta.tau_max, _meta.pcmci_alpha, effective_refresh,
                        )
                        _current_graph = await asyncio.to_thread(
                            discovery_engine.run, matrix, var_names,
                            _meta.tau_max, _meta.pcmci_alpha, current_regime
                        )
                        await asyncio.to_thread(layer1.record_run, _current_graph, current_regime)
                        # Record whether the active chain is still intact in the new graph
                        if _active_hyp and _active_hyp.chain and _active_hyp.chain_key != "unknown":
                            await asyncio.to_thread(
                                layer1.record_chain_run,
                                _active_hyp.chain_key,
                                _active_hyp.chain,
                                _current_graph,
                            )
                        _last_graph_refresh = now
                        # New graph = new causal structure. Allow re-entry after previous breach.
                        _last_breach_graph_id = None
                        logger.info(
                            "CausalAgent: graph refreshed | %s", _current_graph.summary()
                        )
                    except ValueError as exc:
                        logger.debug("CausalAgent: not enough bars yet — %s", exc)
                else:
                    n = feature_bar.n_bars_ready()
                    logger.info(
                        "[ML1 Feature Store] Warm-up progress: %d/60 bars (Granger ready: %d/40) | Features: vol=%.6f, spread=%.6f, trend=%.6f",
                        n, n, state.volatility, state.spread, state.trend_strength
                    )

            # --- Step 2: Monitor active causal trade for trajectory breach ---
            if _active_hyp and _active_traj and _active_do_id:
                elapsed = int(time.monotonic() - _active_entry_time)
                if state.price > 0 and _active_entry_price > 0:
                    actual_return = (state.price - _active_entry_price) / _active_entry_price
                    # Check trajectory divergence (twin sim confidence band)
                    if current_regime == "calm":
                        is_breach = False
                        breach_source = None
                    else:
                        is_breach = twin_sim.check_breach(_active_traj, actual_return, elapsed)
                        breach_source = "trajectory_breach" if is_breach else None

                    # Fast path: queue delivered breach event for this DO (sub-10s response).
                    if not is_breach and _queue_breach_event is not None:
                        is_breach = True
                        breach_source = "assumption_breach"
                        logger.warning(
                            "CausalAgent: fast-path queue breach | "
                            "id=%s | elapsed=%ds | assumptions=%s",
                            _active_do_id[:8], elapsed,
                            _queue_breach_event.get("assumptions", []),
                        )

                    # Fallback: DB breach log — catches cases where queue was full
                    # and the monitor event was dropped (put_nowait QueueFull).
                    if not is_breach:
                        monitor_breach_log = await asyncio.to_thread(
                            ledger.get_breach_log, _active_do_id
                        )
                        if monitor_breach_log:
                            is_breach = True
                            breach_source = "assumption_breach"
                            logger.warning(
                                "CausalAgent: DB breach log detected | "
                                "id=%s | elapsed=%ds | breached=%s",
                                _active_do_id[:8], elapsed, list(monitor_breach_log.keys()),
                            )
                    elif is_breach and breach_source == "trajectory_breach":
                        logger.warning(
                            "CausalAgent: trajectory breach detected | id=%s | elapsed=%ds",
                            _active_do_id[:8], elapsed,
                        )

                    # Unified close handler for BOTH breach types
                    if is_breach:
                        # Close the trade — record outcome and update Layer 2
                        try:
                            active_do = await asyncio.to_thread(
                                ledger.get_decision_object, _active_do_id
                            )
                            if active_do:
                                record = await asyncio.to_thread(
                                    comparator.close_cycle, active_do, breach_source, _current_graph
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
                                    # Chain-level trust update
                                    if _active_hyp.chain_key != "unknown":
                                        await asyncio.to_thread(
                                            trust_updater.update_chain_trust,
                                            _active_hyp.chain_key,
                                            outcome_correct,
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
                                        "CausalAgent: ML2 calibration | avg_breach_risk=%.6f | "
                                        "ml2_calibration=%.6f | assumptions_breached=%s",
                                        avg_breach_risk, ml2_calibration, breach_log,
                                    )

                                    # Update MetaParameterOptimizer
                                    # Pull l2_trust from the DO's meta_params (stored at entry time)
                                    _do_meta = active_do.meta_params or {}
                                    trade_outcome = TradeOutcome(
                                        meta_params_used=_do_meta,
                                        regime=current_regime,
                                        predicted_return=_active_traj.predicted_price_return,
                                        actual_return=actual_return,
                                        pnl_usd=record.actual_pnl,
                                        causal_chain_held=False,
                                        regime_was_stable=True,
                                        avg_breach_risk=avg_breach_risk,
                                        ml2_calibration=ml2_calibration,
                                        l2_trust=_do_meta.get("l2_trust", 0.5),
                                        l2_n_obs=_do_meta.get("l2_n_obs", 0),
                                    )
                                    diagnose_trade(active_do, record, _active_hyp, _active_traj, actual_return, outcome_correct, _current_graph, layer1, current_regime, "data/arivu.db")
                                    await asyncio.to_thread(layer1._optimizer.update, trade_outcome)
                                    # Check if ML2 should retrain
                                    ml2_retrained, ml2_brier = await asyncio.to_thread(
                                        ml2.maybe_retrain, ledger, current_phase[0]
                                    )
                                    if ml2_retrained:
                                        logger.info("ML2 retrained | brier_score=%.6f", ml2_brier or 0.0)
                                        if current_phase[0] == "bootstrap":
                                            current_phase[0] = "trained"
                                            logger.info("ML2 transitioned to trained mode")

                                    # Check if Regime Classifier should retrain (every 10 closed trades, >= 40)
                                    try:
                                        closed_count = await asyncio.to_thread(ledger.count_closed)
                                        if closed_count >= 40 and (not regime_classifier._is_trained or (closed_count - regime_classifier.last_retrain_count) >= 20):
                                            entries = await asyncio.to_thread(ledger.get_closed_entries_for_regime)
                                            regime_retrained = await asyncio.to_thread(regime_classifier.run_kmeans_and_retrain, entries)
                                            if regime_retrained:
                                                regime_classifier.last_retrain_count = closed_count
                                                logger.info(
                                                    "Regime Classifier K-Means & Decision Tree retrained | "
                                                    "closed_count=%d", closed_count
                                                )
                                    except Exception as regime_exc:
                                        logger.error("Failed to retrain Regime Classifier | %s", regime_exc)
                        except Exception as exc:
                            logger.error("CausalAgent: failed to close breach trade | %s", exc)
                        finally:
                            # Record the breached graph so we don't immediately re-enter
                            # on the same stale causal structure.
                            _last_breach_graph_id = _current_graph.version_id if _current_graph else None
                            _active_hyp = None
                            _active_traj = None
                            _active_do_id = None
                            _active_entry_price = 0.0
                            _active_entry_time = 0.0

                # Check mathematical horizon expiry for causal agent trades.
                # Guard: skip if price is 0 (stale feed) — would record -100% return.
                # The check will re-run on the next tick once the feed recovers.
                if _active_traj and elapsed >= _active_traj.time_horizon_s and state.price > 0:
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
                                # Chain-level trust update
                                if _active_hyp.chain_key != "unknown":
                                    await asyncio.to_thread(
                                        trust_updater.update_chain_trust,
                                        _active_hyp.chain_key,
                                        outcome_correct,
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
                                    "CausalAgent: ML2 calibration | avg_breach_risk=%.6f | "
                                    "ml2_calibration=%.6f | assumptions_breached=%s",
                                    avg_breach_risk, ml2_calibration, breach_log,
                                )

                                # Update MetaParameterOptimizer
                                # Pull l2_trust from the DO's meta_params (stored at entry time)
                                _do_meta = active_do.meta_params or {}
                                trade_outcome = TradeOutcome(
                                    meta_params_used=_do_meta,
                                    regime=current_regime,
                                    predicted_return=_active_traj.predicted_price_return if _active_traj else 0.0,
                                    actual_return=actual_return,
                                    pnl_usd=record.actual_pnl,
                                    causal_chain_held=True,
                                    regime_was_stable=True,
                                    avg_breach_risk=avg_breach_risk,
                                    ml2_calibration=ml2_calibration,
                                    l2_trust=_do_meta.get("l2_trust", 0.5),
                                    l2_n_obs=_do_meta.get("l2_n_obs", 0),
                                )
                                diagnose_trade(active_do, record, _active_hyp, _active_traj, actual_return, outcome_correct, _current_graph, layer1, current_regime, "data/arivu.db")
                                await asyncio.to_thread(layer1._optimizer.update, trade_outcome)
                                # Check if ML2 should retrain
                                ml2_retrained, ml2_brier = await asyncio.to_thread(
                                    ml2.maybe_retrain, ledger, current_phase[0]
                                )
                                if ml2_retrained:
                                    logger.info("ML2 retrained | brier_score=%.6f", ml2_brier or 0.0)
                                    if current_phase[0] == "bootstrap":
                                        current_phase[0] = "trained"
                                        logger.info("ML2 transitioned to trained mode")

                                # Check if Regime Classifier should retrain (every 10 closed trades, >= 40)
                                try:
                                    closed_count = await asyncio.to_thread(ledger.count_closed)
                                    if closed_count >= 40 and (not regime_classifier._is_trained or (closed_count - regime_classifier.last_retrain_count) >= 20):
                                        entries = await asyncio.to_thread(ledger.get_closed_entries_for_regime)
                                        regime_retrained = await asyncio.to_thread(regime_classifier.run_kmeans_and_retrain, entries)
                                        if regime_retrained:
                                            regime_classifier.last_retrain_count = closed_count
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
                _obs_write(obs_log_path, state, None, [], None, "no_graph_yet", effective_refresh=effective_refresh, adaptive_fast=adaptive_fast)
                continue

            # get_params_for_trade() rolls the epsilon-greedy dice ONCE per cycle.
            # The returned params object is passed all the way into generate() so
            # every edge is validated under the same threshold — and the DO records
            # the threshold that actually ran (correct attribution for MetaOptimizer).
            params = layer1._optimizer.get_params_for_trade(current_regime)
            logger.info(
                "Causal Agent evaluating hypotheses | regime=%s | Active Params: k_runs=%d, min_runs=%d, threshold=%.6f",
                current_regime, params.k_runs, params.min_runs, params.threshold
            )

            # Block re-entry on the same graph that produced the last breach.
            # Using the same causal structure immediately after it failed is circular.
            # _last_breach_graph_id is cleared when a new PCMCI run completes.
            if _last_breach_graph_id and _current_graph and _current_graph.version_id == _last_breach_graph_id:
                logger.info(
                    "CausalAgent: HOLD | reason=same_graph_as_last_breach | "
                    "graph=%s | waiting for next PCMCI run",
                    _current_graph.version_id[:8],
                )
                continue

            hypotheses = await asyncio.to_thread(hyp_gen.generate, _current_graph, current_regime, state.model_dump(), params=params)

            if not hypotheses:
                continue

            # Log total count (diagnostic: very high = k_runs too loose) then slice
            # to top MAX_HYPOTHESES_ML2 before ML2 annotation.
            # Hypotheses are already sorted by composite_score descending.
            # Annotating all 1240 when only top-1 gets traded is pure waste.
            MAX_HYPOTHESES_ML2 = 30
            n_total_hyps = len(hypotheses)
            hypotheses = hypotheses[:MAX_HYPOTHESES_ML2]
            if n_total_hyps > MAX_HYPOTHESES_ML2:
                logger.info(
                    "CausalAgent: hypothesis pool | total=%d (sliced to top %d for ML2) | "
                    "k_runs=%d threshold=%.6f | high count may indicate loose params",
                    n_total_hyps, MAX_HYPOTHESES_ML2, params.k_runs, params.threshold,
                )

            from core.schemas import Assumption
            from core.decision_utils import annotate_proximity
            for hyp in hypotheses:
                if not hyp.chain:
                    hyp.avg_breach_risk = 0.5
                    continue

                proxy_assumptions = []
                
                # Propagate root sign
                root_node = hyp.chain[-1].source
                root_val = getattr(state, root_node, 0.0)
                root_mean = _current_graph.feature_means.get(root_node, 0.0)
                root_std = max(_current_graph.feature_stds.get(root_node, 1.0), 1e-6)
                root_z = (root_val - root_mean) / root_std
                
                # Pre-calculate expected signs for all nodes
                expected_signs = {}
                current_expected_sign = 1 if root_z > 0 else -1
                for edge in reversed(hyp.chain):
                    expected_signs[edge.source] = current_expected_sign
                    current_expected_sign *= (1 if edge.coeff > 0 else -1)

                for edge in hyp.chain[:3]:  # cap at 3 edges to avoid ML2 overload
                    source_var = edge.source
                    current_val = getattr(state, source_var, None)

                    if current_val is None:
                        continue
                        
                    mean_val = _current_graph.feature_means.get(source_var, 0.0)
                    std_val = max(_current_graph.feature_stds.get(source_var, 1.0), 1e-6)
                    current_z = (current_val - mean_val) / std_val
                    
                    expected_sign = expected_signs.get(source_var, 1)

                    op = "gt" if expected_sign > 0 else "lt"
                    proximity = abs(current_z) / max(abs(current_z) + 0.5, 1.0)

                    proxy_assumptions.append(Assumption(
                        name=f"causal_edge|{edge.source}|{edge.target}|{edge.lag}",
                        variable=source_var,
                        operator=op,
                        threshold=mean_val,
                        current_value=current_val,
                        proximity=proximity,
                        breach_risk=1.0 - hyp.layer2_score,
                    ))

                if not proxy_assumptions:
                    hyp.avg_breach_risk = 0.5
                    continue

                annotated_proxy = ml2.annotate(
                    proxy_assumptions,
                    state,
                    hyp.time_horizon_seconds / 60.0,
                    state.algo_health_vector,
                    regime=current_regime,
                )
                hyp.avg_breach_risk = (
                    sum(a.breach_risk for a in annotated_proxy) / len(annotated_proxy)
                    if annotated_proxy else 0.5
                )

            best_hyp = hypotheses[0]

            if best_hyp is None or (best_hyp.composite_score < MIN_TRADE_SCORE and not best_hyp.is_escape_valve):
                score_str = f"{best_hyp.composite_score:.6f}" if best_hyp else "none"
                _obs_write(obs_log_path, state, _current_graph, hypotheses, None,
                           f"score_too_low:{score_str}",
                           effective_refresh=effective_refresh, adaptive_fast=adaptive_fast)
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

            # BUG2 FIX: Filter DOWN hypotheses when short-selling is not supported.
            # A DOWN hypothesis generates a SELL signal. With no existing long position,
            # executor silently converts SELL → HOLD → actual_pnl=0.0 every time.
            # This wastes a 30-minute horizon and teaches ML2/Layer2 nothing useful.
            # When SUPPORTS_SHORT=False, only consider UP hypotheses for execution.
            if not SUPPORTS_SHORT:
                long_hypotheses = [h for h in hypotheses if h.predicted_direction == "up" or h.is_escape_valve]
                if not long_hypotheses:
                    logger.info(
                        "CausalAgent: HOLD | all %d hypotheses predict DOWN but SUPPORTS_SHORT=False",
                        len(hypotheses),
                    )
                    _obs_write(obs_log_path, state, _current_graph, hypotheses, None, "no_long_hypotheses", effective_refresh=effective_refresh, adaptive_fast=adaptive_fast)
                    continue
                hypotheses = long_hypotheses

            selected_hyp, trajectory = twin_sim.select_best(
                hypotheses, _current_graph, current_state_dict
            )

            if selected_hyp is None or trajectory is None:
                _obs_write(obs_log_path, state, _current_graph, hypotheses, selected_hyp,
                           "zero_expected_value",
                           effective_refresh=effective_refresh, adaptive_fast=adaptive_fast)
                continue

            # Gate: reject trades where expected return is below the fee floor.
            # Round-trip cost ~0.10%; MIN_EXPECTED_RETURN = 0.15% ensures every
            # committed trade has positive expected value after fees.
            # abs() catches both BUY (positive) and CLOSE (negative) signals.
            #
            # ESCAPE VALVE EXEMPTION: escape valve trades are fired specifically
            # to generate closed-trade outcome records so Layer2, MetaOptimizer,
            # and RegimeClassifier get training signal during low-hypothesis periods.
            # Their purpose is NOT profit — it's learning. At 1% capital the fee
            # cost is trivial vs. the learning value of the outcome record.
            # Blocking them here defeats the entire reason the escape valve exists.
            if abs(trajectory.predicted_price_return) < MIN_EXPECTED_RETURN and not selected_hyp.is_escape_valve:
                logger.info(
                    "CausalAgent: HOLD | reason=below_fee_floor | "
                    "expected_return=%.6f%% min=%.6f%% | chain=%s",
                    trajectory.predicted_price_return * 100,
                    MIN_EXPECTED_RETURN * 100,
                    selected_hyp.chain_summary(),
                )
                _obs_write(obs_log_path, state, _current_graph, hypotheses, selected_hyp,
                           f"below_fee_floor:{trajectory.predicted_price_return:.6f}",
                           effective_refresh=effective_refresh, adaptive_fast=adaptive_fast)
                continue
            elif abs(trajectory.predicted_price_return) < MIN_EXPECTED_RETURN and selected_hyp.is_escape_valve:
                logger.info(
                    "CausalAgent: escape valve bypassing fee floor | "
                    "expected_return=%.6f%% (below %.6f%%) | chain=%s | "
                    "firing 0.01%% position to generate Layer2/MetaOptimizer training signal",
                    trajectory.predicted_price_return * 100,
                    MIN_EXPECTED_RETURN * 100,
                    selected_hyp.chain_summary(),
                )


            # --- Step 6: Commit Decision Object ---
            # Only BUY signals reach here (SUPPORTS_SHORT=False filters DOWN above).
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

            # Represent the causal chain as "assumptions" for ledger compatibility.
            # Uses graph variable names (e.g. btc_return) which now exist on CausalState
            # after the schemas.py + causal_state.py updates.
            #
            # ASSUMPTION SEMANTICS FIX:
            # Each causal edge source_var -> price_return carries a coefficient sign.
            # The assumption must monitor whether the causal DRIVER stays in the
            # direction that justified the trade. As soon as the driver reverses,
            # the hypothesis is invalidated → breach.
            #
            # Positive coeff (e.g. btc_return +0.61 → price_return):
            #   Assumption: btc_return must stay > 0  (operator=gt, threshold=0.0)
            #   Breach fires the moment btc_return flips negative.
            #
            # Negative coeff (e.g. volatility -0.29 → price_return):
            #   Assumption: volatility must stay > 0  (always true → use abs threshold)
            #   Actually: the signal was that LOW volatility is bullish, so breach if
            #   volatility jumps above 2× its commit value (spike = regime change).
            #   operator=lt, threshold=2×current_val.
            #
            # Trig vars (session_sin, session_cos) and bounded floats:
            #   Use a 30% directional band — breach if the value crosses zero or
            #   diverges more than 50% from commit value.
            from core.schemas import Assumption
            from pydantic import ValidationError
            causal_assumptions = []
            
            # Propagate root sign for ledger assumptions
            root_node = selected_hyp.chain[-1].source
            root_val = getattr(state, root_node, 0.0)
            root_mean = _current_graph.feature_means.get(root_node, 0.0)
            root_std = max(_current_graph.feature_stds.get(root_node, 1.0), 1e-6)
            root_z = (root_val - root_mean) / root_std
            
            # Pre-calculate expected signs for all nodes
            expected_signs = {}
            current_expected_sign = 1 if root_z > 0 else -1
            for edge in reversed(selected_hyp.chain):
                expected_signs[edge.source] = current_expected_sign
                current_expected_sign *= (1 if edge.coeff > 0 else -1)

            for edge in selected_hyp.chain[:3]:
                source_var = edge.source
                current_val = getattr(state, source_var, None)
                if current_val is None:
                    logger.debug(
                        "CausalAgent: skipping edge %s->%s — var '%s' not in CausalState",
                        edge.source, edge.target, source_var,
                    )
                    continue

                mean_val = _current_graph.feature_means.get(source_var, 0.0)
                std_val = max(_current_graph.feature_stds.get(source_var, 1.0), 1e-6)
                current_z = (current_val - mean_val) / std_val
                
                expected_sign = expected_signs.get(source_var, 1)
                
                op = "gt" if expected_sign > 0 else "lt"
                proximity = abs(current_z) / max(abs(current_z) + 0.5, 1.0)
                
                try:
                    causal_assumptions.append(Assumption(
                        name=f"causal_edge|{edge.source}|{edge.target}|{edge.lag}",
                        variable=source_var,
                        operator=op,
                        threshold=mean_val,
                        current_value=current_val,
                        proximity=proximity,
                        breach_risk=1.0 - selected_hyp.layer2_score,
                    ))
                except ValidationError as ve:
                    logger.debug(
                        "CausalAgent: Assumption build failed for var '%s' | %s",
                        source_var, ve,
                    )
                    continue

            if not causal_assumptions:
                logger.warning(
                    "CausalAgent: no assumptions could be built for chain '%s' "
                    "(all edge source vars missing from CausalState or failed validation) "
                    "— skipping trade this cycle",
                    selected_hyp.chain_summary(),
                )
                continue

            # Annotate with ML2 breach risk predictor
            ml1_vector = state.algo_health_vector
            causal_assumptions = ml2.annotate(
                causal_assumptions,
                state,
                time_horizon=selected_hyp.time_horizon_seconds / 60.0,
                ml1_vector=ml1_vector,
                regime=current_regime,
            )

            # After ml2.annotate() returns causal_assumptions:
            if causal_assumptions:
                avg_breach_risk = sum(
                    a.breach_risk for a in causal_assumptions
                ) / len(causal_assumptions)
            else:
                avg_breach_risk = 0.5

            risk_multiplier = max(0.10, 1.0 - avg_breach_risk)

            original_fraction = 0.0001 if selected_hyp.is_escape_valve else 0.10

            # --- Layer 2 trust multiplier (continuous, no hard gate) ---
            # Layer 2 trust is a Bayesian Beta posterior over directional accuracy.
            # trust=0.5 (untested) → full position. trust=0.3 (proven loser) → 60%.
            # trust=0.7 (proven winner) → full position.
            # Formula: clamp(trust × 2, 0, 1) so neutral=1.0, below-neutral shrinks.
            # This is NOT a threshold — it's a smooth scaling that the MetaOptimizer
            # implicitly learns from: as bad chains get smaller positions, their PnL
            # contribution shrinks and the optimizer hill-climbs toward edges with
            # better Layer 2 trust (higher Layer 1 threshold, different lags, etc).
            l2_trust = selected_hyp.layer2_score          # Bayesian posterior, range [0,1]
            l2_n_obs = selected_hyp.layer2_n_obs
            layer2_multiplier = min(1.0, l2_trust * 2.0)  # 0.5→1.0, 0.3→0.6, 0.7→1.0

            adjusted_params = {
                "position_fraction": max(
                    0.00002,
                    round(original_fraction * risk_multiplier * layer2_multiplier, 7)
                )
            }

            # BUG1 FIX: Fetch REAL account equity to compute projected_pnl correctly.
            # The $1M label in CAPITAL_CAUSAL_AGENT was never what the executor used —
            # executor uses actual account buying_power. Fetch it here so that
            # projected_pnl ≈ actual_pnl (before market moves), making outcome_delta
            # a genuine signal for ML2 training and MetaParameterOptimizer scoring.
            try:
                _account = await asyncio.to_thread(executor._api.get_account)
                _equity = float(_account.equity)
                _buying_power = float(getattr(_account, "buying_power", _equity))
                _effective_capital = min(_equity, _buying_power)
            except Exception as _acc_exc:
                logger.warning(
                    "CausalAgent: could not fetch account equity for projected_pnl | "
                    "falling back to $10,000 | %s", _acc_exc
                )
                _effective_capital = 10_000.0
            _causal_dollar_exposure = min(
                _effective_capital * adjusted_params["position_fraction"],
                95_000.0,  # hard cap matches executor.MAX_ORDER_NOTIONAL
            )
            logger.info(
                "CausalAgent: position sizing | "
                "ML2_breach_risk=%.3f ML2_mult=%.3f | "
                "L2_trust=%.3f(n=%d) L2_mult=%.3f | "
                "fraction %.6f -> %.6f | equity=$%.0f | exposure=$%.0f",
                avg_breach_risk, risk_multiplier,
                l2_trust, l2_n_obs, layer2_multiplier,
                original_fraction, adjusted_params["position_fraction"],
                _effective_capital, _causal_dollar_exposure,
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
                # BUG1 FIX: projected_pnl must use ACTUAL dollar exposure, not the $1M label.
                # Fetch live account equity, apply position_fraction, cap at MAX_ORDER_NOTIONAL.
                # This makes outcome_delta = projected_pnl - actual_pnl meaningful for
                # ML2 training and MetaParameterOptimizer scoring.
                projected_pnl=float(
                    trajectory.predicted_price_return
                    * _causal_dollar_exposure  # computed just above
                ),
                confidence=selected_hyp.composite_score,
                hill_climb_iterations=0,  # causal agent doesn't hill-climb
                phase=current_phase[0],  # will upgrade later when Layer 2 matures
                meta_params={
                    **params.to_dict(),   # reuse params fetched above (line 812) — prevents epsilon-greedy mis-attribution

                    "is_escape_valve": selected_hyp.is_escape_valve,
                    "edge_stability": selected_hyp.layer1_score,
                    "avg_breach_risk": avg_breach_risk,
                    "l2_trust": round(l2_trust, 4),
                    "l2_n_obs": l2_n_obs,
                    "layer2_multiplier": round(layer2_multiplier, 4),
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
                           f"ledger_commit_failed:{exc}",
                           effective_refresh=effective_refresh, adaptive_fast=adaptive_fast)
                continue

            # --- Step 7: Execute ---
            # RACE-CONDITION FIX: set _active_do_id BEFORE calling executor.execute().
            # executor.execute() can take 30–40 s (limit order polling + market fallback).
            # The loop polls every 10 s, so without this guard it re-enters and tries to
            # place a concurrent second order while the first is still executing, causing
            # the file lock to time out and marking the second trade EXECUTION_FAILED.
            _active_do_id = str(do.id)   # lock the slot immediately — prevents duplicate orders
            _active_hyp = selected_hyp
            _active_traj = trajectory
            # NOTE: _active_entry_time and _active_entry_price are set AFTER execute() returns.
            # Setting them here (pre-fill) would start the horizon clock at submission, meaning
            # 15-40 s of fill latency eats into a 10-30 s horizon before monitoring begins.
            try:
                # Detect escape valve trade
                is_escape_valve = selected_hyp.is_escape_valve

                if is_escape_valve:
                    logger.warning(
                        "CausalAgent: escape valve trade | "
                        "chain=%s | position capped at 0.01%% of normal",
                        selected_hyp.chain_summary(),
                    )

                # Use adjusted_params when calling executor
                telemetry = await executor.execute(
                    signal=trade_signal,
                    params=adjusted_params,
                    state=state,
                    decision_object_id=str(do.id),
                )
                # Update ML1 with fresh telemetry — only real orders carry meaningful signal.
                new_vector = ml1.predict_proba(telemetry, is_real_order=(trade_signal == "BUY"))
                await state_manager.update_algo_health(new_vector)

                await asyncio.to_thread(ledger.update_status, str(do.id), "ACTIVE")

                # Stamp fill time NOW — horizon countdown starts from actual fill,
                # not from order submission (15-40 s earlier due to Alpaca latency).
                _active_entry_time = time.monotonic()
                _active_entry_price = state.price

                logger.info(
                    "CausalAgent: trade ACTIVE | id=%s signal=%s price=%.6f",
                    str(do.id)[:8], trade_signal, state.price,
                )


                # Check if ML1 should retrain
                ml1_retrained, ml1_oob = await asyncio.to_thread(ml1.maybe_retrain)
                if ml1_retrained:
                    logger.info("ML1 retrained | oob_accuracy=%.6f", ml1_oob or 0.0)
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
                # Cooldown after lock failure: give the RL arms time to finish their
                # own executor calls before we try again. Without this, the loop
                # immediately retries, times out on the lock again, and spins forever.
                logger.warning(
                    "CausalAgent: cooling down for 90s after EXECUTION_FAILED "
                    "to avoid lock contention with RL arms"
                )
                # Clear all active-trade state so the horizon check and breach check
                # don't fire against a DO that never executed.
                _active_do_id = None
                _active_hyp = None
                _active_traj = None
                _active_entry_price = 0.0
                _active_entry_time = 0.0
                await asyncio.sleep(90)

            # Observability log
            _obs_write(obs_log_path, state, _current_graph, hypotheses, selected_hyp, "traded", effective_refresh=effective_refresh, adaptive_fast=adaptive_fast)

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
    effective_refresh: int = 300,
    adaptive_fast: bool = False,
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
            # Discovery params from graph snapshot (MetaOptimizer-driven)
            "tau_max":          graph.tau_max_used if graph else None,
            "pcmci_alpha":      graph.alpha_used   if graph else None,
            # Adaptive refresh state
            "effective_refresh": effective_refresh,
            "adaptive_fast":     adaptive_fast,
            # Full ranked hypothesis list for THESIS/TWIN stage replay
            "hypotheses": [
                {
                    "id": h.id[:8],
                    "chain": h.chain_summary(),
                    "composite_score": round(h.composite_score, 6),
                    "layer1_score": round(h.layer1_score, 4),
                    "layer2_score": round(h.layer2_score, 4),
                    "predicted_direction": h.predicted_direction,
                    "predicted_magnitude": round(h.predicted_magnitude, 8),
                    "is_escape_valve": h.is_escape_valve,
                    # Multi-hop metadata
                    "hop_count": len(h.chain_edges) if hasattr(h, 'chain_edges') and h.chain_edges else 1,
                    "chain_edges": [
                        {
                            "source": e.source, "target": e.target,
                            "lag": e.lag, "coeff": round(e.coeff, 6),
                        }
                        for e in (h.chain_edges or [])
                    ] if hasattr(h, 'chain_edges') else [],
                    "avg_breach_risk": round(
                        sum(getattr(a, 'breach_risk', 0) for a in (h.assumptions or []))
                        / max(1, len(h.assumptions or [])),
                        4
                    ) if hasattr(h, 'assumptions') else None,
                }
                for h in hypotheses
            ],
            "n_hypotheses": len(hypotheses),
            "best_score": round(hypotheses[0].composite_score, 6) if hypotheses else 0.0,
            "selected_chain": selected.chain_summary() if selected else None,
            "selected_id": selected.id[:8] if selected else None,
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
    executor = Executor(
        api_key=os.getenv("ALPACA_KEY_CAUSAL"),
        api_secret=os.getenv("ALPACA_SECRET_CAUSAL"),
    )

    ml1 = ML1BehaviourClassifier()
    ml2 = ML2BreachPredictor()
    regime = RegimeClassifier()
    regime.last_retrain_count = ledger.count_closed()
    comparator = OutcomeComparator(ledger, executor._api)

    strategies = [EMAStrategy(), BollingerStrategy(), RSIStrategy()]
    simulator = Simulator(strategies=strategies, regime_classifier=regime)  # N25 FIX: pass wrapper, not raw model

    # --- Causal agent components (Task 10/13) ---
    db_path = os.getenv("SQLITE_PATH", "data/arivu.db")
    feature_bar = FeatureBarBuilder(
        regime_classifier=regime,
        on_bar_closed=state_manager.update_graph_features,
    )
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
    current_phase: list[str] = ["trained" if ml2._is_trained else "bootstrap"]


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
                decision_queue=decision_queue,
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
