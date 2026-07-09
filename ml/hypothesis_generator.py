"""ml/hypothesis_generator.py  — Task 5

Hypothesis Generator: reads the current causal graph + Layer 1/2 trust scores
and produces a ranked list of CausalHypothesis objects — trade candidates backed
by discovered causal chains.

ANTI-HOLD DESIGN (mandatory — from implementation_plan.md §10.1):
    1. The qualifying bar is RELATIVE (top-N percentile), not an absolute number.
       There will always be ranked candidates — the generator never returns empty.
    2. Every HOLD cycle logs WHY (no edges / weak edges / chain-penalty too high).
       The difference between "genuinely no signal" and "threshold too strict" must
       be immediately visible in logs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from ml.causal_discovery import CausalEdge, GraphSnapshot
from ml.layer1_tracker import Layer1Tracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_VARIABLE: str = "price_return"   # the variable hypotheses must point to
MAX_HOP_DEPTH: int = 4                  # max causal chain length
CHAIN_LENGTH_PENALTY: float = 0.85      # per-hop confidence discount
MIN_TRADE_SCORE: float = 0.02           # below this → HOLD (logged with reason)
TOP_N_CANDIDATES: int = 5               # always return this many (anti-HOLD)
# Layer 2 observations needed before trust score has full weight in composite_score.
# Below this count, layer2_score is blended toward the neutral 0.5 — the agent
# tries new edges cheaply and lets the data accumulate before fully trusting the score.
# This is a smoothing window, NOT a hard gate. No chain is ever blocked here;
# blocking is done continuously via position sizing in main.py.
MIN_LAYER2_OBS: int = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CausalHypothesis:
    """A fully specified trade hypothesis generated from the causal graph."""
    id: str = field(default_factory=lambda: str(uuid4()))
    graph_version_id: str = ""
    chain: list[CausalEdge] = field(default_factory=list)
    predicted_direction: str = "up"    # "up" or "down"
    predicted_magnitude: float = 0.0   # expected |price_return|
    time_horizon_seconds: int = 0      # sum of lags × bar width
    layer1_score: float = 0.0          # mean stability across chain edges
    layer2_score: float = 0.5          # mean trust score (default 0.5 until data)
    layer2_n_obs: int = 0              # total observations feeding layer2_score
    composite_score: float = 0.0       # final ranked score
    decision_source: str = "causal_agent"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_escape_valve: bool = False
    avg_breach_risk: float = 0.5
    chain_key: str = "unknown"

    def chain_summary(self) -> str:
        if not self.chain:
            return "empty"
        parts = []
        for e in self.chain:
            parts.append(f"{e.source}->(lag={e.lag})->{e.target}")
        return " | ".join(parts)
        
def compute_chain_key(chain: list[CausalEdge]) -> str:
    if not chain:
        return "empty"
    parts = []
    # Reverse to build string from root to target
    for edge in reversed(chain):
        parts.append(f"{edge.source}->(lag={edge.lag})")
    parts.append(chain[0].target)
    return "->".join(parts)


# ---------------------------------------------------------------------------
# Hypothesis Generator
# ---------------------------------------------------------------------------

class HypothesisGenerator:
    """Reads the current graph and generates ranked trade hypotheses.

    Usage:
        gen = HypothesisGenerator(layer1_tracker, layer2_trust_store)
        hypotheses = gen.generate(snapshot)
        # returns sorted list, best first. Never empty (anti-HOLD).
    """

    def __init__(
        self,
        layer1: Layer1Tracker,
        layer2_store: Optional["Layer2TrustStore"] = None,
        bar_width_s: int = 10,
    ) -> None:
        self._layer1 = layer1
        self._layer2 = layer2_store
        self._bar_width_s = bar_width_s

    def generate(
        self,
        snapshot: GraphSnapshot,
        regime: str,
        current_state: dict[str, float],
        params=None,
    ) -> list[CausalHypothesis]:
        """Generate and rank hypotheses from the current graph.

        params: MetaParams from get_params_for_trade() — carries the cycle's
                threshold (possibly epsilon-greedy explored). All edge validation
                and stability scoring in this call will use this same threshold,
                ensuring consistent attribution when MetaOptimizer.update() runs.
                If None, validation falls back to deterministic get_params().

        Always returns all generated hypotheses (never empty if there are chains).
        If no edges point to price_return at all, returns one null hypothesis
        with score=0.0 and a logged HOLD reason.

        Returns:
            List of CausalHypothesis, sorted by composite_score descending.
        """
        price_edges = snapshot.edges_to_target(TARGET_VARIABLE)

        if not price_edges:
            logger.warning(
                "HypothesisGenerator: HOLD | reason=no_edges_to_price_return | "
                "graph=%s | total_edges=%d",
                snapshot.version_id[:8], len(snapshot.edges),
            )
            return [self._null_hypothesis(snapshot.version_id, "no_edges_to_price_return")]

        # ─────────────────────────────────────────────────────────────────
        # PRIMARY EDGE VALIDATION
        # Evidence-based exclusions confirmed by empirical data (80 natural
        # trades across two independent runs) and microstructure literature.
        # These exclusions apply ONLY to the primary (first) edge of a chain.
        # All excluded variables remain in the causal graph and may appear
        # as secondary edges in multi-hop chains.
        # ─────────────────────────────────────────────────────────────────

        # Cross-asset variables confirmed as statistical artifacts at
        # 10-second bar resolution. Institutional arbitrage happens in
        # milliseconds; lags of 40-100 seconds capture retail momentum
        # which is directionally unreliable.
        EXCLUDED_PRIMARY_SOURCES = {"eth_return", "btc_return"}

        # Unsigned variables that carry magnitude/volatility/activity information
        # but no directional (up/down) sign. Allowing them as the primary
        # edge pointing directly to price_return mathematically forces the model
        # to predict direction based on unsigned deviations (e.g. low volume -> down),
        # which is a structural trap (breakout vs. mean reversion coin-flip).
        UNSIGNED_VARIABLES = {
            "volume", "spread", "volatility", "trade_intensity",
            "bollinger_width", "regime_volatile", "regime_trending",
            "algo_health_p_normal", "algo_health_p_stressed", "algo_health_p_degraded",
            "session_sin", "session_cos"
        }

        # Lag caps per variable — beyond these lags the variable's
        # predictive signal for price direction is empirically confirmed
        # to degrade to noise or worse.
        PRIMARY_LAG_CAPS = {
            "volume": 5,            # execution pressure: 50s max
            "trade_intensity": 11,  # order flow momentum: confirmed wins at lag 5,9,10
            "order_book_imbalance": 8,  # LOB signal: confirmed at lag 1,3,7
            "spread": 10,           # liquidity signal: medium duration
            "volatility": 8,        # confirmed win lag=7, loss lag=9 borderline
        }

        def is_valid_primary_edge(edge, current_regime: str) -> bool:
            """
            Returns True if this edge is acceptable as the primary (first)
            edge in a tradeable hypothesis chain.
            """
            if edge.source in EXCLUDED_PRIMARY_SOURCES:
                return False

            if edge.source in UNSIGNED_VARIABLES:
                return False

            if edge.source == "regime_volatile" and current_regime == "calm":
                return False

            if edge.source in PRIMARY_LAG_CAPS:
                if edge.lag > PRIMARY_LAG_CAPS[edge.source]:
                    return False

            return True

        # Find all chains (up to MAX_HOP_DEPTH) ending at price_return
        raw_chains = self._find_chains(snapshot, TARGET_VARIABLE, MAX_HOP_DEPTH, regime, params=params)
        
        chains = []
        for chain in raw_chains:
            # chain[0] is the primary edge (closest to price_return)
            if not is_valid_primary_edge(chain[0], regime):
                logger.info(
                    "HypothesisGenerator: primary edge rejected | "
                    "source=%s lag=%d regime=%s | reason=primary_edge_exclusion",
                    chain[0].source, chain[0].lag, regime
                )
                continue
            chains.append(chain)

        if not chains:
            # --- Escape valve ---
            # If the system has accumulated enough run history but zero edges
            # validate, the MetaOptimizer threshold is too strict for current
            # market conditions. Fire one low-confidence trade on the best
            # available edge to give the optimizer real causal trade feedback.
            n_runs = self._layer1.total_runs_recorded()
            ESCAPE_VALVE_MIN_RUNS = 30
            ESCAPE_VALVE_MIN_STABILITY = 0.10  # edge must appear in at least 10% of runs

            if n_runs >= ESCAPE_VALVE_MIN_RUNS:
                # Score ALL edges above min stability using the full composite
                # (L1 × L2 × magnitude), not just raw stability.
                # This lets Layer 2 trust influence which escape valve edge fires —
                # a less stable but consistently-correct edge can outscore a stable
                # but repeatedly-wrong edge.
                escape_hyps = []
                for edge in price_edges:
                    stability = self._layer1.get_stability_score(
                        edge.source, edge.target, edge.lag, regime
                    )
                    if stability >= ESCAPE_VALVE_MIN_STABILITY:
                        hyp = self._score_chain([edge], snapshot, regime, current_state, params=params)
                        if not hyp.chain:
                            continue
                        hyp.is_escape_valve = True
                        escape_hyps.append((stability, hyp))

                if escape_hyps:
                    escape_hyps.sort(key=lambda x: x[1].composite_score, reverse=True)
                    hypotheses = [h for _, h in escape_hyps]
                    best = hypotheses[0]
                    best_edge = best.chain[0]
                    best_stability = escape_hyps[0][0]
                    logger.warning(
                        "HypothesisGenerator: ESCAPE_VALVE | "
                        "runs=%d candidates=%d best_edge=%s->%s "
                        "stability=%.6f composite=%.6f threshold=%.6f | "
                        "firing %d low-confidence hypothesis(es)",
                        n_runs, len(hypotheses),
                        best_edge.source, best_edge.target,
                        best_stability, best.composite_score,
                        self._layer1._optimizer.get_params(regime).threshold,
                        len(hypotheses),
                    )
                    return hypotheses

            # Normal HOLD path
            logger.warning(
                "HypothesisGenerator: HOLD | reason=no_qualifying_chains | "
                "graph=%s | direct_edges_to_price=%d",
                snapshot.version_id[:8], len(price_edges),
            )
            return [self._null_hypothesis(snapshot.version_id, "no_qualifying_chains")]

        # Score each chain
        hypotheses = []
        for chain in chains:
            hyp = self._score_chain(chain, snapshot, regime, current_state, params=params)
            hypotheses.append(hyp)

        # Direction-accuracy gate: demote chains where the terminal edge (closest
        # to price_return) has empirically predicted the wrong direction more than
        # half the time. L2 trust IS the historical direction win rate (updated via
        # trust_updater.update(outcome_correct=...)). A chain where trust < 0.50
        # with N ≥ 10 observations means it has been wrong > 50% of the time —
        # acting on it as a full natural trade is anti-predictive (confirmed by the
        # 11.4% direction accuracy in empirical analysis).
        # These chains are demoted to escape_valve=True so they still fire for
        # learning purposes but are size-capped at 1% of normal capital.
        # Chains with < MIN_LAYER2_OBS observations are left alone — not enough
        # data yet to condemn them; the dead-zone guard in twin_simulator will
        # handle them via zero EV.
        L2_DIRECTION_ACCURACY_FLOOR = 0.50  # below this = empirically anti-predictive
        L2_MIN_OBS_FOR_GATE = 10            # need at least N obs before condemning
        for hyp in hypotheses:
            if hyp.is_escape_valve:
                continue  # escape valves are already small — don't re-flag
            if not hyp.chain:
                continue
            terminal_edge = hyp.chain[0]  # edge closest to price_return
            if self._layer2 is not None:
                l2_info = self._layer2.get_trust(
                    terminal_edge.source, terminal_edge.target, terminal_edge.lag
                )
                trust = l2_info.get("trust", 0.5)
                n_obs = l2_info.get("n_observations", 0)
                if n_obs >= L2_MIN_OBS_FOR_GATE and trust < L2_DIRECTION_ACCURACY_FLOOR:
                    logger.info(
                        "HypothesisGenerator: direction-accuracy demotion | "
                        "edge=%s->%s lag=%d | trust=%.4f (N=%d) < floor=%.2f | "
                        "demoting to escape_valve",
                        terminal_edge.source, terminal_edge.target, terminal_edge.lag,
                        trust, n_obs, L2_DIRECTION_ACCURACY_FLOOR,
                    )
                    hyp.is_escape_valve = True

        # Sort by composite score descending
        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)

        # Log HOLD warning if even the best is below MIN_TRADE_SCORE
        best = hypotheses[0]
        if best.composite_score < MIN_TRADE_SCORE:
            logger.warning(
                "HypothesisGenerator: HOLD | reason=weak_top_candidate | "
                "top_score=%.6f | threshold=%.6f | chain=%s",
                best.composite_score, MIN_TRADE_SCORE, best.chain_summary(),
            )
        else:
            logger.info(
                "HypothesisGenerator: %d hypotheses | best_score=%.6f | chain=%s",
                len(hypotheses), best.composite_score, best.chain_summary(),
            )

        return hypotheses

    # ------------------------------------------------------------------
    # Chain finding (depth-first graph walk)
    # ------------------------------------------------------------------

    def _find_chains(
        self,
        snapshot: GraphSnapshot,
        target: str,
        max_depth: int,
        regime: str = "unknown",
        params=None,
    ) -> list[list[CausalEdge]]:
        """Find all validated causal chains ending at target, up to max_depth hops.

        params: MetaParams from get_params_for_trade() — every is_validated() call
                in this walk uses the same threshold, so the chain is coherent.

        Chain ordering convention (backward-compatible with _score_chain):
          chain[0]  = edge directly pointing to price_return  (closest to target)
          chain[-1] = edge at the root of the causal chain    (furthest back)

        Example for 3-hop B->A->RSI->price_return:
          chain = [RSI->price_return, A->RSI, B->A]
          chain[0].target == 'price_return'
          chain[-1].source == 'B'

        Only Layer-1 validated edges are included at each hop.
        Cycles are prevented via a visited-source set per branch.
        """
        completed: list[list[CausalEdge]] = []

        def _get_l2(edge) -> tuple[float | None, int]:
            """Fetch L2 trust and observation count for an edge, or (None, 0) if unavailable."""
            if self._layer2 is None:
                return None, 0
            try:
                r = self._layer2.get_trust(edge.source, edge.target, edge.lag)
                return r.get("trust"), r.get("n_observations", 0)
            except Exception:
                return None, 0

        def _walk(
            current_target: str,
            chain_so_far: list[CausalEdge],
            visited: set[str],
            depth: int,
        ) -> None:
            """Recursively extend chain_so_far by one more hop."""
            if depth > max_depth:
                return
            for edge in snapshot.edges_to_target(current_target):
                l2_trust, l2_n_obs = _get_l2(edge)
                if not self._layer1.is_validated(
                    edge.source, edge.target, edge.lag, regime, params=params,
                    l2_trust=l2_trust, l2_n_obs=l2_n_obs,
                ):
                    logger.debug(
                        "Multi-hop: edge pruned | %s->%s lag=%d | "
                        "reason=layer1_not_validated",
                        edge.source, edge.target, edge.lag,
                    )
                    continue
                if edge.source in visited:
                    # Cycle — skip
                    continue
                # Build new chain: current price-facing edge stays at index 0,
                # the new root-side edge is appended at the end.
                new_chain = chain_so_far + [edge]
                completed.append(new_chain)
                logger.debug(
                    "Multi-hop: found %d-hop chain ending %s->%s | "
                    "chain=%s",
                    depth, edge.source, edge.target,
                    " | ".join(f"{e.source}->{e.target}" for e in new_chain),
                )
                # Try to extend further back from this edge's source
                _walk(
                    edge.source,
                    new_chain,
                    visited | {edge.source},
                    depth + 1,
                )

        # Seed: all validated 1-hop edges pointing directly at price_return
        for edge in snapshot.edges_to_target(target):
            l2_trust, l2_n_obs = _get_l2(edge)
            if not self._layer1.is_validated(
                edge.source, edge.target, edge.lag, regime, params=params,
                l2_trust=l2_trust, l2_n_obs=l2_n_obs,
            ):
                score = self._layer1.get_stability_score(
                    edge.source, edge.target, edge.lag, regime, params=params
                )

                logger.debug(
                    "1-hop pruned | %s->%s lag=%d | stability=%.6f | "
                    "reason=layer1_not_validated",
                    edge.source, edge.target, edge.lag, score,
                )
                continue
            chain_1hop = [edge]
            completed.append(chain_1hop)
            # Try to extend this chain back one or more hops
            # visited includes both the target and the edge source to prevent cycles
            _walk(edge.source, chain_1hop, {target, edge.source}, 2)

        return completed

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_chain(
        self,
        chain: list[CausalEdge],
        snapshot: GraphSnapshot,
        regime: str,
        current_state: dict[str, float],
        params=None,
    ) -> CausalHypothesis:
        """Compute composite score for a single causal chain.

        params: MetaParams propagated from generate() — used to pass the same
                params to get_stability_score() so L1 scores are consistent.

        Magnitude uses GEOMETRIC MEAN of edge coefficients, not raw product.

        Why: raw product collapses exponentially with chain length.
          1-hop  coeff=0.30          -> product=0.300, geom_mean=0.300  (same)
          2-hop  coeffs=0.30, 0.20   -> product=0.060, geom_mean=0.245  (fair)
          3-hop  coeffs=0.30,0.20,0.15 -> product=0.009, geom_mean=0.208 (fair)

        CHAIN_LENGTH_PENALTY (0.85 per hop) is the ONE AND ONLY confidence
        discount for chain length. Geometric mean prevents a second hidden
        penalty from making multi-hop chains mathematically unreachable.
        """
        c_key = compute_chain_key(chain)
        n_hops = len(chain)

        # Register the chain in the L1 tracker to initialize/update its history
        self._layer1.register_chain(c_key, snapshot)

        # Layer 1 chain stability
        chain_stability = self._layer1.get_chain_stability(c_key)
        
        # Layer 2 chain trust
        chain_trust = 0.5
        chain_n = 0
        if self._layer2 is not None:
            chain_trust, chain_n = self._layer2.get_chain_trust(c_key)
            
        # Hard gate — block chains with proven wrong direction history
        if chain_n >= 5 and chain_trust < 0.40:
            logger.info("HypothesisGenerator: chain blocked | chain=%s | trust=%.3f n=%d", 
                        c_key, chain_trust, chain_n)
            return self._null_hypothesis(snapshot.version_id, "chain_trust_too_low")

        # Chain length bonus — multi-hop chains rewarded
        length_bonus = min(1.0 + (n_hops - 1) * 0.10, 1.30)
        
        # Effect magnitude: geometric mean of absolute edge coefficients.
        raw_product = 1.0
        for e in chain:
            raw_product *= abs(e.coeff)
        magnitude = raw_product ** (1.0 / n_hops) if n_hops > 0 else 0.0
        magnitude = min(magnitude, 1.0)

        # Base composite score
        composite = chain_stability * chain_trust * length_bonus * magnitude

        # --- JOINT STATE CONSISTENCY & ROOT SIGN PROPAGATION ---
        # 1. Root state sign propagation
        root_node = chain[-1].source
        root_val = current_state.get(root_node, 0.0)
        root_mean = snapshot.feature_means.get(root_node, 0.0)
        root_std = max(snapshot.feature_stds.get(root_node, 1.0), 1e-6)
        root_z = (root_val - root_mean) / root_std
        
        # Propagate through chain
        coeff_product = 1.0
        for edge in chain:
            coeff_product *= edge.coeff
            
        predicted_z = root_z * coeff_product
        direction = "up" if predicted_z > 0 else "down"

        # 2. Joint state consistency penalty
        current_expected_sign = 1 if root_z > 0 else -1
        consistency_penalty = 1.0
        
        for edge in reversed(chain):
            node = edge.target
            if node == "price_return":
                continue 
                
            current_expected_sign = current_expected_sign * (1 if edge.coeff > 0 else -1)
            
            node_val = current_state.get(node, 0.0)
            node_mean = snapshot.feature_means.get(node, 0.0)
            node_std = max(snapshot.feature_stds.get(node, 1.0), 1e-6)
            actual_z = (node_val - node_mean) / node_std
            
            contradiction = max(0.0, -current_expected_sign * actual_z)
            
            if abs(actual_z) > 0.5 and contradiction > 0:
                node_penalty = max(0.20, 1.0 - contradiction * 0.5)
                consistency_penalty *= node_penalty
        
        composite *= consistency_penalty
        
        logger.info(
            "HypothesisGenerator: chain_scored | chain=%s | stability=%.3f | trust=%.3f(n=%d) | length_bonus=%.2f | consistency_penalty=%.2f | composite=%.4f",
            c_key, chain_stability, chain_trust, chain_n, length_bonus, consistency_penalty, composite
        )

        # Time horizon: sum of lags x bar width
        total_lag = sum(e.lag for e in chain)
        time_horizon_s = total_lag * self._bar_width_s
        
        c_key = compute_chain_key(chain)

        return CausalHypothesis(
            graph_version_id=snapshot.version_id,
            chain=list(chain),
            predicted_direction=direction,
            predicted_magnitude=float(magnitude),
            time_horizon_seconds=time_horizon_s,
            layer1_score=round(chain_stability, 4),
            layer2_score=round(chain_trust, 4),
            layer2_n_obs=chain_n,
            composite_score=round(composite, 6),
            chain_key=c_key,
        )

    def _null_hypothesis(self, graph_version_id: str, reason: str) -> CausalHypothesis:
        return CausalHypothesis(
            graph_version_id=graph_version_id,
            chain=[],
            predicted_direction="up",
            predicted_magnitude=0.0,
            time_horizon_seconds=0,
            layer1_score=0.0,
            layer2_score=0.5,
            composite_score=0.0,
        )


# ---------------------------------------------------------------------------
# Layer 2 trust store (stub — full implementation in ml/trust_updater.py)
# ---------------------------------------------------------------------------

class Layer2TrustStore:
    """Read interface for layer2 trust scores (implemented in trust_updater.py)."""

    def get_trust(self, source: str, target: str, lag: int) -> dict:
        raise NotImplementedError
