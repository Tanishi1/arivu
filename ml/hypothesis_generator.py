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

    def chain_summary(self) -> str:
        if not self.chain:
            return "empty"
        parts = []
        for e in self.chain:
            parts.append(f"{e.source}->(lag={e.lag})->{e.target}")
        return " | ".join(parts)


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

    def generate(self, snapshot: GraphSnapshot, regime: str = "unknown") -> list[CausalHypothesis]:
        """Generate and rank hypotheses from the current graph.

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

        # Find all chains (up to MAX_HOP_DEPTH) ending at price_return
        chains = self._find_chains(snapshot, TARGET_VARIABLE, MAX_HOP_DEPTH, regime)

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
                # Find best edge pointing to price_return by raw stability
                best_edge = None
                best_score = 0.0
                for edge in price_edges:
                    score = self._layer1.get_stability_score(
                        edge.source, edge.target, edge.lag, regime
                    )
                    if score > best_score:
                        best_score = score
                        best_edge = edge

                if best_edge is not None and best_score >= ESCAPE_VALVE_MIN_STABILITY:
                    logger.warning(
                        "HypothesisGenerator: ESCAPE_VALVE | "
                        "runs=%d best_edge=%s->%s stability=%.3f "
                        "threshold=%.2f | firing low-confidence hypothesis",
                        n_runs, best_edge.source, best_edge.target,
                        best_score,
                        self._layer1._optimizer.get_params(regime).threshold,
                    )
                    hyp = self._score_chain([best_edge], snapshot.version_id, regime)
                    hyp.is_escape_valve = True
                    return [hyp]

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
            hyp = self._score_chain(chain, snapshot.version_id, regime)
            hypotheses.append(hyp)

        # Sort by composite score descending
        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)

        # Return all hypotheses sorted by score
        
        # Log HOLD warning if even the best is below MIN_TRADE_SCORE
        best = hypotheses[0]
        if best.composite_score < MIN_TRADE_SCORE:
            logger.warning(
                "HypothesisGenerator: HOLD | reason=weak_top_candidate | "
                "top_score=%.4f | threshold=%.4f | chain=%s",
                best.composite_score, MIN_TRADE_SCORE, best.chain_summary(),
            )
        else:
            logger.info(
                "HypothesisGenerator: %d hypotheses | best_score=%.4f | chain=%s",
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
    ) -> list[list[CausalEdge]]:
        """Find all validated causal chains ending at target, up to max_depth hops.

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
                if not self._layer1.is_validated(
                    edge.source, edge.target, edge.lag, regime
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
            if not self._layer1.is_validated(
                edge.source, edge.target, edge.lag, regime
            ):
                score = self._layer1.get_stability_score(
                    edge.source, edge.target, edge.lag, regime
                )
                logger.debug(
                    "1-hop pruned | %s->%s lag=%d | stability=%.2f | "
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
        graph_version_id: str,
        regime: str,
    ) -> CausalHypothesis:
        """Compute composite score for a single causal chain.

        Magnitude uses GEOMETRIC MEAN of edge coefficients, not raw product.

        Why: raw product collapses exponentially with chain length.
          1-hop  coeff=0.30          -> product=0.300, geom_mean=0.300  (same)
          2-hop  coeffs=0.30, 0.20   -> product=0.060, geom_mean=0.245  (fair)
          3-hop  coeffs=0.30,0.20,0.15 -> product=0.009, geom_mean=0.208 (fair)

        CHAIN_LENGTH_PENALTY (0.85 per hop) is the ONE AND ONLY confidence
        discount for chain length. Geometric mean prevents a second hidden
        penalty from making multi-hop chains mathematically unreachable.
        """
        n_hops = len(chain)

        # Layer 1 stability: mean stability score across all edges in chain
        l1_scores = [
            self._layer1.get_stability_score(e.source, e.target, e.lag, regime)
            for e in chain
        ]
        layer1_score = sum(l1_scores) / len(l1_scores) if l1_scores else 0.0

        # Layer 2 trust: mean trust across edges (default 0.5 — neutral until observed)
        layer2_score = 0.5
        layer2_n_obs = 0
        if self._layer2 is not None:
            l2_results = [
                self._layer2.get_trust(e.source, e.target, e.lag)
                for e in chain
            ]
            trust_scores = [r["trust"] for r in l2_results]
            obs_counts = [r["n_observations"] for r in l2_results]
            layer2_score = sum(trust_scores) / len(trust_scores) if trust_scores else 0.5
            layer2_n_obs = min(obs_counts)  # use min — chain is only as trusted as weakest link

            # Down-weight layer2 if insufficient observations
            if layer2_n_obs < MIN_LAYER2_OBS:
                # Blend toward neutral (0.5) proportionally
                weight = layer2_n_obs / MIN_LAYER2_OBS
                layer2_score = layer2_score * weight + 0.5 * (1 - weight)

        # Chain length penalty: 0.85 per hop AFTER the first.
        # This is the ONLY length-based confidence discount.
        length_penalty = CHAIN_LENGTH_PENALTY ** (n_hops - 1)

        # Effect magnitude: geometric mean of absolute edge coefficients.
        # Geometric mean = (product)^(1/n_hops), which normalises for chain length
        # so multi-hop chains aren't exponentially penalised relative to 1-hop.
        raw_product = 1.0
        for e in chain:
            raw_product *= abs(e.coeff)
        # Geometric mean — stays in (0, 1] for typical PCMCI coefficients
        magnitude = raw_product ** (1.0 / n_hops) if n_hops > 0 else 0.0
        magnitude = min(magnitude, 1.0)

        # Composite score
        composite = layer1_score * layer2_score * length_penalty * magnitude

        # Direction: sign of the edge closest to price_return (chain[0] by convention)
        final_edge = chain[0]
        direction = "up" if final_edge.coeff > 0 else "down"

        # Time horizon: sum of lags x bar width
        total_lag = sum(e.lag for e in chain)
        time_horizon_s = total_lag * self._bar_width_s

        return CausalHypothesis(
            graph_version_id=graph_version_id,
            chain=list(chain),
            predicted_direction=direction,
            predicted_magnitude=float(magnitude),
            time_horizon_seconds=time_horizon_s,
            layer1_score=round(layer1_score, 4),
            layer2_score=round(layer2_score, 4),
            layer2_n_obs=layer2_n_obs,
            composite_score=round(composite, 6),
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
