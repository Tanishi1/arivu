"""core/twin_simulator.py  — Task 6

Digital Twin Propagation Simulator.

Given a GraphSnapshot and a CausalHypothesis, simulates the causal chain
forward by composing edge strengths and lags. Produces a SimulatedTrajectory
— the predicted path of price_return and all affected downstream variables.

Two hypotheses run against the same graph produce two distinct trajectories.
The simulator deterministically selects the hypothesis with the higher
expected value.

This is pure arithmetic on the already-discovered graph — runs in milliseconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ml.causal_discovery import GraphSnapshot
from ml.hypothesis_generator import CausalHypothesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BAR_WIDTH_S: int = 10     # must match core/feature_bar.py
CONFIDENCE_BAND_MULTIPLIER: float = 2.0   # breach = deviation > 2× predicted magnitude
MIN_CONFIDENCE_BAND: float = 0.001        # 0.1% floor — SOL must move 0.1% against prediction


# ---------------------------------------------------------------------------
# Simulated trajectory
# ---------------------------------------------------------------------------

@dataclass
class SimulatedTrajectory:
    """The predicted time-series across all affected variables."""
    hypothesis_id: str
    graph_version_id: str
    # Predicted variable values at each future time step (lag 1, lag 2, ..., lag max)
    steps: list[dict[str, float]] = field(default_factory=list)
    # Final predicted price_return at time horizon
    predicted_price_return: float = 0.0
    # Confidence band: breach if |actual - predicted| > band
    confidence_band: float = 0.0
    # Time horizon in seconds
    time_horizon_s: int = 0
    # Composite expected value (used for selection)
    expected_value: float = 0.0


# ---------------------------------------------------------------------------
# Twin Simulator
# ---------------------------------------------------------------------------

class TwinSimulator:
    """Propagates causal hypotheses through the discovered graph.

    Usage:
        sim = TwinSimulator()
        trajectories = sim.simulate_all(hypotheses, snapshot, current_state)
        best = sim.select_best(hypotheses, snapshot, current_state)
    """

    def simulate(
        self,
        hypothesis: CausalHypothesis,
        snapshot: GraphSnapshot,
        current_state: dict[str, float],
    ) -> SimulatedTrajectory:
        """Simulate one hypothesis forward through the graph.

        Args:
            hypothesis:     the causal chain to simulate.
            snapshot:       the current graph (provides edge coefficients).
            current_state:  dict of variable_name → current value (from FeatureBarBuilder).

        Returns:
            SimulatedTrajectory with per-step predictions.
        """
        if not hypothesis.chain:
            return SimulatedTrajectory(
                hypothesis_id=hypothesis.id,
                graph_version_id=hypothesis.graph_version_id,
                predicted_price_return=0.0,
                confidence_band=0.0,
                time_horizon_s=0,
                expected_value=0.0,
            )

        # Work forward from the start of the chain
        # chain[0] is the edge closest to price_return (final edge)
        # chain[-1] is the furthest edge (starting point)
        # We propagate from chain[-1].source through each edge

        steps: list[dict[str, float]] = []
        
        # S-10 FIX: Standardize current state into Z-scores using graph statistics
        simulated_values: dict[str, float] = {}
        for k, v in current_state.items():
            mean = snapshot.feature_means.get(k, 0.0)
            std = snapshot.feature_stds.get(k, 1.0)
            if std == 0.0:
                std = 1.0
            simulated_values[k] = (v - mean) / std

        # Walk the chain in reverse (from root cause → price_return)
        for edge in reversed(hypothesis.chain):
            source_val = simulated_values.get(edge.source, 0.0)
            # Predicted effect: coeff × source_value
            predicted_effect = edge.coeff * source_val
            # Update target variable's predicted value
            target_prev = simulated_values.get(edge.target, 0.0)
            simulated_values[edge.target] = target_prev + predicted_effect

            steps.append({
                "edge": f"{edge.source}→{edge.target}",
                "lag_bars": edge.lag,
                "lag_seconds": edge.lag * BAR_WIDTH_S,
                "coeff": edge.coeff,
                "z_source_value": source_val,
                "z_predicted_effect": predicted_effect,
                "z_new_target_value": simulated_values[edge.target],
            })

        # S-10 FIX: Un-standardize the final predicted price_return back to raw percentage
        z_predicted_return = simulated_values.get("price_return", 0.0)
        mean_ret = snapshot.feature_means.get("price_return", 0.0)
        std_ret = snapshot.feature_stds.get("price_return", 1.0)
        if std_ret == 0.0:
            std_ret = 1.0
            
        predicted_return = (z_predicted_return * std_ret) + mean_ret
        hypothesis.predicted_direction = "up" if predicted_return > 0 else "down"
        confidence_band = max(
            abs(predicted_return) * CONFIDENCE_BAND_MULTIPLIER,
            MIN_CONFIDENCE_BAND,
        )
        time_horizon_s = hypothesis.time_horizon_seconds

        # Expected value: predicted_return weighted by composite_score
        expected_value = predicted_return * hypothesis.composite_score

        return SimulatedTrajectory(
            hypothesis_id=hypothesis.id,
            graph_version_id=hypothesis.graph_version_id,
            steps=steps,
            predicted_price_return=predicted_return,
            confidence_band=confidence_band,
            time_horizon_s=time_horizon_s,
            expected_value=expected_value,
        )

    def simulate_all(
        self,
        hypotheses: list[CausalHypothesis],
        snapshot: GraphSnapshot,
        current_state: dict[str, float],
    ) -> list[SimulatedTrajectory]:
        """Simulate all hypotheses and return trajectories in the same order."""
        return [self.simulate(h, snapshot, current_state) for h in hypotheses]

    def select_best(
        self,
        hypotheses: list[CausalHypothesis],
        snapshot: GraphSnapshot,
        current_state: dict[str, float],
    ) -> tuple[Optional[CausalHypothesis], Optional[SimulatedTrajectory]]:
        """Simulate all hypotheses and return the one with the highest expected value.

        Returns:
            (best_hypothesis, its_trajectory)
        """
        if not hypotheses:
            raise ValueError("TwinSimulator.select_best called with empty hypothesis list")

        trajectories = self.simulate_all(hypotheses, snapshot, current_state)

        best_ev = 0.0
        best_idx = -1

        for i, t in enumerate(trajectories):
            hyp = hypotheses[i]
            avg_breach_risk = getattr(hyp, "avg_breach_risk", 0.5)
            
            # Risk discount: high breach risk reduces expected value
            risk_discount = max(0.10, 1.0 - avg_breach_risk)
            risk_adjusted_ev = abs(t.expected_value) * risk_discount

            logger.debug(
                "TwinSimulator: hypothesis %d | raw_ev=%.6f | "
                "breach_risk=%.3f | discount=%.3f | risk_adj_ev=%.6f | chain=%s",
                i, t.expected_value, avg_breach_risk,
                risk_discount, risk_adjusted_ev,
                hyp.chain_summary(),
            )

            if risk_adjusted_ev > best_ev:
                best_ev = risk_adjusted_ev
                best_idx = i

        if best_idx == -1:
            logger.info("TwinSimulator: no hypothesis with non-zero risk-adjusted EV")
            return None, None

        logger.info(
            "TwinSimulator: selected | chain=%s | raw_ev=%.6f | "
            "breach_risk=%.3f | risk_adj_ev=%.6f",
            hypotheses[best_idx].chain_summary(),
            trajectories[best_idx].expected_value,
            getattr(hypotheses[best_idx], "avg_breach_risk", 0.5),
            best_ev,
        )
        return hypotheses[best_idx], trajectories[best_idx]

    def check_breach(
        self,
        trajectory: SimulatedTrajectory,
        actual_price_return: float,
        elapsed_seconds: int,
    ) -> bool:
        """Return True if the actual return deviates beyond the confidence band.

        Used by core/monitor.py for causal-agent DOs.

        Args:
            trajectory:          the original SimulatedTrajectory for the active trade.
            actual_price_return: the real price return since trade entry.
            elapsed_seconds:     how long the trade has been open.

        Returns:
            True → breach (trade monitoring should trigger exit).
        """
        if elapsed_seconds > trajectory.time_horizon_s:
            # Past time horizon — monitoring by trajectory no longer relevant
            return False

        deviation = abs(actual_price_return - trajectory.predicted_price_return)
        is_breach = deviation > trajectory.confidence_band

        if is_breach:
            logger.warning(
                "TwinSimulator: trajectory breach | "
                "predicted=%.6f actual=%.6f deviation=%.6f band=%.6f | "
                "hyp=%s",
                trajectory.predicted_price_return,
                actual_price_return,
                deviation,
                trajectory.confidence_band,
                trajectory.hypothesis_id[:8],
            )

        return is_breach
