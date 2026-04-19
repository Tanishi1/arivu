"""ml/features.py — Single Source of Truth for Feature Computation
(See DECISIONS.md ADR-005)

This module is the ONLY place where ML2 feature computation lives.
It is imported by:
  - ml/ml2.py       (at training time, reading from training_buffer.csv)
  - core/simulator.py (at inference time, live in decision_cycle_loop)

Any feature change MUST be made here and ONLY here.
Using two different implementations guarantees training-serving skew — a bug
that produces falsely optimistic validation metrics.
"""

from __future__ import annotations

from core.schemas import Assumption, CausalState


def compute_ml2_features(
    causal_state: CausalState,
    assumption: Assumption,
    time_horizon: float,
    ml1_vector: list[float],
) -> dict:
    """Compute the feature dict for ML2 inference and training.

    Must be called with the SAME causal_state snapshot used at decision time.
    Do NOT use any value computed after timestamp_committed — that is data leakage.

    Args:
        causal_state:  snapshot of CausalState at decision time
        assumption:    the Assumption being scored
        time_horizon:  strategy window duration in minutes
        ml1_vector:    [p_normal, p_stressed, p_degraded] from ML1 at decision time

    Returns:
        dict with exactly 9 keys matching training_buffer.csv columns:
        volatility, spread, trend_strength, volume, proximity,
        time_horizon, p_normal, p_stressed, p_degraded
    """
    p_normal, p_stressed, p_degraded = ml1_vector[0], ml1_vector[1], ml1_vector[2]

    return {
        "volatility":    causal_state.volatility,
        "spread":        causal_state.spread,
        "trend_strength": causal_state.trend_strength,
        "volume":        causal_state.volume,
        "proximity":     assumption.proximity,
        "time_horizon":  time_horizon,
        "p_normal":      p_normal,
        "p_stressed":    p_stressed,   # most informative ML1 feature for breach risk
        "p_degraded":    p_degraded,
    }
