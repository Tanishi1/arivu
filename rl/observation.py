"""rl/observation.py
Builds observation vectors for Standard PPO (Arm 2) and Causal RL (Arm 3).

Design rules:
  - build_base_obs() dynamically reads CausalState scalar fields at import time.
    Never hardcode the feature count — CausalState can grow.
  - algo_health_vector is a list[float] of 3 elements: flattened as 3 dims.
  - Excluded from obs: timestamp, active_strategy, price_history,
    last_tick_timestamp (non-float or non-scalar).
  - build_causal_obs() appends causal graph features to the base obs.
  - All arrays are float32 (gymnasium convention).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from core.schemas import CausalState
    from ml.causal_discovery import GraphSnapshot
    from ml.layer1_tracker import Layer1Tracker
    from ml.hypothesis_generator import CausalHypothesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fields excluded from observation (non-float or non-scalar)
# ---------------------------------------------------------------------------
_OBS_EXCLUDED: frozenset[str] = frozenset({
    "timestamp",
    "active_strategy",
    "price_history",
    "last_tick_timestamp",
    "algo_health_vector",   # handled explicitly as 3 separate dims
})

# Compute base obs dim at import time so TradingEnv can use it
# without needing a live CausalState instance.
def _compute_base_obs_dim() -> int:
    """Introspect CausalState to count scalar float fields + 3 for algo_health."""
    from core.schemas import CausalState
    import typing

    count = 0
    for name, field_info in CausalState.model_fields.items():
        if name in _OBS_EXCLUDED:
            continue
        ann = field_info.annotation
        # Accept plain float; also accept Optional[float]
        origin = getattr(ann, "__origin__", None)
        if ann is float:
            count += 1
        elif origin is type(None):
            pass
        # Handle Optional[float] → Union[float, None]
        elif origin is not None:
            args = getattr(ann, "__args__", ())
            if float in args:
                count += 1
    # Add 3 for algo_health_vector dimensions
    count += 3
    return count


BASE_OBS_DIM: int = _compute_base_obs_dim()

# Causal feature dimensions appended for Arm 3:
#   3 hypothesis EV scores + 3 hypothesis breach risk + 1 edge stability + regime one-hot
# Regime one-hot is dynamic; we fix max K=5 for space allocation.
MAX_REGIME_K: int = 5
CAUSAL_EXTRA_DIMS: int = 3 + 3 + 1 + MAX_REGIME_K   # = 12
CAUSAL_OBS_DIM: int = BASE_OBS_DIM + CAUSAL_EXTRA_DIMS


# ---------------------------------------------------------------------------
# Regime integer encoder (tracks regimes seen, assigns stable int IDs)
# ---------------------------------------------------------------------------

_regime_to_idx: dict[str, int] = {}
_KNOWN_REGIMES = ["calm", "volatile", "trending", "unknown", "stressed"]
for _r in _KNOWN_REGIMES:
    _regime_to_idx[_r] = len(_regime_to_idx)


def _regime_one_hot(regime: str) -> np.ndarray:
    """Encode regime as one-hot of length MAX_REGIME_K."""
    vec = np.zeros(MAX_REGIME_K, dtype=np.float32)
    idx = _regime_to_idx.get(regime, _regime_to_idx.get("unknown", 0))
    if idx < MAX_REGIME_K:
        vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_base_obs_dim() -> int:
    """Return the dimensionality of the base (Arm 2) observation vector."""
    return BASE_OBS_DIM


def get_causal_obs_dim() -> int:
    """Return the dimensionality of the Arm 3 (causal RL) observation vector."""
    return CAUSAL_OBS_DIM


def build_base_obs(state: "CausalState") -> np.ndarray:
    """Build the base observation vector from a CausalState snapshot.

    Dynamically reads all scalar float fields from CausalState, excluding
    non-float or compound fields defined in _OBS_EXCLUDED.
    algo_health_vector is flattened as 3 separate dimensions appended at end.

    Returns:
        float32 array of shape (BASE_OBS_DIM,)
    """
    values: list[float] = []
    import typing

    for name, field_info in state.model_fields.items():
        if name in _OBS_EXCLUDED:
            continue
        ann = field_info.annotation
        val = getattr(state, name, 0.0)
        is_float = ann is float
        if not is_float:
            args = getattr(ann, "__args__", ())
            is_float = float in args
        if is_float:
            values.append(float(val) if val is not None else 0.0)

    # Append algo_health_vector as 3 explicit dims
    ahv = state.algo_health_vector
    if len(ahv) == 3:
        values.extend([float(ahv[0]), float(ahv[1]), float(ahv[2])])
    else:
        values.extend([1.0, 0.0, 0.0])

    obs = np.array(values, dtype=np.float32)

    # Guard against NaN/Inf — replace with 0 to protect training
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    if len(obs) != BASE_OBS_DIM:
        # CausalState schema changed — log and zero-pad to maintain shape
        logger.warning(
            "observation.py: obs dim mismatch — expected %d, got %d. "
            "CausalState may have changed. Zero-padding to maintain shape.",
            BASE_OBS_DIM, len(obs),
        )
        padded = np.zeros(BASE_OBS_DIM, dtype=np.float32)
        padded[:min(len(obs), BASE_OBS_DIM)] = obs[:BASE_OBS_DIM]
        obs = padded

    return obs


def build_causal_obs(
    state: "CausalState",
    graph: Optional["GraphSnapshot"],
    layer1: Optional["Layer1Tracker"],
    hypotheses: Optional[list["CausalHypothesis"]],
    regime: str = "unknown",
) -> np.ndarray:
    """Build the Arm 3 (causal RL) observation vector.

    Appends causal graph features to the base obs:
      [base_obs | ev_top3 | breach_risk_top3 | edge_stability | regime_one_hot]

    If no graph exists (early in run before Layer 1 validates), causal dims
    are zero-filled and a DEBUG log is emitted. This is correct Phase A behaviour.

    Returns:
        float32 array of shape (CAUSAL_OBS_DIM,)
    """
    base = build_base_obs(state)

    causal_dims = np.zeros(CAUSAL_EXTRA_DIMS, dtype=np.float32)

    if graph is None or layer1 is None:
        logger.debug("ppo_causal: no graph yet, obs zeroed")
        regime_oh = _regime_one_hot(regime)
        causal_dims[7:7 + MAX_REGIME_K] = regime_oh
        return np.concatenate([base, causal_dims], axis=0)

    # Top 3 hypothesis EV scores
    ev_scores = np.zeros(3, dtype=np.float32)
    breach_scores = np.zeros(3, dtype=np.float32)
    if hypotheses:
        sorted_hyps = sorted(
            hypotheses,
            key=lambda h: abs(getattr(h, "avg_breach_risk", 0.5)),
        )[:3]
        for i, hyp in enumerate(sorted_hyps[:3]):
            ev_scores[i] = float(hyp.composite_score)
            breach_scores[i] = float(getattr(hyp, "avg_breach_risk", 0.5))

    # Edge stability: n_stable / n_total
    validated_edges = layer1.get_validated_edges(regime)
    all_edges = layer1._all_edge_keys
    edge_stability = (
        len(validated_edges) / len(all_edges) if all_edges else 0.0
    )

    # Regime one-hot
    regime_oh = _regime_one_hot(regime)

    causal_dims[:3]              = ev_scores
    causal_dims[3:6]             = breach_scores
    causal_dims[6]               = float(edge_stability)
    causal_dims[7:7 + MAX_REGIME_K] = regime_oh

    full_obs = np.concatenate([base, causal_dims], axis=0)
    full_obs = np.nan_to_num(full_obs, nan=0.0, posinf=0.0, neginf=0.0)
    return full_obs
