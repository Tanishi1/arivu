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
# BASE_OBS_DIM matches the 19 features in the historical dataset.

BASE_OBS_DIM: int = 19

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

    Maps features in the exact order of VARIABLE_NAMES to match historical training.

    Returns:
        float32 array of shape (BASE_OBS_DIM,)
    """
    values = [
        state.price_return,          # 0
        state.volume,                # 1
        state.spread,                # 2
        state.volatility,            # 3
        state.rsi,                   # 4
        state.trade_intensity,       # 5
        state.order_book_imbalance,  # 6
        state.ema_spread,            # 7
        state.bollinger_width,       # 8
        state.price_in_band,         # 9
        state.regime_volatile,       # 10
        state.regime_trending,       # 11
        state.btc_return,            # 12
        state.eth_return,            # 13
        state.algo_health_p_normal,  # 14
        state.algo_health_p_stressed,# 15
        state.algo_health_p_degraded,# 16
        state.session_sin,           # 17
        state.session_cos,           # 18
    ]
    obs = np.array(values, dtype=np.float32)
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


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
