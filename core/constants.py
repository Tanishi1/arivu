"""core/constants.py — Single Source of Truth for Shared Constants

Any constant that is used by more than one module lives here.
If a path, window size, or threshold changes, it changes in ONE place
and every consumer sees it immediately.

Consumers:
  - core/causal_state.py   (window sizes, propagation factors)
  - ml/regime.py           (model paths, K candidates)
  - regime_trainer.py      (all of the above for offline training)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Model file paths — used by ml/regime.py and regime_trainer.py
# ---------------------------------------------------------------------------
REGIME_MODEL_PATH = Path("data/models/regime_dt.joblib")
KMEANS_MODEL_PATH = Path("data/models/kmeans.joblib")
SCALER_MODEL_PATH = Path("data/models/scaler.joblib")

# ---------------------------------------------------------------------------
# Rolling window sizes — used by causal_state.py and regime_trainer.py
# ---------------------------------------------------------------------------
VOLATILITY_WINDOW = 20   # ticks / candles
TREND_WINDOW = 20        # ticks / candles
VOLUME_WINDOW = 20       # ticks / candles

# ---------------------------------------------------------------------------
# Trend slope normalisation — used in trend_strength calculation
# trend_strength = min(1.0, abs(trend_slope) / SLOPE_NORMALISER)
# ---------------------------------------------------------------------------
SLOPE_NORMALISER = 0.01

# ---------------------------------------------------------------------------
# Spread propagation rule constants
# propagated_spread = raw_spread * (1 + max(0, vol - THRESHOLD) * FACTOR)
# ---------------------------------------------------------------------------
SPREAD_PROPAGATION_THRESHOLD = 0.02
SPREAD_PROPAGATION_FACTOR = 5

# ---------------------------------------------------------------------------
# K-Means regime discovery
# ---------------------------------------------------------------------------
K_CANDIDATES = [2, 3, 4]
MIN_ENTRIES_FOR_KMEANS = 40

