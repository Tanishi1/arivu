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
ML1_MODEL_PATH    = Path("data/models/ml1.joblib")
ML2_MODEL_PATH    = Path("data/models/ml2.joblib")

# SQLite — reads SQLITE_PATH env var if set (e.g. SQLITE_PATH=data/prod.db)
# N4 FIX: single source of truth — models.py imports this instead of redefining it
import os as _os
SQLITE_PATH = Path(_os.getenv("SQLITE_PATH", "data/arivu.db"))
del _os  # don't pollute the constants namespace
TRAINING_BUFFER_PATH = Path("data/training_buffer.csv")

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

# Strategy window
STRATEGY_HORIZON_MINUTES = 30

# ML retraining triggers
ML1_RETRAIN_AFTER = 100   # ExecutionTelemetry samples
ML2_RETRAIN_AFTER = 20    # closed OutcomeRecord entries

# Per-instrument assumption thresholds
INSTRUMENTS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

THRESHOLDS = {
    "BTCUSDT": {
        "volatility_limit":   0.04,
        "spread_limit":       0.002,
        "trend_slope_min":    0.0,
        "trend_strength_max": 0.4,
    },
    "ETHUSDT": {
        "volatility_limit":   0.05,
        "spread_limit":       0.003,
        "trend_slope_min":    0.0,
        "trend_strength_max": 0.5,
    },
    "SOLUSDT": {
        "volatility_limit":   0.07,
        "spread_limit":       0.005,
        "trend_slope_min":    0.0,
        "trend_strength_max": 0.6,
    },
}

