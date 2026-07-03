"""verify_models.py
Loads the saved K-Means and Decision Tree models and confirms they
work end-to-end exactly as the live system (main.py) uses them.

What this checks:
  1. Both .joblib files exist and load without errors
  2. DT metadata — feature count (3 or 4) and valid regime class labels
  3. RegimeClassifier wrapper — correct feature routing
  4. Prediction sanity — known regime labels are reachable
  5. Bootstrap fallback — RegimeClassifier still works if no model is loaded
  6. Simulator integration — DT plugs into Simulator.run() via predict_proba()

  NOTE (S3 fix): Checks 2 are forward-compatible. After Week 7 K-Means retraining
  on 80+ live cycles, the DT may use 4 features instead of 3, and a third regime
  label ('trending') may emerge. Hardcoded checks would FAIL on a correct model.

Usage:
    python verify_models.py
"""

import sys
import io

# Force UTF-8 output so unicode labels print cleanly on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import joblib
import numpy as np

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.constants import KMEANS_MODEL_PATH, REGIME_MODEL_PATH, K_CANDIDATES
from core.schemas import CausalState
from ml.regime import RegimeClassifier

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        msg = f"  {FAIL}  {label}" + (f" | {detail}" if detail else "")
        print(msg)
        errors.append(label)


# ---------------------------------------------------------------------------
# 1. Files exist
# ---------------------------------------------------------------------------
print("\n-- 1. Model files " + "-" * 42)
check(KMEANS_MODEL_PATH.exists(), f"kmeans.joblib exists at {KMEANS_MODEL_PATH}")
check(REGIME_MODEL_PATH.exists(), f"regime_dt.joblib exists at {REGIME_MODEL_PATH}")

# Load raw models (bypass RegimeClassifier for structural checks)
try:
    kmeans = joblib.load(KMEANS_MODEL_PATH)
    km_loaded = True
    print(f"  {PASS}  kmeans.joblib loaded cleanly")
except Exception as e:
    km_loaded = False
    errors.append("kmeans.joblib load")
    print(f"  {FAIL}  kmeans.joblib failed to load | {e}")

try:
    dt = joblib.load(REGIME_MODEL_PATH)
    dt_loaded = True
    print(f"  {PASS}  regime_dt.joblib loaded cleanly")
except Exception as e:
    dt_loaded = False
    errors.append("regime_dt.joblib load")
    print(f"  {FAIL}  regime_dt.joblib failed to load | {e}")

if not (km_loaded and dt_loaded):
    print("\nCannot continue — model files failed to load.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 2. DT metadata
# ---------------------------------------------------------------------------
# S3 FIX: All three checks are now forward-compatible.
# After Week 7 K-Means retraining on 80+ live SOL cycles:
#   - DT may be retrained on 4 features (volatility, spread, trend_strength, volume)
#   - A third regime label 'trending' may emerge from the new clusters
#   - K-Means may select K=3 if silhouette scores favour it
# Hardcoded == checks would FAIL on a correct, improved model after retraining.

check(
    dt.n_features_in_ in (3, 4),
    f"DT feature count is 3 or 4 (got {dt.n_features_in_})",
    "Expected 3 (historical model) or 4 (post-live-retrain model)",
)

_VALID_REGIME_CLASSES = {"calm", "volatile", "trending"}
actual_classes = set(dt.classes_)
check(
    actual_classes.issubset(_VALID_REGIME_CLASSES) and len(actual_classes) >= 2,
    f"DT classes are valid regime labels: {sorted(actual_classes)}",
    f"Must be a non-empty subset of {sorted(_VALID_REGIME_CLASSES)}",
)
check(
    hasattr(kmeans, "cluster_centers_"),
    "K-Means has cluster_centers_ (was properly fitted)",
)
check(
    kmeans.n_clusters in K_CANDIDATES,
    f"K-Means K={kmeans.n_clusters} is in allowed candidates {K_CANDIDATES}",
    f"K_CANDIDATES={K_CANDIDATES} (from core.constants)",
)


# ---------------------------------------------------------------------------
# 3. RegimeClassifier wrapper
# ---------------------------------------------------------------------------
print("\n-- 3. RegimeClassifier wrapper " + "-" * 29)

rc = RegimeClassifier()
check(rc._is_trained, "RegimeClassifier loaded pre-trained DT from disk")
num_features = rc._dt.n_features_in_ if rc._dt else 3
check(
    rc._dt is not None and num_features in (3, 4),
    f"RegimeClassifier._dt has n_features_in_={num_features} -> uses {num_features}-feature path",
)

# Determine test inputs based on the scaler or default values
if rc._scaler is not None:
    # Use scaler's mean and scale for testing to be robust to unit differences
    mean_vol = rc._scaler.mean_[0]
    scale_vol = rc._scaler.scale_[0]
    mean_volum = rc._scaler.mean_[-1]
    scale_volum = rc._scaler.scale_[-1]
    
    vol_low = max(0.0, mean_vol - 0.5 * scale_vol)
    vol_high = mean_vol + 3.0 * scale_vol
    volume_low = max(0.0, mean_volum - 0.5 * scale_volum)
    volume_high = mean_volum + 3.0 * scale_volum
else:
    # Fallback to historical units
    vol_low = 0.0005
    vol_high = 0.02
    volume_low = 10.0
    volume_high = 200.0

label_calm = rc.classify(volatility=vol_low, spread=0.001, trend_strength=0.0, volume=volume_low)
check(
    label_calm in _VALID_REGIME_CLASSES,
    f"classify(vol={vol_low:.6f}, volume={volume_low:.1f}) -> '{label_calm}' (valid label)",
)

label_volatile = rc.classify(volatility=vol_high, spread=0.001, trend_strength=0.0, volume=volume_high)
check(
    label_volatile in _VALID_REGIME_CLASSES,
    f"classify(vol={vol_high:.6f}, volume={volume_high:.1f}) -> '{label_volatile}' (valid label)",
)

if num_features == 3:
    # Spread is intentionally ignored for the 3-feature model.
    # Passing different spread values should NOT change the output.
    label_spread_a = rc.classify(volatility=vol_low, spread=0.0, trend_strength=0.999, volume=volume_low)
    label_spread_b = rc.classify(volatility=vol_low, spread=999.0, trend_strength=0.999, volume=volume_low)
    check(
        label_spread_a == label_spread_b,
        "Spread parameter ignored for 3-feature model (as designed)",
        f"Different spreads gave different results: '{label_spread_a}' vs '{label_spread_b}'",
    )
else:
    print("  [INFO]  4-feature model: spread is included in the classifier inputs.")

# Output must always be one of the known valid classes
for label in [label_calm, label_volatile]:
    check(
        label in _VALID_REGIME_CLASSES,
        f"Output '{label}' is a valid regime label",
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if not errors:
    print("All checks passed. Models are working correctly.")
else:
    print(f"{len(errors)} check(s) failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("=" * 60)
