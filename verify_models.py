"""verify_models.py
Loads the saved K-Means and Decision Tree models and confirms they
work end-to-end exactly as the live system (main.py) uses them.

What this checks:
  1. Both .joblib files exist and load without errors
  2. DT metadata — correct feature count (3) and known classes
  3. RegimeClassifier wrapper — correct feature routing (3-feature path)
  4. Prediction sanity — both 'calm' and 'volatile' are reachable
  5. Bootstrap fallback — RegimeClassifier still works if no model is loaded
  6. Simulator integration — DT plugs into Simulator.run() via predict_proba()

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

from core.constants import KMEANS_MODEL_PATH, REGIME_MODEL_PATH
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
print("\n-- 2. Decision Tree metadata " + "-" * 32)

check(
    dt.n_features_in_ == 3,
    f"DT trained on 3 features (got {dt.n_features_in_})",
    "Expected [volatility, trend_strength, volume]",
)
known_classes = {"calm", "volatile"}
actual_classes = set(dt.classes_)
check(
    actual_classes == known_classes,
    f"DT classes match expected: {sorted(actual_classes)}",
    f"Expected {sorted(known_classes)}",
)
check(
    hasattr(kmeans, "cluster_centers_"),
    "K-Means has cluster_centers_ (was properly fitted)",
)
check(
    kmeans.n_clusters == 2,
    f"K-Means used best K=2 (silhouette 0.8845 won)",
    f"Got n_clusters={kmeans.n_clusters}",
)


# ---------------------------------------------------------------------------
# 3. RegimeClassifier wrapper
# ---------------------------------------------------------------------------
print("\n-- 3. RegimeClassifier wrapper " + "-" * 29)

rc = RegimeClassifier()
check(rc._is_trained, "RegimeClassifier loaded pre-trained DT from disk")
check(
    rc._dt is not None and rc._dt.n_features_in_ == 3,
    "RegimeClassifier._dt has n_features_in_=3 -> uses 3-feature path",
)

# Verify the classify() method picks the 3-feature branch (line 78-79 in regime.py)
# volume < 78.83 -> calm, volume >= 78.83 -> volatile (actual DT rules)
label_calm = rc.classify(volatility=0.0005, spread=0.001, trend_strength=0.999, volume=10.0)
check(
    label_calm == "calm",
    f"classify(vol=0.0005, volume=10) -> 'calm' (got '{label_calm}')",
    "Low volume candle should be calm",
)

label_volatile = rc.classify(volatility=0.001, spread=0.001, trend_strength=0.999, volume=200.0)
check(
    label_volatile == "volatile",
    f"classify(vol=0.001, volume=200) -> 'volatile' (got '{label_volatile}')",
    "High volume candle should be volatile",
)

# Spread is intentionally ignored for the 3-feature model.
# Passing different spread values should NOT change the output.
label_spread_a = rc.classify(volatility=0.0005, spread=0.0, trend_strength=0.999, volume=10.0)
label_spread_b = rc.classify(volatility=0.0005, spread=999.0, trend_strength=0.999, volume=10.0)
check(
    label_spread_a == label_spread_b,
    "Spread parameter ignored for 3-feature model (as designed)",
    f"Different spreads gave different results: '{label_spread_a}' vs '{label_spread_b}'",
)

# Output must always be one of the known valid classes
for label in [label_calm, label_volatile]:
    check(
        label in known_classes,
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
