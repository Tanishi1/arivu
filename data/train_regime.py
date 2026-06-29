"""data/train_regime.py
Feature engineering + K-Means + Decision Tree.

Reads:  data/solusdt_1m_raw.csv
Writes: data/models/regime_dt.joblib
        data/models/kmeans.joblib
        data/regime_training_report.txt

Features used (3 — matches what Decision Tree is trained on):
    volatility     — rolling std of 1-min returns over 20 candles
    trend_strength — normalised abs slope of close prices over 20 candles
    volume         — raw volume from kline

NOTE: spread is NOT a training feature because historical kline data
has no bid/ask prices. Spread is only available in the live system.
The Decision Tree is intentionally trained on 3 features only.

Usage:
    python data/train_regime.py
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Add project root to path so core.constants resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.regime import (
    K_CANDIDATES,
    KMEANS_MODEL_PATH,
    REGIME_MODEL_PATH,
    SCALER_MODEL_PATH,
)
from core.causal_state import (
    TREND_WINDOW,
    VOLATILITY_WINDOW,
    VOLUME_WINDOW,
)

SLOPE_NORMALISER = 0.01

REPORT_PATH = Path(__file__).parent / "regime_training_report.txt"

# Exactly the 3 features the Decision Tree is trained on.
# This list is the single source of truth — used during training,
# sanity check, and must match what ml/regime.py passes at inference.
FEATURE_COLS = ["volatility", "trend_strength", "volume"]


# ── Step 1: Load raw klines ───────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["open_time_ms", "close", "volume"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


# ── Step 2: Compute features ──────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    returns = df["close"].pct_change()
    df["volatility"] = returns.rolling(VOLATILITY_WINDOW).std()

    x = np.arange(TREND_WINDOW)
    slopes = df["close"].rolling(TREND_WINDOW).apply(
        lambda y: np.polyfit(x, y, 1)[0], raw=True
    )
    df["trend_strength"] = (slopes.abs() / SLOPE_NORMALISER).clip(upper=1.0)

    df = df.dropna(subset=["volatility", "trend_strength"]).reset_index(drop=True)
    print(f"Features computed | {len(df):,} usable rows after rolling warmup")
    return df


# ── Step 3: K-Means ───────────────────────────────────────────────

def run_kmeans(X: np.ndarray) -> tuple[int, float, object, np.ndarray]:
    best_k, best_score, best_km, best_labels = K_CANDIDATES[0], -1.0, None, None

    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        # Performance optimization for large datasets
        score = silhouette_score(X, labels, sample_size=10000, random_state=42)
        print(f"  K={k} | silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k, best_km, best_labels = score, k, km, labels

    KMEANS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_km, KMEANS_MODEL_PATH)
    print(f"K-Means done | best_k={best_k} | saved to {KMEANS_MODEL_PATH}")
    return best_k, best_score, best_km, best_labels


# ── Step 4: Name clusters ─────────────────────────────────────────

def name_clusters(
    labels: np.ndarray, df: pd.DataFrame
) -> tuple[list[str], dict]:
    cluster_ids = sorted(set(labels))
    stats = {}
    for cid in cluster_ids:
        mask = labels == cid
        stats[cid] = {
            "vol":   df.loc[mask, "volatility"].mean(),
            "trend": df.loc[mask, "trend_strength"].mean(),
            "count": int(mask.sum()),
            "symbols": df.loc[mask, "symbol"].value_counts().to_dict() if "symbol" in df else {},
        }

    sorted_by_vol = sorted(
        cluster_ids, key=lambda c: stats[c]["vol"], reverse=True
    )
    regime_map = {sorted_by_vol[0]: "volatile"}

    if len(sorted_by_vol) > 2:
        remaining = sorted(
            sorted_by_vol[1:],
            key=lambda c: stats[c]["trend"],
            reverse=True,
        )
        regime_map[remaining[0]] = "trending"
        for c in remaining[1:]:
            regime_map[c] = "calm"
    elif len(sorted_by_vol) == 2:
        regime_map[sorted_by_vol[1]] = "calm"

    named = [regime_map.get(lbl, "calm") for lbl in labels]
    total = len(named)
    for regime in sorted(set(named)):
        count = named.count(regime)
        print(f"  Cluster '{regime}': {count:,} rows ({count/total*100:.1f}%)")

    centroid_info = {}
    for cid, regime in regime_map.items():
        centroid_info[regime] = stats[cid]

    return named, centroid_info


# ── Step 5: Train Decision Tree ───────────────────────────────────

def train_dt(
    X: np.ndarray, named_labels: list[str]
) -> tuple[object, float]:
    # Chronological 80/20 split — no shuffle, this is time-series data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = named_labels[:split], named_labels[split:]

    dt = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )
    dt.fit(X_train, y_train)
    accuracy = dt.score(X_test, y_test)

    joblib.dump(dt, REGIME_MODEL_PATH)
    print(
        f"Decision Tree trained | classes={list(dt.classes_)} "
        f"| test accuracy={accuracy:.4f} | saved to {REGIME_MODEL_PATH}"
    )
    return dt, accuracy


# ── Step 6: Write training report ─────────────────────────────────

def write_report(
    n_rows: int,
    best_k: int,
    best_silhouette: float,
    centroid_info: dict,
    dt_accuracy: float,
) -> None:
    lines = [
        "=" * 60,
        "ARIVU — Regime Training Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "Input data",
        f"  Rows processed      : {n_rows:,}",
        f"  Volatility window   : {VOLATILITY_WINDOW} candles",
        f"  Trend window        : {TREND_WINDOW} candles",
        f"  Volume window       : {VOLUME_WINDOW} candles",
        f"  Slope normaliser    : {SLOPE_NORMALISER}",
        "",
        "Features used for training (3)",
        f"  {FEATURE_COLS}",
        "  NOTE: spread excluded — no bid/ask in historical kline data.",
        "  The live RegimeClassifier uses these same 3 features only.",
        "",
        "K-Means results",
        f"  Best K              : {best_k}",
        f"  Silhouette score    : {best_silhouette:.4f}",
        f"  Model saved to      : {KMEANS_MODEL_PATH}",
        "",
        "Cluster centroids",
    ]

    for regime, info in sorted(centroid_info.items()):
        pct = info["count"] / n_rows * 100
        sym_str = " | ".join(f"{s}:{c}" for s, c in info.get("symbols", {}).items())
        lines += [
            f"  [{regime}]",
            f"    rows            : {info['count']:,}  ({pct:.1f}%)",
            f"    mean volatility : {info['vol']:.6f}",
            f"    mean trend_str  : {info['trend']:.6f}",
            f"    symbols         : {sym_str}",
        ]

    lines += [
        "",
        "Decision Tree results",
        f"  max_depth           : 5",
        f"  class_weight        : balanced",
        f"  Train/test split    : 80/20 chronological (no shuffle)",
        f"  Test accuracy       : {dt_accuracy:.4f}",
        f"  Model saved to      : {REGIME_MODEL_PATH}",
        "=" * 60,
    ]

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Training report written to {REPORT_PATH}")


# ── Step 7: Sanity check ──────────────────────────────────────────

def run_sanity_check() -> None:
    print("\nRunning sanity check...")
    from ml.regime import RegimeClassifier
    rc = RegimeClassifier()
    assert rc._is_trained, (
        "FATAL: RegimeClassifier did not load pre-trained model. "
        "Check that REGIME_MODEL_PATH in core/constants.py matches "
        "the path used by ml/regime.py."
    )
    # Pass exactly 3 features — same as FEATURE_COLS above.
    # We pass spread=0.0 to satisfy the method signature, but the modified classify
    # method will dynamically ignore it if the loaded DT only expects 3 features.
    label = rc.classify(
        volatility=0.02,
        spread=0.0,
        trend_strength=0.7,
        volume=1000.0,
    )
    print(f"Sanity check passed — sample prediction: '{label}'")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SOLUSDT", help="Specific symbol (e.g., SOLUSDT)")
    args = parser.parse_args()

    symbols_to_run = [args.symbol.upper()]

    dfs = []
    for symbol in symbols_to_run:
        path = Path(__file__).parent / f"{symbol.lower()}_1m_raw.csv"
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
        df = load_raw(path)
        df["symbol"] = symbol
        df = compute_features(df)
        dfs.append(df)

    if not dfs:
        print("Error: No data files found.")
        sys.exit(1)

    df_all = pd.concat(dfs, ignore_index=True)
    X_raw = df_all[FEATURE_COLS].values

    print("\nScaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    SCALER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_MODEL_PATH)
    print(f"Scaler saved to {SCALER_MODEL_PATH}")

    print("\nRunning K-Means...")
    best_k, best_silhouette, best_km, labels = run_kmeans(X)

    print("\nNaming clusters...")
    named_labels, centroid_info = name_clusters(labels, df_all)

    print("\nTraining Decision Tree...")
    dt, dt_accuracy = train_dt(X, named_labels)

    print("\nWriting training report...")
    write_report(
        n_rows=len(df_all),
        best_k=best_k,
        best_silhouette=best_silhouette,
        centroid_info=centroid_info,
        dt_accuracy=dt_accuracy,
    )

    run_sanity_check()

    print("\nAll done.")
