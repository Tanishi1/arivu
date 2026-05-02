"""data/train_regime.py
Member 2 script — Feature engineering + K-Means + Decision Tree.

Reads:  data/btcusdt_1m_raw.csv  (produced by Member 1)
Writes: data/models/regime_dt.joblib
        data/models/kmeans.joblib

Features used (3 — honest bootstrap set):
    volatility    — rolling std of 1-min returns over 20 candles
    trend_strength — normalised abs slope of close prices over 20 candles
    volume        — raw volume from kline

Usage:
    python data/train_regime.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier

# Add project root to path so core.constants resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.constants import (
    K_CANDIDATES,
    KMEANS_MODEL_PATH,
    REGIME_MODEL_PATH,
    SLOPE_NORMALISER,
    TREND_WINDOW,
    VOLATILITY_WINDOW,
)

RAW_CSV = Path(__file__).parent / "btcusdt_1m_raw.csv"


# ── Step 1: Load raw klines ───────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["open_time_ms", "close", "volume"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


# ── Step 2: Compute features ──────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Returns: (close[i] / close[i-1]) - 1
    returns = df["close"].pct_change()

    # volatility: rolling std of returns
    df["volatility"] = returns.rolling(VOLATILITY_WINDOW).std()

    # trend_strength: normalised abs slope via linear regression on close
    def rolling_slope(series: pd.Series, window: int) -> pd.Series:
        x = np.arange(window)
        slopes = series.rolling(window).apply(
            lambda y: np.polyfit(x, y, 1)[0], raw=True
        )
        return slopes

    slopes = rolling_slope(df["close"], TREND_WINDOW)
    df["trend_strength"] = (slopes.abs() / SLOPE_NORMALISER).clip(upper=1.0)

    # Drop NaN rows from rolling windows
    df = df.dropna(subset=["volatility", "trend_strength"]).reset_index(drop=True)
    print(f"Features computed | {len(df):,} usable rows after rolling warmup")
    return df


# ── Step 3: K-Means ───────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray) -> tuple[int, np.ndarray]:
    best_k, best_score, best_labels = K_CANDIDATES[0], -1.0, None

    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"  K={k} | silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    # Save the winning K-Means model
    KMEANS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
    joblib.dump(km_final, KMEANS_MODEL_PATH)
    print(f"K-Means done | best_k={best_k} | saved to {KMEANS_MODEL_PATH}")
    return best_k, best_labels


# ── Step 4: Name clusters ─────────────────────────────────────────────────────

def name_clusters(labels: np.ndarray, df: pd.DataFrame) -> list[str]:
    cluster_ids = sorted(set(labels))
    stats = {}
    for cid in cluster_ids:
        mask = labels == cid
        stats[cid] = {
            "vol": df.loc[mask, "volatility"].mean(),
            "trend": df.loc[mask, "trend_strength"].mean(),
        }

    sorted_by_vol = sorted(cluster_ids, key=lambda c: stats[c]["vol"], reverse=True)
    regime_map = {sorted_by_vol[0]: "volatile"}

    if len(sorted_by_vol) > 2:
        remaining = sorted(sorted_by_vol[1:], key=lambda c: stats[c]["trend"], reverse=True)
        regime_map[remaining[0]] = "trending"
        for c in remaining[1:]:
            regime_map[c] = "calm"
    elif len(sorted_by_vol) == 2:
        regime_map[sorted_by_vol[1]] = "calm"

    named = [regime_map.get(lbl, "calm") for lbl in labels]
    for regime in set(named):
        print(f"  Cluster '{regime}': {named.count(regime):,} rows")
    return named


# ── Step 5: Train Decision Tree ───────────────────────────────────────────────

def train_dt(X: np.ndarray, named_labels: list[str]) -> None:
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X, named_labels)
    joblib.dump(dt, REGIME_MODEL_PATH)
    print(f"Decision Tree trained | classes={list(dt.classes_)} | saved to {REGIME_MODEL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_raw(RAW_CSV)
    df = compute_features(df)

    feature_cols = ["volatility", "trend_strength", "volume"]
    X = df[feature_cols].values

    print("\nRunning K-Means...")
    best_k, labels = run_kmeans(X)

    print("\nNaming clusters...")
    named_labels = name_clusters(labels, df)

    print("\nTraining Decision Tree...")
    train_dt(X, named_labels)

    print("\nDone. Hand off models to Member 3 for verification.")
