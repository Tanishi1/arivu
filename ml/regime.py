"""ml/regime.py
K-Means regime discovery + Decision Tree regime classifier.

K-Means (Unit V):
  - Runs OFFLINE and PERIODICALLY — NOT in the real-time loop.
  - Clusters closed ledger entries by market features.
  - Discovers which market conditions naturally group together.
  - Trigger: total_closed_entries >= 40 AND total_closed_entries % 10 == 0
  - Validated with elbow method and silhouette score.
  - Discovered labels feed the Decision Tree.

Decision Tree (Unit III):
  - Runs LIVE in Stage 1 of the simulator.
  - Trained offline on K-Means cluster labels.
  - Classifies current market state into a regime.
  - Informs heuristic weights for best-first strategy selection.

Bootstrap:
  Before 40 closed entries, hand-coded thresholds classify regimes:
    volatility > 0.04   → 'volatile'
    trend_slope > 0.005 → 'trending'
    otherwise           → 'calm'

See project_v5.docx Section 9 re: what K-Means does NOT do:
  K-Means does NOT set heuristic weights directly.
  K-Means does NOT run live.
  K-Means does NOT feed into ML1 or ML2.
  K-Means has ONE job: discover regime labels from accumulated ledger data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

from core.constants import (
    REGIME_MODEL_PATH,
    KMEANS_MODEL_PATH,
    MIN_ENTRIES_FOR_KMEANS,
    K_CANDIDATES,
    SCALER_MODEL_PATH,
)

# Bootstrap regime thresholds
VOLATILE_THRESHOLD = 0.04
TRENDING_SLOPE_THRESHOLD = 0.005


class RegimeClassifier:
    """K-Means + Decision Tree for market regime classification.

    The Decision Tree instance is what runs live (in Stage 1).
    K-Means runs offline to produce the labels the DT trains on.
    """

    def __init__(self) -> None:
        self._dt: DecisionTreeClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._is_trained = False
        self._load_models_if_exist()

    def classify(
        self,
        volatility: float,
        spread: float,
        trend_strength: float,
        volume: float,
        trend_slope: float = 0.0,
    ) -> str:
        """Classify current market into a regime.

        In bootstrap: threshold rules.
        In trained: Decision Tree predict().

        HIGH-4 FIX: trend_slope added (default 0.0 for backward compatibility with
        verify_models.py and test callers that use keyword args). The bootstrap
        classifier previously classified ANY strong trend (positive or negative) as
        'trending', giving EMAStrategy a +0.2 Stage 1 boost during downtrends.
        EMA only generates BUY on upward crossovers — selecting it in a downtrend
        produces HOLD cycles that fill the training buffer with zero-signal rows.
        """
        if not self._is_trained:
            return self._bootstrap_classify(volatility, trend_strength, trend_slope)

        # Detect if model expects 3 features (historical bootstrap) or 4 (live retrained)
        if hasattr(self._dt, "n_features_in_") and self._dt.n_features_in_ == 3:
            features = np.array([[volatility, trend_strength, volume]])
        else:
            features = np.array([[volatility, spread, trend_strength, volume]])

        if self._scaler is not None:
            features = self._scaler.transform(features)

        return str(self._dt.predict(features)[0])

    def predict_proba(self, volatility: float, spread: float, trend_strength: float, volume: float) -> list[float]:
        """Return regime probabilities. Handles feature scaling and dimensionality (N25 Fix)."""
        if not self._is_trained:
            # Bootstrap fallback: 100% confidence in the bootstrap classification
            return [1.0]

        if hasattr(self._dt, "n_features_in_") and self._dt.n_features_in_ == 3:
            features = np.array([[volatility, trend_strength, volume]])
        else:
            features = np.array([[volatility, spread, trend_strength, volume]])

        if self._scaler is not None:
            features = self._scaler.transform(features)

        return self._dt.predict_proba(features)[0].tolist()

    def run_kmeans_and_retrain(self, closed_entries: list[dict]) -> bool:
        """Run K-Means offline to discover regimes, then retrain Decision Tree.

        Args:
            closed_entries: list of dicts with keys:
                volatility, spread, trend_strength, volume, strategy_name

        Returns:
            True if the Decision Tree was successfully retrained.
        """
        if len(closed_entries) < MIN_ENTRIES_FOR_KMEANS:
            logger.warning(
                "K-Means skipped | only %d entries (need %d)",
                len(closed_entries), MIN_ENTRIES_FOR_KMEANS,
            )
            return False

        features = ["volatility", "spread", "trend_strength", "volume"]
        X_raw = np.array([[e[f] for f in features] for e in closed_entries])

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        best_k, best_labels = self._select_k(X)
        logger.info("K-Means complete | best_k=%d", best_k)

        # Label assignment: name clusters by dominant strategy performance
        named_labels = self._name_clusters(best_labels, closed_entries)

        # Retrain Decision Tree on discovered labels
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X, named_labels)
        self._dt = dt
        self._scaler = scaler
        self._is_trained = True

        joblib.dump(dt, REGIME_MODEL_PATH)
        joblib.dump(scaler, SCALER_MODEL_PATH)
        logger.info("Decision Tree and Scaler retrained | classes=%s", list(dt.classes_))
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_k(self, X: np.ndarray) -> tuple[int, np.ndarray]:
        """Choose K via elbow method / silhouette score. Returns (K, labels)."""
        best_k = K_CANDIDATES[0]
        best_score = -1.0
        best_labels = None

        for k in K_CANDIDATES:
            if k >= len(X):
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if k > 1 and len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                logger.info("K=%d silhouette=%.4f", k, score)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    best_km = km  # DF1 FIX: track the actual fitted object

        if best_labels is None:
            best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            best_labels = best_km.fit_predict(X)

        # DF1 FIX: save best_km directly (the object used to generate best_labels).
        # Previously: KMeans(n_clusters=best_k).fit(X) — a NEW fit, different object.
        # With random_state=42 both fits are deterministic, but saving the wrong object
        # is architecturally incorrect and fragile if random_state is ever removed.
        joblib.dump(best_km, KMEANS_MODEL_PATH)
        return best_k, best_labels

    def _name_clusters(
        self, labels: np.ndarray, entries: list[dict]
    ) -> list[str]:
        """Assign human-readable regime names to cluster IDs.

        For each cluster, compute mean volatility and trend_strength.
        Highest volatility cluster → 'volatile'
        Highest trend_strength (not volatile) → 'trending'
        Remaining → 'calm'

        Manual review recommended: a team member should inspect centroids
        and confirm the assignment makes intuitive sense.
        """
        cluster_ids = list(set(labels))
        stats: dict[int, dict] = {}

        for cid in cluster_ids:
            mask = labels == cid
            subset = [e for e, m in zip(entries, mask) if m]
            stats[cid] = {
                "vol": np.mean([e["volatility"] for e in subset]),
                "trend": np.mean([e["trend_strength"] for e in subset]),
            }

        # Sort by volatility descending
        sorted_by_vol = sorted(cluster_ids, key=lambda c: stats[c]["vol"], reverse=True)

        regime_map = {}
        regime_map[sorted_by_vol[0]] = "volatile"
        if len(sorted_by_vol) > 2:
            # Secondary: sort remaining by trend
            remaining = sorted_by_vol[1:]
            sorted_by_trend = sorted(remaining, key=lambda c: stats[c]["trend"], reverse=True)
            regime_map[sorted_by_trend[0]] = "trending"
            for c in sorted_by_trend[1:]:
                regime_map[c] = "calm"
        elif len(sorted_by_vol) == 2:
            # With K=2 there is no room for all three regime labels.
            # Promote the second cluster to "trending" if its mean trend_strength is
            # meaningfully positive; otherwise label it "calm".
            # Without this, EMAStrategy can NEVER receive a regime boost when K=2,
            # breaking the regime-aware architecture during the early accumulation phase
            # (Weeks 5–7 when only 40–60 entries exist and K=2 is the best-fit K).
            second_cid = sorted_by_vol[1]
            if stats[second_cid]["trend"] > 0.4:
                regime_map[second_cid] = "trending"
            else:
                regime_map[second_cid] = "calm"

        return [regime_map.get(label, "calm") for label in labels]

    def _bootstrap_classify(
        self, volatility: float, trend_strength: float, trend_slope: float = 0.0
    ) -> str:
        """Bootstrap regime classification via hand-coded thresholds.

        DF2 FIX: trending threshold changed from 0.5 to 0.35.
        EMA heuristic dominates when trend_strength >= ~0.35 (where trend_strength*0.5
        component outweighs Bollinger's (1-trend)*0.5 component). The bootstrap
        classifier was withholding the regime boost until 0.5, causing EMA to miss
        its +0.2 boost in moderately trending markets during Weeks 5-7 before K-Means
        fires at 40+ closed entries.

        HIGH-4 FIX: 'trending' only fires when trend_slope > 0 (uptrend).
        trend_strength = abs(trend_slope)/SLOPE_NORMALISER — it is direction-agnostic.
        Without the slope sign check, a strong downtrend (trend_strength=0.5, slope=-0.01)
        was classified 'trending' and EMAStrategy got a +0.2 Stage 1 boost even though
        EMA only generates BUY on upward crossovers. In a downtrend it always returns
        HOLD, producing zero-outcome cycles that contaminate the training buffer.
        trend_slope defaults to 0.0 so existing callers (verify_models.py, tests) that
        don't pass trend_slope continue to work — 0.0 slope is not > 0, so they fall
        through to 'calm', which is correct for a flat state.
        """
        if volatility > VOLATILE_THRESHOLD:
            return "volatile"
        if trend_strength > 0.35 and trend_slope > 0:   # HIGH-4: must be uptrend
            return "trending"
        return "calm"

    def _load_models_if_exist(self) -> None:
        if REGIME_MODEL_PATH.exists() and SCALER_MODEL_PATH.exists():
            try:
                self._dt = joblib.load(REGIME_MODEL_PATH)
                self._scaler = joblib.load(SCALER_MODEL_PATH)
                self._is_trained = True
                logger.info("Regime DT and Scaler loaded from disk")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Models load failed | bootstrap mode | %s", exc)
        elif REGIME_MODEL_PATH.exists():
            try:
                self._dt = joblib.load(REGIME_MODEL_PATH)
                self._scaler = None
                self._is_trained = True
                logger.info("Regime DT loaded from disk (no scaler)")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Regime DT load failed | bootstrap mode | %s", exc)
