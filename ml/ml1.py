"""ml/ml1.py
ML1 — Algorithm Behavioural State Extractor.
Model: RandomForestClassifier (Unit IV)

Output: [p_normal, p_stressed, p_degraded] via predict_proba()
        ALL THREE VALUES — not a single label. This is the algo_health_vector.

The vector flows INTO:
  - CausalState.algo_health_vector (updates continuously)
  - ML2 feature set (p_stressed is the most informative ML1 feature for breach prediction)
  - DecisionObject (captured at commit time)

Bootstrap mode (< 100 execution telemetry samples):
  latency > 5000ms  → [0.0, 0.0, 1.0]  (degraded)
  latency > 2000ms  → [0.0, 1.0, 0.0]  (stressed)
  otherwise         → [1.0, 0.0, 0.0]  (normal)

Trained mode (>= 100 samples):
  RandomForestClassifier trained on real ExecutionTelemetry data.
  Retrain trigger: every 100 new ExecutionTelemetry samples.

Baseline: always predict 'normal'. ML1 must beat this (macro F1 > 0.33).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from core.schemas import ExecutionTelemetry

logger = logging.getLogger(__name__)

ML1_MODEL_PATH = Path("data/models/ml1.joblib")
RETRAIN_THRESHOLD = 100   # number of telemetry samples before first real train
CLASSES = ["normal", "stressed", "degraded"]


class ML1BehaviourClassifier:
    """Random Forest classifier for algorithm behavioural state.

    Runs inference after every ExecutionTelemetry emission.
    The latest predict_proba() output is captured at ledger commit time.
    """

    def __init__(self) -> None:
        self._model: RandomForestClassifier | None = None
        self._sample_buffer: list[dict] = []
        self._is_trained = False
        self._checkpoint_count = 0
        self._last_sample_count = 0  # S-3: sample count at last retrain, for DB checkpoint
        self._load_model_if_exists()

    @property
    def checkpoint_count(self) -> int:
        """Number of times ML1 has successfully retrained."""
        return self._checkpoint_count

    @property
    def last_sample_count(self) -> int:
        """Sample count used in the most recent retrain (saved before buffer clear)."""
        return self._last_sample_count

    def predict_proba(
        self,
        telemetry: ExecutionTelemetry,
        is_real_order: bool = True,
    ) -> list[float]:
        """Return [p_normal, p_stressed, p_degraded] for the given telemetry.

        Args:
            telemetry: ExecutionTelemetry from the executor.
            is_real_order: Pass False for HOLD/SELL no-op telemetry.
                HOLD telemetry is synthetic (fill_rate=1.0, latency=0.0, slippage=0.0)
                and always classifies as 'normal'. Appending it to _sample_buffer
                prevents ML1 from ever seeing degraded samples, blocking retraining
                and keeping algo_health_vector permanently at [1.0, 0.0, 0.0].

        In bootstrap mode: threshold rules.
        In trained mode: RandomForest predict_proba().

        Always returns a valid probability vector summing to 1.0.
        """
        # Only real BUY orders produce meaningful telemetry for ML1 training.
        if is_real_order:
            self._sample_buffer.append(self._to_features(telemetry))

        if not self._is_trained:
            return self._bootstrap_predict(telemetry)

        features = np.array([list(self._to_features(telemetry).values())])
        proba = self._model.predict_proba(features)[0]  # shape: (n_classes,)
        # Align to [p_normal, p_stressed, p_degraded]
        result = [0.0, 0.0, 0.0]
        for i, cls in enumerate(self._model.classes_):
            result[CLASSES.index(cls)] = float(proba[i])
        return result

    def maybe_retrain(self) -> tuple[bool, float | None]:
        """Retrain if enough new samples have accumulated.

        Returns:
            (retrained: bool, oob_score: float | None)

        S-3 FIX: Returns oob_score so main.py can log a DB checkpoint every time ML1
        retrains. Without this, ML1 training is invisible in model_checkpoints — the
        Week 9 ablation analysis cannot determine when ML1 transitioned from bootstrap
        to trained mode or what OOB accuracy it achieved.
        """
        if len(self._sample_buffer) < RETRAIN_THRESHOLD:
            return False, None

        labels = [self._bootstrap_label(s) for s in self._sample_buffer]
        if len(set(labels)) < 2:
            logger.warning("ML1 retrain skipped | fewer than 2 classes in buffer")
            # Clear buffer to prevent unbounded growth and stale data contamination.
            # Without this, the buffer grows past RETRAIN_THRESHOLD and is re-evaluated
            # every single cycle without ever successfully retraining.
            self._sample_buffer.clear()
            return False, None

        # S-3: save count BEFORE clear so the property is readable in main.py
        self._last_sample_count = len(self._sample_buffer)
        X = np.array([[v for v in s.values()] for s in self._sample_buffer])
        y = np.array(labels)

        self._model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            oob_score=True,
        )
        self._model.fit(X, y)
        self._is_trained = True
        self._sample_buffer.clear()

        oob = getattr(self._model, "oob_score_", None)
        # B7 FIX: dump BEFORE incrementing counter. If joblib.dump() raises (disk full,
        # permission error), the count would be permanently ahead of the saved model,
        # corrupting every subsequent DB checkpoint number.
        joblib.dump(self._model, ML1_MODEL_PATH)
        self._checkpoint_count += 1

        logger.info(
            "ML1 retrained | samples=%d oob_accuracy=%.4f checkpoint=#%d",
            self._last_sample_count, oob or 0.0, self._checkpoint_count,
        )
        return True, oob

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bootstrap_predict(self, telemetry: ExecutionTelemetry) -> list[float]:
        """Threshold-rule predictions for bootstrap phase."""
        if telemetry.order_latency_ms >= 5000 or telemetry.fill_rate < 0.4:
            return [0.0, 0.0, 1.0]  # degraded
        if telemetry.order_latency_ms >= 2000 or telemetry.fill_rate < 0.7:
            return [0.0, 1.0, 0.0]  # stressed
        return [1.0, 0.0, 0.0]     # normal

    def _bootstrap_label(self, features: dict) -> str:
        """Bootstrap label using thresholds on raw feature values."""
        if features["order_latency_ms"] >= 5000 or features["fill_rate"] < 0.4:
            return "degraded"
        if features["order_latency_ms"] >= 2000 or features["fill_rate"] < 0.7:
            return "stressed"
        return "normal"

    def _to_features(self, telemetry: ExecutionTelemetry) -> dict:
        """Convert ExecutionTelemetry to feature dict."""
        return {
            "fill_rate": telemetry.fill_rate,
            "order_latency_ms": telemetry.order_latency_ms,
            "slippage": telemetry.slippage,
            "position_size_deviation": telemetry.position_size_deviation,
        }

    def _load_model_if_exists(self) -> None:
        """Load saved model from disk if available."""
        if ML1_MODEL_PATH.exists():
            try:
                self._model = joblib.load(ML1_MODEL_PATH)
                self._is_trained = True
                logger.info("ML1 model loaded from disk | path=%s", ML1_MODEL_PATH)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ML1 model load failed | starting in bootstrap mode | %s", exc)
