"""ml/ml2.py
ML2 — Assumption Breach Predictor.
Model: LogisticRegression (Unit IV)

Annotates every assumption in a SimulatorResult with a breach probability
BEFORE the DecisionObject is committed to the ledger.

Feature source: ml/features.py — same function at both training and inference.
This prevents training-serving skew (see DECISIONS.md ADR-005).

Bootstrap mode (< 30 closed ledger entries):
  breach_risk = assumption.proximity
  (heuristic baseline — ML2 must beat this to justify its existence)

Trained mode (>= 30 entries):
  LogisticRegression trained on training_buffer.csv data.
  class_weight='balanced' to handle class imbalance (most assumptions hold).
  Retrain trigger: every 20 new closed entries.

Research Metric 2: ML2 Brier score declining across model_checkpoints.
The bootstrap Brier score (proximity heuristic) is the baseline to beat.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from ml.features import compute_ml2_features
from core.schemas import Assumption, CausalState

logger = logging.getLogger(__name__)

ML2_MODEL_PATH = Path("data/models/ml2.joblib")
TRAINING_BUFFER_PATH = Path("data/training_buffer.csv")
RETRAIN_THRESHOLD = 20   # new closed entries before retraining
MIN_SAMPLES = 30         # minimum total samples to attempt training


class ML2BreachPredictor:
    """Logistic Regression breach probability predictor.

    Called by the simulator AFTER Stage 2, BEFORE ledger commit.
    Annotates each Assumption's breach_risk field.
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._is_trained = False
        self._checkpoint_count = 0
        self._new_entries_since_retrain = 0
        self._load_model_if_exists()

    def annotate(
        self,
        assumptions: list[Assumption],
        state: CausalState,
        time_horizon: float,
        ml1_vector: list[float],
    ) -> list[Assumption]:
        """Annotate each assumption with a breach_risk probability.

        The pipeline dependency matters: ML2 knows whether the algorithm is
        stressed (from ml1_vector) because p_stressed is a direct input feature.
        A stressed algorithm in a volatile market has a fundamentally different
        breach profile than a normally executing one in identical conditions.

        Args:
            assumptions: list of Assumption objects to annotate
            state: current CausalState snapshot (at decision time)
            time_horizon: strategy window duration in minutes
            ml1_vector: [p_normal, p_stressed, p_degraded] from ML1

        Returns:
            Same assumptions list with breach_risk fields populated.
        """
        for assumption in assumptions:
            if not self._is_trained:
                risk = self._proximity_baseline(assumption)
            else:
                risk = self._predict_trained(assumption, state, time_horizon, ml1_vector)

            assumption.breach_risk = round(risk, 4)

        logger.info(
            "ML2 annotated | risks=%s",
            [f"{a.name}={a.breach_risk}" for a in assumptions],
        )
        return assumptions

    def maybe_retrain(self, ledger_writer) -> tuple[bool, float | None]:
        """Retrain if enough new closed entries have accumulated.

        Returns (retrained: bool, brier_score: float | None).
        Also logs a checkpoint to the ledger.
        """
        # N23 FIX: Initialize checkpoint count from DB if not yet loaded (across restarts)
        if self._checkpoint_count == 0:
            self._checkpoint_count = ledger_writer.get_max_checkpoint()

        self._new_entries_since_retrain += 1
        if self._new_entries_since_retrain < RETRAIN_THRESHOLD:
            return False, None

        data = self._load_buffer()
        if len(data) < MIN_SAMPLES:
            logger.warning(
                "ML2 retrain skipped | only %d samples (need %d)", len(data), MIN_SAMPLES
            )
            return False, None

        X, y = self._prepare_xy(data)
        if len(set(y)) < 2:
            logger.warning("ML2 retrain skipped | no positive breach labels yet")
            return False, None

        # Time-ordered split: first 80% train, last 20% held-out
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # N9 FIX: Ensure training set has at least 2 classes after the time split
        if len(set(y_train)) < 2:
            logger.warning("ML2 retrain skipped | less than 2 classes in y_train due to time split")
            return False, None

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train, y_train)

        proba_val = model.predict_proba(X_val)[:, 1]
        brier = float(brier_score_loss(y_val, proba_val))

        # Compare against bootstrap (proximity heuristic) Brier score
        proximity_preds = [row["proximity"] for row in data[split:]]
        baseline_brier = float(brier_score_loss(y_val, proximity_preds))

        logger.info(
            "ML2 retrained | samples=%d brier=%.4f baseline=%.4f beats_baseline=%s",
            len(X), brier, baseline_brier, brier < baseline_brier,
        )

        if brier < baseline_brier:
            self._model = model
            self._is_trained = True
            self._checkpoint_count += 1
            joblib.dump(model, ML2_MODEL_PATH)
            
            # N8 FIX: Write the checkpoint to the database so Research Metric 2 exists!
            ledger_writer.save_checkpoint(
                checkpoint_number=self._checkpoint_count,
                phase="bootstrap" if self._checkpoint_count == 1 else "trained",
                ml2_brier_score=brier,
                training_sample_count=len(X),
            )
        else:
            logger.warning("ML2 does not beat baseline — keeping previous model")

        self._new_entries_since_retrain = 0
        return True, brier

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _proximity_baseline(self, assumption: Assumption) -> float:
        """Bootstrap heuristic: breach_risk = proximity (current/threshold)."""
        return min(1.0, assumption.proximity)

    def _predict_trained(
        self,
        assumption: Assumption,
        state: CausalState,
        time_horizon: float,
        ml1_vector: list[float],
    ) -> float:
        features = compute_ml2_features(state, assumption, time_horizon, ml1_vector)
        X = np.array([[v for v in features.values()]])
        proba = self._model.predict_proba(X)[0]
        # index 1 = P(breach=1)
        pos_idx = list(self._model.classes_).index(1) if 1 in self._model.classes_ else 1
        return float(proba[pos_idx])

    def _load_buffer(self) -> list[dict]:
        """Read training_buffer.csv into a list of row dicts."""
        if not TRAINING_BUFFER_PATH.exists():
            return []
        with open(TRAINING_BUFFER_PATH, newline="") as f:
            return list(csv.DictReader(f))

    def _prepare_xy(self, data: list[dict]):
        feature_keys = [
            "volatility", "spread", "trend_strength", "volume",
            "proximity", "time_horizon",
            "p_normal", "p_stressed", "p_degraded",
        ]
        X = np.array([[float(row[k]) for k in feature_keys] for row in data])
        y = np.array([int(row["breached"]) for row in data])
        return X, y

    def _load_model_if_exists(self) -> None:
        if ML2_MODEL_PATH.exists():
            try:
                self._model = joblib.load(ML2_MODEL_PATH)
                self._is_trained = True
                logger.info("ML2 model loaded from disk | path=%s", ML2_MODEL_PATH)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ML2 model load failed | bootstrap mode | %s", exc)
