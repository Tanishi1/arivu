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
        regime: str = "unknown",
    ) -> list[Assumption]:
        """Annotate each assumption with a breach_risk probability.

        The pipeline dependency matters: ML2 knows whether the algorithm is
        stressed (from ml1_vector) because p_stressed is a direct input feature.
        A stressed algorithm in a volatile market has a fundamentally different
        breach profile than a normally executing one in identical conditions.

        CRITICAL: Assumption is frozen (Pydantic frozen=True). Direct attribute
        assignment raises ValidationError. Use model_copy() to produce a new
        Assumption with the updated breach_risk field.

        Args:
            assumptions: list of Assumption objects to annotate
            state: current CausalState snapshot (at decision time)
            time_horizon: strategy window duration in minutes
            ml1_vector: [p_normal, p_stressed, p_degraded] from ML1
            regime: current market regime string (calm/trending/volatile/unknown)

        Returns:
            New list of Assumption objects with breach_risk fields populated.
        """
        annotated: list[Assumption] = []
        for assumption in assumptions:
            if not self._is_trained:
                risk = self._proximity_baseline(assumption)
            else:
                risk = self._predict_trained(assumption, state, time_horizon, ml1_vector, regime)

            # Assumption is frozen — must use model_copy, NOT direct assignment.
            # Direct assignment raises pydantic.ValidationError silently swallowed
            # by the decision cycle, leaving all breach_risk values at 0.5 forever.
            annotated.append(assumption.model_copy(update={"breach_risk": round(risk, 4)}))

        logger.debug(
            "ML2 annotated | risks=%s",
            [f"{a.name}={a.breach_risk}" for a in annotated],
        )
        return annotated


    def maybe_retrain(self, ledger_writer, current_phase: str = "bootstrap") -> tuple[bool, float | None]:
        """Retrain if enough new closed entries have accumulated.

        Returns (retrained: bool, brier_score: float | None).
        Also logs a checkpoint to the ledger.

        B16 FIX: current_phase is now a parameter (not hardcoded as 'trained').
        The 10-week plan requires checkpoint #1 (the bootstrap→trained transition)
        to carry phase='bootstrap' because it was trained on bootstrap data.
        Subsequent retraining checkpoints carry phase='trained'. Week 9 Analysis 3
        plots these as two segments — without this fix there is NO bootstrap segment.
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
            # Reset counter so we don't re-check the buffer on every single close cycle.
            # Without this reset, once _new_entries_since_retrain > RETRAIN_THRESHOLD,
            # we do a full CSV read every cycle indefinitely while the buffer is too small.
            self._new_entries_since_retrain = 0
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
            # B16 FIX: Use the current_phase parameter — NOT hardcoded 'trained'.
            # Checkpoint #1 is the bootstrap→trained boundary, trained on bootstrap data.
            # The 10-week plan requires phase='bootstrap' here so Week 9 Analysis 3
            # can plot the two segments separately. Later retrains in trained phase
            # will pass current_phase='trained'.
            ledger_writer.save_checkpoint(
                checkpoint_number=self._checkpoint_count,
                phase=current_phase,
                ml2_brier_score=brier,
                training_sample_count=len(X),
            )
            logger.info(
                "ML2 beat baseline — transitioning to trained mode | "
                "baseline=%.4f trained=%.4f checkpoint=#%d",
                baseline_brier, brier, self._checkpoint_count,
            )
            self._new_entries_since_retrain = 0
            return True, brier   # C2 FIX: True ONLY when model actually improved

        else:
            logger.warning(
                "ML2 did not beat baseline — keeping previous model | "
                "baseline=%.4f trained=%.4f",
                baseline_brier, brier,
            )

        self._new_entries_since_retrain = 0
        return False, brier   # C2 FIX: False when model failed to improve

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
        regime: str = "unknown",
    ) -> float:
        features = compute_ml2_features(state, assumption, time_horizon, ml1_vector, regime)
        X = np.array([[v for v in features.values()]])
        try:
            proba = self._model.predict_proba(X)[0]
        except ValueError as exc:
            # Model was trained on a different feature schema (e.g. pre-regime-column).
            # Invalidate it immediately so the next retrain cycle replaces it.
            logger.warning(
                "ML2: feature mismatch — invalidating stale model and falling back to "
                "proximity baseline | expected=%s got=%d | %s",
                getattr(self._model, "n_features_in_", "?"), X.shape[1], exc,
            )
            self._model = None
            self._is_trained = False
            return self._proximity_baseline(assumption)
        # index 1 = P(breach=1)
        pos_idx = list(self._model.classes_).index(1) if 1 in self._model.classes_ else 1
        return float(proba[pos_idx])

    def _load_buffer(self) -> list[dict]:
        """Read training_buffer.csv into a list of row dicts.

        B8 FIX: Validate that the CSV header contains all expected columns before
        returning data. A process crash mid-write of the header row leaves a partial
        header. csv.DictReader uses it as column names, causing every _prepare_xy
        call to raise KeyError, permanently blocking ML2 from retraining.
        """
        _REQUIRED_COLS = {
            "volatility", "spread", "trend_strength", "volume",
            "proximity", "time_horizon", "p_normal", "p_stressed", "p_degraded",
            "breached",
        }
        if not TRAINING_BUFFER_PATH.exists():
            return []
        with open(TRAINING_BUFFER_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                logger.error("Training buffer has no header — file may be corrupt. Skipping.")
                return []
            missing = _REQUIRED_COLS - set(reader.fieldnames)
            if missing:
                logger.error(
                    "Training buffer header is corrupt — missing columns: %s. "
                    "Delete data/training_buffer.csv and restart to rebuild it.",
                    sorted(missing),
                )
                return []
            rows = list(reader)

        # Exclude escape valve trades from ML2 training.
        # Escape valve fires when no hypothesis cleared Layer 1 threshold —
        # training on these rows teaches ML2 that specific edges always breach,
        # when really it just means the threshold was too high at that moment.
        # Rows from before this fix lack the column — treat missing as 0 (not EV).
        clean = [r for r in rows if r.get("is_escape_valve", "0") != "1"]
        if len(clean) < len(rows):
            logger.debug(
                "_load_buffer: excluded %d escape-valve rows from ML2 training "
                "(%d clean rows remain)",
                len(rows) - len(clean), len(clean),
            )
        return clean


    def _prepare_xy(self, data: list[dict]):
        feature_keys = [
            "volatility", "spread", "trend_strength", "volume",
            "proximity", "time_horizon",
            "p_normal", "p_stressed", "p_degraded",
        ]
        _REGIME_ENCODING = {"calm": 0, "trending": 1, "volatile": 2}
        # S3-2 FIX: Guard against malformed CSV rows (partial writes from crashes,
        # disk-full mid-append, Ctrl+C during CSV write). A single bad row previously
        # raised KeyError that propagated through maybe_retrain() — marking the active
        # DO as INTERRUPTED instead of CLOSED and permanently blocking ML2 retraining.
        rows_X, rows_y = [], []
        for row in data:
            try:
                x = [float(row[k]) for k in feature_keys]
                # Regime: optional column — old rows without it default to calm (0).
                regime_str = row.get("regime", "calm")
                x.append(float(_REGIME_ENCODING.get(regime_str, 0)))
                rows_X.append(x)
                rows_y.append(int(row["breached"]))
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed training buffer row | %s | row=%s", exc, row)
        n_features = len(feature_keys) + 1  # +1 for regime_encoded
        X = np.array(rows_X) if rows_X else np.empty((0, n_features))
        y = np.array(rows_y)
        return X, y

    def _load_model_if_exists(self) -> None:
        if ML2_MODEL_PATH.exists():
            try:
                model = joblib.load(ML2_MODEL_PATH)
                # Feature count guard: reject models trained on a different schema.
                # n_features_in_ is set by sklearn after fit() — always present on
                # a trained LogisticRegression. Current schema = 10 features.
                expected_n = len([
                    "volatility", "spread", "trend_strength", "volume",
                    "proximity", "time_horizon",
                    "p_normal", "p_stressed", "p_degraded",
                    "regime_encoded",  # added when regime column was introduced
                ])
                model_n = getattr(model, "n_features_in_", None)
                if model_n is not None and model_n != expected_n:
                    logger.warning(
                        "ML2: stale model on disk has %d features, current schema needs %d — "
                        "discarding and entering bootstrap mode. Will retrain on next cycle.",
                        model_n, expected_n,
                    )
                    ML2_MODEL_PATH.unlink(missing_ok=True)
                    return  # leave _is_trained=False

                self._model = model
                self._is_trained = True
                # Estimate entries accumulated since last retrain so process
                # restarts do not reset the counter to zero. CSV has 3 rows per
                # cycle (one per assumption), so divide by 3 for cycle count.
                data = self._load_buffer()
                cycle_count = max(len(data) // 3, 0)
                self._new_entries_since_retrain = cycle_count % RETRAIN_THRESHOLD
                logger.info(
                    "ML2 model loaded from disk | path=%s | n_features=%d | est_entries_since_retrain=%d",
                    ML2_MODEL_PATH, model_n or expected_n, self._new_entries_since_retrain,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ML2 model load failed | bootstrap mode | %s", exc)
