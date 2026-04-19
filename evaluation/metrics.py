"""evaluation/metrics.py
Four research analyses that produce plots from real ledger data.

Run with: python evaluation/metrics.py
Output: evaluation/plots/*.png (created if not present)
Requires: at least 10 closed ledger entries for meaningful output.

Analysis 1 — Full lifecycle plot
  Plot outcome_delta across ALL cycles in chronological order.
  Draw a vertical line at the cycle where phase transitions bootstrap → trained.
  This is the scene-setter, not the primary finding.

Analysis 2 — Trained-phase delta trend (PRIMARY FINDING)
  Filter to phase='trained' only.
  Plot outcome_delta across those cycles.
  Fit a regression line. Negative slope = the system is improving.

Analysis 3 — Brier score progression
  Query model_checkpoints ordered by checkpoint_number.
  Plot Brier score over checkpoints.
  Show bootstrap and trained phase as two colour-coded segments.
  Declining trained-phase Brier score = ML2 is learning.

Analysis 4 — ML1 ablation
  Take trained-phase training data.
  Train ML2 with full feature set (including ML1 vector).
  Train ML2 without ML1 features (p_normal, p_stressed, p_degraded).
  Compare Brier scores on same held-out 20%.
  If with-ML1 scores lower → ML1 contribution independently proven.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss

from ledger.models import SessionLocal, OutcomeRecordRow, DecisionObjectRow, ModelCheckpointRow

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("evaluation/plots")


def _ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Analysis 1 — Full lifecycle plot
# ---------------------------------------------------------------------------

def plot_full_lifecycle() -> None:
    """Plot outcome_delta for all cycles. Vertical line at phase transition.

    Positive slope early (bootstrap) should flatten/invert in trained phase.
    This is the scene-setter for the paper's results section.
    """
    with SessionLocal() as session:
        rows = (
            session.query(OutcomeRecordRow, DecisionObjectRow)
            .join(DecisionObjectRow, OutcomeRecordRow.decision_object_id == DecisionObjectRow.id)
            .order_by(OutcomeRecordRow.timestamp_closed)
            .all()
        )

    if not rows:
        logger.warning("No closed entries for Analysis 1")
        return

    deltas = [abs(r.OutcomeRecordRow.outcome_delta) for r in rows]
    phases = [r.DecisionObjectRow.phase for r in rows]
    cycles = list(range(1, len(deltas) + 1))

    # Find transition point
    transition_cycle = next(
        (i + 1 for i, p in enumerate(phases) if p == "trained"), None
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cycles, deltas, "o-", color="#4A90D9", linewidth=1.5, markersize=4, label="|outcome delta|")

    if transition_cycle:
        ax.axvline(x=transition_cycle, color="#E74C3C", linestyle="--", linewidth=1.5,
                   label=f"Phase transition (cycle {transition_cycle})")

    ax.set_xlabel("Cycle number")
    ax.set_ylabel("|Predicted P&L − Actual P&L|")
    ax.set_title("Analysis 1 — Full Lifecycle: Outcome Delta Over All Cycles")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_plots_dir()
    fig.savefig(PLOTS_DIR / "outcome_delta.png", dpi=150)
    plt.close(fig)
    logger.info("Analysis 1 saved | outcome_delta.png")


# ---------------------------------------------------------------------------
# Analysis 2 — Trained-phase delta trend (primary finding)
# ---------------------------------------------------------------------------

def plot_trained_phase_delta() -> None:
    """Filter to trained phase. Fit regression. Negative slope = improvement.

    This is the primary research finding. A negative slope on the regression
    line means the system's P&L predictions are getting more accurate.
    """
    with SessionLocal() as session:
        rows = (
            session.query(OutcomeRecordRow, DecisionObjectRow)
            .join(DecisionObjectRow, OutcomeRecordRow.decision_object_id == DecisionObjectRow.id)
            .filter(DecisionObjectRow.phase == "trained")
            .order_by(OutcomeRecordRow.timestamp_closed)
            .all()
        )

    if not rows:
        logger.warning("No trained-phase entries for Analysis 2")
        return

    deltas = [abs(r.OutcomeRecordRow.outcome_delta) for r in rows]
    cycles = np.array(range(1, len(deltas) + 1)).reshape(-1, 1)

    reg = LinearRegression().fit(cycles, deltas)
    slope = reg.coef_[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(cycles, deltas, color="#4A90D9", alpha=0.7, s=40, label="Trained-phase cycles")
    ax.plot(cycles, reg.predict(cycles), color="#E74C3C", linewidth=2,
            label=f"Regression (slope={slope:.5f})")

    ax.set_xlabel("Cycle number (trained phase)")
    ax.set_ylabel("|Outcome delta|")
    ax.set_title("Analysis 2 — Trained Phase: Outcome Delta Trend (negative slope = improving)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_plots_dir()
    fig.savefig(PLOTS_DIR / "trained_phase_delta.png", dpi=150)
    plt.close(fig)
    logger.info("Analysis 2 | slope=%.6f | saved trained_phase_delta.png", slope)


# ---------------------------------------------------------------------------
# Analysis 3 — Brier score progression
# ---------------------------------------------------------------------------

def plot_brier_score_progression() -> None:
    """Plot ML2 Brier score over model checkpoints.

    Bootstrap and trained phase shown as separate colour-coded segments.
    Declining trained-phase curve = ML2 is learning as ledger grows.
    """
    with SessionLocal() as session:
        checkpoints = (
            session.query(ModelCheckpointRow)
            .order_by(ModelCheckpointRow.checkpoint_number)
            .all()
        )

    if not checkpoints:
        logger.warning("No model checkpoints for Analysis 3")
        return

    bootstrap = [(c.checkpoint_number, c.ml2_brier_score) for c in checkpoints if c.phase == "bootstrap"]
    trained = [(c.checkpoint_number, c.ml2_brier_score) for c in checkpoints if c.phase == "trained"]

    fig, ax = plt.subplots(figsize=(10, 5))
    if bootstrap:
        bx, by = zip(*bootstrap)
        ax.plot(bx, by, "s--", color="#F39C12", linewidth=1.5, label="Bootstrap phase")
    if trained:
        tx, ty = zip(*trained)
        ax.plot(tx, ty, "o-", color="#27AE60", linewidth=1.5, label="Trained phase")

    ax.set_xlabel("Checkpoint number")
    ax.set_ylabel("ML2 Brier score (lower = better)")
    ax.set_title("Analysis 3 — ML2 Brier Score Over Retraining Checkpoints")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_plots_dir()
    fig.savefig(PLOTS_DIR / "brier_score.png", dpi=150)
    plt.close(fig)
    logger.info("Analysis 3 saved | brier_score.png")


# ---------------------------------------------------------------------------
# Analysis 4 — ML1 ablation
# ---------------------------------------------------------------------------

def run_ml1_ablation() -> tuple[float, float] | None:
    """Train two ML2 instances with and without ML1 features. Compare Brier scores.

    If with-ML1 Brier < without-ML1 Brier → ML1 contribution independently proven.

    Returns:
        (brier_with_ml1, brier_without_ml1) or None if insufficient data.
    """
    buffer_path = Path("data/training_buffer.csv")
    if not buffer_path.exists():
        logger.warning("Training buffer not found for ablation")
        return None

    with open(buffer_path, newline="") as f:
        data = list(csv.DictReader(f))

    # Filter to trained phase only
    trained_data = [r for r in data if r.get("phase") == "trained"]
    if len(trained_data) < 20:
        logger.warning("Insufficient trained-phase data for ablation | rows=%d", len(trained_data))
        return None

    FULL_FEATURES = [
        "volatility", "spread", "trend_strength", "volume",
        "proximity", "time_horizon",
        "p_normal", "p_stressed", "p_degraded",
    ]
    NO_ML1_FEATURES = [
        "volatility", "spread", "trend_strength", "volume",
        "proximity", "time_horizon",
    ]

    def _build_xy(rows: list[dict], feature_keys: list[str]):
        X = np.array([[float(r[k]) for k in feature_keys] for r in rows])
        y = np.array([int(r["breached"]) for r in rows])
        return X, y

    split = int(len(trained_data) * 0.8)
    train, val = trained_data[:split], trained_data[split:]

    X_train_full, y_train = _build_xy(train, FULL_FEATURES)
    X_val_full, y_val = _build_xy(val, FULL_FEATURES)
    X_train_no_ml1, _ = _build_xy(train, NO_ML1_FEATURES)
    X_val_no_ml1, _ = _build_xy(val, NO_ML1_FEATURES)

    def _fit_and_brier(X_tr, y_tr, X_v, y_v) -> float:
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_v)[:, 1]
        return float(brier_score_loss(y_v, proba))

    brier_with = _fit_and_brier(X_train_full, y_train, X_val_full, y_val)
    brier_without = _fit_and_brier(X_train_no_ml1, y_train, X_val_no_ml1, y_val)

    logger.info(
        "Ablation | brier_with_ml1=%.4f brier_without_ml1=%.4f ml1_helps=%s",
        brier_with, brier_without, brier_with < brier_without,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["With ML1 features", "Without ML1 features"],
        [brier_with, brier_without],
        color=["#27AE60", "#E74C3C"],
        alpha=0.8,
    )
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylabel("Brier score (lower = better breach prediction)")
    ax.set_title("Analysis 4 — ML1 Ablation: Does ML1 Improve ML2?")
    ax.set_ylim(0, max(brier_with, brier_without) * 1.3)
    fig.tight_layout()
    _ensure_plots_dir()
    fig.savefig(PLOTS_DIR / "ml1_ablation.png", dpi=150)
    plt.close(fig)
    return brier_with, brier_without


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("Running research analyses...")
    plot_full_lifecycle()
    plot_trained_phase_delta()
    plot_brier_score_progression()
    ablation = run_ml1_ablation()
    if ablation:
        w, wo = ablation
        print(f"\nAblation result: with_ML1={w:.4f}  without_ML1={wo:.4f}  ML1_helps={w < wo}")

    print(f"\nPlots saved to: {PLOTS_DIR.resolve()}")
