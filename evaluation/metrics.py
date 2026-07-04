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
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss

from ledger.models import SessionLocal, OutcomeRecordRow, DecisionObjectRow, ModelCheckpointRow

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("evaluation/plots")
DB_PATH = os.getenv("SQLITE_PATH", "data/arivu.db")


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
# Metric 4 — Regime-Conditional Win Rate Divergence
# ---------------------------------------------------------------------------

def plot_regime_win_rates() -> None:
    """Win rate per arm broken down by market regime (Metric 4).

    Regime inferred from market_state_snapshot:
      regime_volatile > 0.5 → 'volatile', regime_trending > 0.5 → 'trending', else 'calm'.
    Hypothesis: CausalAgent outperforms in trending (stable chains), RL-Standard is flat.
    """
    with SessionLocal() as session:
        rows = (
            session.query(DecisionObjectRow, OutcomeRecordRow)
            .join(OutcomeRecordRow, DecisionObjectRow.id == OutcomeRecordRow.decision_object_id)
            .filter(DecisionObjectRow.status == "CLOSED")
            .all()
        )
    if not rows:
        logger.warning("Metric 4: no closed entries")
        return

    ARM_LABELS = {
        "CausalAgent": "Causal Agent",
        "ppo_standard": "PPO Standard",
        "ppo_causal_feature": "PPO Causal",
        "EMAStrategy": "EMA Legacy",
    }
    REGIMES = ["calm", "trending", "volatile"]
    stats: dict = {}

    for do, out in rows:
        arm = do.strategy_name
        if arm not in ARM_LABELS:
            continue
        try:
            mss = json.loads(do.market_state_snapshot)
        except Exception:
            continue
        v = mss.get("regime_volatile", 0.0)
        t = mss.get("regime_trending", 0.0)
        regime = "volatile" if v > 0.5 else ("trending" if t > 0.5 else "calm")
        key = (arm, regime)
        if key not in stats:
            stats[key] = [0, 0]
        stats[key][1] += 1
        if out.actual_pnl > 0:
            stats[key][0] += 1

    arms = [k for k in ARM_LABELS if any((k, r) in stats for r in REGIMES)]
    if not arms:
        logger.warning("Metric 4: no regime data in snapshots")
        return

    x = np.arange(len(arms))
    width = 0.25
    colors = {"calm": "#4A90D9", "trending": "#27AE60", "volatile": "#E74C3C"}

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, regime in enumerate(REGIMES):
        win_rates = []
        for arm in arms:
            w, n = stats.get((arm, regime), [0, 0])
            win_rates.append((w / n * 100) if n > 0 else 0.0)
        bars = ax.bar(x + i * width, win_rates, width, label=regime.capitalize(),
                      color=colors[regime], alpha=0.85)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms])
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Metric 4 — Regime-Conditional Win Rate per Agent\n"
                 "(CausalAgent should diverge from RL across regimes)")
    ax.legend(title="Regime")
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _ensure_plots_dir()
    fig.savefig(PLOTS_DIR / "regime_win_rates.png", dpi=150)
    plt.close(fig)
    logger.info("Metric 4 saved | regime_win_rates.png")


# ---------------------------------------------------------------------------
# Metric 5 — Inter-Agent Signal Correlation
# ---------------------------------------------------------------------------

def plot_signal_correlation() -> None:
    """Pearson correlation of trade signals between arms (Metric 5).

    Buckets decisions into 10-minute windows. Signal = 1 if arm placed a real
    trade (projected_pnl != 0), else 0. Low correlation = genuine diversity.
    """
    with SessionLocal() as session:
        rows = (
            session.query(DecisionObjectRow)
            .filter(DecisionObjectRow.status.in_(["CLOSED", "ACTIVE", "INTERRUPTED"]))
            .order_by(DecisionObjectRow.timestamp_committed)
            .all()
        )
    if not rows:
        logger.warning("Metric 5: no entries")
        return

    ARM_LABELS = {
        "CausalAgent": "Causal",
        "ppo_standard": "PPO-Std",
        "ppo_causal_feature": "PPO-Causal",
        "EMAStrategy": "EMA",
    }
    arms = list(ARM_LABELS.keys())
    BUCKET_MIN = 10
    pivot: dict = {}

    for do in rows:
        if do.strategy_name not in arms:
            continue
        try:
            ts = datetime.fromisoformat(do.timestamp_committed.replace("Z", "+00:00"))
        except Exception:
            continue
        bucket = ts.replace(minute=(ts.minute // BUCKET_MIN) * BUCKET_MIN,
                             second=0, microsecond=0)
        if bucket not in pivot:
            pivot[bucket] = {a: 0 for a in arms}
        fired = 1 if do.projected_pnl != 0 else 0
        pivot[bucket][do.strategy_name] = max(pivot[bucket][do.strategy_name], fired)

    active_arms = [a for a in arms if any(v.get(a, 0) for v in pivot.values())]
    if len(active_arms) < 2:
        logger.warning("Metric 5: need ≥2 active arms (got %d)", len(active_arms))
        return
    if len(pivot) < 5:
        logger.warning("Metric 5: only %d time buckets — need more data", len(pivot))
        return

    sorted_buckets = sorted(pivot.keys())
    matrix = np.array([[pivot[b].get(a, 0) for a in active_arms] for b in sorted_buckets])
    corr = np.corrcoef(matrix.T)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")
    labels = [ARM_LABELS[a] for a in active_arms]
    ax.set_xticks(range(len(active_arms)))
    ax.set_yticks(range(len(active_arms)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(active_arms)):
        for j in range(len(active_arms)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, color="black")
    ax.set_title("Metric 5 — Inter-Agent Signal Correlation\n"
                 "(lower off-diagonal = more diverse = stronger comparison claim)")
    fig.tight_layout()
    _ensure_plots_dir()
    fig.savefig(PLOTS_DIR / "signal_correlation.png", dpi=150)
    plt.close(fig)
    logger.info("Metric 5 saved | signal_correlation.png | arms=%s", active_arms)


# ---------------------------------------------------------------------------
# Metric 3b — Hill-Climb Convergence vs Graph Stability
# ---------------------------------------------------------------------------

def plot_hillclimb_vs_stability() -> None:
    """Hill-climb iterations vs validated edge count at decision time (Metric 3b).

    Negative slope → causal graph stability reduces optimizer convergence cost,
    proving the graph is genuinely guiding regime adaptation.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        do_rows = conn.execute("""
            SELECT timestamp_committed, hill_climb_iterations
            FROM decision_objects
            WHERE strategy_name = 'CausalAgent'
            AND status IN ('CLOSED', 'INTERRUPTED')
            AND hill_climb_iterations > 0
            ORDER BY timestamp_committed ASC
        """).fetchall()
        l1_rows = conn.execute("""
            SELECT timestamp, edge_key, AVG(present) as rate
            FROM layer1_runs
            GROUP BY timestamp, edge_key
        """).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("Metric 3b: DB error | %s", exc)
        return

    if not do_rows or not l1_rows:
        logger.warning("Metric 3b: insufficient data (causal_decisions=%d, l1_rows=%d)",
                       len(do_rows) if do_rows else 0, len(l1_rows) if l1_rows else 0)
        return

    # Count validated edges per layer1 run timestamp
    ts_rates: dict = defaultdict(lambda: defaultdict(list))
    for row in l1_rows:
        ts_rates[row["timestamp"]][row["edge_key"]].append(row["rate"])
    validated_per_ts = {
        ts: sum(1 for rates in edges.values() if sum(rates) / len(rates) >= 0.65)
        for ts, edges in ts_rates.items()
    }
    l1_timestamps = sorted(validated_per_ts.keys())

    x_pts, y_pts = [], []
    for row in do_rows:
        ts_str = row["timestamp_committed"]
        closest = next((t for t in reversed(l1_timestamps) if t <= ts_str), None)
        if closest is None:
            continue
        x_pts.append(validated_per_ts[closest])
        y_pts.append(row["hill_climb_iterations"])

    if len(x_pts) < 3:
        logger.warning("Metric 3b: only %d data points — need ≥3", len(x_pts))
        return

    x_arr = np.array(x_pts, dtype=float)
    y_arr = np.array(y_pts, dtype=float)
    reg = LinearRegression().fit(x_arr.reshape(-1, 1), y_arr)
    slope = reg.coef_[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_arr, y_arr, color="#4A90D9", alpha=0.7, s=55, label="CausalAgent decisions")
    x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
    ax.plot(x_line, reg.predict(x_line.reshape(-1, 1)), color="#E74C3C", linewidth=2,
            label=f"Regression (slope={slope:.3f})")
    ax.set_xlabel("Validated Edge Count (Layer 1)")
    ax.set_ylabel("Hill-Climb Iterations")
    ax.set_title("Metric 3b — Graph Stability vs Optimizer Convergence Cost\n"
                 "(negative slope = stable graph guides MetaOptimizer)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _ensure_plots_dir()
    fig.savefig(PLOTS_DIR / "hillclimb_vs_stability.png", dpi=150)
    plt.close(fig)
    logger.info("Metric 3b saved | hillclimb_vs_stability.png | slope=%.3f", slope)


# ---------------------------------------------------------------------------
# Metric 6 — Time to Rediscovery
# ---------------------------------------------------------------------------

def plot_time_to_rediscovery() -> None:
    """At which PCMCI run did the agent first rediscover each known-good edge? (Metric 6)

    Earlier rediscovery validates causal signal extraction quality.
    Known edges are those whose logic mirrors the legacy strategies.
    """
    KNOWN_EDGES = {
        "EMA rediscovery":       ("ema_spread",    "price_return"),
        "Bollinger rediscovery": ("price_in_band", "price_return"),
        "RSI rediscovery":       ("rsi",            "price_return"),
        "Spread causal":         ("spread",         "price_return"),
    }
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT ce.source, ce.target, cg.n_bars_used, cg.timestamp, cg.algorithm
            FROM causal_edges ce
            JOIN causal_graphs cg ON ce.graph_version_id = cg.version_id
            ORDER BY cg.timestamp ASC
        """).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("Metric 6: DB error | %s", exc)
        return

    first_seen: dict = {}
    for row in rows:
        for label, (src, tgt) in KNOWN_EDGES.items():
            if row["source"] == src and row["target"] == tgt and label not in first_seen:
                first_seen[label] = {
                    "n_bars": row["n_bars_used"],
                    "algorithm": row["algorithm"],
                }

    fig, ax = plt.subplots(figsize=(9, 5))
    all_labels = list(KNOWN_EDGES.keys())
    colors_bar = ["#27AE60" if lbl in first_seen else "#E74C3C" for lbl in all_labels]
    values = [first_seen.get(lbl, {}).get("n_bars", 0) for lbl in all_labels]

    bars = ax.barh(all_labels, values, color=colors_bar, alpha=0.85)
    for bar, lbl, val in zip(bars, all_labels, values):
        if val > 0:
            alg = first_seen[lbl]["algorithm"]
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val} bars ({alg})", va="center", fontsize=9)
        else:
            ax.text(2, bar.get_y() + bar.get_height() / 2,
                    "Not yet discovered", va="center", fontsize=9, color="white")

    ax.set_xlabel("Bars in PCMCI window at first discovery")
    ax.set_title("Metric 6 — Time to Rediscovery of Known-Good Causal Edges\n"
                 "(green = found, red = pending | earlier = better signal extraction)")
    ax.set_xlim(0, max(values + [200]) * 1.35)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    _ensure_plots_dir()
    fig.savefig(PLOTS_DIR / "time_to_rediscovery.png", dpi=150)
    plt.close(fig)
    logger.info("Metric 6 saved | time_to_rediscovery.png | found=%d/%d",
                len(first_seen), len(KNOWN_EDGES))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("Running research analyses...")
    print("[Original Metrics]")
    plot_full_lifecycle()
    plot_trained_phase_delta()
    plot_brier_score_progression()
    ablation = run_ml1_ablation()
    if ablation:
        w, wo = ablation
        print(f"\nAblation result: with_ML1={w:.4f}  without_ML1={wo:.4f}  ML1_helps={w < wo}")

    print("\n[New Metrics]")
    plot_regime_win_rates()
    plot_signal_correlation()
    plot_hillclimb_vs_stability()
    plot_time_to_rediscovery()

    print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}")
