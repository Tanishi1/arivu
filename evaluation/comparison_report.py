"""evaluation/comparison_report.py  — Task 14

Comparison report: causal agent vs. legacy strategies (EMA/Bollinger/RSI).

Reads the ledger and outputs a per-arm performance table.
Also checks whether any causal agent edge resembles a legacy strategy's logic.

Run:
    python -m evaluation.comparison_report
    # or
    python evaluation/comparison_report.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = os.getenv("SQLITE_PATH", "data/arivu.db")

# Causal edges that, if discovered, would indicate the agent rediscovered
# known-good trading logic from data alone.
KNOWN_STRATEGY_EDGES = {
    "ema_rediscovery":      ("ema_spread", "price_return"),
    "bollinger_rediscovery":("price_in_band", "price_return"),
    "rsi_rediscovery":      ("rsi", "price_return"),
}

ARM_DISPLAY = {
    "CausalAgent":      "Causal Agent",
    "EMAStrategy":      "EMA Legacy",
    "BollingerStrategy":"Bollinger Legacy",
    "RSIStrategy":      "RSI Legacy",
}


def run_report() -> None:
    if not Path(DB_PATH).exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("\n" + "=" * 72)
    print("ARIVU COMPARISON REPORT — Causal Agent vs. Legacy Strategies")
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)

    # -------------------------------------------------------------------
    # Per-arm performance table
    # -------------------------------------------------------------------
    print("\n[1] PER-ARM PERFORMANCE\n")

    arm_stats = _compute_arm_stats(conn)

    header = f"{'Arm':<22} {'Trades':>7} {'Win%':>8} {'Avg PnL':>10} {'Total Return':>14}"
    print(header)
    print("-" * len(header))

    for arm_key, label in ARM_DISPLAY.items():
        stats = arm_stats.get(arm_key, {})
        trades    = stats.get("trades", 0)
        win_rate  = stats.get("win_rate", 0.0)
        avg_pnl   = stats.get("avg_pnl", 0.0)
        total_ret = stats.get("total_return", 0.0)
        if trades == 0:
            print(f"  {label:<20}  (no closed trades yet)")
        else:
            print(
                f"  {label:<20}  {trades:>6}   {win_rate*100:>6.1f}%  "
                f"  ${avg_pnl:>8.2f}   ${total_ret:>12.2f}"
            )

    # -------------------------------------------------------------------
    # Causal agent: top rules by win rate
    # -------------------------------------------------------------------
    print("\n[2] CAUSAL AGENT — TOP RULES\n")
    top_rules = _top_layer2_rules(conn)
    if not top_rules:
        print("  (No Layer 2 trust scores yet — trades still accumulating.)")
    else:
        print(f"  {'Edge':<45} {'Trust':>7} {'N obs':>7}")
        print("  " + "-" * 62)
        for row in top_rules[:10]:
            print(f"  {row['edge_key']:<45} {row['trust']:>7.4f} {row['n_observations']:>7}")

    # -------------------------------------------------------------------
    # Rediscovery check: did the agent find known-good strategy logic?
    # -------------------------------------------------------------------
    print("\n[3] STRATEGY REDISCOVERY CHECK\n")
    print("  Did the causal agent independently discover edges that resemble")
    print("  the logic of the legacy strategies? (Validates the discovery process.)\n")

    found_any = False
    edges = _get_all_causal_edges(conn)
    for label, (src, tgt) in KNOWN_STRATEGY_EDGES.items():
        matching = [e for e in edges if e["source"] == src and e["target"] == tgt]
        if matching:
            best = min(matching, key=lambda x: x["p_value"])
            print(
                f"  [YES] {label}: found edge {src}->{tgt} "
                f"(lag={best['lag']}, coeff={best['coeff']:.4f}, p={best['p_value']:.4f})"
            )
            found_any = True
        else:
            print(f"  [---] {label}: edge {src}->{tgt} NOT YET discovered.")

    if not found_any:
        print("\n  Note: rediscovery requires at least one PCMCI run with enough data.")

    # -------------------------------------------------------------------
    # Layer 1 stability summary
    # -------------------------------------------------------------------
    print("\n[4] LAYER 1 STABILITY SUMMARY\n")
    layer1_stats = _layer1_summary(conn)
    if not layer1_stats:
        print("  (No Layer 1 data yet.)")
    else:
        total = layer1_stats["total_edges"]
        validated = layer1_stats["validated_edges"]
        runs = layer1_stats["total_runs"]
        print(f"  Discovery runs logged:  {runs}")
        print(f"  Unique edges seen:      {total}")
        print(f"  Validated (>=65%):      {validated}")
        if total > 0:
            print(f"  Validation rate:        {validated/total*100:.1f}%")

    conn.close()
    print("\n" + "=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_arm_stats(conn: sqlite3.Connection) -> dict:
    """Compute per-arm win rate, avg PnL, total return from closed decisions."""
    rows = conn.execute("""
        SELECT d.strategy_name, o.actual_pnl
        FROM decision_objects d
        JOIN outcome_records o ON d.id = o.decision_object_id
        WHERE d.status = 'CLOSED'
    """).fetchall()

    stats: dict[str, dict] = {}
    for row in rows:
        arm = row["strategy_name"]
        pnl = row["actual_pnl"]
        if arm not in stats:
            stats[arm] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
        stats[arm]["trades"] += 1
        stats[arm]["total_pnl"] += pnl
        if pnl > 0:
            stats[arm]["wins"] += 1

    result = {}
    for arm, s in stats.items():
        n = s["trades"]
        result[arm] = {
            "trades": n,
            "win_rate": s["wins"] / n if n > 0 else 0.0,
            "avg_pnl": s["total_pnl"] / n if n > 0 else 0.0,
            "total_return": s["total_pnl"],
        }
    return result


def _top_layer2_rules(conn: sqlite3.Connection) -> list[dict]:
    """Return top Layer 2 trust scores sorted by trust descending."""
    try:
        rows = conn.execute(
            "SELECT edge_key, trust, n_observations FROM layer2_trust_scores "
            "ORDER BY trust DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def _get_all_causal_edges(conn: sqlite3.Connection) -> list[dict]:
    """Return all discovered causal edges from the database."""
    try:
        rows = conn.execute(
            "SELECT source, target, lag, coeff, p_value FROM causal_edges"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def _layer1_summary(conn: sqlite3.Connection) -> dict:
    """Summarise Layer 1 run history."""
    try:
        total_runs = conn.execute(
            "SELECT COUNT(DISTINCT graph_version_id) FROM layer1_runs"
        ).fetchone()[0]
        total_edges = conn.execute(
            "SELECT COUNT(DISTINCT edge_key) FROM layer1_runs"
        ).fetchone()[0]
        # Validated: appear in >= 65% of their last 5 runs
        # Approximate by checking presence rate
        rows = conn.execute(
            "SELECT edge_key, AVG(present) as rate FROM layer1_runs "
            "GROUP BY edge_key HAVING rate >= 0.65"
        ).fetchall()
        return {
            "total_runs": total_runs,
            "total_edges": total_edges,
            "validated_edges": len(rows),
        }
    except sqlite3.OperationalError:
        return {}


if __name__ == "__main__":
    run_report()
