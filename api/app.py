"""api/app.py — Arivu read-only REST + WebSocket API

Serves the frontend. Never writes to the DB — all writes are done by
the trading processes (main.py, rl_main.py).

Endpoints:
  GET  /api/state           — current system snapshot (bars, graph, ML1/ML2, regime)
  GET  /api/graph/latest    — latest causal graph + edges + layer1 stability
  GET  /api/ledger          — recent decision_objects + outcome_records
  GET  /api/trust           — all layer2_trust_scores
  GET  /api/optimizer       — meta_optimizer_state (all regime populations)
  GET  /api/replay/last     — full last experiment reconstruction
  GET  /api/comparison      — latest DO for all 3 arms (CausalAgent, ppo_standard, ppo_causal_feature)
  GET  /api/health          — liveness check
  WS   /ws/feed             — tails causal_agent.jsonl for live mode
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# DB path — reads from same env var as the trading system
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("SQLITE_PATH", "data/arivu.db")
JSONL_PATH = Path("logs/causal_agent.jsonl")

app = FastAPI(title="Arivu API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _safe_json(val) -> Optional[dict | list]:
    if not val:
        return None
    try:
        return json.loads(val)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    db_exists = Path(DB_PATH).exists()
    jsonl_exists = JSONL_PATH.exists()
    return {
        "status": "ok",
        "db": db_exists,
        "jsonl": jsonl_exists,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# /api/state  — current system snapshot
# ---------------------------------------------------------------------------

@app.get("/api/state")
def get_state():
    """Latest JSONL line (live feed state) + active/latest DO + graph version."""
    result = {
        "live": None,
        "active_do": None,
        "latest_closed_do": None,
        "bars_ready": None,
        "graph_version": None,
        "regime": None,
        "phase": None,
        "escape_valve": None,
    }

    # Latest JSONL record
    if JSONL_PATH.exists():
        try:
            with open(JSONL_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                result["live"] = last
                result["graph_version"] = last.get("graph_version")
                result["regime"] = last.get("regime") or last.get("hold_reason")
        except Exception:
            pass

    try:
        with _db() as conn:
            # Active DO (ACTIVE or COMMITTED status, CausalAgent only)
            row = conn.execute(
                "SELECT * FROM decision_objects "
                "WHERE strategy_name='CausalAgent' AND status IN ('ACTIVE','COMMITTED') "
                "ORDER BY timestamp_committed DESC LIMIT 1"
            ).fetchone()
            if row:
                result["active_do"] = _do_row_to_dict(row)
                result["phase"] = row["phase"]

            # Latest closed DO
            row = conn.execute(
                "SELECT * FROM decision_objects "
                "WHERE strategy_name='CausalAgent' AND status='CLOSED' "
                "ORDER BY timestamp_committed DESC LIMIT 1"
            ).fetchone()
            if row:
                result["latest_closed_do"] = _do_row_to_dict(row)
                if not result["phase"]:
                    result["phase"] = row["phase"]

            # Latest graph version
            row = conn.execute(
                "SELECT version_id, timestamp, algorithm, n_bars_used "
                "FROM causal_graphs ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row:
                result["bars_ready"] = row["n_bars_used"]
                if not result["graph_version"]:
                    result["graph_version"] = row["version_id"][:8]

            # Escape valve: count consecutive holds from JSONL
            if JSONL_PATH.exists():
                try:
                    with open(JSONL_PATH, "r", encoding="utf-8") as f:
                        lines_all = f.readlines()
                    # Count recent non-traded cycles
                    hold_count = 0
                    for line in reversed(lines_all[-50:]):
                        try:
                            rec = json.loads(line)
                            if rec.get("hold_reason") in ("no_hypothesis", "score_too_low", "no_graph_yet", "ml2_breach_risk"):
                                hold_count += 1
                            else:
                                break
                        except Exception:
                            break
                    fired = any(
                        json.loads(l).get("hypotheses") and
                        any(h.get("is_escape_valve") for h in (json.loads(l).get("hypotheses") or []))
                        for l in lines_all[-20:]
                        if l.strip()
                    )
                    result["escape_valve"] = {
                        "consecutive_holds": hold_count,
                        "armed": hold_count >= 25,
                        "fired": fired,
                    }
                except Exception:
                    pass

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# /api/graph/latest
# ---------------------------------------------------------------------------

@app.get("/api/graph/latest")
def get_latest_graph():
    """Latest causal graph with edges and Layer1 stability scores."""
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM causal_graphs ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if not row:
                return {"graph": None}

            version_id = row["version_id"]
            variables_data = _safe_json(row["variables"]) or {}

            # Edges from causal_edges table
            edges = conn.execute(
                "SELECT source, target, lag, coeff, p_value, conditioning_set "
                "FROM causal_edges WHERE graph_version_id=? ORDER BY p_value ASC",
                (version_id,)
            ).fetchall()

            # Layer1 stability per edge key
            l1_rows = conn.execute(
                "SELECT edge_key, present FROM layer1_runs "
                "ORDER BY id DESC LIMIT 2000"
            ).fetchall()

            # Compute stability from recent history (last 20 runs per edge)
            from collections import defaultdict
            edge_history: dict[str, list] = defaultdict(list)
            for r in reversed(l1_rows):
                key = r["edge_key"]
                if len(edge_history[key]) < 20:
                    edge_history[key].append(bool(r["present"]))

            # Layer2 trust scores
            trust_rows = conn.execute(
                "SELECT edge_key, trust, n_observations FROM layer2_trust_scores"
            ).fetchall()
            trust_map = {r["edge_key"]: {"trust": r["trust"], "n_obs": r["n_observations"]} for r in trust_rows}

            edges_out = []
            for e in edges:
                key = f"{e['source']}|{e['target']}|{e['lag']}"
                history = edge_history.get(key, [])
                stability = sum(history) / len(history) if history else None
                trust_info = trust_map.get(key, {})
                edges_out.append({
                    "source": e["source"],
                    "target": e["target"],
                    "lag": e["lag"],
                    "coeff": round(e["coeff"], 6),
                    "p_value": round(e["p_value"], 6),
                    "conditioning_set": _safe_json(e["conditioning_set"]) or [],
                    "edge_key": key,
                    "stability": round(stability, 4) if stability is not None else None,
                    "trust": trust_info.get("trust"),
                    "trust_n_obs": trust_info.get("n_obs"),
                    "validated": stability is not None and stability >= 0.65,
                })

            return {
                "graph": {
                    "version_id": version_id,
                    "timestamp": row["timestamp"],
                    "algorithm": row["algorithm"],
                    "n_bars_used": row["n_bars_used"],
                    # Discovery params used for this snapshot (MetaOptimizer-driven)
                    "tau_max_used":  row["tau_max_used"]  if "tau_max_used"  in row.keys() else None,
                    "alpha_used":    row["alpha_used"]    if "alpha_used"    in row.keys() else None,
                    "variables": variables_data.get("names", []),
                    "feature_means": variables_data.get("means", {}),
                    "feature_stds": variables_data.get("stds", {}),
                    "edge_count": len(edges_out),
                    "edges": edges_out,
                }
            }
    except Exception as exc:
        return {"error": str(exc), "graph": None}


# ---------------------------------------------------------------------------
# /api/ledger
# ---------------------------------------------------------------------------

@app.get("/api/ledger")
def get_ledger(limit: int = 20, strategy: str = "CausalAgent"):
    """Recent decision objects + their outcome records."""
    try:
        with _db() as conn:
            if strategy == "all":
                rows = conn.execute(
                    "SELECT * FROM decision_objects "
                    "ORDER BY timestamp_committed DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM decision_objects WHERE strategy_name=? "
                    "ORDER BY timestamp_committed DESC LIMIT ?",
                    (strategy, limit)
                ).fetchall()

            results = []
            for row in rows:
                do = _do_row_to_dict(row)
                # Fetch matching outcome record
                out = conn.execute(
                    "SELECT * FROM outcome_records WHERE decision_object_id=? LIMIT 1",
                    (row["id"],)
                ).fetchone()
                do["outcome"] = _outcome_row_to_dict(out) if out else None
                results.append(do)

            return {"decisions": results, "count": len(results)}
    except Exception as exc:
        return {"error": str(exc), "decisions": []}


# ---------------------------------------------------------------------------
# /api/trust
# ---------------------------------------------------------------------------

@app.get("/api/trust")
def get_trust():
    """All Layer2 trust scores + recent history."""
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT edge_key, trust, n_observations, successes, failures "
                "FROM layer2_trust_scores ORDER BY trust DESC"
            ).fetchall()

            scores = []
            for r in rows:
                parts = r["edge_key"].split("|")
                scores.append({
                    "edge_key": r["edge_key"],
                    "source": parts[0] if len(parts) > 0 else "",
                    "target": parts[1] if len(parts) > 1 else "",
                    "lag": int(parts[2]) if len(parts) > 2 else 0,
                    "trust": round(r["trust"], 6),
                    "n_observations": r["n_observations"],
                    "successes": r["successes"] or 0,
                    "failures": r["failures"] or 0,
                })

            # Recent trust history (last 50 updates)
            hist = conn.execute(
                "SELECT edge_key, hypothesis_id, outcome_correct, trust_after, "
                "n_observations, actual_return, predicted_return, timestamp "
                "FROM layer2_trust_history ORDER BY id DESC LIMIT 50"
            ).fetchall()

            return {
                "scores": scores,
                "history": [dict(h) for h in hist],
            }
    except Exception as exc:
        return {"error": str(exc), "scores": []}


# ---------------------------------------------------------------------------
# /api/optimizer
# ---------------------------------------------------------------------------

@app.get("/api/optimizer")
def get_optimizer():
    """MetaParameterOptimizer state — all regime populations."""
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT regime, params_json, history_json, updated_at "
                "FROM meta_optimizer_state"
            ).fetchall()

            regimes = []
            for r in rows:
                population = _safe_json(r["params_json"]) or []
                history = _safe_json(r["history_json"]) or []
                recent_history = history[-20:]
                scored_pop = []
                for i, cand in enumerate(population):
                    scored_pop.append({
                        "index":       i,
                        # Search params
                        "k_runs":      cand.get("k_runs"),
                        "min_runs":    cand.get("min_runs"),
                        "threshold":   cand.get("threshold"),
                        "step_k":      cand.get("step_k"),
                        "step_thresh": cand.get("step_thresh"),
                        # Discovery params (new — tau_max/pcmci_alpha hill-climbed)
                        "tau_max":     cand.get("tau_max"),
                        "pcmci_alpha": cand.get("pcmci_alpha"),
                        "step_tau":    cand.get("step_tau"),
                        "step_alpha":  cand.get("step_alpha"),
                        "regime":      cand.get("regime"),
                        "recent_score": round(
                            sum(h["score"] for h in recent_history) / len(recent_history), 4
                        ) if recent_history else None,
                    })

                regimes.append({
                    "regime":           r["regime"],
                    "updated_at":       r["updated_at"],
                    "population":       scored_pop,
                    "history_count":    len(history),
                    "stagnation_count": None,  # populated from history below
                    "recent_avg_score": round(
                        sum(h["score"] for h in recent_history) / len(recent_history), 4
                    ) if recent_history else None,
                    "history_scores":   [round(h["score"], 4) for h in history[-30:]],
                })

            return {"regimes": regimes}
    except Exception as exc:
        return {"error": str(exc), "regimes": []}


# ---------------------------------------------------------------------------
# /api/replay/last
# ---------------------------------------------------------------------------

@app.get("/api/replay/last")
def get_replay_last():
    """Reconstruct the last complete causal experiment for replay mode."""
    try:
        with _db() as conn:
            # Find the most recent CLOSED CausalAgent decision
            do_row = conn.execute(
                "SELECT * FROM decision_objects "
                "WHERE strategy_name='CausalAgent' AND status='CLOSED' "
                "ORDER BY timestamp_committed DESC LIMIT 1"
            ).fetchone()

            if not do_row:
                return {"replay": None, "reason": "no_closed_trades_yet"}

            do = _do_row_to_dict(do_row)

            # Outcome record
            out_row = conn.execute(
                "SELECT * FROM outcome_records WHERE decision_object_id=? LIMIT 1",
                (do_row["id"],)
            ).fetchone()
            outcome = _outcome_row_to_dict(out_row) if out_row else None

            # Causal graph snapshot — find the graph used for this decision
            graph_version_id = (do.get("tuned_params") or {}).get("graph_version_id")
            graph = None
            if graph_version_id:
                g_row = conn.execute(
                    "SELECT * FROM causal_graphs WHERE version_id=?",
                    (graph_version_id,)
                ).fetchone()
                if g_row:
                    edges = conn.execute(
                        "SELECT source, target, lag, coeff, p_value "
                        "FROM causal_edges WHERE graph_version_id=?",
                        (graph_version_id,)
                    ).fetchall()
                    graph = {
                        "version_id": g_row["version_id"],
                        "algorithm": g_row["algorithm"],
                        "n_bars_used": g_row["n_bars_used"],
                        "edges": [dict(e) for e in edges],
                    }

            # Trust scores for the edges used
            chain_snapshot = do.get("causal_chain_snapshot") or {}
            trust_updates = []
            if out_row:
                # Find trust history entries near this decision's timestamp
                trust_hist = conn.execute(
                    "SELECT edge_key, trust_after, n_observations, outcome_correct, "
                    "actual_return, predicted_return, timestamp "
                    "FROM layer2_trust_history "
                    "WHERE hypothesis_id=? OR timestamp > ? "
                    "ORDER BY id DESC LIMIT 10",
                    (
                        do.get("tuned_params", {}).get("hypothesis_id", ""),
                        do_row["timestamp_committed"],
                    )
                ).fetchall()
                trust_updates = [dict(t) for t in trust_hist]

            # JSONL records around the same timestamp — for hypotheses replay
            jsonl_record = None
            if JSONL_PATH.exists():
                try:
                    do_ts = do_row["timestamp_committed"]
                    with open(JSONL_PATH, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    # Find the line closest in time to the DO timestamp
                    for line in reversed(lines):
                        try:
                            rec = json.loads(line)
                            if rec.get("selected_chain") or rec.get("hold_reason") == "traded":
                                jsonl_record = rec
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

            # Optimizer state
            opt_rows = conn.execute(
                "SELECT regime, params_json, updated_at FROM meta_optimizer_state"
            ).fetchall()
            optimizer_state = [
                {
                    "regime": r["regime"],
                    "population": _safe_json(r["params_json"]) or [],
                    "updated_at": r["updated_at"],
                }
                for r in opt_rows
            ]

            return {
                "replay": {
                    "decision": do,
                    "outcome": outcome,
                    "graph": graph,
                    "trust_updates": trust_updates,
                    "jsonl_record": jsonl_record,
                    "optimizer_state": optimizer_state,
                }
            }
    except Exception as exc:
        return {"error": str(exc), "replay": None}


# ---------------------------------------------------------------------------
# /api/comparison  — all 3 arms for Comparison Arena page
# ---------------------------------------------------------------------------

@app.get("/api/comparison")
def get_comparison():
    """Latest decision + outcome for each of the 3 arms."""
    arms = ["CausalAgent", "ppo_standard", "ppo_causal_feature"]
    result = {}
    try:
        with _db() as conn:
            for arm in arms:
                # Latest DO (any status)
                row = conn.execute(
                    "SELECT * FROM decision_objects WHERE strategy_name=? "
                    "ORDER BY timestamp_committed DESC LIMIT 1",
                    (arm,)
                ).fetchone()
                if not row:
                    result[arm] = None
                    continue

                do = _do_row_to_dict(row)
                out = conn.execute(
                    "SELECT * FROM outcome_records WHERE decision_object_id=? LIMIT 1",
                    (row["id"],)
                ).fetchone()
                do["outcome"] = _outcome_row_to_dict(out) if out else None

                # Summary stats for this arm
                stats_row = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status='CLOSED' THEN 1 ELSE 0 END) as closed,
                        SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(o.actual_pnl) as total_pnl
                    FROM decision_objects d
                    LEFT JOIN outcome_records o ON o.decision_object_id = d.id
                    WHERE d.strategy_name=?
                    """,
                    (arm,)
                ).fetchone()

                result[arm] = {
                    "latest": do,
                    "stats": {
                        "total_decisions": stats_row["total"],
                        "closed": stats_row["closed"],
                        "wins": stats_row["wins"] or 0,
                        "total_pnl": round(stats_row["total_pnl"] or 0.0, 4),
                        "win_rate": round(
                            (stats_row["wins"] or 0) / stats_row["closed"], 4
                        ) if (stats_row["closed"] or 0) > 0 else None,
                    },
                }
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# WebSocket /ws/feed  — tails causal_agent.jsonl for live mode
# ---------------------------------------------------------------------------

@app.websocket("/ws/feed")
async def ws_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        # Send last 5 lines immediately on connect
        if JSONL_PATH.exists():
            with open(JSONL_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-5:]:
                try:
                    await websocket.send_text(line.strip())
                except Exception:
                    break

        # Tail new lines
        pos = JSONL_PATH.stat().st_size if JSONL_PATH.exists() else 0
        while True:
            await asyncio.sleep(2)
            if not JSONL_PATH.exists():
                continue
            size = JSONL_PATH.stat().st_size
            if size > pos:
                with open(JSONL_PATH, "r", encoding="utf-8") as f:
                    f.seek(pos)
                    new_lines = f.readlines()
                pos = size
                for line in new_lines:
                    line = line.strip()
                    if line:
                        try:
                            await websocket.send_text(line)
                        except Exception:
                            return
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Row → dict helpers
# ---------------------------------------------------------------------------

def _do_row_to_dict(row) -> dict:
    d = dict(row)
    for field in ("tuned_params", "market_state_snapshot", "algo_health_vector",
                  "assumptions", "meta_params", "causal_chain_snapshot", "breach_log"):
        d[field] = _safe_json(d.get(field))
    return d


def _outcome_row_to_dict(row) -> dict:
    d = dict(row)
    for field in ("assumptions_held", "assumptions_breached", "breach_timestamps"):
        d[field] = _safe_json(d.get(field))
    return d
