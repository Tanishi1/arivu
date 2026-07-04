"""ml/layer1_tracker.py  — Task 4

Layer 1 stability tracker: tracks whether a discovered causal edge is
statistically real (appearing consistently across K discovery runs) or
just noise from a single window.

An edge must appear in at least VALIDATION_THRESHOLD fraction of its last
K runs to be considered "structurally validated."

ANTI-HOLD DESIGN:
    VALIDATION_THRESHOLD defaults to 0.65 (majority, not unanimous).
    Setting to 1.0 (unanimous) would cause permanent HOLD on noisy markets
    because most edges fail at least one rerun. Do not raise above 0.80.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from ml.causal_discovery import CausalEdge, GraphSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from ml.meta_optimizer import MetaParameterOptimizer, RegimeType



class Layer1Tracker:
    """Tracks edge stability across repeated PCMCI discovery runs.

    Usage:
        tracker = Layer1Tracker(db_path="data/arivu.db")
        tracker.record_run(snapshot)
        validated = tracker.get_validated_edges()
    """

    def __init__(
        self,
        db_path: str = "data/arivu.db",
    ) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        
        # Meta-parameter optimizer instance
        self._optimizer = MetaParameterOptimizer(db_path=db_path)

        # edge_key → deque of booleans (True = present in that run)
        # edge_key = "source|target|lag"
        # We use maxlen=100 to safely store enough history for dynamic k_runs (max 50)
        self._run_history: dict[str, deque[bool]] = {}

        # All unique edge keys seen across all runs (for completeness tracking)
        self._all_edge_keys: set[str] = set()

        self._ensure_schema()
        self._load_from_db()

    # ------------------------------------------------------------------
    # Recording a new discovery run
    # ------------------------------------------------------------------

    def record_run(self, snapshot: GraphSnapshot, regime: str = "unknown") -> None:
        """Update run history for all known edges given a new GraphSnapshot.

        Edges present in this run → True appended.
        Edges known but absent in this run → False appended.
        New edges seen for first time → history initialised with this one True.
        """
        params = self._optimizer.get_params(regime)
        present_keys: set[str] = set()
        for edge in snapshot.edges:
            key = _edge_key(edge.source, edge.target, edge.lag)
            present_keys.add(key)
            self._all_edge_keys.add(key)

        with self._lock:
            # Record presence/absence for all previously known edges
            for key in list(self._all_edge_keys):
                if key not in self._run_history:
                    self._run_history[key] = deque(maxlen=100)
                self._run_history[key].append(key in present_keys)

            # New edges not yet in history
            for key in present_keys - set(self._run_history.keys()):
                self._run_history[key] = deque([True], maxlen=100)
                self._all_edge_keys.add(key)

        # Persist this run's data
        self._persist_run(snapshot, present_keys)

        n_validated = len(self.get_validated_edges())
        logger.info(
            "Layer1: run recorded | graph=%s | validated_edges=%d/%d",
            snapshot.version_id[:8], n_validated, len(self._all_edge_keys),
        )

    # ------------------------------------------------------------------
    # Querying validated edges
    # ------------------------------------------------------------------

    def get_validated_edges(self, regime: str = "unknown") -> list[dict]:
        """Return all edges that meet the validation threshold.

        Returns list of dicts with keys: source, target, lag, stability_score.
        stability_score = fraction of last k_runs where edge was present.
        """
        params = self._optimizer.get_params(regime)
        validated = []
        with self._lock:
            for key, history in self._run_history.items():
                recent_history = list(history)[-params.k_runs:]
                if len(recent_history) < params.min_runs:
                    continue
                stability = sum(recent_history) / len(recent_history)
                if stability >= params.threshold:
                    source, target, lag = _parse_edge_key(key)
                    validated.append({
                        "source": source,
                        "target": target,
                        "lag": int(lag),
                        "stability_score": round(stability, 4),
                        "n_runs_seen": len(recent_history),
                    })
        return validated

    def get_stability_score(self, source: str, target: str, lag: int, regime: str = "unknown") -> float:
        """Return stability fraction for a specific edge (0.0 if never seen or < min runs)."""
        key = _edge_key(source, target, lag)
        params = self._optimizer.get_params(regime)
        with self._lock:
            history = self._run_history.get(key)
        if not history:
            return 0.0
            
        recent_history = list(history)[-params.k_runs:]
        if len(recent_history) < params.min_runs:
            return 0.0
        return sum(recent_history) / len(recent_history)

    def is_validated(self, source: str, target: str, lag: int, regime: str = "unknown") -> bool:
        params = self._optimizer.get_params(regime)
        return self.get_stability_score(source, target, lag, regime) >= params.threshold
    def total_runs_recorded(self) -> int:
        """Return the maximum number of runs recorded for any single edge.
        
        Used by HypothesisGenerator escape valve to detect prolonged HOLD.
        Returns 0 if no history exists yet.
        """
        with self._lock:
            if not self._run_history:
                return 0
            return max(len(h) for h in self._run_history.values())

    def has_stable_edges(self, regime: str = "unknown") -> bool:
        """Return True if at least one validated edge exists.

        Used by legacy_strategy_loop to determine if the causal agent is
        ready to trade independently — at which point legacy strategies
        should stop permanently.
        """
        return len(self.get_validated_edges(regime)) > 0

    def is_legacy_permanently_stopped(self) -> bool:
        """Check the DB flag that marks legacy strategies as permanently stopped.

        Once set, this persists across restarts — legacy strategies never
        resume after the causal graph has been validated for the first time.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT value FROM arivu_flags WHERE key = 'legacy_stopped'"
                ).fetchone()
                return row is not None and row[0] == "1"
        except Exception:
            return False

    def set_legacy_permanently_stopped(self) -> None:
        """Persist the legacy_stopped flag in the DB.

        Called once, the first time has_stable_edges() returns True.
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS arivu_flags "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO arivu_flags (key, value) VALUES ('legacy_stopped', '1')"
                )
                conn.commit()
            logger.info(
                "Layer1: legacy_stopped flag set permanently in DB — "
                "legacy strategies will not restart on future runs"
            )
        except Exception as exc:
            logger.error("Layer1: failed to set legacy_stopped flag | %s", exc)


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS layer1_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_version_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    edge_key TEXT NOT NULL,
                    present INTEGER NOT NULL
                )
            """)
            conn.commit()

    def _persist_run(self, snapshot: GraphSnapshot, present_keys: set[str]) -> None:
        try:
            ts = datetime.now(timezone.utc).isoformat()
            rows = []
            with self._lock:
                for key in self._all_edge_keys:
                    rows.append((
                        snapshot.version_id,
                        ts,
                        key,
                        1 if key in present_keys else 0,
                    ))
            with sqlite3.connect(self._db_path) as conn:
                conn.executemany(
                    "INSERT INTO layer1_runs "
                    "(graph_version_id, timestamp, edge_key, present) VALUES (?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
        except Exception as exc:
            logger.error("Layer1: failed to persist run: %s", exc)

    def _load_from_db(self) -> None:
        """Reconstruct run history from the database (survives restarts)."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                # Get all unique edge keys
                rows = conn.execute(
                    "SELECT edge_key, graph_version_id, present "
                    "FROM layer1_runs "
                    "ORDER BY id ASC"
                ).fetchall()

            if not rows:
                return

            # Rebuild: group by edge_key, keep last K per edge
            history_raw: dict[str, list[bool]] = {}
            for edge_key, _gv_id, present in rows:
                history_raw.setdefault(edge_key, []).append(bool(present))

            with self._lock:
                for key, presences in history_raw.items():
                    # Only keep the most recent 100 runs
                    recent = presences[-100:]
                    self._run_history[key] = deque(recent, maxlen=100)
                    self._all_edge_keys.add(key)

            logger.info(
                "Layer1: loaded history from DB | edges=%d",
                len(self._run_history),
            )
        except Exception as exc:
            logger.warning("Layer1: could not load from DB: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_key(source: str, target: str, lag: int) -> str:
    return f"{source}|{target}|{lag}"


def _parse_edge_key(key: str) -> tuple[str, str, int]:
    parts = key.split("|")
    return parts[0], parts[1], int(parts[2])
