"""ml/trust_updater.py  — Task 8

Layer 2 trust updater: after each closed causal-agent trade, updates the
trust score of every causal edge that was part of the reasoning chain.

Rules that predicted correctly become more trusted.
Rules that didn't become less trusted.
This is the self-correction loop — no human ever rewrites a rule.

Trust score: initialises at 0.5 (neutral), converges toward 1.0 or 0.0
as evidence accumulates. Score is meaningful after ~15-20 observations.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Optional

from ml.hypothesis_generator import CausalHypothesis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LEARNING_RATE: float = 0.10         # how fast trust scores update
INITIAL_TRUST: float = 0.50         # neutral starting point
MIN_OBS_FOR_TRUST: int = 15         # trust is "meaningful" above this count


class Layer2TrustUpdater:
    """Manages per-edge trust scores updated from real trade outcomes.

    Usage:
        updater = Layer2TrustUpdater(db_path="data/arivu.db")
        # After each closed trade:
        updater.update(hypothesis, outcome_correct=True)
        # When generating hypotheses:
        trust_info = updater.get_trust("ema_spread", "price_return", lag=1)
    """

    def __init__(
        self,
        db_path: str = "data/arivu.db",
        learning_rate: float = LEARNING_RATE,
    ) -> None:
        self._db_path = db_path
        self._lr = learning_rate
        self._lock = threading.Lock()
        # In-memory cache: edge_key → {"trust": float, "n_observations": int}
        self._scores: dict[str, dict] = {}
        self._ensure_schema()
        self._load_from_db()

    # ------------------------------------------------------------------
    # Updating trust after a closed trade
    # ------------------------------------------------------------------

    def update(
        self,
        hypothesis: CausalHypothesis,
        outcome_correct: bool,
        actual_price_return: float,
        predicted_price_return: float,
    ) -> None:
        """Update trust for every edge in the hypothesis chain.

        Args:
            hypothesis:              the causal reasoning used for the trade.
            outcome_correct:         True if actual direction matched predicted.
            actual_price_return:     real price_return over the trade horizon.
            predicted_price_return:  what the hypothesis predicted.
        """
        outcome = 1.0 if outcome_correct else 0.0

        for edge in hypothesis.chain:
            key = _edge_key(edge.source, edge.target, edge.lag)
            with self._lock:
                if key not in self._scores:
                    self._scores[key] = {
                        "trust": INITIAL_TRUST, 
                        "n_observations": 0,
                        "successes": 0,
                        "failures": 0,
                    }
                rec = self._scores[key]
                old_trust = rec["trust"]
                
                if outcome_correct:
                    rec["successes"] += 1
                else:
                    rec["failures"] += 1
                
                rec["n_observations"] += 1
                
                # Bayesian Beta Posterior Mean (Uniform prior: alpha=1, beta=1)
                new_trust = (1.0 + rec["successes"]) / (2.0 + rec["n_observations"])
                
                rec["trust"] = round(new_trust, 6)

            self._persist_update(
                edge_key=key,
                trust=new_trust,
                n_observations=rec["n_observations"],
                successes=rec["successes"],
                failures=rec["failures"],
                hypothesis_id=hypothesis.id,
                outcome_correct=outcome_correct,
                actual_return=actual_price_return,
                predicted_return=predicted_price_return,
            )

            logger.info(
                "Layer2: trust updated | edge=%s | outcome=%s | "
                "old=%.4f new=%.4f n=%d",
                key, "correct" if outcome_correct else "wrong",
                old_trust, new_trust, rec["n_observations"],
            )

    # ------------------------------------------------------------------
    # Reading trust scores (called by HypothesisGenerator)
    # ------------------------------------------------------------------

    def get_trust(self, source: str, target: str, lag: int) -> dict:
        """Return trust score + observation count for an edge.

        Returns:
            {"trust": float, "n_observations": int}
            If edge has never been observed: neutral defaults.
        """
        key = _edge_key(source, target, lag)
        with self._lock:
            return dict(self._scores.get(key, {"trust": INITIAL_TRUST, "n_observations": 0}))

    def get_all_scores(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._scores.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS layer2_trust_scores (
                    edge_key TEXT PRIMARY KEY,
                    trust REAL NOT NULL,
                    n_observations INTEGER NOT NULL,
                    successes INTEGER DEFAULT 0,
                    failures INTEGER DEFAULT 0
                )
            """)
            # Fix 5: Add successes and failures for Bayesian Beta updates if they don't exist.
            try:
                conn.execute("ALTER TABLE layer2_trust_scores ADD COLUMN successes INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE layer2_trust_scores ADD COLUMN failures INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS layer2_trust_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    edge_key TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    outcome_correct INTEGER NOT NULL,
                    trust_after REAL NOT NULL,
                    n_observations INTEGER NOT NULL,
                    actual_return REAL NOT NULL,
                    predicted_return REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_from_db(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT edge_key, trust, n_observations, successes, failures FROM layer2_trust_scores"
                ).fetchall()
            with self._lock:
                for edge_key, trust, n_obs, succ, fail in rows:
                    self._scores[edge_key] = {
                        "trust": trust, 
                        "n_observations": n_obs,
                        "successes": succ if succ is not None else 0,
                        "failures": fail if fail is not None else 0,
                    }
            logger.info("Layer2: loaded %d trust scores from DB", len(self._scores))
        except Exception as exc:
            logger.warning("Layer2: could not load from DB: %s", exc)

    def _persist_update(
        self,
        edge_key: str,
        trust: float,
        n_observations: int,
        successes: int,
        failures: int,
        hypothesis_id: str,
        outcome_correct: bool,
        actual_return: float,
        predicted_return: float,
    ) -> None:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO layer2_trust_scores "
                    "(edge_key, trust, n_observations, successes, failures) VALUES (?, ?, ?, ?, ?)",
                    (edge_key, trust, n_observations, successes, failures),
                )
                conn.execute(
                    "INSERT INTO layer2_trust_history "
                    "(edge_key, hypothesis_id, outcome_correct, trust_after, "
                    "n_observations, actual_return, predicted_return, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        edge_key, hypothesis_id, int(outcome_correct),
                        trust, n_observations, actual_return, predicted_return, ts,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("Layer2: persist failed: %s", exc)


def _edge_key(source: str, target: str, lag: int) -> str:
    return f"{source}|{target}|{lag}"
