"""ml/meta_optimizer.py — Agent Self-Parameter Optimizer

INTEGRATION NOTES FOR AGENT:
─────────────────────────────────────────────────────────────────────────────
This module is fully self-contained. It imports nothing from the existing
codebase and has no side effects until explicitly instantiated.

WIRING CHECKLIST (agent must do all 4 to activate):

  1. layer1_tracker.py
     ─ Import: from ml.meta_optimizer import MetaParameterOptimizer, RegimeType
     ─ In __init__: self._optimizer = MetaParameterOptimizer(db_path=db_path)
     ─ In get_validated_edges() / is_validated(): replace the three hardcoded
       constants (K_RUNS, MIN_RUNS_FOR_VALIDATION, VALIDATION_THRESHOLD) with:
           regime = <however you currently read regime from CausalState>
           params = self._optimizer.get_params(regime)
           k      = params.k_runs
           min_r  = params.min_runs
           thresh = params.threshold
     ─ Pass params to record_run() so they are stored per-run (see below).

  2. main.py (or wherever closed-trade callback lives)
     ─ Import: from ml.meta_optimizer import TradeOutcome
     ─ After every trade closes, build a TradeOutcome and call:
           self._layer1_tracker._optimizer.update(outcome)

  3. ledger / DecisionObject schema
     ─ Add a meta_params field (dict) to DecisionObject.
     ─ Before committing, call:
           params = optimizer.get_params(current_regime)
           decision.meta_params = params.to_dict()
     ─ Agent must check whether DecisionObject is a dataclass, Pydantic model,
       or plain dict and add the field accordingly — do NOT alter this file.

  4. trust_updater.py (Layer 2)
     ─ At the end of Layer2TrustUpdater.update(), build a TradeOutcome and
       call optimizer.update(outcome) — same call as in main.py.
     ─ Only one caller needed; choose whichever already has P&L data.

SCHEMA COMPATIBILITY:
     The optimizer creates its own table (meta_optimizer_state) and never
     touches existing tables. Safe to add to any existing arivu.db.

WHAT THE AGENT MUST NOT CHANGE IN THIS FILE:
     ─ The scoring function (_score_config) — mathematical contract.
     ─ The Beta posterior formula — must stay (1+s)/(2+n).
     ─ The random initialization ranges — intentionally wide/uninformed.
     ─ DB table name — other modules may reference it for reporting.

WHAT THE AGENT MAY CHANGE:
     ─ STEP_DECAY_RATE: how fast hill-climbing steps shrink (default 0.80).
     ─ MIN_STEP_SIZES: floor below which a random restart triggers.
     ─ RESTART_THRESHOLD: score floor that triggers random restart.
     ─ RegimeType values: if your regime classifier uses different labels,
       update the Enum values to match exactly — string must equal classifier
       output or get_params() will fall back to UNKNOWN.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public Enums & Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class RegimeType(str, Enum):
    """
    AGENT: these string values must exactly match what your regime classifier
    outputs (e.g. RegimeClassifier.classify() return values).
    If your classifier returns 'calm', 'volatile', 'trending' — leave as-is.
    If it returns e.g. 'CALM', 'RANGING' — update the values here only.
    """
    CALM      = "calm"
    VOLATILE  = "volatile"
    TRENDING  = "trending"
    UNKNOWN   = "unknown"   # fallback when regime is not yet determined


@dataclass
class MetaParams:
    """
    The three operational parameters the optimizer controls.
    These replace the hardcoded constants in layer1_tracker.py.
    """
    k_runs:    int   # size of the rolling run history window
    min_runs:  int   # minimum runs observed before validation is eligible
    threshold: float # fraction of k_runs an edge must appear in to validate
    regime:    str   = "unknown"

    # Internal hill-climbing state — agent should not read these directly
    step_k:     int   = 5
    step_min_r: int   = 1
    step_thresh: float = 0.05

    def to_dict(self) -> dict:
        """Serialise for ledger / DecisionObject meta_params field."""
        return {
            "k_runs":      self.k_runs,
            "min_runs":    self.min_runs,
            "threshold":   round(self.threshold, 4),
            "regime":      self.regime,
            "step_k":      self.step_k,
            "step_min_r":  self.step_min_r,
            "step_thresh": round(self.step_thresh, 4),
        }

    def validate(self) -> "MetaParams":
        """
        Enforce hard mathematical constraints that must always hold.
        Agent: do not remove these — they prevent invalid states.
        """
        self.k_runs    = max(3, int(self.k_runs))
        self.min_runs  = max(2, min(int(self.min_runs), self.k_runs))
        self.threshold = max(0.40, min(float(self.threshold), 0.95))
        self.step_k     = max(1, int(self.step_k))
        self.step_min_r = max(1, int(self.step_min_r))
        self.step_thresh = max(0.01, float(self.step_thresh))
        return self


@dataclass
class TradeOutcome:
    """
    Everything the optimizer needs to score the configuration that produced
    a trade. Agent must populate all fields after a trade closes.

    AGENT WIRING NOTES:
    ─ meta_params_used: copy from the DecisionObject.meta_params dict that
      was committed before this trade executed.
    ─ regime: regime active at trade entry time (string matching RegimeType).
    ─ predicted_return: what TwinSimulator predicted (from DecisionObject).
    ─ actual_return: real price_return over the trade horizon (from monitor).
    ─ pnl_usd: actual realised P&L in dollars (from executor/ledger).
    ─ causal_chain_held: True if monitor did NOT flag a trajectory breach.
    ─ regime_was_stable: True if regime classifier agreed on same regime for
      the full duration of the trade (agent must track this separately).
    """
    meta_params_used:   dict
    regime:             str
    predicted_return:   float
    actual_return:      float
    pnl_usd:            float
    causal_chain_held:  bool
    regime_was_stable:  bool


# ─────────────────────────────────────────────────────────────────────────────
# Hill-climbing constants — agent MAY adjust these
# ─────────────────────────────────────────────────────────────────────────────

STEP_DECAY_RATE    = 0.90   # slow down decay to allow exploration
MIN_STEP_K         = 2      # floor for k_runs step
MIN_STEP_THRESH    = 0.01   # floor for threshold step
RESTART_THRESHOLD  = 0.30   # score floor — restart if best score stays below this

# Stagnation detection constants
STAGNATION_DELTA_THRESHOLD = 0.005  # minimum score improvement to reset stagnation counter
STAGNATION_LIMIT = 20               # consecutive non-improving updates before restart



# ─────────────────────────────────────────────────────────────────────────────
# MetaParameterOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class MetaParameterOptimizer:
    """
    Hill-climbs on (k_runs, min_runs, threshold) independently per market
    regime. Initialized with random values; converges purely from trade
    outcomes — no human-specified bounds or target values.

    One instance shared across the system; per-regime state is maintained
    internally.
    """

    def __init__(self, db_path: str = "data/arivu.db") -> None:
        self._db_path = db_path
        self._lock    = threading.Lock()

        # Per-regime optimizer state
        # { regime_str -> { "params": MetaParams, "history": [...] } }
        self._state: dict[str, dict] = {}

        # Stagnation detection: tracks consecutive non-improving updates per regime
        # Reset to 0 whenever score improves by >= STAGNATION_DELTA_THRESHOLD
        self._stagnation_counts: dict[str, int] = {}
        self._last_scores: dict[str, float] = {}

        self._ensure_schema()
        self._load_from_db()

        # Initialise any regime not yet in DB with random params
        for regime in RegimeType:
            if regime.value not in self._state:
                self._state[regime.value] = {
                    "params":  self._random_init(regime.value),
                    "history": [],   # list of scored outcomes for this regime
                }
                logger.info(
                    "MetaOptimizer: random init | regime=%s | %s",
                    regime.value,
                    self._state[regime.value]["params"].to_dict(),
                )

        # Persist newly initialized regimes immediately so params
        # survive restarts before any trade closes
        for regime in RegimeType:
            if regime.value in self._state:
                self._persist(regime.value)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def get_params(self, regime: str) -> MetaParams:
        """
        Called by layer1_tracker on every validation check.
        Returns the current best-known params for this regime.
        Falls back to UNKNOWN if regime string is unrecognised.
        """
        key = regime if regime in self._state else RegimeType.UNKNOWN.value
        with self._lock:
            return self._state[key]["params"]

    def update(self, outcome: TradeOutcome) -> None:
        """
        Called after every closed trade.
        Scores the configuration that was active, hill-climbs one step.
        """
        regime = outcome.regime
        if regime not in self._state:
            regime = RegimeType.UNKNOWN.value

        score = self._score_outcome(outcome)

        with self._lock:
            bucket  = self._state[regime]
            current = bucket["params"]

            bucket["history"].append({
                "params": outcome.meta_params_used,
                "score":  score,
            })
            # Keep history bounded to last 200 outcomes
            if len(bucket["history"]) > 200:
                bucket["history"] = bucket["history"][-200:]

            current_score = self._score_config(
                outcome.meta_params_used, bucket["history"]
            )

            # Generate all neighbours (±step for each parameter)
            neighbours = self._get_neighbours(current)
            best_neighbour      = None
            best_neighbour_score = current_score

            for neighbour in neighbours:
                n_dict = neighbour.to_dict()
                n_score = self._score_config(n_dict, bucket["history"])
                if n_score > best_neighbour_score:
                    best_neighbour       = neighbour
                    best_neighbour_score = n_score

            if best_neighbour is not None:
                # Improvement found — move to neighbour
                # Reset stagnation counter since we made progress
                self._stagnation_counts[regime] = 0
                self._last_scores[regime] = best_neighbour_score

                old_dict = current.to_dict()
                new_dict = best_neighbour.to_dict()
                logger.info(
                    "MetaOptimizer: Hill-climbing regime '%s' (score %.4f -> %.4f) | "
                    "threshold: %.4f -> %.4f | k_runs: %d -> %d | min_runs: %d -> %d",
                    regime, current_score, best_neighbour_score,
                    old_dict["threshold"], new_dict["threshold"],
                    old_dict["k_runs"], new_dict["k_runs"],
                    old_dict["min_runs"], new_dict["min_runs"]
                )
                bucket["params"] = best_neighbour
            else:
                # No improvement — shrink step sizes (converging)
                old_k = current.step_k
                old_thresh = current.step_thresh
                current.step_k      = max(
                    MIN_STEP_K,
                    int(current.step_k * STEP_DECAY_RATE)
                )
                current.step_thresh = max(
                    MIN_STEP_THRESH,
                    current.step_thresh * STEP_DECAY_RATE
                )
                logger.info(
                    "MetaOptimizer: Local optimum reached for regime '%s' (score: %.4f) | "
                    "Shrinking step sizes: step_k: %d -> %d | step_thresh: %.4f -> %.4f",
                    regime, current_score, old_k, current.step_k, old_thresh, current.step_thresh
                )

                # --- Stagnation detection ---
                # Check whether score has meaningfully improved since last update.
                # Small oscillations (< STAGNATION_DELTA_THRESHOLD) count as stagnation.
                last_score = self._last_scores.get(regime, current_score)
                score_delta = abs(current_score - last_score)

                if score_delta >= STAGNATION_DELTA_THRESHOLD:
                    # Real improvement — reset stagnation counter
                    self._stagnation_counts[regime] = 0
                    logger.debug(
                        "MetaOptimizer: stagnation counter reset | regime=%s | "
                        "score delta=%.4f", regime, score_delta
                    )
                else:
                    # No meaningful improvement — increment stagnation counter
                    self._stagnation_counts[regime] = (
                        self._stagnation_counts.get(regime, 0) + 1
                    )

                self._last_scores[regime] = current_score

                # Trigger restart if stuck for STAGNATION_LIMIT consecutive updates
                # AND step sizes are already at minimum (fully converged)
                stagnation_count = self._stagnation_counts.get(regime, 0)
                if (stagnation_count >= STAGNATION_LIMIT
                        and current.step_k <= MIN_STEP_K
                        and current.step_thresh <= MIN_STEP_THRESH):

                    old_params = current.to_dict()
                    new_params = self._random_init(regime)
                    bucket["params"] = new_params
                    self._stagnation_counts[regime] = 0
                    self._last_scores[regime] = 0.5  # reset to uninformed prior

                    logger.info(
                        "MetaOptimizer: STAGNATION RESTART | regime=%s | "
                        "stuck for %d updates | score_range=[%.4f] | "
                        "escaped_params=k_runs=%d,min_runs=%d,threshold=%.2f | "
                        "new_params=k_runs=%d,min_runs=%d,threshold=%.2f",
                        regime, stagnation_count, current_score,
                        old_params["k_runs"], old_params["min_runs"],
                        old_params["threshold"],
                        new_params.k_runs, new_params.min_runs,
                        new_params.threshold,
                    )

        self._persist(regime)

    # ─────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────

    def _score_outcome(self, outcome: TradeOutcome) -> float:
        """
        Score a single trade outcome. Returns float in [0, 1].

        Components:
          - direction_correct: did we predict the right direction?
          - prediction_accuracy: how close was predicted vs actual return?
          - regime_purity: was the regime stable during this trade?
          - chain_held: did the causal chain hold (no trajectory breach)?
        """
        # Direction correctness (binary but weighted heavily)
        direction_correct = (
            math.copysign(1, outcome.predicted_return) ==
            math.copysign(1, outcome.actual_return)
        ) if outcome.predicted_return != 0 else False

        direction_score = 1.0 if direction_correct else 0.0

        # Prediction accuracy: ratio of actual to predicted, capped at 1
        if outcome.predicted_return != 0:
            accuracy = min(
                1.0,
                abs(outcome.actual_return) / abs(outcome.predicted_return)
            )
        else:
            accuracy = 0.0

        # Regime purity bonus
        regime_purity = 1.0 if outcome.regime_was_stable else 0.6

        # Causal chain integrity bonus
        chain_bonus = 1.0 if outcome.causal_chain_held else 0.5

        # Composite
        raw = (
            direction_score * 0.45 +
            accuracy        * 0.30 +
            regime_purity   * 0.15 +
            chain_bonus     * 0.10
        )
        return round(raw, 6)

    def _score_config(
        self,
        config_dict: dict,
        history: list[dict],
    ) -> float:
        """
        Score a parameter configuration against outcome history using a Gaussian Kernel.
        Each historical trade outcome is weighted by its normalized distance to the
        target configuration, and the final score is the weighted average combined
        with a Bayesian prior (alpha = 1.0 at score = 0.5) to handle low-data areas.
        """
        if not history:
            return 0.5

        # Normalization bandwidths
        sigma_k = 5.0
        sigma_min_r = 2.0
        sigma_thresh = 0.05
        
        # Bayesian prior weight (equivalent to 1 observation at score 0.5)
        alpha = 1.0
        
        target_k = float(config_dict.get("k_runs", 0))
        target_min_r = float(config_dict.get("min_runs", 0))
        target_thresh = float(config_dict.get("threshold", 0.0))
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for h in history:
            p = h["params"]
            s = h["score"]
            
            diff_k = (target_k - float(p.get("k_runs", 0))) / sigma_k
            diff_min_r = (target_min_r - float(p.get("min_runs", 0))) / sigma_min_r
            diff_thresh = (target_thresh - float(p.get("threshold", 0.0))) / sigma_thresh
            
            d_squared = diff_k * diff_k + diff_min_r * diff_min_r + diff_thresh * diff_thresh
            weight = math.exp(-d_squared / 2.0)
            
            weighted_sum += weight * s
            weight_total += weight
            
        score = (weighted_sum + alpha * 0.5) / (weight_total + alpha)
        return round(score, 6)

    # ─────────────────────────────────────────────────────────────────────
    # Hill-climbing helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_neighbours(self, params: MetaParams) -> list[MetaParams]:
        """Generate all ±1 step neighbours for each parameter."""
        neighbours = []
        for dk in (-params.step_k, 0, params.step_k):
            for dm in (-params.step_min_r, 0, params.step_min_r):
                for dt in (-params.step_thresh, 0, params.step_thresh):
                    if dk == 0 and dm == 0 and dt == 0:
                        continue
                    n = MetaParams(
                        k_runs     = params.k_runs     + dk,
                        min_runs   = params.min_runs   + dm,
                        threshold  = params.threshold  + dt,
                        regime     = params.regime,
                        step_k     = params.step_k,
                        step_min_r = params.step_min_r,
                        step_thresh= params.step_thresh,
                    ).validate()
                    neighbours.append(n)
        return neighbours

    @staticmethod
    def _random_init(regime: str) -> MetaParams:
        """
        Uninformed random initialization.
        Wide ranges — the optimizer discovers what works, not us.
        """
        k     = random.randint(3, 50)
        min_r = random.randint(2, k)
        thresh = round(random.uniform(0.40, 0.95), 2)
        return MetaParams(
            k_runs     = k,
            min_runs   = min_r,
            threshold  = thresh,
            regime     = regime,
            step_k     = random.randint(5, 12),
            step_min_r = random.randint(1, 3),
            step_thresh= round(random.uniform(0.05, 0.12), 2),
        ).validate()

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_optimizer_state (
                    regime      TEXT PRIMARY KEY,
                    params_json TEXT NOT NULL,
                    history_json TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_from_db(self) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT regime, params_json, history_json "
                    "FROM meta_optimizer_state"
                ).fetchall()
            for regime, params_json, history_json in rows:
                p = json.loads(params_json)
                self._state[regime] = {
                    "params": MetaParams(
                        k_runs      = p["k_runs"],
                        min_runs    = p["min_runs"],
                        threshold   = p["threshold"],
                        regime      = p["regime"],
                        step_k      = p.get("step_k",      5),
                        step_min_r  = p.get("step_min_r",  1),
                        step_thresh = p.get("step_thresh",  0.05),
                    ).validate(),
                    "history": json.loads(history_json),
                }
            logger.info(
                "MetaOptimizer: loaded %d regime states from DB",
                len(self._state),
            )
        except Exception as exc:
            logger.warning("MetaOptimizer: could not load from DB: %s", exc)

    def _persist(self, regime: str) -> None:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                bucket = self._state[regime]
                params_json  = json.dumps(bucket["params"].to_dict())
                history_json = json.dumps(bucket["history"][-200:])
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO meta_optimizer_state "
                    "(regime, params_json, history_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (regime, params_json, history_json, ts),
                )
                conn.commit()
        except Exception as exc:
            logger.error("MetaOptimizer: persist failed: %s", exc)
