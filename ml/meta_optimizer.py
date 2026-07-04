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
    The operational parameters the optimizer controls.
    These replace the hardcoded constants in layer1_tracker.py (k, min_r, thresh)
    AND the hardcoded constants in causal_discovery.py (TAU_MAX, ALPHA).
    """
    k_runs:    int   # size of the rolling run history window
    min_runs:  int   # minimum runs observed before validation is eligible
    threshold: float # fraction of k_runs an edge must appear in to validate
    regime:    str   = "unknown"

    # PCMCI discovery hyperparameters — hill-climbed by MetaOptimizer
    # tau_max:     maximum lag in bars to search (range [2, 12])
    # pcmci_alpha: significance threshold for edge inclusion (range [0.01, 0.10])
    # These let the optimizer discover the real causal timescale for the asset.
    tau_max:     int   = 4
    pcmci_alpha: float = 0.05

    # Internal hill-climbing state — agent should not read these directly
    step_k:      int   = 5
    step_min_r:  int   = 1
    step_thresh: float = 0.05
    step_tau:    int   = 1    # step for tau_max hill-climbing
    step_alpha:  float = 0.01 # step for pcmci_alpha hill-climbing

    def to_dict(self) -> dict:
        """Serialise for ledger / DecisionObject meta_params field."""
        return {
            "k_runs":      self.k_runs,
            "min_runs":    self.min_runs,
            "threshold":   round(self.threshold, 4),
            "regime":      self.regime,
            "tau_max":     self.tau_max,
            "pcmci_alpha": round(self.pcmci_alpha, 4),
            "step_k":      self.step_k,
            "step_min_r":  self.step_min_r,
            "step_thresh": round(self.step_thresh, 4),
            "step_tau":    self.step_tau,
            "step_alpha":  round(self.step_alpha, 4),
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
        self.tau_max     = max(2, min(12, int(self.tau_max)))
        self.pcmci_alpha = max(0.005, min(0.15, float(self.pcmci_alpha)))
        self.step_tau    = max(1, int(self.step_tau))
        self.step_alpha  = max(0.005, float(self.step_alpha))
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
    is_escape_valve:    bool = False
    edge_stability:     float = 0.0
    avg_breach_risk:    float = 0.5
    ml2_calibration:    float = 0.5
    # Layer 2 trust of the chain AT TIME OF TRADE ENTRY.
    # This closes the feedback loop: MetaOptimizer sees whether the params
    # it selected produced hypotheses with high or low Bayesian trust.
    # Hill climbing then moves toward params that discover higher-trust edges.
    l2_trust:           float = 0.5
    l2_n_obs:           int   = 0


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
IMMUNITY_CYCLES = STAGNATION_LIMIT // 2  # cycles a newly injected candidate is protected
                                          # from eviction so it can accumulate fair history



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
        # Immunity counter: after a replacement fires, the new random candidate is
        # protected for IMMUNITY_CYCLES updates so it has time to accumulate history
        # and be scored fairly before it can itself become the eviction target.
        self._candidate_immunity: dict[str, int] = {}

        self._ensure_schema()
        self._load_from_db()

        # Initialise any regime not yet in DB with random params
        for regime in RegimeType:
            if regime.value not in self._state:
                self._state[regime.value] = {
                    "population": [self._random_init(regime.value) for _ in range(3)],
                    "history": [],   # list of scored outcomes for this regime
                }
                logger.info(
                    "MetaOptimizer: random init | regime=%s | 3 candidates created",
                    regime.value,
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
            bucket = self._state[key]
            if "history" not in bucket or not bucket["history"]:
                return bucket["population"][0]
            
            best_cand = None
            best_score = -1.0
            for cand in bucket["population"]:
                score = self._score_config(cand.to_dict(), bucket["history"])
                if score > best_score:
                    best_score = score
                    best_cand = cand
            return best_cand

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
            population = bucket.get("population", [])
            if not population and "params" in bucket:
                population = [bucket["params"]]
                bucket["population"] = population

            bucket["history"].append({
                "params": outcome.meta_params_used,
                "score":  score,
            })
            # Keep history bounded to last 200 outcomes
            if len(bucket["history"]) > 200:
                bucket["history"] = bucket["history"][-200:]

            # Tick down immunity counter once per trade cycle (outside candidate loop).
            self._candidate_immunity[regime] = max(
                0, self._candidate_immunity.get(regime, 0) - 1
            )

            candidate_scores = []
            for cand in population:
                cand_score = self._score_config(cand.to_dict(), bucket["history"])
                candidate_scores.append((cand_score, cand))

            candidate_scores.sort(key=lambda x: x[0], reverse=True)

            # Capture stagnant candidate's (idx=0) diagnostic state for the replacement
            # log — these are the step sizes that CAUSED stagnation, not the worst
            # candidate's info (which would tell us nothing about why we're replacing).
            _stagnant_step_k: int = 0
            _stagnant_step_thresh: float = 0.0
            _stagnant_score: float = 0.0

            new_population = []
            for idx, (c_score, current) in enumerate(candidate_scores):
                # Generate all neighbours (±step for each parameter)
                neighbours = self._get_neighbours(current)
                best_neighbour      = None
                best_neighbour_score = c_score

                for neighbour in neighbours:
                    n_dict = neighbour.to_dict()
                    n_score = self._score_config(n_dict, bucket["history"])
                    if n_score > best_neighbour_score:
                        best_neighbour       = neighbour
                        best_neighbour_score = n_score

                if best_neighbour is not None:
                    # Improvement found — move to neighbour
                    if idx == 0:
                        self._stagnation_counts[regime] = 0
                        self._last_scores[regime] = best_neighbour_score

                        old_dict = current.to_dict()
                        new_dict = best_neighbour.to_dict()
                        logger.info(
                            "MetaOptimizer: Hill-climbing regime '%s' (score %.4f -> %.4f) | "
                            "threshold: %.4f -> %.4f | k_runs: %d -> %d | min_runs: %d -> %d",
                            regime, c_score, best_neighbour_score,
                            old_dict["threshold"], new_dict["threshold"],
                            old_dict["k_runs"], new_dict["k_runs"],
                            old_dict["min_runs"], new_dict["min_runs"]
                        )
                    new_population.append(best_neighbour)
                else:
                    # No improvement — shrink step sizes (converging)
                    old_k = current.step_k
                    old_thresh = current.step_thresh
                    current.step_k      = max(MIN_STEP_K, int(current.step_k * STEP_DECAY_RATE))
                    current.step_thresh = max(MIN_STEP_THRESH, current.step_thresh * STEP_DECAY_RATE)

                    if idx == 0:
                        logger.info(
                            "MetaOptimizer: Local optimum reached for regime '%s' (score: %.4f) | "
                            "Shrinking step sizes: step_k: %d -> %d | step_thresh: %.4f -> %.4f",
                            regime, c_score, old_k, current.step_k, old_thresh, current.step_thresh
                        )
                        if current.step_k <= MIN_STEP_K and current.step_thresh <= MIN_STEP_THRESH:
                            self._stagnation_counts[regime] = self._stagnation_counts.get(regime, 0) + 1
                            # Record stagnant candidate's state for the replacement log
                            _stagnant_step_k    = current.step_k
                            _stagnant_step_thresh = current.step_thresh
                            _stagnant_score     = c_score
                        else:
                            self._stagnation_counts[regime] = 0
                        self._last_scores[regime] = c_score

                    stagnation_count = self._stagnation_counts.get(regime, 0)
                    immunity         = self._candidate_immunity.get(regime, 0)
                    is_last          = idx == len(candidate_scores) - 1

                    if is_last and stagnation_count >= STAGNATION_LIMIT and immunity == 0:
                        # Replace the worst candidate with a fresh random to re-inject
                        # diversity.  Log the STAGNANT candidate's (idx=0) step sizes —
                        # those explain WHY replacement fired, not the worst candidate's
                        # score.  Grant immunity so the new random isn't immediately
                        # evicted before it has a chance to accumulate scoring history.
                        new_params = self._random_init(regime)
                        new_population.append(new_params)
                        self._stagnation_counts[regime] = 0
                        self._candidate_immunity[regime] = IMMUNITY_CYCLES
                        logger.warning(
                            "MetaOptimizer: REPLACING worst candidate | regime=%s | "
                            "stagnant_candidate: score=%.4f step_k=%d step_thresh=%.4f "
                            "stuck for %d updates | immunity granted for %d cycles",
                            regime,
                            _stagnant_score, _stagnant_step_k, _stagnant_step_thresh,
                            stagnation_count, IMMUNITY_CYCLES,
                        )

                    elif is_last and stagnation_count >= STAGNATION_LIMIT and immunity > 0:
                        # Stagnation limit reached but the recently injected candidate
                        # is still in its immunity window — wait until it has accumulated
                        # enough history to be scored fairly before replacing again.
                        logger.info(
                            "MetaOptimizer: Stagnation limit reached — replacement suppressed | "
                            "regime=%s | immunity=%d cycles remaining | "
                            "stagnant_candidate: score=%.4f step_k=%d step_thresh=%.4f",
                            regime, immunity,
                            _stagnant_score, _stagnant_step_k, _stagnant_step_thresh,
                        )
                        new_population.append(current)

                    else:
                        new_population.append(current)

            bucket["population"] = new_population

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
          - ml2_calibration: how well did ML2 predict breaches?
          - l2_trust_quality: was the chain well-trusted by Layer 2 at entry?
            This is the key signal: MetaOptimizer hill-climbs toward Layer 1
            params that produce high-trust hypotheses, which indirectly filters
            spurious edges without any hard-coded trust threshold.
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

        # ML2 calibration reward
        ml2_bonus = outcome.ml2_calibration

        # Layer 2 trust quality signal.
        # l2_trust=0.5 (untested) → neutral contribution 0.5
        # l2_trust=0.3 (proven loser) → pulls score down → optimizer avoids these params
        # l2_trust=0.7 (proven winner) → pulls score up → optimizer seeks these params
        # Scaled so 0.5 trust = 0.5 contribution (neutral), linear above/below.
        # l2_n_obs=0 → fully neutral (no data yet, don't penalise bootstrap trades)
        if outcome.l2_n_obs > 0:
            l2_quality = outcome.l2_trust
        else:
            l2_quality = 0.5  # untested chain: neutral, don't penalise exploration

        # Penalty for high breach risk on a losing trade
        risk_penalty = 1.0
        if outcome.avg_breach_risk > 0.70 and outcome.pnl_usd < 0:
            risk_penalty = 0.6

        # Composite — weights sum to 1.0
        # l2_quality added at 10%, ml2_bonus reduced from 15%→10%, chain_bonus 10%→5%
        raw = (
            direction_score * 0.35 +
            accuracy        * 0.25 +
            regime_purity   * 0.15 +
            chain_bonus     * 0.05 +
            ml2_bonus       * 0.10 +
            l2_quality      * 0.10
        ) * risk_penalty
        return round(raw, 6)

    def _score_config(
        self,
        config_dict: dict,
        history: list[dict],
    ) -> float:
        """
        Score a parameter configuration against outcome history using a Gaussian Kernel.

        Operates in a 5-dimensional space: k_runs, min_runs, threshold, tau_max, pcmci_alpha.
        Historical records without tau_max/alpha (old format) default to module constants
        so they score correctly against outcomes produced under those parameters.

        Each historical trade outcome is weighted by its normalized distance to the
        target configuration, and the final score is the weighted average combined
        with a Bayesian prior (alpha_prior = 1.0 at score = 0.5) to handle low-data areas.
        """
        if not history:
            return 0.5

        # Normalization bandwidths — controls how quickly kernel weight drops off
        sigma_k      = 5.0    # k_runs: 5-run difference halves the weight
        sigma_min_r  = 2.0    # min_runs: 2-run difference halves the weight
        sigma_thresh = 0.05   # threshold: 5% difference halves the weight
        sigma_tau    = 2.0    # tau_max: 2-bar difference halves the weight
        sigma_alpha  = 0.02   # pcmci_alpha: 0.02 difference halves the weight

        # Bayesian prior weight (equivalent to 1 observation at score 0.5)
        alpha_prior = 1.0

        target_k      = float(config_dict.get("k_runs",      0))
        target_min_r  = float(config_dict.get("min_runs",    0))
        target_thresh = float(config_dict.get("threshold",   0.0))
        target_tau    = float(config_dict.get("tau_max",     4))   # default = module constant
        target_palpha = float(config_dict.get("pcmci_alpha", 0.05)) # default = module constant

        weighted_sum = 0.0
        weight_total = 0.0

        for h in history:
            p = h["params"]
            s = h["score"]

            is_ev = p.get("is_escape_valve", False)
            edge_stability = float(p.get("edge_stability", 0.0))

            h_thresh = edge_stability if is_ev else float(p.get("threshold", 0.0))

            if is_ev and target_thresh > edge_stability:
                effective_score = 0.25
            else:
                effective_score = s

            # 5-dimensional kernel: all dimensions contribute to distance
            diff_k      = (target_k      - float(p.get("k_runs",      0)))    / sigma_k
            diff_min_r  = (target_min_r  - float(p.get("min_runs",    0)))    / sigma_min_r
            diff_thresh = (target_thresh - h_thresh)                           / sigma_thresh
            diff_tau    = (target_tau    - float(p.get("tau_max",     4)))    / sigma_tau
            diff_alpha  = (target_palpha - float(p.get("pcmci_alpha", 0.05))) / sigma_alpha

            d_squared = (
                diff_k      * diff_k +
                diff_min_r  * diff_min_r +
                diff_thresh * diff_thresh +
                diff_tau    * diff_tau +
                diff_alpha  * diff_alpha
            )
            weight = math.exp(-d_squared / 2.0)

            weighted_sum += weight * effective_score
            weight_total += weight

        score = (weighted_sum + alpha_prior * 0.5) / (weight_total + alpha_prior)
        return round(score, 6)

    # ─────────────────────────────────────────────────────────────────────
    # Hill-climbing helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_neighbours(self, params: MetaParams) -> list[MetaParams]:
        """Generate neighbours for hill-climbing.

        Strategy:
          - Cross-product over (k_runs, min_runs, threshold) — 26 combos as before.
          - Independent +/- steps for tau_max and pcmci_alpha (4 additional combos).
          - Total: 30 neighbours. Avoids the 3^5=243 explosion of a full cross-product.

        tau and alpha are perturbed independently because they are independent
        dimensions of the discovery space, not correlated with k/m/thresh.
        """
        neighbours = []
        # Existing 3-param cross-product (k_runs, min_runs, threshold)
        for dk in (-params.step_k, 0, params.step_k):
            for dm in (-params.step_min_r, 0, params.step_min_r):
                for dt in (-params.step_thresh, 0, params.step_thresh):
                    if dk == 0 and dm == 0 and dt == 0:
                        continue
                    n = MetaParams(
                        k_runs      = params.k_runs      + dk,
                        min_runs    = params.min_runs    + dm,
                        threshold   = params.threshold   + dt,
                        regime      = params.regime,
                        tau_max     = params.tau_max,
                        pcmci_alpha = params.pcmci_alpha,
                        step_k      = params.step_k,
                        step_min_r  = params.step_min_r,
                        step_thresh = params.step_thresh,
                        step_tau    = params.step_tau,
                        step_alpha  = params.step_alpha,
                    ).validate()
                    neighbours.append(n)
        # Independent tau_max perturbations
        for dtau in (-params.step_tau, params.step_tau):
            n = MetaParams(
                k_runs      = params.k_runs,
                min_runs    = params.min_runs,
                threshold   = params.threshold,
                regime      = params.regime,
                tau_max     = params.tau_max + dtau,
                pcmci_alpha = params.pcmci_alpha,
                step_k      = params.step_k,
                step_min_r  = params.step_min_r,
                step_thresh = params.step_thresh,
                step_tau    = params.step_tau,
                step_alpha  = params.step_alpha,
            ).validate()
            neighbours.append(n)
        # Independent pcmci_alpha perturbations
        for da in (-params.step_alpha, params.step_alpha):
            n = MetaParams(
                k_runs      = params.k_runs,
                min_runs    = params.min_runs,
                threshold   = params.threshold,
                regime      = params.regime,
                tau_max     = params.tau_max,
                pcmci_alpha = params.pcmci_alpha + da,
                step_k      = params.step_k,
                step_min_r  = params.step_min_r,
                step_thresh = params.step_thresh,
                step_tau    = params.step_tau,
                step_alpha  = params.step_alpha,
            ).validate()
            neighbours.append(n)
        return neighbours

    @staticmethod
    def _random_init(regime: str) -> MetaParams:
        """
        Uninformed random initialization.
        Wide ranges — the optimizer discovers what works, not us.
        Includes tau_max and pcmci_alpha so the discovery timescale
        is also searched from scratch.
        """
        k     = random.randint(3, 50)
        min_r = random.randint(2, k)
        thresh = round(random.uniform(0.40, 0.95), 2)
        tau   = random.randint(2, 10)
        alpha = round(random.uniform(0.01, 0.10), 3)
        return MetaParams(
            k_runs      = k,
            min_runs    = min_r,
            threshold   = thresh,
            regime      = regime,
            tau_max     = tau,
            pcmci_alpha = alpha,
            step_k      = random.randint(5, 12),
            step_min_r  = random.randint(1, 3),
            step_thresh = round(random.uniform(0.05, 0.12), 2),
            step_tau    = 1,
            step_alpha  = round(random.uniform(0.005, 0.02), 3),
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
                p_list = json.loads(params_json)
                if isinstance(p_list, dict):
                    p_list = [p_list]
                
                population = []
                for p in p_list:
                    population.append(MetaParams(
                        k_runs      = p["k_runs"],
                        min_runs    = p["min_runs"],
                        threshold   = p["threshold"],
                        regime      = p["regime"],
                        # New fields — default to module constants for old records
                        tau_max     = p.get("tau_max",     4),
                        pcmci_alpha = p.get("pcmci_alpha", 0.05),
                        step_k      = p.get("step_k",      5),
                        step_min_r  = p.get("step_min_r",  1),
                        step_thresh = p.get("step_thresh", 0.05),
                        step_tau    = p.get("step_tau",    1),
                        step_alpha  = p.get("step_alpha",  0.01),
                    ).validate())
                    
                self._state[regime] = {
                    "population": population,
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
                population_dicts = [p.to_dict() for p in bucket.get("population", [])]
                params_json  = json.dumps(population_dicts)
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
