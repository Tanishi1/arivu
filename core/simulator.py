"""shared/simulator.py
Two-stage simulator: strategy selection + parameter tuning.

Stage 1 — Best-First Search (Unit II + III)
    Scores all three strategies using heuristic functions.
    Heuristic weights are informed by the Decision Tree regime classifier.
    Highest-scoring strategy is selected first.

Stage 2 — Hill-Climbing (Unit II)
    Custom ~30-line implementation (NOT scipy — see DECISIONS.md ADR-002).
    Nudges one parameter at a time from defaults, keeps improvements,
    discards regressions, stops at local optimum.
    Logs hill_climb_iterations as Research Metric 3.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from core.schemas import CausalState, SimulatorResult

if TYPE_CHECKING:
    from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

MAX_HILL_CLIMB_ITERATIONS = 20
# Random restart: if the default-params score is below this, try alternative
# starting points to escape poor regions of the parameter space.
# Without restarts, a flat objective landscape always produces iterations=1,
# making Research Metric 3 uninformative.
HILL_CLIMB_RESTART_THRESHOLD = 0.15
HILL_CLIMB_RANDOM_RESTARTS = 2


# ---------------------------------------------------------------------------
# Stage 1 — Heuristic functions
# ---------------------------------------------------------------------------

def _h_ema_crossover(state: CausalState) -> float:
    """Heuristic for EMA Crossover (momentum) strategy.

    Scores high when: strong trend, low volatility, tight spread.

    HIGH-2 FIX: 3rd factor changed from volume to (1-spread_norm).
    EMA's 3rd assumption is spread_constraint (spread < 0.005). The heuristic
    must penalise the same signal the assumption monitors.

    HIGH-3 FIX: vol_norm threshold changed from 0.1 to 0.07 (SOLUSDT limit).
    Previously EMA was 30% less penalised than RSI at SOLUSDT's actual vol limit.
    """
    vol_norm    = min(1.0, state.volatility / 0.07)   # HIGH-3: was 0.1
    spread_norm = min(1.0, state.spread / 0.005)       # HIGH-2: aligned with spread_constraint
    return (
        state.trend_strength * 0.5
        + (1 - vol_norm) * 0.3
        + (1 - spread_norm) * 0.2                      # HIGH-2: was vol_pct (volume/100_000)
    )


def _h_bollinger(state: CausalState) -> float:
    """Heuristic for Bollinger Band Reversion (mean-reversion) strategy.

    Scores high when: weak trend, elevated volatility (wider bands = better entry),
    normal spread.

    HIGH-3 FIX: vol_norm threshold changed from 0.1 to 0.07 (SOLUSDT limit).
    """
    vol_norm    = min(1.0, state.volatility / 0.07)   # HIGH-3: was 0.1
    spread_norm = min(1.0, state.spread / 0.005)
    return (
        (1 - state.trend_strength) * 0.5
        + vol_norm * 0.3
        + spread_norm * 0.2
    )


def _h_rsi_divergence(state: CausalState) -> float:
    """Heuristic for RSI Divergence strategy — the uncertainty hedge.

    RSI divergence is most valuable when the market is neither clearly
    trending (EMA's domain) nor clearly ranging/calm (Bollinger's domain).
    Score peaks in ambiguous conditions: moderate trend + above-normal volatility.

    Formula: RSI scores as (1 - average of EMA-comfort and Bollinger-comfort).
    When both momentum and mean-reversion strategies are uncomfortable, RSI wins.
    Uses SOLUSDT vol scale (0.07) — not the legacy BTC threshold (0.05).
    """
    vol_norm   = min(1.0, state.volatility / 0.07)   # SOLUSDT volatility_limit
    trend_norm = state.trend_strength

    ema_comfort = trend_norm * 0.5 + (1.0 - vol_norm) * 0.3
    bol_comfort = (1.0 - trend_norm) * 0.5 + vol_norm * 0.3

    uncertainty = 1.0 - (ema_comfort + bol_comfort) / 2.0
    # vol bonus: RSI divergence patterns are more structurally significant
    # in elevated-volatility markets (wider swings → cleaner extremes)
    return max(0.1, min(1.0, uncertainty + vol_norm * 0.2))


HEURISTICS = {
    "EMAStrategy": _h_ema_crossover,
    "BollingerStrategy": _h_bollinger,
    "RSIStrategy": _h_rsi_divergence,
}


# ---------------------------------------------------------------------------
# Stage 2 — Custom Hill-Climbing (~30 lines, see ADR-002)
# ---------------------------------------------------------------------------

def _random_params(bounds: dict) -> dict:
    """Generate a random parameter dict within all strategy bounds.

    Snaps each sampled value to the nearest legal step so the resulting dict
    is always valid input for strategy.evaluate().
    """
    params = {}
    for name, bound in bounds.items():
        # S1 FIX: Guard against step=0 to prevent ZeroDivisionError.
        # No current strategy has step=0, but there is no contract preventing it.
        # A team member adding a new strategy with a continuous parameter could
        # accidentally trigger this and crash the decision cycle.
        if bound.get("step", 0) <= 0:
            params[name] = bound["min"]
            continue
        steps = int(round((bound["max"] - bound["min"]) / bound["step"]))
        step_idx = random.randint(0, steps) if steps > 0 else 0
        params[name] = round(bound["min"] + step_idx * bound["step"], 8)
        params[name] = max(bound["min"], min(bound["max"], params[name]))
    return params


def _hill_climb(
    strategy: "BaseStrategy",
    state: CausalState,
    max_iterations: int = MAX_HILL_CLIMB_ITERATIONS,
) -> tuple[dict, int]:
    """Search the strategy's parameter space for a local optimum.

    Returns: (best_params, iterations_taken)
    Logs iterations for Research Metric 3.
    If defaults are already locally optimal, returns defaults after 1 iteration.
    """
    bounds = strategy.get_parameter_bounds()
    current_params = strategy.get_default_params()
    current_score = strategy.evaluate(state, current_params)

    # Random restarts — if defaults score poorly, sample alternative starting
    # points before committing to the hill-climb scan. This prevents the search
    # from getting stuck in a flat or suboptimal region of the parameter space
    # and ensures Research Metric 3 reflects real convergence dynamics.
    if current_score < HILL_CLIMB_RESTART_THRESHOLD:
        for attempt in range(HILL_CLIMB_RANDOM_RESTARTS):
            candidate = _random_params(bounds)
            score = strategy.evaluate(state, candidate)
            if score > current_score:
                old_score = current_score  # S2 FIX: capture BEFORE overwriting
                current_params = candidate
                current_score = score
                logger.debug(
                    "Hill-climb random restart %d improved score | %.4f → %.4f",
                    attempt + 1, old_score, current_score,
                )

    iterations = 0
    improved = True

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        for param_name, bound in bounds.items():
            step = bound["step"]
            for delta in (step, -step):
                candidate = dict(current_params)
                new_val = candidate[param_name] + delta

                # Respect parameter bounds
                if not (bound["min"] <= new_val <= bound["max"]):
                    continue

                candidate[param_name] = new_val
                score = strategy.evaluate(state, candidate)

                if score > current_score:
                    current_params = candidate
                    current_score = score
                    improved = True
                    break  # restart scan from new position
            if improved:
                break

    logger.info(
        "Stage 2 complete | strategy=%s iterations=%d score=%.4f",
        strategy.__class__.__name__, iterations, current_score,
    )
    return current_params, iterations


# ---------------------------------------------------------------------------
# Simulator entry point
# ---------------------------------------------------------------------------

class Simulator:
    """Runs both stages and produces a SimulatorResult.

    Usage:
        sim = Simulator(strategies=[ema, bollinger, rsi], regime_classifier=dt)
        result = sim.run(state)
    """

    def __init__(
        self,
        strategies: list["BaseStrategy"],
        regime_classifier=None,  # ml.regime.RegimeClassifier or None
    ) -> None:
        self._strategies = {s.__class__.__name__: s for s in strategies}
        self._regime_classifier = regime_classifier

    def run(self, state: CausalState) -> SimulatorResult:
        """Run Stage 1 then Stage 2 and return the full SimulatorResult."""

        # Stage 1 — best-first search
        scores: dict[str, float] = {}
        # N31 FIX: Get the dominant regime string explicitly to map to strategies
        dominant_regime = "bootstrap"
        if self._regime_classifier is not None:
            try:
                # predict returns the string label ("trending", "volatile", "calm")
                dominant_regime = self._regime_classifier.classify(
                    state.volatility, state.spread, state.trend_strength, state.volume,
                    trend_slope=state.trend_slope,  # HIGH-4: needed for bootstrap downtrend guard
                )
            except Exception:
                pass

        # Map regime to the strategy best suited for it
        regime_boost_map = {
            "trending": "EMAStrategy",
            "volatile": "RSIStrategy",
            "calm": "BollingerStrategy"
        }

        for name, strategy in self._strategies.items():
            h_fn = HEURISTICS.get(name)
            base_score = h_fn(state) if h_fn else 0.5

            # N31 FIX: Additive regime nudge (+0.2, capped at 1.0) instead of
            # multiplicative 1.5×. A multiplicative boost on an already-high base
            # score makes the regime-aligned strategy win unconditionally, overriding
            # strong heuristic differentiation. Additive ensures a clear heuristic
            # winner still beats a regime-aligned but poor-fit strategy.
            if dominant_regime in regime_boost_map and regime_boost_map[dominant_regime] == name:
                base_score = min(1.0, base_score + 0.2)  # additive nudge, capped

            scores[name] = base_score

        best_name = max(scores, key=scores.__getitem__)
        best_strategy = self._strategies[best_name]
        best_score = scores[best_name]

        logger.info(
            "Stage 1 selected | strategy=%s score=%.4f | all_scores=%s",
            best_name, best_score, scores,
        )

        # Stage 2 — hill-climbing on selected strategy
        tuned_params, iterations = _hill_climb(best_strategy, state)

        # Build assumptions from strategy
        assumptions = best_strategy.get_assumptions(state, tuned_params)

        # projected_pnl from evaluate() on tuned params
        projected_pnl = best_strategy.evaluate(state, tuned_params)

        return SimulatorResult(
            strategy_name=best_name,
            tuned_params=tuned_params,
            heuristic_score=best_score,
            hill_climb_iterations=iterations,
            projected_pnl=projected_pnl,
            confidence=best_score,
            assumptions=assumptions,
        )
