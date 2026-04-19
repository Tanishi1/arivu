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
from typing import TYPE_CHECKING

from core.schemas import CausalState, SimulatorResult

if TYPE_CHECKING:
    from strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

MAX_HILL_CLIMB_ITERATIONS = 20


# ---------------------------------------------------------------------------
# Stage 1 — Heuristic functions
# ---------------------------------------------------------------------------

def _h_ema_crossover(state: CausalState) -> float:
    """Heuristic for EMA Crossover (momentum) strategy.

    Scores high when: strong trend, low volatility, decent volume.
    """
    vol_norm = min(1.0, state.volatility / 0.1)
    vol_pct = min(1.0, state.volume / 100_000)
    return (
        state.trend_strength * 0.5
        + (1 - vol_norm) * 0.3
        + vol_pct * 0.2
    )


def _h_bollinger(state: CausalState) -> float:
    """Heuristic for Bollinger Band Reversion (mean-reversion) strategy.

    Scores high when: weak trend, moderate volatility, normal spread.
    """
    vol_norm = min(1.0, state.volatility / 0.1)
    spread_norm = min(1.0, state.spread / 0.005)
    return (
        (1 - state.trend_strength) * 0.5
        + vol_norm * 0.3
        + spread_norm * 0.2
    )


def _h_rsi_divergence(state: CausalState) -> float:
    """Heuristic for RSI Divergence strategy.

    Conservative — always viable but weighted lower in strong trending/volatile conditions.
    """
    vol_penalty = min(1.0, state.volatility / 0.05)
    return max(0.1, 0.5 - vol_penalty * 0.2 - state.trend_strength * 0.1)


HEURISTICS = {
    "EMAStrategy": _h_ema_crossover,
    "BollingerStrategy": _h_bollinger,
    "RSIStrategy": _h_rsi_divergence,
}


# ---------------------------------------------------------------------------
# Stage 2 — Custom Hill-Climbing (~30 lines, see ADR-002)
# ---------------------------------------------------------------------------

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
        regime_classifier=None,  # sklearn DecisionTreeClassifier or None (bootstrap)
    ) -> None:
        self._strategies = {s.__class__.__name__: s for s in strategies}
        self._regime_classifier = regime_classifier

    def run(self, state: CausalState) -> SimulatorResult:
        """Run Stage 1 then Stage 2 and return the full SimulatorResult."""

        # Stage 1 — best-first search
        scores: dict[str, float] = {}
        for name, strategy in self._strategies.items():
            h_fn = HEURISTICS.get(name)
            base_score = h_fn(state) if h_fn else 0.5

            # Decision Tree optionally re-weights heuristics by regime
            if self._regime_classifier is not None:
                try:
                    features = [[
                        state.volatility, state.spread,
                        state.trend_strength, state.volume,
                    ]]
                    regime_proba = self._regime_classifier.predict_proba(features)[0]
                    # Weight applied per regime — expand as needed
                    base_score *= float(max(regime_proba))
                except Exception:  # noqa: BLE001
                    pass  # fall back to raw heuristic

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
