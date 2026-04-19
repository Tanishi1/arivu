"""strategies/base.py
Abstract base class for all trading strategies.

Every strategy MUST implement all 5 abstract methods below.
Read these docstrings before implementing your strategy.

The strategy interface is a plug-in. Phase 1 is complete with stub
implementations. A missing strategy means the simulator runs with 2
strategies, which is still valid (ADR — Risk Register).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.schemas import Assumption, CausalState


class BaseStrategy(ABC):
    """Abstract base class for Arivu trading strategies.

    All strategies must implement exactly these 5 methods. No more, no less.
    evaluate() MUST NOT have side effects — it is called repeatedly by Hill-climbing.
    """

    # ------------------------------------------------------------------
    # 1. Signal generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signal(self, state: "CausalState", params: dict) -> str:
        """Generate a trading signal based on market state.

        Args:
            state: current CausalState snapshot
            params: tuned parameters dict (from get_default_params or Hill-climbing)

        Returns:
            "BUY", "SELL", or "HOLD"

        Must NOT have side effects. Called repeatedly by the simulator.
        """

    # ------------------------------------------------------------------
    # 2. Assumption definition
    # ------------------------------------------------------------------

    @abstractmethod
    def get_assumptions(
        self, state: "CausalState", params: dict
    ) -> list["Assumption"]:
        """Return exactly 3 Assumption objects for this strategy + params.

        Assumptions define the causal conditions that must hold for this
        strategy's expected performance to be valid.

        Returns:
            List of exactly 3 Assumption objects. The ledger, monitor, and
            ML2 all depend on there being exactly 3. Do not return fewer.
        """

    # ------------------------------------------------------------------
    # 3. Default parameters
    # ------------------------------------------------------------------

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return the default parameter dictionary.

        Hill-climbing starts from these values. Defaults should be
        empirically reasonable, not arbitrary.

        Returns:
            dict with param_name → default_value
        """

    # ------------------------------------------------------------------
    # 4. Parameter bounds
    # ------------------------------------------------------------------

    @abstractmethod
    def get_parameter_bounds(self) -> dict:
        """Return bounds for each tunable parameter.

        Used by Hill-climbing Stage 2 to constrain the search space.

        Returns:
            dict with param_name → {"min": float, "max": float, "step": float}
        """

    # ------------------------------------------------------------------
    # 5. Evaluation function
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, state: "CausalState", params: dict) -> float:
        """Score a parameter configuration against the current market state.

        This is the objective function for Hill-climbing. Higher = better.

        MUST NOT have side effects — called many times per decision cycle.
        MUST return 0.0 if generate_signal returns "HOLD" (not worth tuning).
        MUST return a penalty (negative or near-zero) for clearly bad states.

        Args:
            state: current CausalState snapshot
            params: parameter dict to evaluate

        Returns:
            float score — higher is better
        """
