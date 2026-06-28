"""core/decision_utils.py
Shared utility functions for proximity annotation.

S3-3 DESIGN DECISION: This module is the single source of truth for proximity
computation. Strategies' get_assumptions() return proximity=0.0 (placeholder).
Main.py calls annotate_proximity() before ml2.annotate() and before ledger commit.

Rationale: Proximity computation lives in ONE place so the H4 special case
(SLOPE_NORMALISER for threshold=0.0 EMA trend_persistence) cannot diverge between
what is committed to the ledger and what ML2 trains on. Tests can import this
function directly without depending on the full main.py orchestration layer.
"""

from __future__ import annotations

from core.schemas import Assumption, CausalState
from core.constants import SLOPE_NORMALISER


def annotate_proximity(
    assumptions: list[Assumption],
    state: CausalState,
) -> list[Assumption]:
    """Compute current_value and proximity for each assumption from live state.

    Called by decision_cycle_loop BEFORE ml2.annotate() to ensure ML2
    bootstrap mode sees real proximity values, not the Pydantic defaults (0.0).
    ML2 bootstrap mode returns breach_risk = assumption.proximity — so a 0.0
    proximity would corrupt all bootstrap training data.

    H4 FIX: threshold=0.0 with operator='gt' (EMA trend_persistence) uses
    SLOPE_NORMALISER for proximity so:
      - slope=0.01 (= SLOPE_NORMALISER, strong trend)  → proximity ≈ 0.0 (safe)
      - slope=0.001 (10% of SLOPE_NORMALISER, weak)    → proximity ≈ 0.9 (at risk)

    Args:
        assumptions: list from SimulatorResult.assumptions (proximity=0.0 placeholders)
        state: current CausalState snapshot at decision time

    Returns:
        New list with current_value and proximity populated. Assumption is frozen —
        returns model_copy() of each, does not mutate.
    """
    annotated: list[Assumption] = []
    for a in assumptions:
        current_val = getattr(state, a.variable, 0.0)

        if a.threshold == 0 and a.operator == "gt":
            # H4 FIX: operator='gt', threshold=0.0 (EMA trend_persistence).
            # Proximity normalised by SLOPE_NORMALISER.
            prox = max(0.0, 1.0 - min(1.0, current_val / SLOPE_NORMALISER))
        elif a.threshold == 0:
            prox = 1.0
        elif a.operator == "gt":
            # B11 FIX: for 'gt' assumptions the breach point is current_val <= threshold.
            # Safe = current_val >> threshold (prox near 0).
            # At risk = current_val ≈ threshold (prox near 1).
            # Formula: threshold / current_val  (safe when current >> threshold).
            # Guard: if current_val <= 0, we're already at or below breach → prox=1.0.
            prox = (a.threshold / current_val) if current_val > 0 else 1.0
        else:
            # operator='lt': breach when current_val >= threshold.
            # prox = current_val / threshold (approaching from below).
            prox = current_val / a.threshold

        prox = round(min(1.0, max(0.0, prox)), 6)
        annotated.append(a.model_copy(update={
            "current_value": current_val,
            "proximity": prox,
        }))
    return annotated
