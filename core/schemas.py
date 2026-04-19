"""shared/schemas.py
All 6 pydantic contracts for Arivu.

These schemas are the single source of truth for all inter-component data.
Every team member MUST read this file before touching any other file.
Pydantic v2 enforces all types at every component boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Contract 1 — MarketTick
# ---------------------------------------------------------------------------

class MarketTick(BaseModel):
    """A single parsed message from the Binance WebSocket stream.

    Produced by: shared/stream.py
    Consumed by: shared/causal_state.py
    """

    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    spread: float  # ask - bid, computed on parse


# ---------------------------------------------------------------------------
# Contract 2 — Assumption
# (embedded inside DecisionObject — not transmitted separately)
# ---------------------------------------------------------------------------

class Assumption(BaseModel):
    """A single causal assumption attached to a strategy decision.

    name:       human-readable identifier  e.g. 'volatility_control'
    variable:   the CausalState field to check  e.g. 'volatility'
    operator:   'gt' (variable must be > threshold) or 'lt' (< threshold)
    threshold:  the boundary value
    proximity:  current_value / threshold  — filled by monitor at check time
    breach_risk: P(breach) from ML2  — filled before ledger commit
    """

    name: str
    variable: str
    operator: Literal["gt", "lt"]
    threshold: float
    proximity: float = 0.0
    breach_risk: float = 0.0  # annotated by ML2 pre-commit


# ---------------------------------------------------------------------------
# Contract 3 — CausalState
# ---------------------------------------------------------------------------

class CausalState(BaseModel):
    """The live internal model of market + algorithm state.

    Single instance, lives for the entire run, updated on every tick.

    IMPORTANT:
    - Only market_feed_loop may WRITE to this object.
    - decision_cycle_loop and assumption_monitor_loop only READ it.
    - Writes must be protected with asyncio.Lock.
    - algo_health_vector is the ML1 output: [p_normal, p_stressed, p_degraded].
      It is NOT a one-time snapshot — ML1 runs continuously in the background
      and the latest vector is captured at the moment of each ledger commit.
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    price: float = 0.0
    volatility: float = 0.0         # rolling std of returns
    spread: float = 0.0             # ask - bid (propagated from market)
    trend_slope: float = 0.0        # slope of price over last N periods
    trend_strength: float = 0.0     # normalised trend magnitude [0, 1]
    volume: float = 0.0             # rolling average volume

    # ML1 output — updated continuously in background, captured at commit time
    # [p_normal, p_stressed, p_degraded] — all three values, not a single label
    algo_health_vector: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0])

    active_strategy: str = "none"
    position_size: float = 0.0
    capital_deployed: float = 0.0

    # Stale feed detection — monitor pauses if this is > 10s ago
    last_tick_timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Contract 4 — ExecutionTelemetry
# ---------------------------------------------------------------------------

class ExecutionTelemetry(BaseModel):
    """Emitted by the executor after every order attempt.

    Produced by: execution/executor.py
    Consumed by: ml/ml1.py (for ML1 retraining + inference)
    """

    fill_rate: float                # fills / orders placed in last 10 orders
    order_latency_ms: float         # time from order submit to fill
    slippage: float                 # intended_price − actual_fill_price
    position_size_deviation: float  # actual_size / intended_size − 1
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Contract 5 — SimulatorResult
# ---------------------------------------------------------------------------

class SimulatorResult(BaseModel):
    """Output of the two-stage simulator.

    Produced by: shared/simulator.py
    Consumed by: shared/comparator.py, ledger/writer.py
    hill_climb_iterations is logged per cycle for Research Metric 3.
    """

    strategy_name: str
    tuned_params: dict
    heuristic_score: float
    hill_climb_iterations: int      # Research Metric 3: must be > 0
    projected_pnl: float
    confidence: float
    assumptions: list[Assumption]


# ---------------------------------------------------------------------------
# Contract 6 — DecisionObject
# ---------------------------------------------------------------------------

class DecisionObject(BaseModel):
    """The immutable record committed to the ledger BEFORE any actuation.

    IMMUTABILITY RULE:
    Once status=COMMITTED, only the 'status' field may change.
    All other fields are write-once. ledger/writer.py enforces this with
    a ValueError guard. This is the most important invariant in the system.

    phase: 'bootstrap' (before ML2 has been trained on real data)
           'trained'   (after ML2 beats the proximity heuristic baseline)
    The phase column must be present from cycle one. Do not add it later.
    """

    id: UUID = Field(default_factory=uuid4)
    timestamp_committed: datetime = Field(default_factory=datetime.utcnow)
    strategy_name: str
    tuned_params: dict
    market_state_snapshot: dict     # snapshot of CausalState at commit time
    algo_health_vector: list[float] # ML1 vector captured at commit time
    assumptions: list[Assumption]
    projected_pnl: float
    confidence: float
    status: Literal[
        "COMMITTED", "ACTIVE", "CLOSED", "INTERRUPTED", "EXECUTION_FAILED"
    ] = "COMMITTED"
    phase: Literal["bootstrap", "trained"] = "bootstrap"


# ---------------------------------------------------------------------------
# Contract 6b — OutcomeRecord
# ---------------------------------------------------------------------------

class OutcomeRecord(BaseModel):
    """Created when a DecisionObject moves to CLOSED. One per DecisionObject.

    outcome_delta = projected_pnl - actual_pnl  (Research Metric 1)
    hill_climb_iterations carried over from the SimulatorResult (Metric 3)
    Both assumptions_held and assumptions_breached must be fully populated —
    missing labels break ML2 training.
    """

    id: UUID = Field(default_factory=uuid4)
    decision_object_id: UUID
    timestamp_closed: datetime = Field(default_factory=datetime.utcnow)
    actual_pnl: float
    outcome_delta: float                    # projected − actual
    assumptions_held: list[str]             # assumption names that held
    assumptions_breached: list[str]         # assumption names that breached
    breach_timestamps: dict                 # {assumption_name: iso_timestamp}
    hill_climb_iterations: int
