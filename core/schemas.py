"""core/schemas.py
All 6 pydantic contracts for Arivu.

These schemas are the single source of truth for all inter-component data.
Every team member MUST read this file before touching any other file.
Pydantic v2 enforces all types at every component boundary.

Key design decisions enforced here:
  - DecisionObject is frozen (immutable in Python, not just in the DB).
  - algo_health_vector is validated to have exactly 3 elements summing to ~1.0.
  - Assumption.variable is constrained to actual CausalState field names.
  - All timestamps are timezone-aware (UTC). Never use datetime.utcnow().
  - OutcomeRecord carries phase so training queries never need a JOIN.
  - DecisionObject carries hill_climb_iterations so Metric 3 survives INTERRUPTED cycles.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Type alias — constrains Assumption.variable to real CausalState fields.
# Keep this in sync with VARIABLE_NAMES in core/feature_bar.py and the
# fields added to CausalState below.
# ---------------------------------------------------------------------------

CausalStateField = Literal[
    # --- tick-level fields (always populated) ---
    "price",
    "volatility",
    "spread",
    "trend_slope",
    "trend_strength",
    "volume",
    "rsi_current",
    "divergence_candle_span",
    # --- 10-second bar features (populated after first bar closes) ---
    "price_return",
    "rsi",                    # bar-level RSI (same Wilder method, bar granularity)
    "trade_intensity",
    "order_book_imbalance",
    "ema_spread",
    "bollinger_width",
    "price_in_band",
    "regime_volatile",
    "regime_trending",
    "btc_return",
    "eth_return",
    "algo_health_p_normal",
    "algo_health_p_stressed",
    "algo_health_p_degraded",
    "session_sin",
    "session_cos",
]


# ---------------------------------------------------------------------------
# Contract 1 — MarketTick
# ---------------------------------------------------------------------------

class MarketTick(BaseModel):
    """A single parsed message from the Binance WebSocket stream.

    Produced by: core/stream.py
    Consumed by: core/causal_state.py
    """
    model_config = ConfigDict(frozen=True)

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
    """A single causal assumption attached to a strategy decision."""
    model_config = ConfigDict(frozen=True)

    name: str
    variable: CausalStateField          # constrained — no silent typos
    operator: Literal["gt", "lt"]
    threshold: float
    current_value: float = 0.0
    proximity: float = 0.0
    breach_risk: float = 0.5           # annotated by ML2 pre-commit


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
    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    price: float = 0.0
    volatility: float = 0.0         # rolling std of returns
    spread: float = 0.0             # ask - bid (propagated from market)
    trend_slope: float = 0.0        # slope of price over last N periods
    trend_strength: float = 0.0     # normalised trend magnitude [0, 1]
    volume: float = 0.0             # rolling average volume

    # ML1 output — updated continuously in background, captured at commit time
    # [p_normal, p_stressed, p_degraded] — all three values, not a single label
    algo_health_vector: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0])

    # RSI current value — computed by causal_state.py from rolling price history.
    rsi_current: float = 50.0

    # Candles since last local price low/high — computed by causal_state.py.
    divergence_candle_span: float = 3.5

    # -----------------------------------------------------------------------
    # 10-second bar features — populated by FeatureBarBuilder after each bar
    # closes via CausalStateManager.update_graph_features().
    # Defaults are safe neutrals so the causal agent can build assumptions
    # even during the initial warm-up period (before the first bar closes).
    # -----------------------------------------------------------------------
    price_return: float = 0.0
    rsi: float = 50.0                    # bar-level RSI (same Wilder, bar granularity)
    trade_intensity: float = 0.0
    order_book_imbalance: float = 0.0
    ema_spread: float = 0.0
    bollinger_width: float = 0.0
    price_in_band: float = 0.5
    regime_volatile: float = 0.0
    regime_trending: float = 0.0
    btc_return: float = 0.0
    eth_return: float = 0.0
    algo_health_p_normal: float = 1.0
    algo_health_p_stressed: float = 0.0
    algo_health_p_degraded: float = 0.0
    session_sin: float = 0.0
    session_cos: float = 1.0

    active_strategy: str = "none"
    position_size: float = 0.0
    capital_deployed: float = 0.0
    price_history: list[float] = Field(default_factory=list)

    # Stale feed detection — monitor pauses if this is > 10s ago
    last_tick_timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Contract 4 — ExecutionTelemetry
# ---------------------------------------------------------------------------

class ExecutionTelemetry(BaseModel):
    """Emitted by the executor after every order attempt."""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    symbol: str
    fill_rate: float                # fills / orders placed in last 10 orders
    order_latency_ms: float         # time from order submit to fill
    slippage: float                 # intended_price − actual_fill_price
    position_size_deviation: float  # actual_size / intended_size − 1


# ---------------------------------------------------------------------------
# Contract 5 — SimulatorResult
# ---------------------------------------------------------------------------

class SimulatorResult(BaseModel):
    """Output of the two-stage simulator."""
    model_config = ConfigDict(frozen=True)

    strategy_name: str
    symbol: str = "SOLUSDT"
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
    Once committed, this object is frozen both in Python (model_config frozen=True)
    and in the database (only the 'status' SQL column may be updated via
    LedgerWriter.update_status()). Any attempt to mutate a field after
    creation raises a Pydantic ValidationError immediately.

    phase: 'bootstrap' (before ML2 has been trained on real data)
           'trained'   (after ML2 beats the proximity heuristic baseline)
    The phase column must be present from cycle one. Do not add it later.

    hill_climb_iterations: carried directly from SimulatorResult so that
    Research Metric 3 is preserved even if the cycle is INTERRUPTED before
    an OutcomeRecord is created.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    timestamp_committed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    strategy_name: str
    symbol: str = "SOLUSDT"
    tuned_params: dict
    market_state_snapshot: dict     # snapshot of CausalState at commit time
    algo_health_vector: list[float] # ML1 vector captured at commit time
    assumptions: list[Assumption]
    projected_pnl: float
    confidence: float
    hill_climb_iterations: int      # Research Metric 3 — preserved on INTERRUPTED
    status: Literal[
        "COMMITTED", "ACTIVE", "CLOSED", "INTERRUPTED", "EXECUTION_FAILED"
    ] = "COMMITTED"
    phase: Literal["bootstrap", "trained"] = "bootstrap"
    meta_params: Optional[dict] = None      
    causal_chain_snapshot: Optional[dict] = None  
    schema_version: int = 1

    @field_validator("algo_health_vector")
    @classmethod
    def validate_algo_health_vector(cls, v: list[float]) -> list[float]:
        """Enforce [p_normal, p_stressed, p_degraded] contract.

        Exactly 3 elements summing to ~1.0.
        A corrupt vector here corrupts ML2 feature set silently — prevent at source.
        """
        if len(v) != 3:
            raise ValueError(
                f"algo_health_vector must have exactly 3 elements "
                f"[p_normal, p_stressed, p_degraded], got {len(v)}"
            )
        total = sum(v)
        if abs(total - 1.0) > 0.02:
            raise ValueError(
                f"algo_health_vector must sum to ~1.0 (tolerance ±0.02), "
                f"got {total:.4f}. Values: {v}"
            )
        return v

    @field_validator("assumptions")
    @classmethod
    def validate_assumption_count(cls, v: list) -> list:
        """Enforce 1-3 assumptions per DecisionObject.

        The monitor, ML2 annotator, and training_buffer.csv all depend on
        assumptions per cycle.
        """
        if not (1 <= len(v) <= 3):
            raise ValueError(
                f"Each strategy must return 1-3 Assumption objects, got {len(v)}."
            )
        return v


# ---------------------------------------------------------------------------
# Contract 6b — OutcomeRecord
# ---------------------------------------------------------------------------

class OutcomeRecord(BaseModel):
    """Created when a DecisionObject moves to CLOSED."""
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    decision_object_id: UUID
    timestamp_closed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    symbol: str = "SOLUSDT"
    actual_pnl: float
    outcome_delta: float                    # projected − actual
    assumptions_held: list[str]             # assumption names that held
    assumptions_breached: list[str]         # assumption names that breached
    breach_timestamps: dict                 # {assumption_name: iso_timestamp}
    hill_climb_iterations: int
    close_reason: Literal[
        "horizon_expired",
        "assumption_breach",
        "trajectory_breach",
        "system_shutdown",
        "manual",
        "incomplete",   # sentinel: DB row written but close_reason never set (crash mid-write)
    ]
    phase: Literal["bootstrap", "trained"]  # carried from DecisionObject
    schema_version: int = 1
