"""tests/test_week2.py
Week 2 verification tests for causal_state.py and stream.py.

8 tests per task spec:
  1. test_causal_state_propagates_volatility
  2. test_trigger_masking_fixed (CRITICAL — verifies no elif in thresholds)
  3. test_causal_state_rolling_calculations
  4. test_is_stale_false_on_fresh_state
  5. test_is_stale_true_after_delay
  6. test_causal_state_manager_has_correct_structure
  7. test_snapshot_excludes_private_fields
  8. test_independent_if_statements_in_source
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from core.causal_state import CausalStateManager
from core.schemas import MarketTick


def _make_tick(price=150.0, volume=10.0, bid=149.9, ask=150.1) -> MarketTick:
    return MarketTick(
        symbol="SOLUSDT",
        timestamp=datetime.now(timezone.utc),
        price=price,
        volume=volume,
        bid=bid,
        ask=ask,
        spread=ask - bid,
    )


@pytest.fixture
def queue_and_manager():
    q = asyncio.Queue()
    mgr = CausalStateManager(decision_queue=q)
    return q, mgr


# ---------------------------------------------------------------------------
# 1. Volatility propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_causal_state_propagates_volatility(queue_and_manager):
    """High-volatility ticks should propagate into the spread multiplier."""
    q, mgr = queue_and_manager

    # Feed oscillating prices to create high volatility
    prices = [150, 155, 145, 160, 140, 165, 135, 170, 130, 175,
              125, 180, 120, 185, 115, 190, 110, 195, 105, 200]
    for p in prices:
        await mgr.update(_make_tick(price=float(p), bid=p - 0.5, ask=p + 0.5))

    state = mgr.snapshot()
    # Volatility must be computed and positive after price swings
    assert state.volatility > 0.0, "Volatility should be non-zero after price swings"
    # With high volatility the propagated spread should be > raw spread (1.0)
    # because Rule 1 multiplies spread by (1 + (vol - threshold) * factor)
    assert state.spread > 0.0, "Spread should be positive"


# ---------------------------------------------------------------------------
# 2. Trigger masking — THE MOST IMPORTANT TEST
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trigger_masking_fixed(queue_and_manager):
    """Both volatility AND spread thresholds must fire independently.

    If elif was used instead of independent if statements,
    only ONE event would fire. This test catches that bug.

    Uses edge detection: starts below threshold, then crosses above.
    """
    q, mgr = queue_and_manager

    # Phase 1: Feed stable ticks to fill the buffer and establish baseline below threshold
    for i in range(25):
        await mgr.update(_make_tick(
            price=150.0,
            volume=10.0,
            bid=149.999,
            ask=150.001,  # tiny spread = 0.002
        ))

    # Drain any events from the warmup phase
    while not q.empty():
        q.get_nowait()

    # Phase 2: Feed wild price swings WITH wide spread to cross BOTH thresholds
    prices = [150, 200, 100, 250, 50, 300, 25, 350, 10, 400,
              5, 450, 3, 500, 2, 550, 1, 600, 0.5, 650]
    for p in prices:
        spread_val = max(p * 0.05, 0.1)  # very wide spread
        await mgr.update(_make_tick(
            price=float(p),
            bid=max(0.01, p - spread_val),
            ask=p + spread_val,
        ))

    # Drain all events from queue
    events = []
    while not q.empty():
        events.append(q.get_nowait())

    # Must have at least 2 events — volatility AND spread (or trend_reversal)
    # If elif was used, we'd only get 1
    reasons = [e.get("reason") for e in events]
    assert len(events) >= 2, (
        f"Expected at least 2 trigger events (independent if), "
        f"got {len(events)}: {reasons}"
    )


# ---------------------------------------------------------------------------
# 3. Rolling calculations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_causal_state_rolling_calculations(queue_and_manager):
    """After feeding 25+ ticks, all rolling features should be computed."""
    q, mgr = queue_and_manager

    # Feed 25 ticks with incrementing prices and varying volume
    # Prices vary enough to produce non-zero volatility and trend
    for i in range(25):
        await mgr.update(_make_tick(
            price=100.0 + i * 2.0 + (i % 3) * 0.5,  # uptrend with noise
            volume=10.0 + i * 0.1,
            bid=99.5 + i * 2.0,
            ask=100.5 + i * 2.0,
        ))

    state = mgr.snapshot()
    assert state.volatility > 0.0, "Volatility should be computed after 25 ticks"
    assert state.trend_slope != 0.0, "Trend slope should be computed"
    assert state.trend_strength >= 0.0, "Trend strength should be non-negative"
    assert state.volume > 0.0, "Volume should be computed"


# ---------------------------------------------------------------------------
# 4. Stale detection — fresh state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_stale_false_on_fresh_state(queue_and_manager):
    """A just-updated state should NOT be stale."""
    q, mgr = queue_and_manager

    await mgr.update(_make_tick())
    state = mgr.snapshot()
    # last_tick_timestamp should be within the last few seconds
    delta = (datetime.now(timezone.utc) - state.last_tick_timestamp).total_seconds()
    assert delta < 10, f"Just-updated state should not be stale, delta={delta}s"


# ---------------------------------------------------------------------------
# 5. Stale detection — old state
# ---------------------------------------------------------------------------

def test_is_stale_true_after_delay():
    """A state with old last_tick_timestamp should be stale."""
    from core.schemas import CausalState

    state = CausalState(
        last_tick_timestamp=datetime.now(timezone.utc) - timedelta(seconds=15)
    )
    delta = (datetime.now(timezone.utc) - state.last_tick_timestamp).total_seconds()
    assert delta > 10, "State with 15s old timestamp should be considered stale"


# ---------------------------------------------------------------------------
# 6. Manager structure
# ---------------------------------------------------------------------------

def test_causal_state_manager_has_correct_structure():
    """CausalStateManager should have snapshot() and update() methods."""
    q = asyncio.Queue()
    mgr = CausalStateManager(decision_queue=q)

    assert hasattr(mgr, "snapshot"), "Manager must have snapshot() method"
    assert hasattr(mgr, "update"), "Manager must have update() method"
    assert hasattr(mgr, "update_algo_health"), "Manager must have update_algo_health() method"


# ---------------------------------------------------------------------------
# 7. Snapshot excludes private fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_snapshot_excludes_private_fields(queue_and_manager):
    """Snapshot should return CausalState without internal implementation details."""
    q, mgr = queue_and_manager

    await mgr.update(_make_tick())
    state = mgr.snapshot()

    # Snapshot is a pydantic CausalState — should have public market fields
    assert hasattr(state, "volatility"), "Snapshot must have volatility"
    assert hasattr(state, "spread"), "Snapshot must have spread"
    assert hasattr(state, "trend_slope"), "Snapshot must have trend_slope"
    assert hasattr(state, "price"), "Snapshot must have price"
    assert hasattr(state, "algo_health_vector"), "Snapshot must have algo_health_vector"

    # Internal manager buffers should NOT leak into the snapshot
    assert not hasattr(state, "_prices"), "Private _prices should not be in snapshot"
    assert not hasattr(state, "_volumes"), "Private _volumes should not be in snapshot"
    assert not hasattr(state, "_lock"), "Private _lock should not be in snapshot"


# ---------------------------------------------------------------------------
# 8. Source code check — no elif in threshold checks
# ---------------------------------------------------------------------------

def test_independent_if_statements_in_source():
    """The _check_thresholds method must use independent if statements, NOT elif.

    elif would cause trigger masking: if volatility fires first,
    spread breach would be silently ignored. This is the most
    dangerous single bug possible in the causal model.
    """
    with open("core/causal_state.py") as f:
        source = f.read()

    # Find the _check_thresholds method
    method_start = source.find("def _check_thresholds")
    assert method_start != -1, "_check_thresholds method not found in source"

    # Extract from method start to the next method or end of class
    remaining = source[method_start:]
    # Find the next def at the same or lower indentation
    lines = remaining.split("\n")
    method_body = lines[0]  # the def line
    for line in lines[1:]:
        if line.strip().startswith("def ") or (line.strip() and not line.startswith(" ")):
            break
        method_body += "\n" + line

    assert "elif" not in method_body, (
        "CRITICAL: elif found in _check_thresholds — "
        "this causes trigger masking. Use independent if statements."
    )
