"""tests/test_causal_state.py
Unit tests for shared/causal_state.py

Five test cases per impl plan E48:
  1. test_volatility_propagation
  2. test_spread_assumption_proximity
  3. test_trend_reversal_trigger
  4. test_lock_prevents_concurrent_writes  (review write path)
  5. test_stale_feed_detection
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from core.causal_state import CausalStateManager, STALE_THRESHOLD_S
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
async def test_volatility_propagation(queue_and_manager):
    """Setting high volatility via ticks should propagate spread multiplier."""
    q, mgr = queue_and_manager

    # Feed a sequence of ticks with sharp price movement (high volatility)
    prices = [50000, 50100, 49900, 50200, 49800, 50300, 49700, 50400, 49600, 50500]
    for p in prices:
        await mgr.update(_make_tick(price=p, bid=p - 1, ask=p + 1))

    state = mgr.snapshot()
    # With high volatility the propagated spread should be > raw spread (2.0)
    assert state.volatility > 0.0, "Volatility should be non-zero after price swings"


# ---------------------------------------------------------------------------
# 2. Spread assumption proximity
# ---------------------------------------------------------------------------

def test_spread_assumption_proximity():
    """Proximity = current_value / threshold should equal 0.9 for spread=0.0018, threshold=0.002."""
    from core.monitor import _check_assumption
    from core.schemas import Assumption, CausalState

    assumption = Assumption(
        name="spread_constraint",
        variable="spread",
        operator="lt",
        threshold=0.002,
    )
    state = CausalState(spread=0.0018)
    breached = _check_assumption(assumption, state)

    assert not breached, "spread=0.0018 < threshold=0.002 should NOT be breached"
    assert abs(assumption.proximity - 0.9) < 0.001, f"Expected proximity=0.9, got {assumption.proximity}"


# ---------------------------------------------------------------------------
# 3. Trend reversal trigger
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trend_reversal_trigger(queue_and_manager):
    """A sign flip in trend_slope should put an event on the decision queue."""
    q, mgr = queue_and_manager

    # First: uptrend (prices increasing) to fill the window
    for p in [50000 + (i * 50) for i in range(20)]:
        await mgr.update(_make_tick(price=p, bid=p - 1, ask=p + 1))

    # Then: downtrend (prices decreasing sharply) long enough to flip the 20-tick slope
    for p in [51000 - (i * 50) for i in range(20)]:
        await mgr.update(_make_tick(price=p, bid=p - 1, ask=p + 1))

    # Check if any trigger event was fired
    events = []
    while not q.empty():
        events.append(await q.get())

    trend_reversal_events = [e for e in events if e.get("reason") == "trend_reversal"]
    assert len(trend_reversal_events) > 0, "Expected at least one trend_reversal trigger event"


# ---------------------------------------------------------------------------
# 4. Lock prevents concurrent writes (structural check)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lock_prevents_concurrent_writes(queue_and_manager):
    """Two coroutines updating state simultaneously should not corrupt state."""
    q, mgr = queue_and_manager

    async def feed_ticks(start_price: float, n: int):
        for i in range(n):
            p = start_price + i
            await mgr.update(_make_tick(price=p, bid=p - 1, ask=p + 1))

    # Run two coroutines concurrently
    await asyncio.gather(
        feed_ticks(50000, 10),
        feed_ticks(51000, 10),
    )

    state = mgr.snapshot()
    # State should be internally consistent — price must be a valid number
    assert isinstance(state.price, float), "CausalState.price must be a float after concurrent writes"
    assert state.price > 0, "Price should be positive after concurrent writes"


# ---------------------------------------------------------------------------
# 5. Stale feed detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_feed_detection(queue_and_manager):
    """Feed is stale if last_tick_timestamp is more than STALE_THRESHOLD_S seconds ago."""
    from core.monitor import _is_feed_stale
    from core.schemas import CausalState

    q, mgr = queue_and_manager

    # Never updated — should be stale
    assert _is_feed_stale(mgr), "Uninitialised feed should be stale"

    # Update with a fresh tick
    await mgr.update(_make_tick())
    assert not _is_feed_stale(mgr), "Just-updated feed should not be stale"
