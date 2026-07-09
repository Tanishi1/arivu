"""tests/test_causal_breaches.py
Unit tests for true causal edge breach tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from core.schemas import DecisionObject, Assumption
from ml.causal_discovery import GraphSnapshot, CausalEdge
from core.comparator import OutcomeComparator
import core.comparator
import pathlib
core.comparator.TRAINING_BUFFER_PATH = pathlib.Path("data/temp_test_training_buffer.csv")


def test_causal_edge_breach_detection():
    # 1. Create mock ledger and alpaca clients
    mock_ledger = MagicMock()
    mock_ledger.get_decision_object.return_value = None
    mock_ledger.get_breach_log.return_value = {}  # start with empty breach log
    
    mock_alpaca = MagicMock()
    mock_alpaca.get_position.return_value.unrealized_pl = 100.0

    comparator = OutcomeComparator(mock_ledger, mock_alpaca)

    # 2. Define the decision object with a CausalAgent strategy and 3 assumptions:
    #    - Assumption 1: causal_edge|volatility|price_return|2 (expected coeff +0.5)
    #    - Assumption 2: causal_edge|spread|price_return|1 (expected coeff -0.2)
    #    - Assumption 3: causal_edge_padding|1 (padding)
    do = DecisionObject(
        strategy_name="CausalAgent",
        tuned_params={
            "decision_source": "causal_agent",
        },
        market_state_snapshot={},
        algo_health_vector=[1.0, 0.0, 0.0],
        assumptions=[
            Assumption(
                name="causal_edge|volatility|price_return|2",
                variable="volatility",
                operator="gt",
                threshold=1.0,
                current_value=0.5,
                proximity=0.5,
                breach_risk=0.5,
            ),
            Assumption(
                name="causal_edge|spread|price_return|1",
                variable="volatility",
                operator="lt",
                threshold=1.0,
                current_value=-0.2,
                proximity=0.8,
                breach_risk=0.8,
            ),
            Assumption(
                name="causal_edge_padding|1",
                variable="volatility",
                operator="lt",
                threshold=1.0,
                current_value=0.0,
                proximity=0.0,
                breach_risk=0.0,
            ),
        ],
        projected_pnl=100.0,
        confidence=0.5,
        phase="bootstrap",
        hill_climb_iterations=0,
    )

    # Case A: Missing Edge Breach + Sign Flip Breach
    # In the live graph, volatility->price_return(lag 2) is missing.
    # spread->price_return(lag 1) has coeff +0.4 (sign flipped from -0.2 to +0.4).
    # Thus, both real edges should breach. Padding should hold.
    live_graph_a = GraphSnapshot(
        version_id="v1",
        timestamp=datetime.now(timezone.utc),
        edges=[
            CausalEdge("spread", "price_return", 1, 0.4, 0.01, [], "v1"),
        ],
        variables=["spread", "price_return"],
        n_bars_used=100,
        algorithm="pcmci"
    )

    record_a = comparator.close_cycle(do, "horizon_expired", live_graph_a)
    assert record_a is not None
    assert "causal_edge|volatility|price_return|2" in record_a.assumptions_breached  # missing edge
    assert "causal_edge|spread|price_return|1" in record_a.assumptions_breached      # flipped sign
    assert "causal_edge_padding|1" in record_a.assumptions_held                     # padding holds

    # Case B: Held Edges
    # In this live graph, both edges exist and match the expected sign:
    # volatility->price_return(lag 2) coeff is +0.1 (matching +0.5 sign)
    # spread->price_return(lag 1) coeff is -0.3 (matching -0.2 sign)
    # Thus, all assumptions (including padding) should hold.
    live_graph_b = GraphSnapshot(
        version_id="v2",
        timestamp=datetime.now(timezone.utc),
        edges=[
            CausalEdge("volatility", "price_return", 2, 0.1, 0.01, [], "v2"),
            CausalEdge("spread", "price_return", 1, -0.3, 0.01, [], "v2"),
        ],
        variables=["volatility", "spread", "price_return"],
        n_bars_used=100,
        algorithm="pcmci"
    )

    mock_ledger.get_breach_log.return_value = {}  # reset
    record_b = comparator.close_cycle(do, "horizon_expired", live_graph_b)
    assert record_b is not None
    assert len(record_b.assumptions_breached) == 0
    assert len(record_b.assumptions_held) == 3
    assert "causal_edge|volatility|price_return|2" in record_b.assumptions_held
    assert "causal_edge|spread|price_return|1" in record_b.assumptions_held
    assert "causal_edge_padding|1" in record_b.assumptions_held
