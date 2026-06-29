"""execution/executor.py
Alpaca smart order executor.

Three execution mechanics:
  1. Limit chasing — place limit order; if not filled in 30s, send market fallback
  2. Slippage tracking — intended_price minus actual fill price, logged per order
  3. Position sizing — 20% capital exposure cap (MAX_POSITION_FRACTION)

After every order attempt, emits an ExecutionTelemetry object.
ML1 consumes ExecutionTelemetry to classify algorithm behavioural state.

Error handling per impl plan E25 / E11 MUST NOTs:
  - If Alpaca API is unreachable: retry 3× with 1s backoff, then log WARNING and mark EXECUTION_FAILED
  - Fallback to market order if limit times out after LIMIT_TIMEOUT_S
  - Never use real money — ALPACA_BASE_URL must be paper-api endpoint
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from core.schemas import CausalState, ExecutionTelemetry

load_dotenv()

logger = logging.getLogger(__name__)

# Position sizing — cap at 20% of account equity
MAX_POSITION_FRACTION = 0.20
LIMIT_TIMEOUT_S = 30
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1

SYMBOL = "SOLUSD"


class Executor:
    """Places orders on Alpaca paper trading and emits ExecutionTelemetry."""

    def __init__(self) -> None:
        self._api = tradeapi.REST(
            key_id=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )
        self._order_history: list[dict] = []  # last 10 orders for fill_rate calc

    async def execute(
        self,
        signal: str,
        params: dict,
        state: CausalState,
        decision_object_id: str,
        precomputed_qty: float | None = None,
    ) -> ExecutionTelemetry:
        """Execute a trading signal and emit telemetry.

        Args:
            signal: "BUY", "SELL", or "HOLD"
            params: tuned parameters from Hill-climbing (includes position_fraction)
            state: current CausalState snapshot
            decision_object_id: for logging context
            precomputed_qty: if provided, skip the internal _compute_quantity() REST
                call. Pass this when the caller has already fetched the qty to avoid
                a redundant Alpaca get_account() call per cycle.

        Returns:
            ExecutionTelemetry for ML1 to consume.
        """
        if signal == "HOLD":
            return self._no_op_telemetry()

        # N27 FIX: Alpaca crypto paper trading does not support short selling.
        # Since Arivu starts every cycle flat, a SELL signal is an illicit short attempt.
        if signal == "SELL":
            logger.warning("Alpaca crypto does not support short selling. Converting SELL to HOLD.")
            return self._no_op_telemetry()

        intended_price = state.price
        side = "buy" if signal == "BUY" else "sell"
        qty = precomputed_qty if precomputed_qty is not None else await self._compute_quantity(params, state)

        start_time = time.monotonic()
        fill_price, filled = await self._place_limit_with_fallback(
            side, qty, intended_price
        )
        latency_ms = (time.monotonic() - start_time) * 1000

        slippage = intended_price - fill_price if filled else 0.0
        self._order_history.append({"filled": filled})
        if len(self._order_history) > 10:
            self._order_history.pop(0)

        fill_rate = sum(1 for o in self._order_history if o["filled"]) / len(self._order_history)
        intended_qty = qty
        actual_qty = qty if filled else 0
        pos_size_dev = (actual_qty / intended_qty - 1) if intended_qty > 0 else 0.0

        # N26 FIX: Add missing symbol to prevent Pydantic ValidationError crash
        telemetry = ExecutionTelemetry(
            symbol=SYMBOL,
            fill_rate=fill_rate,
            order_latency_ms=latency_ms,
            slippage=slippage,
            position_size_deviation=pos_size_dev,
        )

        logger.info(
            "Order placed | side=%s qty=%.6f fill=%.2f slippage=%.4f latency=%.0fms",
            side, qty, fill_price, slippage, latency_ms,
        )
        return telemetry

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _place_limit_with_fallback(
        self, side: str, qty: float, limit_price: float
    ) -> tuple[float, bool]:
        """Try limit order. Fall back to market order after LIMIT_TIMEOUT_S."""
        for attempt in range(MAX_RETRIES):
            try:
                order = await asyncio.to_thread(
                    self._api.submit_order,
                    symbol=SYMBOL,
                    qty=qty,
                    side=side,
                    type="limit",
                    time_in_force="day",  # HIGH-3 FIX: 'gtc' rejected by Alpaca paper crypto; 'day' is universally supported
                    limit_price=round(limit_price, 2),
                )

                # Poll for fill
                deadline = time.monotonic() + LIMIT_TIMEOUT_S
                while time.monotonic() < deadline:
                    await asyncio.sleep(2)
                    o = await asyncio.to_thread(self._api.get_order, order.id)
                    if o.status == "filled":
                        fill_price = float(o.filled_avg_price or limit_price)
                        return fill_price, True

                # Timeout — cancel and fall back to market
                await asyncio.to_thread(self._api.cancel_order, order.id)
                logger.warning("Limit order timed out | falling back to market order")
                return await self._place_market_order(side, qty)

            except tradeapi.rest.APIError as exc:
                logger.error("Alpaca API error (attempt %d) | %s", attempt + 1, exc)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF_S)

        logger.warning("All retries exhausted | order not placed")
        return limit_price, False

    async def _place_market_order(self, side: str, qty: float) -> tuple[float, bool]:
        """Market order fallback — higher slippage but ensures execution."""
        try:
            order = await asyncio.to_thread(
                self._api.submit_order,
                symbol=SYMBOL,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",
            )
            # H_NEW_3 FIX: Poll until filled or 10-second timeout.
            # Previously: sleep(3) then read once. If fill takes > 3s,
            # filled_avg_price is None → fill_price=0.0 → slippage=intended_price
            # (e.g. 150.0) → ML1 classifies as severely degraded. Silent corruption.
            fill_price = 0.0
            for _ in range(10):
                await asyncio.sleep(1)
                o = await asyncio.to_thread(self._api.get_order, order.id)
                if o.status == "filled":
                    fill_price = float(o.filled_avg_price or 0.0)
                    break
            else:
                logger.error(
                    "Market order not filled after 10s poll | "
                    "fill_price=0.0 — ExecutionTelemetry slippage will be unreliable"
                )
            logger.warning("Market order fallback filled | price=%.2f", fill_price)
            return fill_price, True

        except Exception as exc:  # noqa: BLE001
            logger.error("Market fallback failed | %s", exc)
            return 0.0, False

    async def _compute_quantity(self, params: dict, state: CausalState) -> float:
        """Compute order quantity respecting the 20% position cap."""
        try:
            account = await asyncio.to_thread(self._api.get_account)
            equity = float(account.equity)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not fetch Alpaca account equity | falling back to $10,000 | "
                "ORDER MAY BE OVERSIZED if real equity is lower | %s", exc
            )
            equity = 10_000.0  # fallback for testing only — never silently use in live session

        fraction = min(
            params.get("position_fraction", 0.10),
            MAX_POSITION_FRACTION,
        )
        dollar_exposure = equity * fraction
        qty = dollar_exposure / state.price if state.price > 0 else 0.0
        return round(qty, 6)

    def _no_op_telemetry(self) -> ExecutionTelemetry:
        """Return neutral telemetry for HOLD signals."""
        # N26 FIX: Add missing symbol
        return ExecutionTelemetry(
            symbol=SYMBOL,
            fill_rate=1.0,
            order_latency_ms=0.0,
            slippage=0.0,
            position_size_deviation=0.0,
        )
