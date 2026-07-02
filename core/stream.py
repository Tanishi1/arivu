"""core/stream.py
Binance WebSocket market data feed for SOL/USDT + BTC/ETH macro + L2 depth.

Connects to six simultaneous streams:
  - solusdt@kline_1m    — 1-minute candlestick data (SOL price, volume)
  - solusdt@bookTicker  — best bid/ask (for spread calculation)
  - solusdt@aggTrade    — aggregate trade stream (volume, trade-side aggression)
  - solusdt@depth5@100ms — L2 top-5 order book (order book imbalance)
  - btcusdt@aggTrade    — BTC macro price feed
  - ethusdt@aggTrade    — ETH macro price feed

Reconnection handler fires within 2 seconds of any drop.
Stale feed detection: last_tick_timestamp tracked in CausalState.
If feed is stale > 10s, assumption_monitor_loop pauses breach alerts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from core.causal_state import CausalStateManager
from core.schemas import MarketTick

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
SYMBOL = "solusdt"
RECONNECT_DELAY_S = 2
STALE_THRESHOLD_S = 10

STREAMS = [
    f"{SYMBOL}@kline_1m",
    f"{SYMBOL}@bookTicker",
    f"{SYMBOL}@aggTrade",
    f"{SYMBOL}@depth5@100ms",  # L2 order book imbalance (Task 1 confirmed available)
    "btcusdt@aggTrade",         # BTC macro lead signal
    "ethusdt@aggTrade",         # ETH macro secondary signal
]


class BinanceFeed:
    """Manages the Binance WebSocket connection and parses incoming messages
    into MarketTick objects for the causal state manager.

    Also feeds the FeatureBarBuilder with macro prices, trade aggression,
    and L2 order book imbalance for causal discovery.
    """

    def __init__(
        self,
        state_manager: CausalStateManager,
        feature_bar_builder=None,  # core.feature_bar.FeatureBarBuilder | None
    ) -> None:
        self._state_manager = state_manager
        self._feature_bar = feature_bar_builder
        self._running = False
        # B10 FIX: store the live WebSocket so stop() can close it.
        # Previously stop() only set _running=False, which only prevents
        # RECONNECTION — the active `async for raw_message in ws:` loop
        # continues waiting for the next Binance message indefinitely.
        # Without closing the connection, market_feed_loop never returns,
        # asyncio.gather blocks forever, and _graceful_shutdown never fires.
        self._ws = None

        # Latest values — merged from multiple stream types
        self._latest_price: float = 0.0
        self._latest_volume: float = 0.0
        self._latest_bid: float = 0.0
        self._latest_ask: float = 0.0

    async def run(self) -> None:
        """Main feed loop — connects and reconnects indefinitely."""
        self._running = True
        stream_path = "/".join(STREAMS)
        url = f"{BINANCE_WS_BASE}?streams={stream_path}"

        while self._running:
            try:
                logger.info("Connecting to Binance WebSocket...")
                async with websockets.connect(url) as ws:
                    self._ws = ws   # B10 FIX: expose handle for stop()
                    logger.info("Market feed connected | streams=%s", STREAMS)
                    async for raw_message in ws:
                        await self._handle_message(raw_message)
                    self._ws = None

            except ConnectionClosed as exc:
                # Differentiate clean shutdown (code 1000) from unexpected drops
                # so that monitoring logs are not polluted with false error alerts.
                if getattr(exc.rcvd, "code", None) == 1000:
                    logger.info(
                        "Market feed closed cleanly (1000) | reconnecting in %ds",
                        RECONNECT_DELAY_S,
                    )
                else:
                    logger.warning(
                        "Market feed disconnected | reason=%s | reconnecting in %ds",
                        exc, RECONNECT_DELAY_S,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Feed unexpected error | %s | reconnecting in %ds", exc, RECONNECT_DELAY_S)

            if self._running:
                await asyncio.sleep(RECONNECT_DELAY_S)

    async def stop(self) -> None:
        """Signal the feed to stop and close the active WebSocket connection.

        B10 FIX: closing self._ws causes the `async for raw_message in ws:`
        loop to raise ConnectionClosed immediately, exiting the context manager
        and allowing market_feed_loop to return. Without this, _running=False
        only prevents reconnection — the current ws stays open forever.
        """
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass  # already closed or connection error — feed will exit regardless

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str) -> None:
        """Parse one WebSocket message and update causal state + feature bar."""
        try:
            envelope = json.loads(raw)
            stream_name: str = envelope.get("stream", "")
            data: dict = envelope.get("data", {})

            should_update = False

            if "kline" in stream_name and SYMBOL in stream_name:
                self._parse_kline(data)
                should_update = True

            elif "bookTicker" in stream_name and SYMBOL in stream_name:
                self._parse_book_ticker(data)
                should_update = True  # spread changes matter for propagation

            elif "aggTrade" in stream_name and SYMBOL in stream_name:
                self._parse_agg_trade(data)
                # Feed trade aggression to feature bar builder
                if self._feature_bar is not None:
                    is_buyer_maker = bool(data.get("m", True))
                    self._feature_bar.update_trade(is_buyer_maker)

            elif "depth5" in stream_name and SYMBOL in stream_name:
                # L2 order book imbalance
                if self._feature_bar is not None:
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    if bids and asks:
                        bid_qty = sum(float(b[1]) for b in bids)
                        ask_qty = sum(float(a[1]) for a in asks)
                        total = bid_qty + ask_qty
                        if total > 0:
                            imbalance = (bid_qty - ask_qty) / total
                            self._feature_bar.update_imbalance(imbalance)

            elif "btcusdt" in stream_name:
                btc_price = float(data.get("p", 0))
                if btc_price > 0 and self._feature_bar is not None:
                    self._feature_bar.update_macro("BTC", btc_price)

            elif "ethusdt" in stream_name:
                eth_price = float(data.get("p", 0))
                if eth_price > 0 and self._feature_bar is not None:
                    self._feature_bar.update_macro("ETH", eth_price)

            # Only update causal state on kline/bookTicker (not every aggTrade)
            if should_update and self._latest_price > 0 and self._latest_bid > 0:
                tick = MarketTick(
                    symbol=SYMBOL.upper(),
                    timestamp=datetime.now(timezone.utc),
                    price=self._latest_price,
                    volume=self._latest_volume,
                    bid=self._latest_bid,
                    ask=self._latest_ask,
                    spread=self._latest_ask - self._latest_bid,
                )
                await self._state_manager.update(tick)

                # Feed SOL price/volume/spread to feature bar builder
                if self._feature_bar is not None:
                    self._feature_bar.update_sol(
                        self._latest_price,
                        self._latest_volume,
                        self._latest_ask - self._latest_bid,
                    )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Malformed tick discarded | %s", exc)

    def _parse_kline(self, data: dict) -> None:
        k = data.get("k", {})
        self._latest_price = float(k.get("c", self._latest_price))  # close price
        self._latest_volume = float(k.get("v", self._latest_volume))

    def _parse_book_ticker(self, data: dict) -> None:
        self._latest_bid = float(data.get("b", self._latest_bid))
        self._latest_ask = float(data.get("a", self._latest_ask))

    def _parse_agg_trade(self, data: dict) -> None:
        self._latest_price = float(data.get("p", self._latest_price))
        # N20 FIX: Do not overwrite 1-minute cumulative volume with single trade quantity
        # self._latest_volume = float(data.get("q", self._latest_volume))
