"""scripts/check_imbalance.py  — Task 1 acceptance test.

Subscribes to Binance WebSocket for 60 seconds and logs order book
imbalance (or trade-side aggression proxy) for SOLUSDT.

Run:
    python scripts/check_imbalance.py

Pass criterion: at least 3 different non-degenerate values observed.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time

import websockets

SYMBOL = "solusdt"
DURATION_S = 60
WS_BASE = "wss://stream.binance.com:9443/stream"

STREAMS = [
    f"{SYMBOL}@depth5@100ms",   # L2 top-5 levels
    f"{SYMBOL}@aggTrade",       # trade-side aggression fallback
    f"btcusdt@aggTrade",        # BTC macro feed check
    f"ethusdt@aggTrade",        # ETH macro feed check
]


async def run() -> None:
    url = f"{WS_BASE}?streams={'/'.join(STREAMS)}"
    imbalance_values: list[float] = []
    aggression_values: list[float] = []
    btc_prices: list[float] = []
    eth_prices: list[float] = []

    buyer_trades = 0
    total_trades = 0

    deadline = time.monotonic() + DURATION_S
    print(f"Connecting to Binance WebSocket for {DURATION_S}s ...\n")

    try:
        async with websockets.connect(url) as ws:
            async for raw in ws:
                if time.monotonic() > deadline:
                    break

                try:
                    envelope = json.loads(raw)
                    stream: str = envelope.get("stream", "")
                    data: dict = envelope.get("data", {})

                    if "depth5" in stream:
                        bids = data.get("bids", [])
                        asks = data.get("asks", [])
                        if bids and asks:
                            bid_qty = sum(float(b[1]) for b in bids)
                            ask_qty = sum(float(a[1]) for a in asks)
                            total_qty = bid_qty + ask_qty
                            if total_qty > 0:
                                imb = (bid_qty - ask_qty) / total_qty
                                imbalance_values.append(imb)

                    elif "aggTrade" in stream and SYMBOL in stream:
                        total_trades += 1
                        if not data.get("m", True):  # m=False → buyer is aggressor
                            buyer_trades += 1
                        ratio = buyer_trades / total_trades if total_trades else 0.5
                        aggression_values.append(ratio)

                    elif "btcusdt" in stream:
                        btc_prices.append(float(data.get("p", 0)))

                    elif "ethusdt" in stream:
                        eth_prices.append(float(data.get("p", 0)))

                except Exception as exc:
                    print(f"  [WARN] parse error: {exc}")

    except Exception as exc:
        print(f"[ERROR] WebSocket error: {exc}")
        return

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    # --- L2 imbalance ---
    if imbalance_values:
        unique = len(set(round(v, 4) for v in imbalance_values))
        print(f"\n[L2 Depth]  samples={len(imbalance_values)}, unique={unique}")
        print(f"  mean={statistics.mean(imbalance_values):.4f}  "
              f"stdev={statistics.stdev(imbalance_values) if len(imbalance_values) > 1 else 0:.4f}  "
              f"min={min(imbalance_values):.4f}  max={max(imbalance_values):.4f}")
        if unique >= 3:
            print("  ✓ PASS — L2 true imbalance is available and non-degenerate.")
            print("  → Use: order_book_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)")
        else:
            print("  ✗ FAIL — L2 feed is degenerate (falling back to aggression proxy).")
    else:
        print("\n[L2 Depth]  NO DATA — depth stream not available.")
        print("  → Falling back to trade-side aggression proxy.")

    # --- Aggression proxy ---
    if aggression_values:
        unique_agg = len(set(round(v, 3) for v in aggression_values))
        print(f"\n[Aggression Proxy]  trades={total_trades}, buyer_trades={buyer_trades}")
        print(f"  buyer_ratio  mean={statistics.mean(aggression_values):.4f}  "
              f"stdev={statistics.stdev(aggression_values) if len(aggression_values) > 1 else 0:.4f}")
        if unique_agg >= 3:
            print("  ✓ PASS — aggression proxy is non-degenerate (valid fallback).")
        else:
            print("  ✗ WARN — aggression proxy is degenerate.")
    else:
        print("\n[Aggression Proxy]  NO DATA.")

    # --- BTC / ETH macro feeds ---
    if btc_prices:
        print(f"\n[BTC feed]  samples={len(btc_prices)}  "
              f"price_range=[{min(btc_prices):.2f}, {max(btc_prices):.2f}]  ✓ available")
    else:
        print("\n[BTC feed]  ✗ NOT AVAILABLE")

    if eth_prices:
        print(f"[ETH feed]  samples={len(eth_prices)}  "
              f"price_range=[{min(eth_prices):.2f}, {max(eth_prices):.2f}]  ✓ available")
    else:
        print("[ETH feed]  ✗ NOT AVAILABLE")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run())
