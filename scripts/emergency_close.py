"""scripts/emergency_close.py
One-shot script to chunk-close the stuck SOLUSD position.

Run BEFORE restarting main.py or rl_main.py:
    python scripts/emergency_close.py

Also resets the paper account balance tracking by cancelling all open orders first.
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

import alpaca_trade_api as tradeapi

MAX_NOTIONAL_PER_ORDER = 180_000.0

def get_api(key_env, secret_env, label):
    key    = os.getenv(key_env, "")
    secret = os.getenv(secret_env, "")
    if not key or not secret:
        print(f"[{label}] No credentials found ({key_env}), skipping.")
        return None
    return tradeapi.REST(
        key_id=key,
        secret_key=secret,
        base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    )

def close_chunked(api, label):
    # 1. Cancel all open orders first
    try:
        orders = api.list_orders(status="open")
        for o in orders:
            api.cancel_order(o.id)
            print(f"[{label}] Cancelled open order {o.id}")
    except Exception as e:
        print(f"[{label}] Could not cancel open orders: {e}")

    # 2. Check position
    try:
        pos = api.get_position("SOLUSD")
    except Exception as e:
        if "position does not exist" in str(e).lower():
            print(f"[{label}] No open position. ✓")
            return
        print(f"[{label}] Error reading position: {e}")
        return

    qty = float(pos.qty)
    price = float(pos.current_price)
    notional = qty * price
    unrealized = float(pos.unrealized_pl)
    print(f"[{label}] Position: {qty:.2f} SOL @ ${price:.2f} = ${notional:,.0f} | unrealized_pl=${unrealized:,.2f}")

    if qty <= 0:
        print(f"[{label}] No long position. ✓")
        return

    chunk_qty = MAX_NOTIONAL_PER_ORDER / price
    remaining = qty
    chunk_num = 0

    while remaining > 0.001:
        sell_qty = round(min(chunk_qty, remaining), 6)
        chunk_num += 1
        try:
            order = api.submit_order(
                symbol="SOLUSD",
                qty=sell_qty,
                side="sell",
                type="market",
                time_in_force="gtc",
            )
            print(f"[{label}] Chunk {chunk_num}: sell {sell_qty:.4f} SOL | order {order.id} submitted")
            remaining -= sell_qty
            if remaining > 0.001:
                print(f"[{label}] Waiting 3s before next chunk... (remaining: {remaining:.4f} SOL)")
                time.sleep(3)
        except Exception as e:
            print(f"[{label}] Chunk {chunk_num} FAILED: {e}")
            break

    print(f"[{label}] Close complete. {chunk_num} chunk(s) submitted.")

    # 3. Print account summary
    try:
        acct = api.get_account()
        print(f"[{label}] Account after close: equity=${float(acct.equity):,.2f} | cash=${float(acct.cash):,.2f} | buying_power=${float(acct.buying_power):,.2f}")
    except Exception as e:
        print(f"[{label}] Could not read account: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("ARIVU EMERGENCY POSITION CLOSE")
    print("=" * 60)

    # Main causal agent account
    api_main = get_api("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "CausalAgent")
    if api_main:
        close_chunked(api_main, "CausalAgent")

    print()

    # RL Standard account
    api_std = get_api("ALPACA_KEY_PPO_STANDARD", "ALPACA_SECRET_PPO_STANDARD", "ppo_standard")
    if api_std:
        close_chunked(api_std, "ppo_standard")

    print()

    # RL Causal account
    api_csl = get_api("ALPACA_KEY_PPO_CAUSAL", "ALPACA_SECRET_PPO_CAUSAL", "ppo_causal")
    if api_csl:
        close_chunked(api_csl, "ppo_causal")

    print()
    print("Done. Verify on Alpaca paper dashboard, then restart main.py and rl_main.py.")
