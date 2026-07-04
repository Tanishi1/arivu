"""rl/train_ppo.py
Phase A training for Arm 2 (Standard PPO) and Arm 3 (Causal RL).

Protocol:
  1. Load data/solusdt_historical.npz (produced by scripts/fetch_historical.py).
  2. Train Arm 2 (mode="standard") on training split, evaluate on validation.
  3. Train Arm 3 (mode="causal", causal dims zeroed) on same splits.
  4. Log Sharpe ratio and win rate on validation for both models.
  5. Assert no NaN in model weights before saving.
  6. Save to data/models/ppo_standard.zip and data/models/ppo_causal.zip.

Usage:
    python rl/train_ppo.py [--timesteps 1000000] [--data data/solusdt_historical.npz]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _compute_sharpe(returns: list[float], periods_per_year: int = 525960) -> float:
    """Compute annualised Sharpe ratio from a list of per-bar returns."""
    arr = np.array(returns, dtype=np.float64)
    if len(arr) < 2 or np.std(arr) == 0.0:
        return 0.0
    return float(np.mean(arr) / np.std(arr) * np.sqrt(periods_per_year))


def _evaluate(model, env_class, data: np.ndarray, mode: str, n_episodes: int = 20) -> dict:
    """Run n_episodes of evaluation and return metrics."""
    from rl.trading_env import ACTION_BUY, ACTION_SELL, ACTION_HOLD

    all_returns: list[float] = []
    wins = 0
    total_trades = 0

    for ep in range(n_episodes):
        env = env_class(data=data, mode=mode, seed=ep)
        obs, _ = env.reset(seed=ep)
        done = False
        ep_pnl = 0.0
        trade_pnls: list[float] = []
        in_pos = False
        entry_ret = 0.0
        step = 0

        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            if int(action) == ACTION_BUY and not in_pos:
                in_pos = True
            elif int(action) == ACTION_SELL and in_pos:
                trade_pnls.append(reward)
                in_pos = False
            step += 1

        for r in trade_pnls:
            total_trades += 1
            all_returns.append(r)
            if r > 0:
                wins += 1

        env.close()

    sharpe = _compute_sharpe(all_returns)
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "n_episodes": n_episodes,
        "mean_return": round(float(np.mean(all_returns)) if all_returns else 0.0, 6),
    }


def _assert_no_nan_in_model(model, name: str) -> None:
    """Raise if any model parameter contains NaN — catch degenerate training."""
    import torch
    for param_name, param in model.policy.named_parameters():
        if torch.isnan(param).any():
            raise ValueError(
                f"NaN detected in {name} parameter '{param_name}' — "
                "training produced a degenerate model. Do not save."
            )
    logger.info("%s: NaN check passed — all parameters finite", name)


def train_arm(
    arm_name: str,
    mode: str,
    train_data: np.ndarray,
    val_data: np.ndarray,
    save_path: str,
    total_timesteps: int,
) -> dict:
    """Train a single MaskablePPO arm and return evaluation metrics."""
    # BUG7 FIX: Use MaskablePPO from sb3_contrib so that TradingEnv.action_masks()
    # is respected during training. Standard PPO silently ignores action_masks().
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.utils import get_action_masks
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError:
        logger.error(
            "sb3_contrib not installed. Run: pip install sb3-contrib"
        )
        raise
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl.trading_env import TradingEnv

    logger.info("=" * 60)
    logger.info("Training %s | mode=%s | timesteps=%d", arm_name, mode, total_timesteps)
    logger.info("=" * 60)

    # DummyVecEnv preserves action_masks() method (make_vec_env wrapping loses it).
    vec_env = DummyVecEnv([lambda: TradingEnv(data=train_data, mode=mode)] * 4)

    # Check if tensorboard is installed to make logging optional
    try:
        import tensorboard  # noqa: F401
        tb_log = f"logs/tensorboard_{arm_name}"
    except ImportError:
        tb_log = None

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=tb_log,
    )

    model.learn(total_timesteps=total_timesteps)
    vec_env.close()

    # NaN check before saving
    _assert_no_nan_in_model(model, arm_name)

    # Evaluate on validation split
    logger.info("Evaluating %s on validation split...", arm_name)
    metrics = _evaluate(model, TradingEnv, val_data, mode=mode, n_episodes=30)

    logger.info(
        "%s validation | Sharpe=%.4f | WinRate=%.4f | Trades=%d | MeanReturn=%.6f",
        arm_name, metrics["sharpe"], metrics["win_rate"],
        metrics["total_trades"], metrics["mean_return"],
    )

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.info("%s saved to %s", arm_name, save_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO comparison arms (Phase A)")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--data", default="data/solusdt_historical.npz")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(
            "Historical data not found: %s\n"
            "Run: python scripts/fetch_historical.py first", args.data
        )
        sys.exit(1)

    logger.info("Loading historical data from %s", args.data)
    npz = np.load(args.data, allow_pickle=True)
    train_data = npz["train"].astype(np.float32)
    val_data   = npz["val"].astype(np.float32)

    # Sanitise any residual NaN/Inf (should not exist, but guard)
    train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
    val_data   = np.nan_to_num(val_data,   nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "Data loaded | train=%d bars | val=%d bars | features=%d",
        len(train_data), len(val_data), train_data.shape[1],
    )

    # --- Arm 2: Standard PPO ---
    metrics_standard = train_arm(
        arm_name="ppo_standard",
        mode="standard",
        train_data=train_data,
        val_data=val_data,
        save_path="data/models/ppo_standard",
        total_timesteps=args.timesteps,
    )

    # --- Arm 3: Causal RL Phase A ---
    metrics_causal = train_arm(
        arm_name="ppo_causal",
        mode="causal",
        train_data=train_data,
        val_data=val_data,
        save_path="data/models/ppo_causal",
        total_timesteps=args.timesteps,
    )

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE — PHASE A SUMMARY")
    logger.info("  ppo_standard | Sharpe=%.4f | WinRate=%.4f", metrics_standard["sharpe"], metrics_standard["win_rate"])
    logger.info("  ppo_causal   | Sharpe=%.4f | WinRate=%.4f", metrics_causal["sharpe"],   metrics_causal["win_rate"])
    logger.info("Note: ppo_causal was trained with causal dims ZEROED (Phase A).")
    logger.info("Phase B live fine-tuning will activate real causal features.")
    logger.info("=" * 60)

    # Acceptance gate: both models must exist
    assert os.path.exists("data/models/ppo_standard.zip"), "ppo_standard.zip not found after training"
    assert os.path.exists("data/models/ppo_causal.zip"),   "ppo_causal.zip not found after training"
    logger.info("Acceptance check passed — both model files exist.")


if __name__ == "__main__":
    main()
