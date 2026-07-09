"""rl/trading_env.py
Gymnasium environment for training PPO on SOLUSDT historical bar data.

Two modes:
  "standard" — Arm 2: base CausalState obs only
  "causal"   — Arm 3: base obs + causal graph features (zeroed in Phase A)

Episode = one 4-hour window (240 1-minute bars from the historical dataset).
One 1m bar = one step. The 10-second resolution of live trading is approximated
by 1m bars here — this is a known approximation documented in fetch_historical.py.

Reward structure (designed to prevent churning):
  - HOLD with no position:      0.0
  - In position, unrealised PnL >= 0:  0.0
  - In position, unrealised PnL < 0:   STEP_PENALTY_NEGATIVE (-0.0001)
  - On close (BUY → SELL or episode end): realised PnL percentage
  - NaN guard: any NaN/Inf in obs or reward raises ValueError immediately

Observation space: Box(shape=(obs_dim,), dtype=float32)
Action space: Discrete(3) — 0=HOLD, 1=BUY, 2=SELL
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.observation import get_base_obs_dim, get_causal_obs_dim

logger = logging.getLogger(__name__)

EPISODE_BARS: int = 240          # 4 hours at 1 bar/minute
STEP_PENALTY_NEGATIVE: float = -0.0001   # per-step penalty when in a losing unrealised position
POSITION_FRACTION: float = 0.05          # fixed for both RL arms
MIN_HOLD_BARS: int = 10         # minimum bars to hold before SELL is allowed (10 bars = 100s)
SLIPPAGE_BPS: float = 5.0       # slippage per side (5 bps = 0.05%)

# Action encoding
ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2


class TradingEnv(gym.Env):
    """Gymnasium trading environment for PPO training on historical SOLUSDT bars.

    Args:
        data:      numpy array of shape (N_bars, 19) feature matrix.
        mode:      "standard" (Arm 2) or "causal" (Arm 3, causal dims zeroed in Phase A).
        seed:      optional random seed for reproducible episode selection.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: np.ndarray,
        mode: str = "standard",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert mode in ("standard", "causal"), f"mode must be 'standard' or 'causal', got {mode!r}"
        self._data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self._mode = mode
        self._n_bars = len(self._data)
        self._rng = np.random.default_rng(seed)

        obs_dim = get_base_obs_dim() if mode == "standard" else get_causal_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Episode state
        self._start_idx: int = 0
        self._current_step: int = 0
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._episode_pnl: float = 0.0
        self._steps_since_entry: int = 0   # for MIN_HOLD_BARS enforcement
        self._steps_since_exit: int = 999  # cooldown after SELL before BUY allowed
        self._current_price: float = 100.0

        logger.debug(
            "TradingEnv init | mode=%s | obs_dim=%d | n_bars=%d | episode_bars=%d",
            mode, obs_dim, self._n_bars, EPISODE_BARS,
        )

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        max_start = max(0, self._n_bars - EPISODE_BARS - 1)
        self._start_idx = int(self._rng.integers(0, max_start + 1))
        self._current_step = 0
        self._in_position = False
        self._entry_price = 0.0
        self._episode_pnl = 0.0
        self._steps_since_entry = 0
        self._steps_since_exit = 999
        self._current_price = 100.0  # BUG6 FIX: reset reconstructed price each episode

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        bar_idx = self._start_idx + self._current_step
        # BUG6 FIX: Reconstruct price level by applying the bar's price_return to
        # the running price. This gives a realistic price trajectory for PnL calculation.
        price_return_bar = float(self._data[min(bar_idx, self._n_bars - 1), 0])
        self._current_price *= (1.0 + price_return_bar)
        price = self._current_price

        reward = 0.0
        terminated = False
        truncated = False

        # Minimum hold enforcement: mask SELL if < MIN_HOLD_BARS since entry
        if action == ACTION_SELL and self._in_position and self._steps_since_entry < MIN_HOLD_BARS:
            action = ACTION_HOLD  # too early to sell — force hold

        if action == ACTION_BUY and not self._in_position:
            self._in_position = True
            slippage_factor = 1.0 + (SLIPPAGE_BPS / 10000.0)
            self._entry_price = price * slippage_factor
            self._steps_since_entry = 0
            self._steps_since_exit = 999  # reset exit timer while in position

        elif action == ACTION_SELL and self._in_position:
            slippage_factor = 1.0 - (SLIPPAGE_BPS / 10000.0)
            exit_price = price * slippage_factor
            pnl_pct = (exit_price - self._entry_price) / self._entry_price if self._entry_price > 0 else 0.0
            reward = float(pnl_pct)
            self._episode_pnl += reward
            self._in_position = False
            self._entry_price = 0.0
            self._steps_since_entry = 0
            self._steps_since_exit = 0   # start post-sell cooldown

        elif self._in_position:
            self._steps_since_entry += 1
            # Shaped step penalty for unrealised losing position (accounting for exit slippage)
            unrealised_close = price * (1.0 - (SLIPPAGE_BPS / 10000.0))
            unrealised_pnl = (unrealised_close - self._entry_price) / self._entry_price if self._entry_price > 0 else 0.0
            if unrealised_pnl < 0:
                reward = STEP_PENALTY_NEGATIVE
        else:
            # Flat, not buying: increment exit cooldown
            self._steps_since_exit += 1

        self._current_step += 1

        # Episode end
        if self._current_step >= EPISODE_BARS:
            if self._in_position:
                # Force-close at final bar
                price_return_last = float(self._data[min(self._start_idx + self._current_step - 1, self._n_bars - 1), 0])
                self._current_price *= (1.0 + price_return_last)
                close_price = self._current_price * (1.0 - (SLIPPAGE_BPS / 10000.0))
                pnl_pct = (close_price - self._entry_price) / self._entry_price if self._entry_price > 0 else 0.0
                reward += float(pnl_pct)
                self._episode_pnl += float(pnl_pct)
                self._in_position = False
            terminated = True

        obs = self._get_obs()

        # NaN guard — training must not silently corrupt on bad data
        if np.isnan(reward) or np.isinf(reward):
            raise ValueError(
                f"TradingEnv: NaN or Inf reward at step {self._current_step} | "
                f"price={price} entry={self._entry_price} action={action}"
            )
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            raise ValueError(
                f"TradingEnv: NaN or Inf in obs at step {self._current_step}"
            )

        info = {
            "episode_pnl": self._episode_pnl,
            "in_position": self._in_position,
            "price": price,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def action_masks(self) -> list[bool]:
        """MaskablePPO action masks — prevents logically invalid actions.

        Masks:
          - BUY  (action 1) when already in position
          - BUY  (action 1) within MIN_HOLD_BARS steps after a SELL (post-sell cooldown)
          - SELL (action 2) when not in position
          - SELL (action 2) also masked within MIN_HOLD_BARS of entry to prevent churn
        """
        can_sell = self._in_position and self._steps_since_entry >= MIN_HOLD_BARS
        can_buy  = not self._in_position and self._steps_since_exit >= MIN_HOLD_BARS
        return [
            True,      # HOLD always valid
            can_buy,   # BUY only when flat AND past post-sell cooldown
            can_sell,  # SELL only when long AND past min hold
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_price(self, bar_idx: int) -> float:
        """[DEPRECATED — price is now tracked via self._current_price in step()]

        Kept for backward compatibility with force-close path.
        Returns self._current_price which is updated each step.
        """
        return self._current_price

    def _get_obs(self) -> np.ndarray:
        """Build observation from current bar's feature row."""
        bar_idx = min(
            self._start_idx + self._current_step,
            self._n_bars - 1,
        )
        row = self._data[bar_idx]  # shape (19,) — VARIABLE_NAMES order

        base = row[:19].astype(np.float32)

        if self._mode == "standard":
            return base

        from rl.observation import CAUSAL_EXTRA_DIMS
        # Causal dims: zeroed in Phase A (historical training)
        causal = np.zeros(CAUSAL_EXTRA_DIMS, dtype=np.float32)
        return np.concatenate([base, causal], axis=0)
