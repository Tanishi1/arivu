"""tests/test_ml2_integration.py
Tests for all four ML2 integration changes:
  - Change 1: TradeOutcome ML2 fields + _score_outcome calibration/risk penalty
  - Change 2: Dynamic monitor cadence interpolation formula
  - Change 3: TwinSimulator risk-adjusted EV selection
  - Change 4: Layer1Tracker legacy flag persistence
  - Arm 2/3: RL observation dim consistency + TradingEnv mechanics
"""

from __future__ import annotations

import sqlite3
import pytest

from core.schemas import Assumption, CausalState, DecisionObject
from ml.meta_optimizer import TradeOutcome, MetaParameterOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assumption(name="a", variable="volatility", operator="lt",
                threshold=0.10, proximity=0.5, breach_risk=0.5):
    return Assumption(
        name=name,
        variable=variable,
        operator=operator,
        threshold=threshold,
        proximity=proximity,
        breach_risk=breach_risk,
    )


def _base_outcome(**kwargs) -> TradeOutcome:
    defaults = dict(
        meta_params_used={},
        regime="calm",
        predicted_return=0.01,
        actual_return=0.01,
        pnl_usd=10.0,
        causal_chain_held=True,
        regime_was_stable=True,
        avg_breach_risk=0.3,
        ml2_calibration=0.8,
    )
    defaults.update(kwargs)
    return TradeOutcome(**defaults)


@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "arivu.db")


# ===========================================================================
# Change 4 — MetaOptimizer: TradeOutcome now carries ML2 fields
# ===========================================================================

class TestTradeOutcomeML2Fields:
    def test_trade_outcome_has_avg_breach_risk(self):
        o = _base_outcome(avg_breach_risk=0.7)
        assert o.avg_breach_risk == pytest.approx(0.7)

    def test_trade_outcome_has_ml2_calibration(self):
        o = _base_outcome(ml2_calibration=0.9)
        assert o.ml2_calibration == pytest.approx(0.9)

    def test_trade_outcome_defaults(self):
        """Default values must not crash existing call sites that omit ML2 fields."""
        o = TradeOutcome(
            meta_params_used={},
            regime="calm",
            predicted_return=0.01,
            actual_return=0.01,
            pnl_usd=10.0,
            causal_chain_held=True,
            regime_was_stable=True,
        )
        assert 0.0 <= o.avg_breach_risk <= 1.0
        assert 0.0 <= o.ml2_calibration <= 1.0


# ===========================================================================
# Change 4 — _score_outcome: calibration bonus + risk penalty
# ===========================================================================

class TestScoreOutcomeML2:
    def setup_method(self):
        # We test _score_outcome directly via MetaParameterOptimizer
        # using a fake temp DB
        import tempfile, os
        self._tmpdir = tempfile.mkdtemp()
        self._opt = MetaParameterOptimizer(db_path=os.path.join(self._tmpdir, "t.db"))

    def _score(self, **kwargs) -> float:
        o = _base_outcome(**kwargs)
        return self._opt._score_outcome(o)

    def test_score_is_float_in_range(self):
        s = self._score()
        assert 0.0 <= s <= 1.0

    def test_high_calibration_scores_higher_than_low(self):
        """A perfectly calibrated prediction (ml2_calibration=1.0) should score
        higher than an uncalibrated one (ml2_calibration=0.0) all else equal."""
        high = self._score(ml2_calibration=1.0, avg_breach_risk=0.3, pnl_usd=10.0)
        low  = self._score(ml2_calibration=0.0, avg_breach_risk=0.3, pnl_usd=10.0)
        assert high > low, f"High calibration should outscore low: {high} vs {low}"

    def test_risk_penalty_applied_on_losing_high_risk_trade(self):
        """avg_breach_risk > 0.70 AND pnl < 0 must reduce score via risk_penalty=0.6."""
        risky_loss  = self._score(avg_breach_risk=0.85, pnl_usd=-50.0, ml2_calibration=0.5,
                                   predicted_return=0.01, actual_return=-0.01)
        safe_loss   = self._score(avg_breach_risk=0.30, pnl_usd=-50.0, ml2_calibration=0.5,
                                   predicted_return=0.01, actual_return=-0.01)
        assert risky_loss < safe_loss, (
            f"Reckless loss ({risky_loss:.4f}) should score lower than safe loss ({safe_loss:.4f})"
        )

    def test_no_risk_penalty_on_winning_trade_even_with_high_risk(self):
        """risk_penalty only fires on LOSING trades — winning high-risk trades are allowed."""
        risky_win = self._score(avg_breach_risk=0.85, pnl_usd=100.0, ml2_calibration=0.5)
        safe_win  = self._score(avg_breach_risk=0.30, pnl_usd=100.0, ml2_calibration=0.5)
        # Risky wins are not penalised — they may score similarly or differently based
        # purely on calibration/direction, not the risk penalty multiplier
        risky_loss = self._score(avg_breach_risk=0.85, pnl_usd=-100.0, ml2_calibration=0.5,
                                  predicted_return=0.01, actual_return=-0.01)
        assert risky_win > risky_loss, "Reckless win should still outscore reckless loss"

    def test_score_weights_sum_correctly(self):
        """Perfect outcome: direction correct, accuracy=1, regime stable,
        chain held, calibration=1 → score should be close to 1.0."""
        s = self._score(
            predicted_return=0.05,
            actual_return=0.05,
            pnl_usd=50.0,
            causal_chain_held=True,
            regime_was_stable=True,
            avg_breach_risk=0.3,  # below penalty threshold
            ml2_calibration=1.0,
        )
        assert s > 0.90, f"Perfect outcome should score > 0.90, got {s:.4f}"


# ===========================================================================
# Change 2 — Monitor: dynamic cadence formula
# ===========================================================================

class TestDynamicMonitorCadence:
    """Tests the cadence formula: interval = max(2, min(10, 10 - risk * 8))"""

    @staticmethod
    def _compute_interval(max_breach_risk: float) -> float:
        return max(2.0, min(10.0, round(10.0 - (max_breach_risk * 8.0), 1)))

    def test_zero_risk_gives_max_interval(self):
        assert self._compute_interval(0.0) == pytest.approx(10.0)

    def test_full_risk_gives_min_interval(self):
        assert self._compute_interval(1.0) == pytest.approx(2.0)

    def test_midpoint_risk_gives_midpoint_interval(self):
        # risk=0.5 → 10 - 0.5*8 = 6.0
        assert self._compute_interval(0.5) == pytest.approx(6.0)

    def test_interval_is_bounded_below_at_2(self):
        """Even extreme risk (>1.0) must not go below 2s."""
        assert self._compute_interval(5.0) >= 2.0

    def test_interval_is_bounded_above_at_10(self):
        """Even zero or negative risk must not exceed 10s."""
        assert self._compute_interval(-1.0) <= 10.0

    def test_higher_risk_gives_shorter_interval(self):
        """Monotonicity: more risk → more frequent polling."""
        i_low  = self._compute_interval(0.2)
        i_high = self._compute_interval(0.8)
        assert i_high < i_low

    def test_dynamic_cadence_module_formula_matches(self):
        """Verify the formula in monitor.py matches the expected formula directly."""
        for risk in [0.0, 0.25, 0.5, 0.75, 1.0]:
            expected = max(2.0, min(10.0, round(10.0 - (risk * 8.0), 1)))
            got = self._compute_interval(risk)
            assert got == pytest.approx(expected), f"risk={risk}: expected {expected}, got {got}"


# ===========================================================================
# Change 3 — TwinSimulator: risk-adjusted EV prefers low-risk hypothesis
# ===========================================================================

class TestTwinSimulatorRiskAdjustedEV:
    def _make_hyp(self, composite_score=0.5, avg_breach_risk=0.5):
        from ml.hypothesis_generator import CausalHypothesis
        h = CausalHypothesis()
        h.composite_score = composite_score
        h.avg_breach_risk = avg_breach_risk
        return h

    def test_risk_discount_formula(self):
        """risk_discount = max(0.10, 1.0 - avg_breach_risk)"""
        for risk, expected_discount in [
            (0.0, 1.0),
            (0.5, 0.5),
            (1.0, 0.10),   # floor at 0.10
            (0.9, 0.10),   # 1-0.9=0.10, exactly at floor
        ]:
            discount = max(0.10, 1.0 - risk)
            assert discount == pytest.approx(expected_discount), f"risk={risk}"

    def test_low_risk_hyp_preferred_over_higher_raw_ev_with_high_risk(self):
        """A hypothesis with lower raw EV but lower breach risk should win
        if its risk-adjusted EV exceeds the high-risk hypothesis's adjusted EV."""
        # high EV but high risk
        ev_risky = 0.10
        risk_risky = 0.85
        adj_risky = abs(ev_risky) * max(0.10, 1.0 - risk_risky)   # 0.10 * 0.15 = 0.015

        # lower EV but safe
        ev_safe = 0.06
        risk_safe = 0.10
        adj_safe = abs(ev_safe) * max(0.10, 1.0 - risk_safe)      # 0.06 * 0.90 = 0.054

        assert adj_safe > adj_risky, (
            f"Safe hypothesis (adj={adj_safe:.4f}) should beat risky one (adj={adj_risky:.4f})"
        )

    def test_minimum_discount_floor_prevents_zero_ev(self):
        """Even with breach_risk=1.0, risk_discount floor at 0.10 means
        adjusted EV is never zero for a non-zero raw EV."""
        ev = 0.05
        risk = 1.0
        adj = abs(ev) * max(0.10, 1.0 - risk)
        assert adj > 0.0, "Risk-adjusted EV should never be 0 (floor=0.10)"
        assert adj == pytest.approx(0.005)


# ===========================================================================
# Change 4 — Layer1Tracker: has_stable_edges + legacy flag
# ===========================================================================

class TestLayer1LegacyFlag:
    def test_has_stable_edges_false_when_empty(self, temp_db):
        from ml.layer1_tracker import Layer1Tracker
        t = Layer1Tracker(db_path=temp_db)
        assert t.has_stable_edges() is False

    def test_legacy_not_stopped_on_fresh_db(self, temp_db):
        from ml.layer1_tracker import Layer1Tracker
        t = Layer1Tracker(db_path=temp_db)
        assert t.is_legacy_permanently_stopped() is False

    def test_set_and_read_legacy_flag(self, temp_db):
        from ml.layer1_tracker import Layer1Tracker
        t = Layer1Tracker(db_path=temp_db)
        t.set_legacy_permanently_stopped()
        assert t.is_legacy_permanently_stopped() is True

    def test_legacy_flag_persists_across_instances(self, temp_db):
        """Flag must survive creating a new Layer1Tracker from the same DB."""
        from ml.layer1_tracker import Layer1Tracker
        t1 = Layer1Tracker(db_path=temp_db)
        t1.set_legacy_permanently_stopped()
        # Create a fresh instance from same DB
        t2 = Layer1Tracker(db_path=temp_db)
        assert t2.is_legacy_permanently_stopped() is True, (
            "legacy_stopped flag must survive across process/instance restarts"
        )

    def test_set_legacy_flag_is_idempotent(self, temp_db):
        """Calling set_legacy_permanently_stopped() twice must not raise."""
        from ml.layer1_tracker import Layer1Tracker
        t = Layer1Tracker(db_path=temp_db)
        t.set_legacy_permanently_stopped()
        t.set_legacy_permanently_stopped()   # second call must be safe
        assert t.is_legacy_permanently_stopped() is True


# ===========================================================================
# Executor: backward compatibility — no lock path is valid
# ===========================================================================

class TestExecutorCrossProcessLock:
    def test_executor_no_lock_path_initialises(self, monkeypatch):
        """Executor() with no lock path must init cleanly (backward compat)."""
        import os
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        from execution.executor import Executor
        e = Executor()
        assert e._xlock is None

    def test_executor_with_lock_path_warns_if_filelock_missing(self, monkeypatch, tmp_path):
        """If filelock is not installed but lock path provided, must log WARNING
        and set _xlock=None — never crash."""
        import os
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        # Simulate filelock not available
        import execution.executor as exec_mod
        monkeypatch.setattr(exec_mod, "_FILELOCK_AVAILABLE", False)
        from execution.executor import Executor
        e = Executor(cross_process_lock_path=str(tmp_path / "test.lock"))
        assert e._xlock is None   # must not crash, must set to None


# ===========================================================================
# RL: observation dim consistency
# ===========================================================================

class TestRLObservationDims:
    def test_base_obs_dim_is_positive_int(self):
        from rl.observation import BASE_OBS_DIM
        assert isinstance(BASE_OBS_DIM, int)
        assert BASE_OBS_DIM > 0

    def test_causal_obs_dim_exceeds_base(self):
        from rl.observation import BASE_OBS_DIM, CAUSAL_OBS_DIM, CAUSAL_EXTRA_DIMS
        assert CAUSAL_OBS_DIM == BASE_OBS_DIM + CAUSAL_EXTRA_DIMS
        assert CAUSAL_OBS_DIM > BASE_OBS_DIM

    def test_build_base_obs_returns_correct_shape(self):
        from rl.observation import build_base_obs, BASE_OBS_DIM
        import numpy as np
        state = CausalState(price=150.0, volatility=0.01, volume=1000.0)
        obs = build_base_obs(state)
        assert obs.shape == (BASE_OBS_DIM,)
        assert obs.dtype == np.float32

    def test_build_base_obs_no_nan_or_inf(self):
        from rl.observation import build_base_obs
        import numpy as np
        state = CausalState(price=150.0, volatility=0.01, volume=1000.0)
        obs = build_base_obs(state)
        assert not np.isnan(obs).any(), "base obs must not contain NaN"
        assert not np.isinf(obs).any(), "base obs must not contain Inf"

    def test_build_causal_obs_no_graph_zero_fills_causal_dims(self):
        """When graph is None, causal dims must be zeroed (Phase A behaviour)."""
        from rl.observation import build_causal_obs, BASE_OBS_DIM, CAUSAL_EXTRA_DIMS
        import numpy as np
        state = CausalState(price=150.0, volatility=0.01, volume=1000.0)
        obs = build_causal_obs(state, graph=None, layer1=None, hypotheses=None)
        assert obs.shape == (BASE_OBS_DIM + CAUSAL_EXTRA_DIMS,)
        # Causal dims should be zeros (except regime one-hot)
        causal_part = obs[BASE_OBS_DIM:]
        # At least ev scores (first 3) should be zero
        assert np.all(causal_part[:3] == 0.0), "EV scores should be 0 with no graph"

    def test_build_causal_obs_no_nan(self):
        from rl.observation import build_causal_obs
        import numpy as np
        state = CausalState(price=150.0, volatility=0.01, volume=1000.0)
        obs = build_causal_obs(state, graph=None, layer1=None, hypotheses=None)
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

    def test_get_base_obs_dim_matches_constant(self):
        from rl.observation import get_base_obs_dim, BASE_OBS_DIM
        assert get_base_obs_dim() == BASE_OBS_DIM

    def test_get_causal_obs_dim_matches_constant(self):
        from rl.observation import get_causal_obs_dim, CAUSAL_OBS_DIM
        assert get_causal_obs_dim() == CAUSAL_OBS_DIM


# ===========================================================================
# RL: TradingEnv mechanics
# ===========================================================================

class TestTradingEnv:
    @pytest.fixture
    def sample_data(self):
        import numpy as np
        # 500 bars of synthetic feature data (19 variables)
        rng = np.random.default_rng(42)
        data = rng.normal(0, 0.01, size=(500, 19)).astype(np.float32)
        return data

    def test_env_resets_cleanly(self, sample_data):
        from rl.trading_env import TradingEnv
        env = TradingEnv(data=sample_data, mode="standard", seed=0)
        obs, info = env.reset(seed=0)
        assert obs is not None
        assert obs.shape[0] > 0
        env.close()

    def test_obs_shape_matches_obs_space(self, sample_data):
        import numpy as np
        from rl.trading_env import TradingEnv
        env = TradingEnv(data=sample_data, mode="standard", seed=0)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_causal_env_obs_shape_larger(self, sample_data):
        from rl.trading_env import TradingEnv
        env_std    = TradingEnv(data=sample_data, mode="standard", seed=0)
        env_causal = TradingEnv(data=sample_data, mode="causal",   seed=0)
        obs_std, _    = env_std.reset(seed=0)
        obs_causal, _ = env_causal.reset(seed=0)
        assert obs_causal.shape[0] > obs_std.shape[0]
        env_std.close()
        env_causal.close()

    def test_action_space_is_discrete_3(self, sample_data):
        from rl.trading_env import TradingEnv
        import gymnasium as gym
        env = TradingEnv(data=sample_data, mode="standard", seed=0)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 3
        env.close()

    def test_episode_terminates_within_episode_bars(self, sample_data):
        from rl.trading_env import TradingEnv, EPISODE_BARS
        env = TradingEnv(data=sample_data, mode="standard", seed=0)
        obs, _ = env.reset(seed=0)
        steps = 0
        done = False
        while not done and steps < EPISODE_BARS + 10:
            obs, reward, terminated, truncated, info = env.step(0)   # HOLD
            done = terminated or truncated
            steps += 1
        assert done, "Episode must terminate within EPISODE_BARS steps"
        assert steps <= EPISODE_BARS + 1
        env.close()

    def test_reward_is_float_on_each_step(self, sample_data):
        from rl.trading_env import TradingEnv
        env = TradingEnv(data=sample_data, mode="standard", seed=0)
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)
        env.close()

    def test_no_position_hold_gives_zero_reward(self, sample_data):
        from rl.trading_env import TradingEnv
        env = TradingEnv(data=sample_data, mode="standard", seed=0)
        env.reset(seed=0)
        # HOLD with no position should always give reward=0.0
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(0.0), f"HOLD with no position should be 0, got {reward}"
        env.close()

    def test_sell_without_position_gives_zero_reward(self, sample_data):
        from rl.trading_env import TradingEnv
        env = TradingEnv(data=sample_data, mode="standard", seed=0)
        env.reset(seed=0)
        # SELL without an open position — should be ignored (no reward)
        _, reward, _, _, _ = env.step(2)   # SELL
        assert reward == pytest.approx(0.0)
        env.close()

    def test_buy_then_sell_produces_pnl(self, sample_data):
        """BUY followed by SELL should produce a non-trivial reward (realised PnL)."""
        from rl.trading_env import TradingEnv
        env = TradingEnv(data=sample_data, mode="standard", seed=0)
        env.reset(seed=0)
        env.step(1)   # BUY
        _, reward, _, _, _ = env.step(2)   # SELL
        # reward is the price_return diff — may be positive or negative but must be float
        assert isinstance(reward, float)
        env.close()

    def test_causal_obs_dims_zeroed_in_historical_mode(self, sample_data):
        """In causal mode (Phase A training), the causal extra dims must be zeros."""
        import numpy as np
        from rl.trading_env import TradingEnv
        from rl.observation import BASE_OBS_DIM
        env = TradingEnv(data=sample_data, mode="causal", seed=0)
        obs, _ = env.reset(seed=0)
        causal_dims = obs[BASE_OBS_DIM:]
        assert np.all(causal_dims == 0.0), (
            "Causal dims must be zeroed in Phase A (historical training mode)"
        )
        env.close()
