# env/xauusd_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class XAUUSDTradingEnv(gym.Env):
    """
    Discrete trading env.

    Long-only actions:
      0 = Flat
      1 = Long

    Long-short actions:
      0 = Short
      1 = Flat
      2 = Long

    Position is applied on the NEXT step to avoid look-ahead.
    Reward = pnl - trade_cost - turnover_penalty - flat_penalty + hold_bonus
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,             # (T, F)
        returns: np.ndarray,              # (T,)
        window: int = 64,
        cost_per_trade: float = 0.0001,   # cost per unit position change
        turnover_coef: float = 0.0002,    # extra penalty per unit position change
        flat_penalty: float = 0.0,
        hold_bonus: float = 0.0,
        position_mode: str = "long_only",
        random_start: bool = False,
        decision_interval: int = 1,
        force_flat_mask: np.ndarray | None = None,
        max_episode_steps: int | None = None,
    ):
        super().__init__()
        assert features.ndim == 2
        assert returns.ndim == 1
        assert len(features) == len(returns)

        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)

        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.turnover_coef = float(turnover_coef)
        self.flat_penalty = float(flat_penalty)
        self.hold_bonus = float(hold_bonus)
        if position_mode not in {"long_only", "long_short"}:
            raise ValueError("position_mode must be 'long_only' or 'long_short'")
        self.position_mode = position_mode
        self.random_start = bool(random_start)
        self.decision_interval = max(1, int(decision_interval))
        if force_flat_mask is not None and len(force_flat_mask) != len(self.r):
            raise ValueError("force_flat_mask must have the same length as returns")
        self.force_flat_mask = None if force_flat_mask is None else np.asarray(force_flat_mask, dtype=bool)

        self.T = len(self.r)
        self.max_episode_steps = max_episode_steps

        obs_dim = self.window * self.X.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(2 if self.position_mode == "long_only" else 3)

        self._reset_state()

    def _reset_state(self):
        if self.random_start and self.max_episode_steps is not None:
            latest_start = max(self.window, self.T - self.max_episode_steps - 1)
            self.t = int(self.np_random.integers(self.window, latest_start + 1))
        else:
            self.t = self.window
        self.pos = 0
        self.steps = 0
        self.equity = 1.0

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]  # (window, F)
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        action = int(action)
        if self.position_mode == "long_short":
            requested_pos = {0: -1, 1: 0, 2: 1}.get(action, 0)
        else:
            requested_pos = 1 if action == 1 else 0

        can_change_position = (self.steps % self.decision_interval) == 0
        force_flat = self.force_flat_mask is not None and bool(self.force_flat_mask[self.t])
        if force_flat:
            new_pos = 0
        else:
            new_pos = requested_pos if can_change_position else self.pos

        # position change magnitude. Direct short->long flip costs 2 units.
        delta = abs(new_pos - self.pos)

        # costs/penalties for changing position
        trade_cost = self.cost * delta
        turnover_penalty = self.turnover_coef * delta

        # pnl from holding PREVIOUS position over this bar
        pnl = self.pos * self.r[self.t]

        # penalize being flat (nudges agent to stay exposed in drift markets)
        flat_pen = self.flat_penalty if new_pos == 0 else 0.0

        # bonus for holding the same position (nudges stability, reduces flip-flop)
        hold_bonus = self.hold_bonus if delta == 0 else 0.0

        reward = pnl - trade_cost - turnover_penalty - flat_pen + hold_bonus

        # track equity
        self.equity *= (1.0 + reward)

        # update position after reward (avoid look-ahead)
        self.pos = new_pos

        # advance time
        self.t += 1
        self.steps += 1

        terminated = self.t >= self.T
        truncated = False
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True

        info = {
            "equity": float(self.equity),
            "pos": int(self.pos),
            "trade_cost": float(trade_cost),
            "can_change_position": bool(can_change_position),
            "force_flat": bool(force_flat),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
