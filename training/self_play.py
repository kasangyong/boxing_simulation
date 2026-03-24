"""
training/self_play.py
=====================
Self-play environment and callback for two-agent boxing.

- SelfPlayEnv   : single-agent Gymnasium wrapper around BoxingEnv (two agents live)
                  fighter_0 = learning agent (SB3 trains this)
                  fighter_1 = frozen opponent (updated every update_freq steps)
- SelfPlayCallback : updates the opponent policy periodically
"""

from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.boxing_env import BoxingEnv, TORSO_KO_Z, ACT_DIM
from training.reward import compute_reward

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3 = True
except ImportError:
    _SB3 = False


# ─────────────────────────────────────────────────────────────────────────────
class SelfPlayEnv(gym.Env):
    """
    Wraps BoxingEnv for self-play training.

    Observation : fighter_0's 75-dim view
    Action      : fighter_0's 21-dim action
    Reward      : full boxing reward (reward.py) for fighter_0

    fighter_1 acts using a frozen copy of the policy, updated via SelfPlayCallback.
    Before the first opponent is loaded, fighter_1 uses zero actions (static dummy).
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, opponent_path: str | None = None, render_mode: str | None = None):
        super().__init__()
        self.base_env    = BoxingEnv(single_agent_mode=False, render_mode=render_mode)
        self.render_mode = render_mode

        self.observation_space = self.base_env.observation_space  # Box(75,)
        self.action_space      = self.base_env.action_space       # Box(21,)

        self.opponent_model   = None
        self._opp_obs         = np.zeros(75, dtype=np.float32)
        self._prev_sim_state: dict = {}
        self._ep_return       = 0.0

        if opponent_path and os.path.exists(opponent_path):
            self.load_opponent(opponent_path)

    # ── opponent management ────────────────────────────────────────────────────
    def load_opponent(self, path: str):
        """Load a PPO checkpoint as the frozen opponent."""
        self.opponent_model = PPO.load(path)

    # ── Gymnasium API ──────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        self._ep_return = 0.0
        obs_dict, info  = self.base_env.reset(seed=seed, options=options)
        self._opp_obs         = obs_dict["fighter_1"]
        self._prev_sim_state  = self.base_env.get_sim_state()
        return obs_dict["fighter_0"], info

    def step(self, action: np.ndarray):
        prev_hp    = self.base_env.hp.copy()
        prev_state = self._prev_sim_state

        # ── opponent acts ──────────────────────────────────────────────────────
        if self.opponent_model is not None:
            opp_action, _ = self.opponent_model.predict(
                self._opp_obs, deterministic=False
            )
        else:
            opp_action = np.zeros(ACT_DIM, dtype=np.float32)

        action_dict = {
            "fighter_0": np.asarray(action,     dtype=np.float32),
            "fighter_1": np.asarray(opp_action, dtype=np.float32),
        }

        obs_dict, _, terminated, truncated, info = self.base_env.step(action_dict)

        cur_state        = self.base_env.get_sim_state()
        self._opp_obs    = obs_dict["fighter_1"]
        self._prev_sim_state = cur_state

        # ── reward (full boxing, fighter_0 perspective) ───────────────────────
        reward, components = compute_reward(
            "fighter_0", cur_state, prev_state,
            action_dict["fighter_0"], self.base_env.hp, prev_hp,
        )

        self._ep_return += reward
        info["reward_components"] = components
        info["ep_return"]         = self._ep_return

        return obs_dict["fighter_0"], float(reward), terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


# ─────────────────────────────────────────────────────────────────────────────
class SelfPlayCallback(BaseCallback):
    """
    Every `update_freq` training steps:
      1. Save the current policy to checkpoints/selfplay_opponent.zip
      2. Load it as fighter_1's frozen policy in all SelfPlayEnv instances
    This creates an automatic self-play curriculum: the opponent gets stronger
    as the learning agent improves.
    """

    def __init__(
        self,
        selfplay_env=None,      # unused (kept for backward compat)
        update_freq: int = 50_000,
        save_dir: str = "checkpoints",
        log_every_ep: int = 20,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.update_freq  = update_freq
        self.save_dir     = save_dir
        self.log_every_ep = log_every_ep

        self._last_update  = 0
        self._ep_count     = 0
        self._last_log_ep  = -1
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int]   = []

    def _on_step(self) -> bool:
        # ── episode tracking ──────────────────────────────────────────────────
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self._ep_rewards.append(float(ep["r"]))
                self._ep_lengths.append(int(ep["l"]))
                self._ep_count += 1

        if (
            self._ep_rewards
            and self._ep_count % self.log_every_ep == 0
            and self._ep_count != self._last_log_ep
        ):
            self._last_log_ep = self._ep_count
            self._print_episode_line()

        # ── opponent update ───────────────────────────────────────────────────
        if self.num_timesteps - self._last_update >= self.update_freq:
            self._last_update = self.num_timesteps
            path = os.path.join(self.save_dir, "selfplay_opponent")
            self.model.save(path)
            # update opponent in ALL parallel envs via VecEnv interface
            self.training_env.env_method("load_opponent", path + ".zip")
            if self.verbose:
                mean_r = float(np.mean(self._ep_rewards[-50:])) if self._ep_rewards else 0.0
                print(
                    f"\n  [SelfPlay] Opponent updated @ step {self.num_timesteps:,}"
                    f"  |  mean_R(50ep)={mean_r:.1f}\n"
                )
        return True

    def _print_episode_line(self):
        mean_r = float(np.mean(self._ep_rewards[-100:]))
        mean_l = float(np.mean(self._ep_lengths[-100:]))
        bar    = self._bar(mean_r, lo=-150, hi=300, width=20)
        print(
            f"  [SelfPlay] step={self.num_timesteps:>8,}  ep={self._ep_count:>5}  "
            f"R={mean_r:>7.1f} {bar}  len={mean_l:>5.0f}"
        )

    @staticmethod
    def _bar(val: float, lo: float, hi: float, width: int) -> str:
        frac   = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        filled = int(frac * width)
        return "[" + "#" * filled + "." * (width - filled) + "]"
