"""
training/curriculum.py
======================
Phase-based curriculum learning manager + Gymnasium wrapper.

Phases
------
1  Balance     (0 – 1 M steps)  : single agent, upright + survival only
2  Locomotion  (1 M – 3 M steps): single agent, + velocity toward ring centre
3  Strike      (3 M – 8 M steps): single agent vs static dummy, + punch reward
4  Self-Play   (8 M + steps)    : two live agents, full reward (→ self_play.py)

Phase advances when  mean episode reward > threshold  OR  step budget exceeded.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco
from envs.boxing_env import BoxingEnv, _PREFIXES, N_JOINTS, OBS_DIM, ACT_DIM, TORSO_KO_Z

# ── phase thresholds ───────────────────────────────────────────────────────────
PHASE_CONFIG = {
    1: {"name": "Balance",    "step_budget": 1_000_000, "reward_threshold": 80.0},
    2: {"name": "Locomotion", "step_budget": 2_000_000, "reward_threshold": 120.0},
    3: {"name": "Strike",     "step_budget": 5_000_000, "reward_threshold": 180.0},
    4: {"name": "Self-Play",  "step_budget": float("inf"), "reward_threshold": float("inf")},
}

# ring centre (x=0, y=0)
RING_CENTRE = np.array([0.0, 0.0], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
class CurriculumEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper around BoxingEnv with phase-specific rewards.
    Compatible with stable-baselines3.

    Observation: same 75-dim as BoxingEnv (fighter_0 view).
    Action     : 21-dim torque control for fighter_0.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        phase: int = 1,
        render_mode: str | None = None,
        xml_path: str | None = None,
    ):
        super().__init__()
        kwargs = dict(
            single_agent_mode=True,
            active_agent="fighter_0",
            render_mode=render_mode,
        )
        if xml_path:
            kwargs["xml_path"] = xml_path

        self.base_env   = BoxingEnv(**kwargs)
        self.phase      = phase
        self.render_mode = render_mode

        self.observation_space = self.base_env.observation_space  # Box(75,)
        self.action_space      = self.base_env.action_space       # Box(21,)

        # running reward window for phase-advance detection
        self._ep_rewards: deque[float] = deque(maxlen=50)
        self._ep_return  = 0.0
        self._total_steps = 0

        self._prev_sim_state: dict = {}

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._ep_rewards.append(self._ep_return)
        self._ep_return = 0.0
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._prev_sim_state = self.base_env.get_sim_state()
        return obs, info

    def step(self, action: np.ndarray):
        prev_hp     = self.base_env.hp.copy()
        prev_state  = self._prev_sim_state

        obs, _, terminated, truncated, info = self.base_env.step(action)
        self._total_steps += 1

        cur_state = self.base_env.get_sim_state()
        reward    = self._phase_reward(action, cur_state, prev_state,
                                       self.base_env.hp, prev_hp, info)

        self._ep_return      += reward
        self._prev_sim_state  = cur_state
        info["phase"]         = self.phase
        info["phase_name"]    = PHASE_CONFIG[self.phase]["name"]

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()

    # ── phase-specific reward ──────────────────────────────────────────────────

    def _phase_reward(
        self,
        action: np.ndarray,
        cur: dict,
        prev: dict,
        hp: dict,
        prev_hp: dict,
        info: dict,
    ) -> float:
        agent  = "fighter_0"
        torso_z     = cur.get(f"{agent}_torso_z", 1.0)
        torso_z_pre = prev.get(f"{agent}_torso_z", 1.0)
        standing    = torso_z >= TORSO_KO_Z

        # ── Phase 1: NO energy penalty (allows agent to use full control) ──
        # Phase 2+: small penalty to discourage flailing
        energy_pen = 0.0 if self.phase == 1 else -0.001 * float(np.sum(np.square(action)))

        # ── upright: quadratic, full bonus only when truly standing tall ───
        target_z = 0.90
        crouch_z = 0.70
        if torso_z >= target_z:
            upright = 0.5
        elif torso_z >= crouch_z:
            frac    = (torso_z - crouch_z) / (target_z - crouch_z)
            upright = 0.5 * (frac ** 2)
        else:
            upright = -0.5 * ((crouch_z - torso_z) / crouch_z)

        survival = 0.2 if standing else 0.0

        # ── foot contact bonus: reward both feet touching the ground ───────
        foot_bonus = self._foot_contact_bonus()

        # KO penalty
        own_ko = torso_z < TORSO_KO_Z and torso_z_pre >= TORSO_KO_Z
        ko_pen = -50.0 if own_ko else 0.0

        r = energy_pen + upright + survival + foot_bonus + ko_pen

        if self.phase >= 2:
            # velocity toward ring centre
            pelvis_vel = cur.get(f"{agent}_pelvis_vel", np.zeros(3))
            pelvis_pos = self.base_env.data.qpos[
                self.base_env._root_qposadr["b0"] : self.base_env._root_qposadr["b0"] + 2
            ]
            to_centre = RING_CENTRE - pelvis_pos.astype(np.float32)
            dist      = float(np.linalg.norm(to_centre))
            if dist > 0.1:
                direction  = to_centre / dist
                vel_toward = float(np.dot(pelvis_vel[:2], direction))
                r += 0.5 * np.clip(vel_toward, -1.0, 1.0)

        if self.phase >= 3:
            # proximity bonus: reward for approaching the dummy (max +0.5 at contact, 0 at 2m+)
            b0_pos = self.base_env.data.qpos[
                self.base_env._root_qposadr["b0"]:self.base_env._root_qposadr["b0"] + 2
            ]
            b1_pos = self.base_env.data.qpos[
                self.base_env._root_qposadr["b1"]:self.base_env._root_qposadr["b1"] + 2
            ]
            dist = float(np.linalg.norm(b0_pos - b1_pos))
            r += 0.5 * max(0.0, 1.0 - dist / 2.0)

            # punch reward on dummy (opponent is static → fist contacts still detected)
            for attacker, defender, force in cur.get("contacts", []):
                if attacker == agent:
                    r += 5.0 * float(force) * 0.01
            hp_dealt = max(0.0, prev_hp.get("fighter_1", 100.0) - hp.get("fighter_1", 100.0))
            r += 10.0 * hp_dealt
            opp_ko = cur.get("fighter_1_torso_z", 1.0) < TORSO_KO_Z
            if opp_ko:
                r += 100.0

        return float(r)

    # ── phase management ───────────────────────────────────────────────────────

    def _foot_contact_bonus(self) -> float:
        """Reward for each foot that is in contact with the floor."""
        env   = self.base_env
        bonus = 0.0
        floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_geoms = {
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "b0_left_foot_geom"),
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "b0_right_foot_geom"),
        }
        feet_touching = set()
        for ci in range(env.data.ncon):
            c  = env.data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if floor_id in (g1, g2):
                other = g2 if g1 == floor_id else g1
                if other in foot_geoms:
                    feet_touching.add(other)
        bonus = 0.15 * len(feet_touching)   # +0.15 per foot touching ground
        return bonus

    def should_advance(self) -> bool:
        """True when current phase's criterion is met."""
        if self.phase >= 4:
            return False
        cfg = PHASE_CONFIG[self.phase]
        mean_r = float(np.mean(self._ep_rewards)) if len(self._ep_rewards) >= 10 else -9999
        return (
            mean_r >= cfg["reward_threshold"]
            or self._total_steps >= cfg["step_budget"]
        )

    def advance_phase(self) -> int:
        """Increment phase and return new phase number."""
        if self.phase < 4:
            self.phase += 1
            print(
                f"\n[Curriculum] *** Phase {self.phase}: "
                f"{PHASE_CONFIG[self.phase]['name']} ***\n"
            )
        return self.phase

    @property
    def mean_ep_reward(self) -> float:
        if len(self._ep_rewards) == 0:
            return 0.0
        return float(np.mean(self._ep_rewards))


# ─────────────────────────────────────────────────────────────────────────────
class CurriculumCallback:
    """
    Minimal callback compatible with SB3's callback API.
    Call .on_step() after every env step; it checks phase advancement.
    """

    def __init__(self, curriculum_env: CurriculumEnv, verbose: int = 1):
        self.env     = curriculum_env
        self.verbose = verbose

    def on_step(self) -> bool:
        if self.env.should_advance():
            old = self.env.phase
            self.env.advance_phase()
            if self.verbose and old != self.env.phase:
                print(
                    f"  mean_ep_reward={self.env.mean_ep_reward:.1f}  "
                    f"total_steps={self.env._total_steps:,}"
                )
            if self.env.phase == 4:
                return False   # signal caller to switch to self-play
        return True
