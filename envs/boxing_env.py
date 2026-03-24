"""
boxing_env.py
=============
Gymnasium-compatible MultiAgentEnv for the 3-D humanoid boxing simulation.

Observation layout (75-dim per agent)
--------------------------------------
[0:3]   own pelvis world position          (x, y, z)
[3:7]   own pelvis orientation quaternion  (w, x, y, z)
[7:10]  own pelvis linear velocity         (vx, vy, vz)
[10:13] own pelvis angular velocity        (wx, wy, wz)
[13:34] own 21 joint angles
[34:55] own 21 joint velocities
[55:58] opponent pelvis relative position  (dx, dy, dz)
[58:61] opponent pelvis relative velocity  (dvx, dvy, dvz)
[61:64] own left-fist world position       (x, y, z)
[64:67] own right-fist world position      (x, y, z)
[67:70] opponent left-fist relative pos    (dx, dy, dz)
[70:73] opponent right-fist relative pos   (dx, dy, dz)
[73]    own HP / 100
[74]    opponent HP / 100

Action layout (21-dim per agent, torque control ±1 mapped through gear)
------------------------------------------------------------------------
 0  abdomen_y
 1-3  left_hip_x/y/z
 4  left_knee
 5-6  left_ankle_x/y
 7-9  right_hip_x/y/z
 10  right_knee
 11-12  right_ankle_x/y
 13-15  left_shoulder_x/y/z
 16  left_elbow
 17-19  right_shoulder_x/y/z
 20  right_elbow

Reference State Initialization (DeepMimic §3.1)
-------------------------------------------------
  10 % probability → pure rest pose (all joints = 0, standing upright)
  90 % probability → standing pose + random perturbation
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco

# reward module (optional import so env itself stays usable without training/)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from training.reward import compute_reward as _compute_reward
    _REWARD_AVAILABLE = True
except ImportError:
    _REWARD_AVAILABLE = False

_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", "boxer.xml")

# ── anatomical constants ──────────────────────────────────────────────────────
N_JOINTS      = 21          # actuated joints per fighter
OBS_DIM       = 75          # observation dimension per fighter
ACT_DIM       = 21          # action dimension per fighter
MAX_STEPS     = 1000
TORSO_KO_Z    = 0.45        # torso centre below this → KO
INIT_HP       = 100.0
HP_FORCE_SCALE = 0.08       # contact_force → HP damage
HP_DAMAGE_MAX  = 8.0        # cap per single contact

# ── body-name look-up helpers ─────────────────────────────────────────────────
_PREFIXES = ("b0", "b1")

# Joints per fighter in order (must match actuator order in XML)
_JOINT_NAMES = [
    "{p}_abdomen_y",
    "{p}_left_hip_x",  "{p}_left_hip_y",  "{p}_left_hip_z",
    "{p}_left_knee",
    "{p}_left_ankle_x", "{p}_left_ankle_y",
    "{p}_right_hip_x", "{p}_right_hip_y", "{p}_right_hip_z",
    "{p}_right_knee",
    "{p}_right_ankle_x", "{p}_right_ankle_y",
    "{p}_left_shoulder_x", "{p}_left_shoulder_y", "{p}_left_shoulder_z",
    "{p}_left_elbow",
    "{p}_right_shoulder_x", "{p}_right_shoulder_y", "{p}_right_shoulder_z",
    "{p}_right_elbow",
]


def _joint_names_for(prefix: str):
    return [n.format(p=prefix) for n in _JOINT_NAMES]


class BoxingEnv(gym.Env):
    """
    Two-agent humanoid boxing environment.

    Usage (single-agent compatible wrapper expected for SB3):
        env = BoxingEnv()
        obs, info = env.reset()
        # obs is a dict: {"fighter_0": array(75,), "fighter_1": array(75,)}

    For curriculum Phase 1 (single-agent balance), use:
        env = BoxingEnv(single_agent_mode=True, active_agent="fighter_0")
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = _XML_PATH,
        max_steps: int = MAX_STEPS,
        single_agent_mode: bool = False,
        active_agent: str = "fighter_0",
        render_mode: str | None = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.max_steps         = max_steps
        self.single_agent_mode = single_agent_mode
        self.active_agent      = active_agent
        self.render_mode       = render_mode

        self.agents = ["fighter_0", "fighter_1"]

        # ── spaces ───────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        # ── cache joint/body indices ──────────────────────────────────────────
        self._jnt_qposadr = {}   # joint_name → qpos address
        self._jnt_dofadr  = {}   # joint_name → qvel address
        for prefix in _PREFIXES:
            for jname in _joint_names_for(prefix):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    raise ValueError(f"Joint not found in XML: {jname}")
                self._jnt_qposadr[jname] = self.model.jnt_qposadr[jid]
                self._jnt_dofadr[jname]  = self.model.jnt_dofadr[jid]

        # freejoint qpos/qvel addresses
        self._root_qposadr = {}
        self._root_dofadr  = {}
        for i, prefix in enumerate(_PREFIXES):
            rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_root")
            self._root_qposadr[prefix] = self.model.jnt_qposadr[rid]
            self._root_dofadr[prefix]  = self.model.jnt_dofadr[rid]

        # body ids
        self._bid = {}
        for prefix in _PREFIXES:
            for bname in [
                f"{prefix}_pelvis", f"{prefix}_torso",
                f"{prefix}_left_fist", f"{prefix}_right_fist",
            ]:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
                if bid < 0:
                    raise ValueError(f"Body not found in XML: {bname}")
                self._bid[bname] = bid

        # geom ids for contact detection
        self._fist_geoms  = {}   # prefix → {geom_id}
        self._torso_geoms = {}   # prefix → {geom_id}
        ngeom = self.model.ngeom
        for gi in range(ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if gname is None:
                continue
            for prefix in _PREFIXES:
                if gname.startswith(prefix):
                    if "fist" in gname:
                        self._fist_geoms.setdefault(prefix, set()).add(gi)
                    if "torso" in gname or "head" in gname or "pelvis" in gname:
                        self._torso_geoms.setdefault(prefix, set()).add(gi)

        # ── runtime state ─────────────────────────────────────────────────────
        self.hp: dict[str, float] = {}
        self.step_count = 0
        self._contact_log: list[tuple[str, str, float]] = []  # (attacker, defender, force)

        # renderer (lazy init)
        self._renderer = None

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._apply_rsi()
        mujoco.mj_forward(self.model, self.data)

        self.hp = {"fighter_0": INIT_HP, "fighter_1": INIT_HP}
        self.step_count  = 0
        self._contact_log = []

        obs  = self._get_obs_dict()
        info = self._make_info()

        if self.single_agent_mode:
            return obs[self.active_agent], info
        return obs, info

    def step(self, action):
        """
        In multi-agent mode:  action = {"fighter_0": arr21, "fighter_1": arr21}
        In single-agent mode: action = arr21  (opponent is zeroed out)
        """
        if self.single_agent_mode:
            actions = {self.active_agent: np.asarray(action, dtype=np.float32)}
            opp = "fighter_1" if self.active_agent == "fighter_0" else "fighter_0"
            actions[opp] = np.zeros(ACT_DIM, dtype=np.float32)
        else:
            actions = {k: np.asarray(v, dtype=np.float32) for k, v in action.items()}

        # ── apply controls ────────────────────────────────────────────────────
        self.data.ctrl[: ACT_DIM]            = np.clip(actions["fighter_0"], -1, 1)
        self.data.ctrl[ACT_DIM : 2 * ACT_DIM] = np.clip(actions["fighter_1"], -1, 1)

        prev_hp = self.hp.copy()
        self._contact_log.clear()

        prev_state = self.get_sim_state()

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        self._process_contacts()

        obs  = self._get_obs_dict()
        info = self._make_info(prev_hp=prev_hp)

        terminated = self._check_ko()
        truncated  = self.step_count >= self.max_steps

        cur_state = self.get_sim_state()

        if _REWARD_AVAILABLE:
            rewards = {}
            reward_components = {}
            for agent in self.agents:
                act = actions[agent]
                r, comps = _compute_reward(agent, cur_state, prev_state, act, self.hp, prev_hp)
                rewards[agent] = r
                reward_components[agent] = comps
            info["reward_components"] = reward_components
        else:
            rewards = {"fighter_0": 0.0, "fighter_1": 0.0}

        if self.single_agent_mode:
            return (
                obs[self.active_agent],
                rewards[self.active_agent],
                terminated,
                truncated,
                info,
            )
        return obs, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "human":
            self._render_human()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs_dict(self) -> dict[str, np.ndarray]:
        return {
            "fighter_0": self._get_obs("fighter_0"),
            "fighter_1": self._get_obs("fighter_1"),
        }

    def _get_obs(self, agent_id: str) -> np.ndarray:
        idx     = 0 if agent_id == "fighter_0" else 1
        prefix  = _PREFIXES[idx]
        oprefix = _PREFIXES[1 - idx]

        # ── own freejoint ─────────────────────────────────────────────────────
        rpa = self._root_qposadr[prefix]
        rda = self._root_dofadr[prefix]
        pelvis_pos  = self.data.qpos[rpa     : rpa + 3].copy()   # xyz
        pelvis_quat = self.data.qpos[rpa + 3 : rpa + 7].copy()   # w,x,y,z
        pelvis_linv = self.data.qvel[rda     : rda + 3].copy()   # vx,vy,vz
        pelvis_angv = self.data.qvel[rda + 3 : rda + 6].copy()   # wx,wy,wz

        # ── own joint angles & velocities ─────────────────────────────────────
        jnames  = _joint_names_for(prefix)
        jangles = np.array([self.data.qpos[self._jnt_qposadr[j]] for j in jnames], dtype=np.float32)
        jvels   = np.array([self.data.qvel[self._jnt_dofadr[j]]  for j in jnames], dtype=np.float32)

        # ── opponent freejoint ────────────────────────────────────────────────
        orpa = self._root_qposadr[oprefix]
        orda = self._root_dofadr[oprefix]
        opp_pos  = self.data.qpos[orpa : orpa + 3]
        opp_linv = self.data.qvel[orda : orda + 3]

        opp_rel_pos = (opp_pos - pelvis_pos).astype(np.float32)
        opp_rel_vel = (opp_linv - pelvis_linv).astype(np.float32)

        # ── fist positions ────────────────────────────────────────────────────
        own_lf_pos = self.data.xpos[self._bid[f"{prefix}_left_fist"]].astype(np.float32)
        own_rf_pos = self.data.xpos[self._bid[f"{prefix}_right_fist"]].astype(np.float32)
        opp_lf_pos = self.data.xpos[self._bid[f"{oprefix}_left_fist"]].astype(np.float32)
        opp_rf_pos = self.data.xpos[self._bid[f"{oprefix}_right_fist"]].astype(np.float32)
        opp_lf_rel = opp_lf_pos - pelvis_pos.astype(np.float32)
        opp_rf_rel = opp_rf_pos - pelvis_pos.astype(np.float32)

        # ── HP ────────────────────────────────────────────────────────────────
        own_hp = np.float32(self.hp[agent_id]     / INIT_HP)
        opp_hp = np.float32(self.hp[
            "fighter_1" if agent_id == "fighter_0" else "fighter_0"
        ] / INIT_HP)

        obs = np.concatenate([
            pelvis_pos.astype(np.float32),   # 3
            pelvis_quat.astype(np.float32),  # 4
            pelvis_linv.astype(np.float32),  # 3
            pelvis_angv.astype(np.float32),  # 3
            jangles,                          # 21
            jvels,                            # 21
            opp_rel_pos,                      # 3
            opp_rel_vel,                      # 3
            own_lf_pos,                       # 3
            own_rf_pos,                       # 3
            opp_lf_rel,                       # 3
            opp_rf_rel,                       # 3
            [own_hp, opp_hp],                 # 2
        ]).astype(np.float32)                 # total = 75

        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != ({OBS_DIM},)"  # nq=56(7+21)*2 nv=54
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # Contact / HP
    # ──────────────────────────────────────────────────────────────────────────

    def _process_contacts(self):
        force_buf = np.zeros(6, dtype=np.float64)

        for ci in range(self.data.ncon):
            c  = self.data.contact[ci]
            g1 = c.geom1
            g2 = c.geom2

            for attacker_p, defender_p, fg, tg in [
                ("b0", "b1", g1, g2),
                ("b0", "b1", g2, g1),
                ("b1", "b0", g1, g2),
                ("b1", "b0", g2, g1),
            ]:
                if (
                    fg in self._fist_geoms.get(attacker_p, set())
                    and tg in self._torso_geoms.get(defender_p, set())
                ):
                    mujoco.mj_contactForce(self.model, self.data, ci, force_buf)
                    normal_force = abs(force_buf[0])
                    damage = min(normal_force * HP_FORCE_SCALE, HP_DAMAGE_MAX)

                    defender_agent = "fighter_0" if defender_p == "b0" else "fighter_1"
                    attacker_agent = "fighter_0" if attacker_p == "b0" else "fighter_1"
                    self.hp[defender_agent] = max(0.0, self.hp[defender_agent] - damage)
                    self._contact_log.append((attacker_agent, defender_agent, normal_force))
                    break  # count each contact once

    def _check_ko(self) -> bool:
        for prefix in _PREFIXES:
            bid  = self._bid[f"{prefix}_torso"]
            torso_z = self.data.xpos[bid][2]
            if torso_z < TORSO_KO_Z:
                return True
        if self.hp["fighter_0"] <= 0 or self.hp["fighter_1"] <= 0:
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Reference State Initialization  (DeepMimic §3.1)
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_rsi(self):
        """
        Rest pose with 10 % probability; otherwise standing pose + noise.
        Only hinge joints are perturbed (not freejoint translation/height).
        """
        if np.random.random() < 0.10:
            # Pure rest – data is already reset; just set initial pelvis heights.
            self._set_pelvis_heights()
            return

        self._set_pelvis_heights()

        # Perturb hinge joint angles with small noise
        # Kept tiny so the agent always starts near-upright (easier to learn balance)
        noise_scale = np.random.uniform(0.01, 0.05)
        for prefix in _PREFIXES:
            for jname in _joint_names_for(prefix):
                jid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                adr  = self.model.jnt_qposadr[jid]
                lo   = self.model.jnt_range[jid, 0]
                hi   = self.model.jnt_range[jid, 1]
                noise = np.random.randn() * noise_scale
                self.data.qpos[adr] = float(np.clip(noise, lo, hi))

        # Small initial velocity noise
        self.data.qvel[:] = np.random.randn(self.model.nv) * 0.05

    def _set_pelvis_heights(self):
        """Place both pelvis freejoints at standing height and correct x offset."""
        for prefix, x_pos in zip(_PREFIXES, [-0.75, 0.75]):
            rpa = self._root_qposadr[prefix]
            self.data.qpos[rpa + 0] = x_pos
            self.data.qpos[rpa + 1] = 0.0
            self.data.qpos[rpa + 2] = 0.83    # standing height
            # quaternion: identity for b0, 180° around z for b1
            if prefix == "b0":
                self.data.qpos[rpa + 3] = 1.0  # w
                self.data.qpos[rpa + 4] = 0.0
                self.data.qpos[rpa + 5] = 0.0
                self.data.qpos[rpa + 6] = 0.0
            else:
                # 180° around z → quat = (0, 0, 0, 1) → w=0, x=0, y=0, z=1
                self.data.qpos[rpa + 3] = 0.0
                self.data.qpos[rpa + 4] = 0.0
                self.data.qpos[rpa + 5] = 0.0
                self.data.qpos[rpa + 6] = 1.0

    # ──────────────────────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────────────────────

    def _make_info(self, prev_hp: dict | None = None) -> dict:
        info: dict = {
            "step":       self.step_count,
            "hp":         self.hp.copy(),
            "contacts":   list(self._contact_log),
        }
        if prev_hp is not None:
            info["hp_delta"] = {
                k: prev_hp[k] - self.hp[k] for k in self.agents
            }
        for prefix, agent in zip(_PREFIXES, self.agents):
            bid = self._bid[f"{prefix}_torso"]
            info[f"{agent}_torso_z"] = float(self.data.xpos[bid][2])
        return info

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────────

    def _get_renderer(self, width=480, height=360):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderer

    def _render_rgb(self, width=480, height=360) -> np.ndarray:
        renderer = self._get_renderer(width, height)
        renderer.update_scene(self.data, camera="fixed_cam" if self._has_cam("fixed_cam") else -1)
        return renderer.render()

    def _render_human(self):
        """Launch MuJoCo passive viewer (blocking)."""
        import mujoco.viewer as mjviewer
        with mjviewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

    def _has_cam(self, name: str) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────────

    def get_sim_state(self) -> dict:
        """Return a snapshot of sim state for reward computation."""
        state: dict = {"hp": self.hp.copy(), "contacts": list(self._contact_log)}
        for prefix, agent in zip(_PREFIXES, self.agents):
            bid = self._bid[f"{prefix}_torso"]
            torso_pos = self.data.xpos[bid].copy()
            state[f"{agent}_torso_z"]   = float(torso_pos[2])
            state[f"{agent}_pos_xy"]    = torso_pos[:2].copy()
            state[f"{agent}_torso_pos"] = torso_pos          # 3D for fist reward
            # pelvis velocity
            rda = self._root_dofadr[prefix]
            state[f"{agent}_pelvis_vel"] = self.data.qvel[rda : rda + 3].copy()
            # fist positions (3D)
            for side in ("left", "right"):
                fid = self._bid[f"{prefix}_{side}_fist"]
                state[f"{agent}_{side}_fist_pos"] = self.data.xpos[fid].copy()
        return state
