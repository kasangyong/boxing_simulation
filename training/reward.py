"""
training/reward.py
==================
Modular reward shaping for the humanoid boxing simulation.

compute_reward(agent_id, sim_state, prev_sim_state, action, hp, prev_hp)
  → float

Reward components
-----------------
1. punch_reward    : +5 * punch_velocity  when fist hits opponent torso
2. hp_damage       : +(prev_opp_hp - cur_opp_hp) * 10
3. getting_hit     : -(prev_own_hp - cur_own_hp)  * 7
4. survival_bonus  : +0.1 per step while own torso is above KO threshold
5. ko_reward       : +100 / -100 for knocking out or being KO'd
6. energy_penalty  : -0.005 * sum(action ** 2)
7. upright_bonus   : +0.2 * (torso_z / 0.9), clamped to [0, 0.2]
8. proximity_bonus : +0.8 * max(0, 1 - dist/2.0)  — reward for being close
9. approach_bonus  : +0.5 * max(0, prev_dist - cur_dist)  — reward for closing distance
"""

from __future__ import annotations

import numpy as np

# ── thresholds ────────────────────────────────────────────────────────────────
TORSO_KO_Z       = 0.45    # must match boxing_env.py
STANDING_TORSO_Z = 0.90    # reference height for upright bonus

# ── reward weights ────────────────────────────────────────────────────────────
W_PUNCH_VEL   =  5.0
W_HP_DEALT    = 10.0
W_HP_RECEIVED =  7.0
W_SURVIVAL    =  0.1
W_KO_WIN      = 100.0
W_KO_LOSE     = 100.0
W_ENERGY      =  0.005
W_UPRIGHT     =  0.2
W_PROXIMITY   =  0.8   # max +0.8/step when adjacent, 0 at 2m+
W_APPROACH    =  0.5   # +0.5 per meter closed toward opponent this step
W_FIST_VEL    =  5.0   # reward per meter of fist movement toward opponent per step
PROXIMITY_MAX_DIST   = 2.0
FIST_STRIKE_RANGE    = 1.5  # only reward fist velocity when within this distance


def compute_reward(
    agent_id: str,
    sim_state: dict,
    prev_sim_state: dict,
    action: np.ndarray,
    hp: dict[str, float],
    prev_hp: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Parameters
    ----------
    agent_id       : "fighter_0" or "fighter_1"
    sim_state      : dict returned by env.get_sim_state()  (after step)
    prev_sim_state : same dict captured before step
    action         : (21,) array, the action taken this step
    hp             : current HP  {agent_id: float}
    prev_hp        : HP before this step

    Returns
    -------
    total_reward   : float
    components     : dict with individual reward terms (for logging)
    """
    opp_id = "fighter_1" if agent_id == "fighter_0" else "fighter_0"

    own_torso_z     = sim_state.get(f"{agent_id}_torso_z", 1.0)
    own_torso_z_pre = prev_sim_state.get(f"{agent_id}_torso_z", 1.0)
    opp_torso_z     = sim_state.get(f"{opp_id}_torso_z", 1.0)
    opp_torso_z_pre = prev_sim_state.get(f"{opp_id}_torso_z", 1.0)

    cur_own_hp  = hp[agent_id]
    cur_opp_hp  = hp[opp_id]
    pre_own_hp  = prev_hp[agent_id]
    pre_opp_hp  = prev_hp[opp_id]

    # ── 1. Punch reward ───────────────────────────────────────────────────────
    # For each fist-contact by this agent, reward proportional to contact force.
    # Velocity proxy: we use the normal contact force (already computed as HP damage).
    punch_r = 0.0
    for attacker, defender, force in sim_state.get("contacts", []):
        if attacker == agent_id:
            punch_r += W_PUNCH_VEL * float(force) * 0.01  # scale force → velocity proxy

    # ── 2. HP damage dealt ────────────────────────────────────────────────────
    hp_dealt   = max(0.0, pre_opp_hp - cur_opp_hp)
    hp_dealt_r = W_HP_DEALT * hp_dealt

    # ── 3. Getting hit penalty ────────────────────────────────────────────────
    hp_received   = max(0.0, pre_own_hp - cur_own_hp)
    hp_received_r = -W_HP_RECEIVED * hp_received

    # ── 4. Survival bonus (only while standing) ───────────────────────────────
    standing = own_torso_z >= TORSO_KO_Z
    survival_r = W_SURVIVAL if standing else 0.0

    # ── 5. KO reward / penalty ────────────────────────────────────────────────
    ko_r = 0.0
    own_ko  = own_torso_z < TORSO_KO_Z and own_torso_z_pre >= TORSO_KO_Z
    opp_ko  = opp_torso_z < TORSO_KO_Z and opp_torso_z_pre >= TORSO_KO_Z  # edge-triggered
    own_hp_ko = cur_own_hp <= 0.0 and pre_own_hp > 0.0
    opp_hp_ko = cur_opp_hp <= 0.0 and pre_opp_hp > 0.0

    if opp_ko or opp_hp_ko:
        ko_r += W_KO_WIN
    if own_ko or own_hp_ko:
        ko_r -= W_KO_LOSE

    # ── 6. Energy penalty ─────────────────────────────────────────────────────
    energy_r = -W_ENERGY * float(np.sum(np.square(action)))

    # ── 7. Upright bonus ──────────────────────────────────────────────────────
    upright_r = W_UPRIGHT * float(np.clip(own_torso_z / STANDING_TORSO_Z, 0.0, 1.0))

    # ── 8. Proximity bonus + approach bonus ──────────────────────────────────
    own_pos      = sim_state.get(f"{agent_id}_pos_xy")
    opp_pos      = sim_state.get(f"{opp_id}_pos_xy")
    own_pos_prev = prev_sim_state.get(f"{agent_id}_pos_xy")
    opp_pos_prev = prev_sim_state.get(f"{opp_id}_pos_xy")
    if own_pos is not None and opp_pos is not None:
        dist = float(np.linalg.norm(own_pos - opp_pos))
        proximity_r = W_PROXIMITY * max(0.0, 1.0 - dist / PROXIMITY_MAX_DIST)
        if own_pos_prev is not None and opp_pos_prev is not None:
            prev_dist = float(np.linalg.norm(own_pos_prev - opp_pos_prev))
            approach_r = W_APPROACH * max(0.0, prev_dist - dist)  # reward closing gap
        else:
            approach_r = 0.0
    else:
        proximity_r = 0.0
        approach_r  = 0.0

    # ── 9. Fist-toward-opponent velocity reward ───────────────────────────────
    # Rewards fist movement directed toward the opponent (within striking range).
    # Encourages punching motions even before contact is made.
    fist_r = 0.0
    opp_torso_pos = sim_state.get(f"{opp_id}_torso_pos")
    if opp_torso_pos is not None:
        for side in ("left", "right"):
            fist_cur  = sim_state.get(f"{agent_id}_{side}_fist_pos")
            fist_prev = prev_sim_state.get(f"{agent_id}_{side}_fist_pos")
            if fist_cur is not None and fist_prev is not None:
                to_opp = opp_torso_pos - fist_cur
                dist_fist_to_opp = float(np.linalg.norm(to_opp))
                if 0 < dist_fist_to_opp < FIST_STRIKE_RANGE:
                    to_opp_unit = to_opp / dist_fist_to_opp
                    fist_delta = fist_cur - fist_prev          # movement this step
                    vel_toward = float(np.dot(fist_delta, to_opp_unit))
                    fist_r += W_FIST_VEL * max(0.0, vel_toward)

    total = punch_r + hp_dealt_r + hp_received_r + survival_r + ko_r + energy_r + upright_r + proximity_r + approach_r + fist_r

    components = {
        "punch":     punch_r,
        "hp_dealt":  hp_dealt_r,
        "hp_recv":   hp_received_r,
        "survival":  survival_r,
        "ko":        ko_r,
        "energy":    energy_r,
        "upright":   upright_r,
        "proximity": proximity_r,
        "approach":  approach_r,
        "fist_vel":  fist_r,
    }
    return total, components
