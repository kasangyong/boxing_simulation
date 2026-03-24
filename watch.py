"""
watch.py
========
Standalone MuJoCo viewer — auto-detects self-play vs curriculum checkpoints.

Run in a SEPARATE terminal:
  py -3.11 watch.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
import time
from collections import deque

import mujoco
import mujoco.viewer as mjv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from envs.boxing_env import BoxingEnv, TORSO_KO_Z

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

SUMMARY_EVERY = 10


def _ckpt_step_num(path: str) -> int:
    m = re.search(r"_(\d+)_steps", os.path.basename(path))
    return int(m.group(1)) if m else 0


def latest_checkpoint() -> str | None:
    selfplay = glob.glob("checkpoints/boxer_selfplay*.zip")
    if selfplay:
        return max(selfplay, key=_ckpt_step_num)
    others = glob.glob("checkpoints/boxer_p*.zip") + glob.glob("checkpoints/phase*.zip")
    if others:
        return max(others, key=_ckpt_step_num)
    return None


def is_selfplay(path: str) -> bool:
    return "selfplay" in os.path.basename(path)


def ckpt_steps(path: str) -> str:
    m = re.search(r"_(\d+)_steps", os.path.basename(path))
    return f"{int(m.group(1)):,}" if m else "?"


def bar(val: float, lo: float, hi: float, w: int = 20) -> str:
    f = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    return "[" + "#" * int(f * w) + "." * (w - int(f * w)) + "]"


def run_selfplay_episode(model, env: BoxingEnv, viewer, ep: int, ckpt: str) -> dict:
    obs_dict, _ = env.reset()
    obs0, obs1  = obs_dict["fighter_0"], obs_dict["fighter_1"]
    total_r0 = 0.0
    steps = 0
    done = False
    z0_list, z1_list, hp0_log, hp1_log = [], [], [], []

    while not done:
        if not viewer.is_running():
            break
        a0, _ = model.predict(obs0, deterministic=True)
        a1, _ = model.predict(obs1, deterministic=True)
        obs_dict, rewards, terminated, truncated, info = env.step(
            {"fighter_0": a0, "fighter_1": a1}
        )
        obs0, obs1 = obs_dict["fighter_0"], obs_dict["fighter_1"]
        total_r0 += float(rewards["fighter_0"])
        steps += 1
        done = terminated or truncated
        z0_list.append(info.get("fighter_0_torso_z", 1.0))
        z1_list.append(info.get("fighter_1_torso_z", 1.0))
        hp0_log.append(info["hp"]["fighter_0"])
        hp1_log.append(info["hp"]["fighter_1"])
        viewer.sync()
        time.sleep(0.005)

        if steps % 100 == 0 and not done:
            print(
                f"    step={steps:>4}  "
                f"RED  z={z0_list[-1]:.2f} HP={hp0_log[-1]:5.1f}  |  "
                f"BLUE z={z1_list[-1]:.2f} HP={hp1_log[-1]:5.1f}  |  cumR={total_r0:>6.1f}"
            )

    z0, z1 = np.array(z0_list), np.array(z1_list)
    ko0 = z0.min() < TORSO_KO_Z
    ko1 = z1.min() < TORSO_KO_Z
    winner = "BLUE wins" if ko0 and not ko1 else ("RED  wins" if ko1 and not ko0 else "Draw     ")
    hp0f = hp0_log[-1] if hp0_log else 100.0
    hp1f = hp1_log[-1] if hp1_log else 100.0

    print(
        f"\n  [ep {ep:>4}] {ckpt_steps(ckpt):>8} train-steps | {winner} | steps={steps}\n"
        f"           RED  z_min={z0.min():.2f}  HP={hp0f:.0f}  ko={ko0}\n"
        f"           BLUE z_min={z1.min():.2f}  HP={hp1f:.0f}  ko={ko1}\n"
        f"           R0={total_r0:>7.1f} {bar(total_r0, -200, 400)}"
    )
    return {"steps": steps, "reward": total_r0, "ko0": ko0, "ko1": ko1}


def run_solo_episode(model, env, viewer, ep: int, ckpt: str) -> dict:
    obs, _ = env.reset()
    total_r, steps, done = 0.0, 0, False
    while not done:
        if not viewer.is_running():
            break
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        total_r += r; steps += 1; done = term or trunc
        viewer.sync()
        time.sleep(0.005)
    print(f"  [ep {ep:>4}] {ckpt_steps(ckpt):>8} steps | "
          f"solo  steps={steps}  R={total_r:>7.1f} {bar(total_r,-50,200)}")
    return {"steps": steps, "reward": total_r}


def main():
    print("=" * 62)
    print("  Boxing RL - Live Viewer")
    print("  Self-play checkpoints auto-detected.")
    print("  Close the MuJoCo window to exit.")
    print("=" * 62 + "\n")

    base_env  = BoxingEnv(single_agent_mode=False)
    solo_env  = None
    model     = None
    last_ckpt = None
    selfplay_mode = False
    episode   = 0
    history: deque = deque(maxlen=SUMMARY_EVERY)

    with mjv.launch_passive(base_env.model, base_env.data) as viewer:
        viewer.cam.distance  = 4.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 160

        while viewer.is_running():
            ckpt = latest_checkpoint()

            if ckpt is None:
                print("  [watch] No checkpoint yet. Waiting...")
                for _ in range(200):
                    if not viewer.is_running():
                        break
                    base_env.data.ctrl[:] = 0
                    mujoco.mj_step(base_env.model, base_env.data)
                    viewer.sync()
                    time.sleep(0.005)
                continue

            if ckpt != last_ckpt:
                mode = "SELF-PLAY" if is_selfplay(ckpt) else "CURRICULUM"
                print(f"\n  [watch] [{mode}] {os.path.basename(ckpt)}"
                      f"  ({ckpt_steps(ckpt)} train steps)\n")
                try:
                    model         = PPO.load(ckpt)
                    last_ckpt     = ckpt
                    selfplay_mode = is_selfplay(ckpt)
                except Exception as e:
                    print(f"  [watch] Load failed: {e}")
                    time.sleep(2)
                    continue

            if model is None:
                continue

            episode += 1
            if selfplay_mode:
                stats = run_selfplay_episode(model, base_env, viewer, episode, last_ckpt)
            else:
                if solo_env is None:
                    from training.curriculum import CurriculumEnv
                    solo_env = CurriculumEnv(phase=1)
                stats = run_solo_episode(model, solo_env, viewer, episode, last_ckpt)

            history.append(stats)
            if episode % SUMMARY_EVERY == 0:
                rewards = [h["reward"] for h in history]
                lengths = [h["steps"]  for h in history]
                print(
                    f"\n  {'='*50}\n"
                    f"  SUMMARY last {len(history)} eps\n"
                    f"    mean_R={np.mean(rewards):>7.1f}  max_R={np.max(rewards):>7.1f}"
                    f"  mean_len={np.mean(lengths):>5.0f}\n"
                    f"  {'='*50}\n"
                )

    base_env.close()
    if solo_env:
        solo_env.close()
    print("\n[watch] Viewer closed.")


if __name__ == "__main__":
    main()
