"""
train.py
========
Main training entry point with live MuJoCo viewer.

Usage
-----
  # Full curriculum (Phase 1 → 4), viewer ON:
  python train.py

  # Phase 1 only, 1M steps:
  python train.py --phase 1 --total-steps 1000000

  # Resume from checkpoint:
  python train.py --phase 2 --load checkpoints/phase1_final.zip

Viewer behaviour
----------------
  Every --eval-freq steps the current policy is evaluated for 1 episode
  and rendered in the MuJoCo passive viewer window.
  The viewer stays open the entire training run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

sys.path.insert(0, os.path.dirname(__file__))
from training.curriculum import CurriculumEnv, PHASE_CONFIG
from training.self_play import SelfPlayEnv, SelfPlayCallback

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

# ── directories ───────────────────────────────────────────────────────────────
CKPT_DIR = "checkpoints"
LOG_DIR  = "logs"

# ── PPO hyper-parameters (spec §Phase-5) ──────────────────────────────────────
PPO_KWARGS = dict(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.001,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    target_kl     = 0.05,       # stop inner loop early if KL exceeds this
    verbose       = 1,
    policy_kwargs = {"net_arch": [256, 256, 128]},
)


# ─────────────────────────────────────────────────────────────────────────────
class VisualEvalCallback(BaseCallback):
    """
    Every `eval_freq` training steps, runs one evaluation episode
    with the MuJoCo passive viewer so the user can watch learning progress.
    """

    def __init__(self, curriculum_env: CurriculumEnv, eval_freq: int = 10_000):
        super().__init__(verbose=0)
        self.cenv      = curriculum_env
        self.eval_freq = eval_freq
        self._viewer   = None
        self._last_eval = 0

    def _init_callback(self) -> None:
        import mujoco.viewer as mjv
        # attach viewer to the training env's model/data so geometry is correct
        self._viewer = mjv.launch_passive(
            self.cenv.base_env.model,
            self.cenv.base_env.data,
        )
        print(
            "\n[Viewer] MuJoCo window opened."
            "  Every 10 000 training steps the current policy plays one episode.\n"
        )

    def _on_step(self) -> bool:
        if self._viewer is None or not self._viewer.is_running():
            return True

        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            self._run_eval_episode()

        return True

    def _run_eval_episode(self):
        obs, _ = self.cenv.reset()
        total_r, steps = 0.0, 0
        done = False
        while not done:
            if not self._viewer.is_running():
                break
            action, _ = self.model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = self.cenv.step(action)
            total_r += r
            steps   += 1
            done     = terminated or truncated
            self._viewer.sync()
            time.sleep(0.005)           # real-time playback (sim dt = 0.005 s)

        phase = self.cenv.phase
        print(
            f"  [Viewer | Phase {phase} | {PHASE_CONFIG[phase]['name']}] "
            f"step={self.num_timesteps:>8,}  "
            f"eval_ep: {steps} steps  reward={total_r:.1f}"
        )

    def _on_training_end(self) -> None:
        if self._viewer and self._viewer.is_running():
            self._viewer.close()


# ─────────────────────────────────────────────────────────────────────────────
class PhaseCallback(BaseCallback):
    """
    Tracks episode stats + PPO training diagnostics, logs progress,
    handles phase transitions.

    Episode-level log  (every log_every_ep episodes):
      Phase / steps / ep / mean_reward bar / ep_len

    Training diagnostic log  (every log_every_update updates = after each PPO batch):
      policy_loss  value_loss  entropy  approx_kl  clip_frac  explained_var  std
    """

    def __init__(
        self,
        curriculum_env: CurriculumEnv,
        use_wandb: bool = False,
        log_every_ep: int = 20,
        log_every_update: int = 5,   # print diagnostics every N PPO updates
    ):
        super().__init__(verbose=1)
        self.cenv             = curriculum_env
        self.use_wandb        = use_wandb
        self.log_every_ep     = log_every_ep
        self.log_every_update = log_every_update

        self._ep_count    = 0
        self._last_log_ep = -1
        self._update_count = 0
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int]   = []

    # ── called every env step ─────────────────────────────────────────────────
    def _on_step(self) -> bool:
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

        if self.cenv.should_advance():
            self.cenv.advance_phase()
            path = os.path.join(CKPT_DIR, f"phase{self.cenv.phase - 1}_final")
            self.model.save(path)
            print(f"  [Checkpoint] saved -> {path}.zip")
            if self.cenv.phase == 4:
                print("\n[train.py] Phase 4 (Self-Play) reached. Stopping curriculum.\n")
                return False

        return True

    # ── called after each PPO rollout + update ────────────────────────────────
    def _on_rollout_end(self) -> None:
        self._update_count += 1
        if self._update_count % self.log_every_update == 0:
            self._print_diagnostic_line()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _print_episode_line(self):
        mean_r = float(np.mean(self._ep_rewards[-100:]))
        mean_l = float(np.mean(self._ep_lengths[-100:]))
        phase  = self.cenv.phase
        bar    = self._progress_bar(mean_r, lo=-50, hi=200, width=20)
        print(
            f"  [Ph{phase} {PHASE_CONFIG[phase]['name']:10s}] "
            f"steps={self.num_timesteps:>8,}  ep={self._ep_count:>5}  "
            f"R={mean_r:>7.1f} {bar}  len={mean_l:>5.0f}"
        )
        if self.use_wandb and _WANDB:
            wandb.log({
                "phase": phase, "mean_reward": mean_r,
                "mean_ep_len": mean_l, "total_steps": self.num_timesteps,
            })

    def _print_diagnostic_line(self):
        """Read PPO internals from SB3 logger and print a compact diagnostics row."""
        lv = self.model.logger.name_to_value   # dict filled after each update

        def g(key, default=float("nan")):
            return float(lv.get(key, default))

        pg_loss = g("train/policy_gradient_loss")
        vf_loss = g("train/value_loss")
        ent     = g("train/entropy_loss")
        kl      = g("train/approx_kl")
        clip    = g("train/clip_fraction")
        ev      = g("train/explained_variance")
        std     = g("train/std")
        lr      = g("train/learning_rate")

        # colour-code explained_variance: <0=bad, 0-0.5=ok, >0.5=good
        ev_tag = "EV:BAD " if ev < 0.05 else ("EV:OK  " if ev < 0.5 else "EV:GOOD")

        print(
            f"    [Diag upd={self._update_count:>4}] "
            f"pg={pg_loss:+.4f}  vf={vf_loss:.4f}  "
            f"ent={ent:.4f}  kl={kl:.5f}  "
            f"clip={clip:.3f}  {ev_tag}={ev:.3f}  "
            f"std={std:.3f}  lr={lr:.2e}"
        )
        if self.use_wandb and _WANDB:
            wandb.log({
                "train/pg_loss": pg_loss, "train/vf_loss": vf_loss,
                "train/entropy": ent,     "train/approx_kl": kl,
                "train/clip_frac": clip,  "train/explained_var": ev,
                "train/std": std,         "total_steps": self.num_timesteps,
            })

    @staticmethod
    def _progress_bar(val: float, lo: float, hi: float, width: int) -> str:
        frac   = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        filled = int(frac * width)
        return "[" + "#" * filled + "." * (width - filled) + "]"


# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    if args.wandb and _WANDB:
        wandb.init(
            project = "boxing-rl",
            name    = f"curriculum_phase{args.phase}_{int(time.time())}",
            config  = vars(args),
        )

    # ── environment setup ─────────────────────────────────────────────────────
    if args.selfplay:
        n_envs = args.n_envs
        if n_envs > 1:
            vec_env = SubprocVecEnv([lambda: SelfPlayEnv() for _ in range(n_envs)],
                                    start_method="spawn")
        else:
            vec_env = DummyVecEnv([lambda: SelfPlayEnv()])
        vec_env = VecMonitor(vec_env)
        train_env = None  # not used directly; SelfPlayCallback uses training_env
    else:
        # Single-agent curriculum (Phase 1 balance)
        train_env = CurriculumEnv(phase=args.phase)
        vec_env   = DummyVecEnv([lambda: train_env])
        vec_env   = VecMonitor(vec_env)

    # ── PPO model ─────────────────────────────────────────────────────────────
    if args.load:
        print(f"[train.py] Resuming from: {args.load}")
        kwargs = {**PPO_KWARGS}
        if args.lr:
            kwargs["learning_rate"] = args.lr
            print(f"[train.py] Overriding learning_rate -> {args.lr}")
        if args.ent_coef is not None:
            kwargs["ent_coef"] = args.ent_coef
            print(f"[train.py] Overriding ent_coef -> {args.ent_coef}")
        model = PPO.load(args.load, env=vec_env, **{k: v for k, v in kwargs.items()
                                                     if k in ("learning_rate", "clip_range",
                                                               "ent_coef", "target_kl")})
        model.set_env(vec_env)
    else:
        model = PPO("MlpPolicy", vec_env, **PPO_KWARGS)

    # ── callbacks ─────────────────────────────────────────────────────────────
    if args.selfplay:
        callbacks = [
            CheckpointCallback(
                save_freq   = 50_000,
                save_path   = CKPT_DIR,
                name_prefix = "boxer_selfplay",
            ),
            SelfPlayCallback(
                update_freq  = 50_000,
                save_dir     = CKPT_DIR,
            ),
        ]
    else:
        curriculum_env = train_env
        callbacks = [
            CheckpointCallback(
                save_freq   = 50_000,
                save_path   = CKPT_DIR,
                name_prefix = f"boxer_p{args.phase}",
            ),
            PhaseCallback(
                curriculum_env = curriculum_env,
                use_wandb      = args.wandb,
                log_every_ep   = 20,
            ),
        ]

    # ── run ───────────────────────────────────────────────────────────────────
    mode_label = "Self-Play" if args.selfplay else f"Phase {args.phase}: {PHASE_CONFIG[args.phase]['name']}"
    print(
        f"\n{'='*60}\n"
        f"  Boxing RL  |  {mode_label}\n"
        f"  Total steps : {args.total_steps:,}\n"
        f"{'='*60}\n"
    )

    model.learn(
        total_timesteps = args.total_steps,
        callback        = callbacks,
        progress_bar    = True,
        reset_num_timesteps = not bool(args.load),
    )

    # ── final save ────────────────────────────────────────────────────────────
    tag        = "selfplay_final" if args.selfplay else f"phase{args.phase}_final"
    final_path = os.path.join(CKPT_DIR, tag)
    model.save(final_path)
    print(f"\n[train.py] Final model saved -> {final_path}.zip\n")

    vec_env.close()
    if args.wandb and _WANDB:
        wandb.finish()
    return model


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Boxing RL – curriculum trainer with live viewer")
    p.add_argument("--phase",       type=int, default=1,
                   help="Starting curriculum phase (1–4)")
    p.add_argument("--total-steps", type=int, default=500_000_000,
                   help="Total training steps (default: 500M)")
    p.add_argument("--eval-freq",   type=int, default=10_000,
                   help="Steps between viewer eval episodes (default: 10 000)")
    p.add_argument("--load",        type=str, default=None,
                   help="Checkpoint .zip to resume from")
    p.add_argument("--wandb",       action="store_true",
                   help="Enable W&B logging")
    p.add_argument("--lr",          type=float, default=None,
                   help="Override learning rate (useful for phase transfer)")
    p.add_argument("--ent-coef",    type=float, default=None,
                   help="Override entropy coefficient (default: 0.001)")
    p.add_argument("--selfplay",    action="store_true",
                   help="Two-agent self-play mode (fighter_1 = frozen copy of current policy)")
    p.add_argument("--n-envs",      type=int, default=1,
                   help="Number of parallel environments (default: 1)")
    args = p.parse_args()

    train(args)
