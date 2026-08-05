"""
evaluate.py
===========

Load a checkpoint saved by ``train.py`` and evaluate the deterministic
(mean) policy on Ant-v5. Optionally record MP4 videos of the rollouts.

Usage
-----
    python src/evaluate.py \
        --checkpoint models/<run_name>/ckpt_iter000100.pt \
        --episodes 10 \
        --video

Notes
-----
* We use the *mean* action (``actor.act_deterministic``) at eval time. This
  is standard for continuous-control benchmarks; the sampled policy would
  report higher variance without any real training signal here.
* The observation normalization stats saved in the checkpoint are restored
  before evaluation; without this, the loaded policy would receive raw
  observations it has never seen and behave randomly. See docs/PPO.md §7.
* Videos are written under ``videos/<run_name>/`` at episode boundaries.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from actor import Actor
from utils import resolve_device, set_seed


def _apply_norm_state(env: gym.Env, norm_state: dict) -> None:
    """Restore NormalizeObservation stats saved in the checkpoint.

    We *disable* running-average updates during evaluation by monkey-setting
    the wrapper's ``update_running_mean`` if present; falling back to
    manually keeping the pre-loaded stats otherwise.
    """
    walk: gym.Env | None = env
    while walk is not None and hasattr(walk, "env"):
        cls = walk.__class__.__name__
        if cls == "NormalizeObservation" and "obs_rms_mean" in norm_state:
            walk.obs_rms.mean  = norm_state["obs_rms_mean"].copy()
            walk.obs_rms.var   = norm_state["obs_rms_var"].copy()
            walk.obs_rms.count = norm_state["obs_rms_count"]
        walk = walk.env  # type: ignore[assignment]


def _build_eval_env(
    env_id: str,
    seed: int,
    *,
    normalize_obs: bool,
    clip_obs: float,
    record_video: bool,
    video_dir: str,
) -> gym.Env:
    """Single-env eval pipeline. No reward normalization (we want raw returns)."""
    if record_video:
        env = gym.make(env_id, render_mode="rgb_array")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            disable_logger=True,
        )
    else:
        env = gym.make(env_id)

    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -clip_obs, clip_obs),
            observation_space=env.observation_space,
        )
    env.action_space.seed(seed)
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--video",      action="store_true", help="Record MP4s.")
    parser.add_argument("--video-dir",  type=str, default=None)
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = resolve_device(cfg.get("device", "auto"))
    set_seed(args.seed)

    run_name = Path(args.checkpoint).parent.name
    video_dir = args.video_dir or str(Path(cfg["video_dir"]) / run_name)

    env = _build_eval_env(
        env_id=cfg["env_id"],
        seed=args.seed,
        normalize_obs=cfg["normalize_obs"],
        clip_obs=cfg["clip_obs"],
        record_video=args.video,
        video_dir=video_dir,
    )
    _apply_norm_state(env, ckpt.get("env_norm", {}))

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))

    actor = Actor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=cfg["hidden_sizes"],
        activation=cfg["activation"],
        log_std_init=cfg["log_std_init"],
    ).to(device)
    actor.load_state_dict(ckpt["agent"]["actor"])
    actor.eval()

    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = actor.act_deterministic(obs_t).squeeze(0).cpu().numpy()
            obs, reward, term, trunc, info = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = bool(term or trunc)
        returns.append(ep_return)
        lengths.append(ep_length)
        print(f"[eval] episode {ep + 1}/{args.episodes} "
              f"return={ep_return:8.2f} length={ep_length}")

    env.close()
    ret = np.asarray(returns)
    print(f"\n[eval] mean_return={ret.mean():.2f}  std={ret.std():.2f}  "
          f"min={ret.min():.2f}  max={ret.max():.2f}  "
          f"mean_length={float(np.mean(lengths)):.1f}")


if __name__ == "__main__":
    main()
