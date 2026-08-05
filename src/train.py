"""
train.py
========

Main training entry point.

Usage
-----
    python src/train.py --config configs/ant_v5.yaml

What this script does (see docs/PPO.md §8 for the algorithm):

1.  Build a SyncVectorEnv of ``num_envs`` Ant-v5 instances, each wrapped with
    observation/reward normalization, episode-stats logging, and action
    clipping (see ``utils.make_env``).
2.  Build the PPO agent (actor + critic + Adam).
3.  For each iteration:
       a) run the policy for ``n_steps`` steps in each env, storing the
          transitions in a :class:`RolloutBuffer`;
       b) bootstrap ``V(s_T)`` and compute GAE advantages + return targets;
       c) run ``n_epochs`` passes of minibatch updates;
       d) linearly anneal the learning rate;
       e) log metrics to TensorBoard;
       f) periodically save a checkpoint that includes the env's running
          normalization statistics.

The "first-iter ratio == 1" assertion inside the update is the single most
useful sanity check we can afford; see docs/PPO.md §11.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange

from buffer import RolloutBuffer
from ppo import PPOAgent
from utils import load_config, make_env, resolve_device, set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_schedule(progress: float) -> float:
    """1.0 -> 0.0 as progress goes 0.0 -> 1.0. Clamped."""
    return max(0.0, 1.0 - progress)


def _obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def _extract_norm_state(envs: gym.vector.VectorEnv) -> dict:
    """Pull running mean/std out of NormalizeObservation / NormalizeReward so
    we can persist them in the checkpoint. Best-effort: if the wrappers are
    disabled we return an empty dict.
    """
    state: dict = {}
    unwrapped = envs
    # gymnasium vector envs expose a list of `env_fns`; we introspect the
    # first sub-env's wrapper chain to grab a snapshot.
    try:
        sub_env = envs.envs[0]  # type: ignore[attr-defined]
    except AttributeError:
        return state
    walk = sub_env
    while hasattr(walk, "env"):
        cls = walk.__class__.__name__
        if cls == "NormalizeObservation":
            state["obs_rms_mean"] = np.array(walk.obs_rms.mean, copy=True)
            state["obs_rms_var"]  = np.array(walk.obs_rms.var, copy=True)
            state["obs_rms_count"] = float(walk.obs_rms.count)
        elif cls == "NormalizeReward":
            state["ret_rms_mean"] = np.array(walk.return_rms.mean, copy=True)
            state["ret_rms_var"]  = np.array(walk.return_rms.var, copy=True)
            state["ret_rms_count"] = float(walk.return_rms.count)
        walk = walk.env
    _ = unwrapped  # silence linter
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ant_v5.yaml")
    parser.add_argument("--override-total-timesteps", type=int, default=None,
                        help="Optional short-run override for smoke tests.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override_total_timesteps is not None:
        cfg["total_timesteps"] = args.override_total_timesteps

    # --- reproducibility -----------------------------------------------------
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    # --- run identity --------------------------------------------------------
    run_name = f"{cfg['exp_name']}__seed{cfg['seed']}__{int(time.time())}"
    log_dir  = Path(cfg["log_dir"])  / run_name
    ckpt_dir = Path(cfg["ckpt_dir"]) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    writer.add_text(
        "config",
        "|param|value|\n|---|---|\n" +
        "\n".join(f"|{k}|{v}|" for k, v in cfg.items()),
        0,
    )

    # --- vector envs --------------------------------------------------------
    env_fns = [
        make_env(
            env_id=cfg["env_id"],
            seed=cfg["seed"],
            idx=i,
            normalize_obs=cfg["normalize_obs"],
            normalize_reward=cfg["normalize_reward"],
            clip_obs=cfg["clip_obs"],
            clip_reward=cfg["clip_reward"],
            gamma=cfg["gamma"],
            capture_video=False,
            run_name=run_name,
        )
        for i in range(cfg["num_envs"])
    ]
    envs = gym.vector.SyncVectorEnv(env_fns)
    assert isinstance(envs.single_action_space, gym.spaces.Box), \
        "This implementation only supports continuous (Box) action spaces."

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    # --- agent --------------------------------------------------------------
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=cfg["hidden_sizes"],
        activation=cfg["activation"],
        log_std_init=cfg["log_std_init"],
        learning_rate=cfg["learning_rate"],
        clip_coef=cfg["clip_coef"],
        clip_vloss=cfg["clip_vloss"],
        vf_coef=cfg["vf_coef"],
        ent_coef=cfg["ent_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        n_epochs=cfg["n_epochs"],
        minibatch_size=cfg["minibatch_size"],
        normalize_advantage=cfg["normalize_advantage"],
        target_kl=cfg["target_kl"],
        device=device,
    )

    # --- buffer -------------------------------------------------------------
    buffer = RolloutBuffer(
        n_steps=cfg["n_steps"],
        num_envs=cfg["num_envs"],
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )

    # --- iteration bookkeeping ---------------------------------------------
    batch_size    = cfg["n_steps"] * cfg["num_envs"]
    n_iterations  = cfg["total_timesteps"] // batch_size
    global_step   = 0
    start_time    = time.time()
    rng           = np.random.default_rng(cfg["seed"])

    # Episode-return tracking: NormalizeReward wraps the reward but
    # RecordEpisodeStatistics sees the *raw* reward, so `info["episode"]["r"]`
    # is the number we actually care about.
    ep_returns: list[float] = []
    ep_lengths: list[int]   = []

    # --- initial reset ------------------------------------------------------
    next_obs, _ = envs.reset(seed=cfg["seed"])
    next_obs = _obs_to_tensor(next_obs, device)
    next_terminated = torch.zeros(cfg["num_envs"], dtype=torch.bool, device=device)

    first_update_ratio_checked = False

    for iteration in trange(1, n_iterations + 1, desc="iters"):
        # ------------------------- linear LR anneal ------------------------
        if cfg["anneal_lr"]:
            progress = (iteration - 1) / max(1, n_iterations)
            agent.set_lr(cfg["learning_rate"] * _linear_schedule(progress))

        # ------------------------- rollout ---------------------------------
        buffer.reset()
        for _t in range(cfg["n_steps"]):
            global_step += cfg["num_envs"]

            # Sample action, log-prob, value. No grad here.
            action, logprob, value = agent.select_action(next_obs)
            action_np = action.detach().cpu().numpy()

            # Step. ClipAction wrapper enforces the [-1, 1] bounds *without*
            # touching the log-prob we stored.
            next_obs_np, reward_np, term_np, trunc_np, info = envs.step(action_np)

            # Store step. NOTE: `terminated` (not term|trunc) is the GAE mask.
            buffer.add(
                obs=next_obs,  # obs *before* the step
                actions=action,
                logprobs=logprob,
                rewards=torch.as_tensor(reward_np, dtype=torch.float32, device=device),
                values=value,
                terminated=torch.as_tensor(term_np, dtype=torch.bool, device=device),
            )

            next_obs = _obs_to_tensor(next_obs_np, device)
            next_terminated = torch.as_tensor(term_np, dtype=torch.bool, device=device)

            # Log any completed episodes (RecordEpisodeStatistics puts them in info).
            if "episode" in info:
                ep_info = info["episode"]
                # In vector envs, ep_info["r"] is an array with valid entries
                # where an episode ended this step (masked by _final_info).
                mask = info.get("_episode", np.ones_like(ep_info["r"], dtype=bool))
                for r_val, l_val, m in zip(ep_info["r"], ep_info["l"], mask):
                    if m:
                        ep_returns.append(float(r_val))
                        ep_lengths.append(int(l_val))

        # ------------------------- GAE -------------------------------------
        last_value = agent.value(next_obs)
        buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_terminated=next_terminated,
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
        )

        # ------------------------- sanity check ----------------------------
        # First minibatch of the very first update: ratio must be ~1 because
        # theta hasn't moved yet. See docs/PPO.md §11.
        if not first_update_ratio_checked:
            with torch.no_grad():
                first_mb = next(iter(buffer.get_minibatches(cfg["minibatch_size"], rng=rng)))
                new_logp, _ = agent.actor.evaluate_actions(first_mb.obs, first_mb.actions)
                ratio0 = (new_logp - first_mb.old_logprobs).exp()
                assert torch.allclose(
                    ratio0, torch.ones_like(ratio0), atol=1e-4
                ), f"First-iter ratio != 1: mean={ratio0.mean():.6f}"
            first_update_ratio_checked = True

        # ------------------------- PPO update ------------------------------
        metrics = agent.update(buffer, rng=rng)

        # ------------------------- logging --------------------------------
        if iteration % cfg["log_every_iters"] == 0:
            recent_return = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            recent_length = float(np.mean(ep_lengths[-20:])) if ep_lengths else float("nan")
            sps = int(global_step / (time.time() - start_time))

            writer.add_scalar("charts/episodic_return", recent_return, global_step)
            writer.add_scalar("charts/episodic_length", recent_length, global_step)
            writer.add_scalar("charts/sps", sps, global_step)
            writer.add_scalar("charts/lr", metrics.lr, global_step)
            writer.add_scalar("losses/policy_loss", metrics.policy_loss, global_step)
            writer.add_scalar("losses/value_loss",  metrics.value_loss, global_step)
            writer.add_scalar("losses/entropy",     metrics.entropy, global_step)
            writer.add_scalar("losses/approx_kl",   metrics.approx_kl, global_step)
            writer.add_scalar("losses/clip_frac",   metrics.clip_fraction, global_step)
            writer.add_scalar("losses/explained_variance",
                              metrics.explained_variance, global_step)
            writer.add_scalar("losses/grad_norm",   metrics.grad_norm, global_step)
            writer.add_scalar("policy/log_std_mean", metrics.log_std_mean, global_step)

        # ------------------------- checkpoint -----------------------------
        if iteration % cfg["save_every_iters"] == 0 or iteration == n_iterations:
            ckpt_path = ckpt_dir / f"ckpt_iter{iteration:06d}.pt"
            torch.save(
                {
                    "iteration":     iteration,
                    "global_step":   global_step,
                    "config":        cfg,
                    "agent":         agent.state_dict(),
                    "env_norm":      _extract_norm_state(envs),
                },
                ckpt_path,
            )

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
