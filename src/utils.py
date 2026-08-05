"""
utils.py
========

Small building blocks shared by training and evaluation:

* Reproducible seeding.
* A gymnasium env factory that stacks the *correct* order of wrappers.
* Orthogonal weight init helper used by both networks.
* A tiny YAML -> dataclass-ish config loader (kept dict-based on purpose:
  we do not want a leaky abstraction over a 30-line config file).

See ``docs/PPO.md`` §7 for why observation/reward normalization matters.
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config into a plain dict. No validation on purpose."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return cfg


def resolve_device(pref: str) -> torch.device:
    """`"auto"` -> cuda if available, else cpu. Otherwise take user string."""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (cpu+cuda).

    Note: perfect determinism with CUDA also requires
    ``torch.use_deterministic_algorithms(True)`` and env var
    ``CUBLAS_WORKSPACE_CONFIG=:16:8`` — we skip that because it hurts
    throughput significantly and PPO is already stochastic through the env.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    env_id: str,
    seed: int,
    idx: int,
    *,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    gamma: float = 0.99,
    capture_video: bool = False,
    video_folder: str | None = None,
    run_name: str = "run",
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a single wrapped gymnasium env.

    Wrapper order (outer -> inner as seen by the agent's ``env.step`` call):

        FlattenObservation
        -> ClipAction                    # actor outputs unbounded R^k
        -> RecordEpisodeStatistics       # logs *raw* returns
        -> NormalizeObservation          # running mean/std of s
        -> TransformObservation(clip)
        -> NormalizeReward(gamma)        # running std of returns
        -> TransformReward(clip)

    Rationale: episode stats must capture the *true* reward, not the
    normalized one, so ``RecordEpisodeStatistics`` sits *above* the reward
    normalizer. ``ClipAction`` sits above everything else so the log-prob is
    computed on the raw Gaussian sample (see docs/PPO.md §5.1).
    """

    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            if video_folder is not None:
                env = gym.wrappers.RecordVideo(
                    env,
                    video_folder=os.path.join(video_folder, run_name),
                    episode_trigger=lambda ep: ep % 50 == 0,
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
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(
                env, lambda r: float(np.clip(r, -clip_reward, clip_reward))
            )

        env.action_space.seed(seed + idx)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal init with a configurable gain.

    Following the PPO reference implementation:
    - hidden layers: ``std = sqrt(2)``
    - policy mean head: ``std = 0.01``
    - value head: ``std = 1.0``
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(
    in_dim: int,
    hidden_sizes: list[int],
    activation: str = "tanh",
    hidden_gain: float = np.sqrt(2),
) -> nn.Sequential:
    """Small helper to avoid repeating the same 6 lines in actor + critic.

    Returns a Sequential of ``[Linear, act, Linear, act, ...]`` — *no* final
    output layer. The caller adds its own head so it can use a custom gain.
    """
    act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(layer_init(nn.Linear(last, h), std=hidden_gain))
        layers.append(act_cls())
        last = h
    return nn.Sequential(*layers)


def hidden_out_dim(hidden_sizes: list[int]) -> int:
    """Return the trailing dim of the MLP produced by :func:`build_mlp`."""
    return hidden_sizes[-1]
