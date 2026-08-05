"""
buffer.py
=========

Fixed-size on-policy rollout buffer.

Layout: all fields are stored as ``(T, N, ...)`` tensors where

    T = ``n_steps``       (steps per env per iteration)
    N = ``num_envs``      (parallel envs)

At the end of a rollout we call :meth:`compute_returns_and_advantages`,
then iterate minibatches over the flat ``T * N`` axis.

Only the caller decides *when* to advance the write pointer (via
:meth:`add`) and *when* to reset (via :meth:`reset`). The buffer itself
is dumb storage + GAE glue + a minibatch iterator.

See docs/PPO.md §8 for how this fits into the full training loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from gae import compute_gae


@dataclass
class Minibatch:
    """One minibatch of flattened rollout samples."""
    obs: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Stores one PPO iteration worth of on-policy samples."""

    def __init__(
        self,
        n_steps: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
    ) -> None:
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        shape_TN = (n_steps, num_envs)
        self.obs         = torch.zeros((*shape_TN, obs_dim), device=device)
        self.actions     = torch.zeros((*shape_TN, act_dim), device=device)
        self.logprobs    = torch.zeros(shape_TN, device=device)
        self.rewards     = torch.zeros(shape_TN, device=device)
        self.values      = torch.zeros(shape_TN, device=device)
        # `terminated[t]` = did the transition producing obs[t] end an episode?
        # We use it as the GAE mask for step t (see docs/PPO.md §4.1).
        self.terminated  = torch.zeros(shape_TN, device=device, dtype=torch.bool)

        # Filled by compute_returns_and_advantages
        self.advantages: torch.Tensor | None = None
        self.returns:    torch.Tensor | None = None

        self.ptr = 0

    # ---- fill --------------------------------------------------------------

    def reset(self) -> None:
        self.ptr = 0
        self.advantages = None
        self.returns = None

    def add(
        self,
        obs:        torch.Tensor,   # (N, obs_dim)
        actions:    torch.Tensor,   # (N, act_dim)
        logprobs:   torch.Tensor,   # (N,)
        rewards:    torch.Tensor,   # (N,)
        values:     torch.Tensor,   # (N,)
        terminated: torch.Tensor,   # (N,) bool
    ) -> None:
        if self.ptr >= self.n_steps:
            raise RuntimeError("RolloutBuffer is full; call reset() before add().")
        t = self.ptr
        self.obs[t]        = obs
        self.actions[t]    = actions
        self.logprobs[t]   = logprobs
        self.rewards[t]    = rewards
        self.values[t]     = values
        self.terminated[t] = terminated
        self.ptr += 1

    # ---- GAE ---------------------------------------------------------------

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,       # (N,) V(s_T)
        last_terminated: torch.Tensor,  # (N,) bool
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Populate ``advantages`` and ``returns`` in-place using GAE."""
        if self.ptr != self.n_steps:
            raise RuntimeError(
                f"Buffer only has {self.ptr}/{self.n_steps} steps; "
                "cannot compute GAE."
            )
        advantages, returns = compute_gae(
            rewards=self.rewards,
            values=self.values,
            terminated=self.terminated,
            last_value=last_value,
            last_terminated=last_terminated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self.advantages = advantages
        self.returns = returns

    # ---- minibatch iterator -----------------------------------------------

    def get_minibatches(
        self,
        minibatch_size: int,
        rng: np.random.Generator | None = None,
    ) -> Iterator[Minibatch]:
        """Yield randomly shuffled minibatches over the flat (T*N,) axis.

        We build one permutation for this epoch and slice it in blocks — that
        guarantees every sample is seen exactly once per epoch.
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_returns_and_advantages() first.")

        batch = self.n_steps * self.num_envs
        if minibatch_size > batch:
            raise ValueError(
                f"minibatch_size ({minibatch_size}) > buffer size ({batch})."
            )

        # Flatten (T, N, ...) -> (T*N, ...). Use reshape() (not view()) because
        # some fields will be non-contiguous after slicing.
        flat_obs        = self.obs.reshape(batch, self.obs_dim)
        flat_actions    = self.actions.reshape(batch, self.act_dim)
        flat_logprobs   = self.logprobs.reshape(batch)
        flat_values     = self.values.reshape(batch)
        flat_advantages = self.advantages.reshape(batch)
        flat_returns    = self.returns.reshape(batch)

        if rng is None:
            perm = torch.randperm(batch, device=self.device)
        else:
            perm = torch.as_tensor(
                rng.permutation(batch), device=self.device, dtype=torch.long
            )

        for start in range(0, batch, minibatch_size):
            idx = perm[start : start + minibatch_size]
            yield Minibatch(
                obs=flat_obs[idx],
                actions=flat_actions[idx],
                old_logprobs=flat_logprobs[idx],
                old_values=flat_values[idx],
                advantages=flat_advantages[idx],
                returns=flat_returns[idx],
            )
