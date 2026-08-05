"""
gae.py
======

Generalized Advantage Estimation (Schulman et al., 2016, arXiv:1506.02438).

Given per-timestep tensors of shape ``(T, N)`` where T is the rollout length
and N is the number of parallel envs, plus a bootstrap value ``last_value``
of shape ``(N,)``, compute the GAE advantages and value targets.

The recursion is:

    delta_t = r_t + gamma * (1 - d_{t+1}) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - d_{t+1}) * A_{t+1}
    R_t     = A_t + V(s_t)

Notes
-----
* ``d_{t+1}`` is the ``terminated`` flag *of the transition that produced
  s_{t+1}*, **not** ``terminated | truncated``. On truncation we still want
  to bootstrap with V(s_{t+1}). See docs/PPO.md §4.1.
* We implement this as a plain Python loop over T. T is small (e.g. 2048)
  so this is not a hotspot; the K-epoch minibatch loop dominates runtime.
"""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,          # (T, N)
    values: torch.Tensor,           # (T, N)  values at s_t
    terminated: torch.Tensor,       # (T, N)  True if s_t was terminal after step
    last_value: torch.Tensor,       # (N,)    V(s_T) for bootstrap
    last_terminated: torch.Tensor,  # (N,)    was the *very last* step terminal?
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(advantages, returns)``, both shape ``(T, N)``.

    Convention: ``terminated[t]`` refers to whether the transition ending at
    ``s_{t+1}`` was terminal. In the loop below we therefore look one step
    ahead when computing the mask.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - last_terminated.float()
            next_value = last_value
        else:
            next_nonterminal = 1.0 - terminated[t + 1].float()
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns
