"""
actor.py
========

Diagonal Gaussian policy for continuous control.

    pi(a|s) = N( mu_theta(s), diag(sigma^2) )

Design choices (see docs/PPO.md §5 for the full rationale):

* ``mu_theta`` is an MLP with ``tanh`` activations.
* ``sigma = exp(log_std)`` where ``log_std`` is a **state-independent**
  learnable parameter (matches the original PPO/TRPO implementations and
  is empirically more stable on MuJoCo than a state-dependent head).
* We wrap the ``Normal`` distribution in ``Independent(..., 1)`` so that
  ``log_prob`` and ``entropy`` sum across the action dimensions.
* The mean head uses orthogonal init with gain ``0.01``, so the initial
  policy is nearly the origin action → gentle exploration at start.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from utils import build_mlp, hidden_out_dim, layer_init


class Actor(nn.Module):
    """MLP actor producing a diagonal Gaussian distribution over actions."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int],
        activation: str = "tanh",
        log_std_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.trunk = build_mlp(obs_dim, hidden_sizes, activation=activation)
        # Small last-layer gain: initial mean near 0 for stable exploration.
        self.mean_head = layer_init(
            nn.Linear(hidden_out_dim(hidden_sizes), act_dim), std=0.01
        )
        # State-independent std, parameterized in log-space for positivity.
        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))

    # ---- distribution ----------------------------------------------------

    def _distribution(self, obs: torch.Tensor) -> Independent:
        mean = self.mean_head(self.trunk(obs))
        # expand_as so batching works regardless of obs.shape[:-1]
        std = self.log_std.exp().expand_as(mean)
        return Independent(Normal(mean, std), 1)

    # ---- interfaces used by training / rollout ---------------------------

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action for rollout. Returns (action, log_prob).

        No grad here on purpose: the sampled log-prob is stored in the buffer
        and treated as ``log pi_old`` — it must be a *constant* w.r.t. theta
        in the PPO ratio.
        """
        dist = self._distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-prob and entropy of ``actions`` under current pi.

        Used inside the PPO update. Returns:
            log_prob : shape (B,)
            entropy  : shape (B,)
        """
        dist = self._distribution(obs)
        return dist.log_prob(actions), dist.entropy()

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the mean action (used at evaluation)."""
        return self.mean_head(self.trunk(obs))
