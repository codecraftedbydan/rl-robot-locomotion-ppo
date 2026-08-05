"""
critic.py
=========

Scalar state-value function V_phi(s) as a small MLP.

* Separate from the actor (no shared trunk). Engstrom et al. 2020 show that
  a shared trunk hurts on MuJoCo continuous control.
* Output head uses orthogonal init with gain 1.0 — the value scale should
  match the return scale (which is normalized by NormalizeReward).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils import build_mlp, hidden_out_dim, layer_init


class Critic(nn.Module):
    """MLP critic returning V(s) of shape (B,)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int],
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.trunk = build_mlp(obs_dim, hidden_sizes, activation=activation)
        self.value_head = layer_init(
            nn.Linear(hidden_out_dim(hidden_sizes), 1), std=1.0
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(obs)).squeeze(-1)
