"""
ppo.py
======

The PPO update: clipped surrogate objective, optionally clipped value loss,
entropy bonus, per-minibatch advantage normalization, global gradient
clipping, and Schulman's approximate-KL estimator for early stopping.

Follows Schulman et al. 2017, eqs. (7) and (9). See docs/PPO.md §3 and §8
for the derivation and the exact loop structure.

The agent owns:
    * the actor,
    * the critic,
    * a single Adam optimizer chained over both networks' parameters.

Public interface used by ``train.py``:

    agent.select_action(obs)          -> (action, logprob, value)  # rollout
    agent.value(obs)                  -> V(s)                      # bootstrap
    agent.update(buffer, ...)         -> metrics dict
    agent.set_lr(new_lr)                                          # annealing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import torch
import torch.nn as nn

from actor import Actor
from buffer import RolloutBuffer
from critic import Critic


@dataclass
class PPOMetrics:
    """One update's diagnostics — everything you'd want in TensorBoard."""
    policy_loss:        float = 0.0
    value_loss:         float = 0.0
    entropy:            float = 0.0
    approx_kl:          float = 0.0
    clip_fraction:      float = 0.0
    explained_variance: float = 0.0
    grad_norm:          float = 0.0
    n_updates:          int   = 0
    early_stopped:      bool  = False
    lr:                 float = 0.0
    log_std_mean:       float = 0.0
    per_epoch_kl:       list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        d = {k: v for k, v in self.__dict__.items() if k != "per_epoch_kl"}
        return d


class PPOAgent:
    """Actor + critic + PPO optimizer. Stateless w.r.t. rollouts."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden_sizes: list[int],
        activation: str,
        log_std_init: float,
        learning_rate: float,
        clip_coef: float,
        clip_vloss: bool,
        vf_coef: float,
        ent_coef: float,
        max_grad_norm: float,
        n_epochs: int,
        minibatch_size: int,
        normalize_advantage: bool,
        target_kl: float | None,
        device: torch.device,
    ) -> None:
        self.actor  = Actor(obs_dim, act_dim, hidden_sizes, activation, log_std_init).to(device)
        self.critic = Critic(obs_dim, hidden_sizes, activation).to(device)

        # Single optimizer chained across both networks (docs/PPO.md §10 bug #5).
        self.optimizer = torch.optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=learning_rate,
            eps=1e-5,  # PPO reference default — larger eps stabilizes early updates.
        )

        self.clip_coef           = clip_coef
        self.clip_vloss          = clip_vloss
        self.vf_coef             = vf_coef
        self.ent_coef            = ent_coef
        self.max_grad_norm       = max_grad_norm
        self.n_epochs            = n_epochs
        self.minibatch_size      = minibatch_size
        self.normalize_advantage = normalize_advantage
        self.target_kl           = target_kl
        self.device              = device

    # ---- rollout-time helpers ---------------------------------------------

    @torch.no_grad()
    def select_action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and also compute V(s). Used inside the rollout."""
        action, logprob = self.actor.act(obs)
        value = self.critic(obs)
        return action, logprob, value

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    # ---- LR annealing ------------------------------------------------------

    def set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ---- update ------------------------------------------------------------

    def update(self, buffer: RolloutBuffer, rng: np.random.Generator) -> PPOMetrics:
        """Run K epochs of PPO minibatch updates over ``buffer``."""
        m = PPOMetrics(lr=self.optimizer.param_groups[0]["lr"])
        params = list(chain(self.actor.parameters(), self.critic.parameters()))

        all_losses_pg:  list[float] = []
        all_losses_v:   list[float] = []
        all_entropy:    list[float] = []
        all_approx_kl:  list[float] = []
        all_clip_frac:  list[float] = []
        all_grad_norms: list[float] = []

        early_stop = False
        for epoch in range(self.n_epochs):
            epoch_kls: list[float] = []
            for mb in buffer.get_minibatches(self.minibatch_size, rng=rng):
                # --- forward: new log-prob, entropy, value ------------------
                new_logp, entropy = self.actor.evaluate_actions(mb.obs, mb.actions)
                new_value         = self.critic(mb.obs)

                # --- probability ratio r_t(theta) ---------------------------
                logratio = new_logp - mb.old_logprobs
                ratio    = logratio.exp()

                # --- advantage (optionally normalized per-minibatch) --------
                adv = mb.advantages
                if self.normalize_advantage:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # --- clipped surrogate objective L^CLIP ---------------------
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- value loss L^VF, optionally clipped -------------------
                if self.clip_vloss:
                    v_clipped = mb.old_values + torch.clamp(
                        new_value - mb.old_values,
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_unclipped = (new_value - mb.returns).pow(2)
                    v_loss_clipped   = (v_clipped - mb.returns).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * (new_value - mb.returns).pow(2).mean()

                # --- entropy bonus L^S -------------------------------------
                ent_loss = entropy.mean()

                # --- combined loss -----------------------------------------
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent_loss

                # --- step ---------------------------------------------------
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()

                # --- diagnostics -------------------------------------------
                with torch.no_grad():
                    # Schulman's estimator: E[(r-1) - log r]  (nonnegative, low-var).
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clip_frac = (
                        (ratio - 1.0).abs() > self.clip_coef
                    ).float().mean().item()

                all_losses_pg.append(pg_loss.item())
                all_losses_v.append(v_loss.item())
                all_entropy.append(ent_loss.item())
                all_approx_kl.append(approx_kl)
                all_clip_frac.append(clip_frac)
                all_grad_norms.append(float(grad_norm))
                epoch_kls.append(approx_kl)
                m.n_updates += 1

            mean_epoch_kl = float(np.mean(epoch_kls))
            m.per_epoch_kl.append(mean_epoch_kl)
            if self.target_kl is not None and mean_epoch_kl > 1.5 * self.target_kl:
                # Optional early-stop — matches the SB3-style trigger threshold.
                early_stop = True
                break

        # --- aggregate metrics -------------------------------------------------
        m.policy_loss   = float(np.mean(all_losses_pg))
        m.value_loss    = float(np.mean(all_losses_v))
        m.entropy       = float(np.mean(all_entropy))
        m.approx_kl     = float(np.mean(all_approx_kl))
        m.clip_fraction = float(np.mean(all_clip_frac))
        m.grad_norm     = float(np.mean(all_grad_norms))
        m.log_std_mean  = float(self.actor.log_std.mean().item())
        m.early_stopped = early_stop

        # Explained variance of the value function on this iteration's buffer:
        # 1 - Var(returns - values) / Var(returns).
        with torch.no_grad():
            assert buffer.returns is not None
            y_true = buffer.returns.reshape(-1)
            y_pred = buffer.values.reshape(-1)
            var_y = y_true.var()
            m.explained_variance = (
                float("nan") if var_y == 0 else float(1.0 - (y_true - y_pred).var() / var_y)
            )
        return m

    # ---- checkpointing -----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "actor":     self.actor.state_dict(),
            "critic":    self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.optimizer.load_state_dict(sd["optimizer"])
