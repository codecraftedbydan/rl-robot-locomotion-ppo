# PPO - Theory, Derivations, and Implementation Notes

This document explains **every algorithmic decision** in the codebase.
It is intentionally verbose: the goal is Research-Engineer level understanding,
not just working code.

Primary references:

- **PPO paper**: Schulman, Wolski, Dhariwal, Radford, Klimov (2017),
  *Proximal Policy Optimization Algorithms*, arXiv:1707.06347.
- **GAE paper**: Schulman, Moritz, Levine, Jordan, Abbeel (2016),
  *High-Dimensional Continuous Control Using Generalized Advantage Estimation*,
  arXiv:1506.02438.
- **TRPO paper**: Schulman, Levine, Moritz, Jordan, Abbeel (2015),
  *Trust Region Policy Optimization*, arXiv:1502.05477.
- **Implementation-details survey**: Engstrom et al. (2020),
  *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*
  and Huang et al. (2022), *The 37 Implementation Details of PPO*.

---

## 1. Problem setting

We model the environment as a Markov Decision Process
$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0)$.
A stochastic policy $\pi_\theta(a \mid s)$ selects actions.
We want to maximize the expected discounted return

$$
J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right].
$$

For **Ant-v5** (MuJoCo, gymnasium):

- $\dim(\mathcal{S}) = 105$ (torso pose + joint positions + joint velocities + contact forces).
- $\dim(\mathcal{A}) = 8$, $a \in [-1, 1]^8$ (torques on the 8 hinges).
- Reward = forward velocity + healthy bonus − control cost − contact cost.
- Episode ends on unhealthy state (z-height out of range) or timeout (1000 steps).

---

## 2. Vanilla policy gradient (recap)

The policy-gradient theorem gives

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s, a \sim \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a \mid s) \, A^{\pi_\theta}(s, a)\right],
$$

where $A^{\pi}(s,a) = Q^\pi(s,a) - V^\pi(s)$ is the advantage.
This is *on-policy*: samples come from $\pi_\theta$, so after **one gradient step**
they are stale. That is wasteful for expensive rollouts (like MuJoCo).

---

## 3. From TRPO to PPO

TRPO reuses samples by optimizing a **surrogate objective** with an
importance-sampling ratio $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$:

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t\!\left[r_t(\theta) \hat{A}_t\right],
$$

subject to a KL trust region $\mathbb{E}_t[\text{KL}(\pi_{\text{old}} \Vert \pi_\theta)] \le \delta$.
Solving that constrained problem needs conjugate-gradient + line-search - complex.

**PPO's key insight**: enforce the trust region *implicitly* by clipping the
probability ratio. This gives a first-order method that any deep-learning
framework can optimize with SGD.

### 3.1 Clipped surrogate objective (Eq. 7 of the paper)

$$
\boxed{\;L^{\text{CLIP}}(\theta) = \mathbb{E}_t \!\left[
\min\!\big(\, r_t(\theta)\hat{A}_t,\; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, \hat{A}_t \big)\right]\;}
$$

Intuition:

- If $\hat{A}_t > 0$: we want to *increase* $\pi_\theta(a_t \mid s_t)$, but only up
  to $r_t = 1 + \epsilon$; beyond that the objective is clipped and the gradient
  vanishes → no incentive to move further away from $\pi_{\text{old}}$.
- If $\hat{A}_t < 0$: same idea on the other side. We stop decreasing the
  probability once $r_t \le 1 - \epsilon$.
- The `min` ensures the objective is a **pessimistic (lower) bound** on the
  unclipped surrogate - this is what gives the trust-region-like guarantee.

Typical $\epsilon = 0.2$.

### 3.2 Value loss

We fit the critic $V_\phi(s)$ to the empirical returns $\hat{R}_t$ (see §4):

$$
L^{\text{VF}}(\phi) = \mathbb{E}_t\!\left[(V_\phi(s_t) - \hat{R}_t)^2\right].
$$

Optional **clipped value loss** (used in the paper's public implementation):

$$
L^{\text{VF-CLIP}} = \mathbb{E}_t\!\left[\max\!\big((V_\phi(s_t) - \hat{R}_t)^2,\;(V_\phi^{\text{clip}}(s_t) - \hat{R}_t)^2\big)\right],
$$

with $V_\phi^{\text{clip}} = V_{\phi_{\text{old}}} + \operatorname{clip}(V_\phi - V_{\phi_{\text{old}}}, -\epsilon, +\epsilon)$.
This mirrors the policy clip on the value function.

### 3.3 Entropy bonus

To encourage exploration:

$$
L^{\text{S}}(\theta) = \mathbb{E}_t[\, H(\pi_\theta(\cdot \mid s_t))\, ].
$$

For a diagonal Gaussian in $k$ dims, $H = \tfrac{k}{2}(1 + \ln 2\pi) + \sum_i \ln \sigma_i$.
For MuJoCo continuous control the bonus is often set to **0** - the Gaussian
policy is already stochastic enough. We keep the coefficient exposed via config.

### 3.4 Total loss

$$
L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 \, L^{\text{VF}}(\phi) - c_2 \, L^{\text{S}}(\theta).
$$

Typical: $c_1 = 0.5$, $c_2 = 0.0$ (Ant-v5 continuous).

---

## 4. Generalized Advantage Estimation (GAE)

We need $\hat{A}_t$ from samples. Using Monte-Carlo returns has low bias but
high variance; using the one-step TD residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
is the opposite. GAE trades them off with $\lambda$:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}.
$$

Equivalent efficient backward recursion (this is what we implement):

$$
\hat{A}_t = \delta_t + \gamma \lambda\, (1 - d_{t+1})\, \hat{A}_{t+1},
$$

where $d_{t+1} \in \{0,1\}$ is the terminal flag at the *next* step (mask so
advantages don't leak across episode boundaries).

The **returns targets** for the critic are then

$$
\hat{R}_t = \hat{A}_t + V(s_t).
$$

Typical: $\gamma = 0.99$, $\lambda = 0.95$.

### 4.1 Terminal vs truncation - the subtle bit

`gymnasium` distinguishes:

- **`terminated=True`**: the MDP genuinely ended (Ant fell over → unhealthy).
  Bootstrap value = 0.
- **`truncated=True`**: time-limit only; the MDP would have continued.
  Bootstrap value = $V(s_{T+1})$.

The GAE mask must use `terminated`, not `terminated | truncated`. Getting this
wrong biases the value function and is a **classic silent bug**.

---

## 5. Gaussian policy for continuous action spaces

We factor the policy as a diagonal Gaussian:

$$
\pi_\theta(a \mid s) = \mathcal{N}(a;\; \mu_\theta(s),\; \operatorname{diag}(\sigma^2)),
$$

with:

- **Mean** $\mu_\theta(s)$: output of an MLP with `tanh` activations, orthogonal
  init, and a *small* last-layer gain (0.01) so initial actions are near zero.
- **Std** $\sigma$: **state-independent**, a single learnable parameter vector
  `log_std ∈ R^k`, initialized to 0 (i.e. $\sigma = 1$).
  - Rationale: matches the original PPO/TRPO paper implementations.
    A state-dependent std often collapses to a deterministic policy early
    in training on MuJoCo.

We use `torch.distributions.Independent(Normal(mean, std), 1)` so that
`log_prob` and `entropy` **sum over action dimensions** - otherwise you get
per-dim tensors, which subtly breaks the ratio.

### 5.1 Action bounds

Ant-v5 requires $a \in [-1, 1]^8$. The Gaussian is unbounded. Standard PPO
handling is:

1. Sample $a \sim \mathcal{N}(\mu, \sigma)$.
2. Compute `log_prob(a)` on the **unclipped** action.
3. **Clip** the action *only* before passing to the env (or use gymnasium's
   `ClipAction` wrapper).

Do **not** recompute `log_prob` on the clipped action - that biases the
ratio. `tanh`-squashed policies (SAC-style) exist and are cleaner, but PPO's
paper uses plain Gaussian + clip, so we do too.

---

## 6. Network architecture

Two separate MLPs, no weight sharing (Engstrom et al. 2020 shows separate
networks match or beat shared ones on MuJoCo):

- **Actor** (`src/actor.py`): `obs (105)` → `Linear 64` → `tanh` → `Linear 64`
  → `tanh` → `Linear 8` (mean). Plus `log_std ∈ R^8` parameter.
- **Critic** (`src/critic.py`): `obs (105)` → `Linear 64` → `tanh`
  → `Linear 64` → `tanh` → `Linear 1`.

**Weight init** (from the paper's reference implementation):

- Hidden `Linear`: orthogonal, gain $\sqrt{2}$, bias 0.
- Policy mean head: orthogonal, gain **0.01**, bias 0. (Small init so the
  initial policy is near-uniform on the action space.)
- Value head: orthogonal, gain **1.0**, bias 0.
- `log_std`: initialized to 0 → $\sigma = 1$.

Activation: `tanh` is the historical PPO default and empirically stable on
MuJoCo. `ReLU` also works but can cause dead units.

---

## 7. Observation and reward normalization

Two wrappers, both from `gymnasium.wrappers`:

- **`NormalizeObservation`**: subtract running mean, divide by running std
  (per-dim). Critical for MuJoCo - raw observations span many orders of
  magnitude.
- **`NormalizeReward`**: divides rewards by a running estimate of the *return*
  standard deviation (not the reward std). Prevents the value function from
  needing to fit huge scales. Note: this changes the *scale* of $\gamma$'s
  effect - but not the sign of the gradient, so learning is unaffected. This
  is discussed in Engstrom et al. 2020.

Both must be **saved at checkpoint time** (their running stats) or evaluation
will be broken.

---

## 8. The full PPO loop

```
for iteration in range(N_iterations):
    # --- rollout: run policy for T steps in each of N envs ---
    for t in range(T):
        with torch.no_grad():
            action, logprob = actor.sample(obs)
            value          = critic(obs)
        next_obs, reward, term, trunc, info = envs.step(clip(action))
        buffer.add(obs, action, logprob, reward, value, term)
        obs = next_obs

    # --- compute GAE ---
    with torch.no_grad():
        last_value = critic(obs)
    advantages, returns = compute_gae(buffer, last_value, gamma, lambda_)

    # --- optimize for K epochs over minibatches ---
    for epoch in range(K):
        for mb in buffer.get_minibatches(mb_size):
            new_logp, entropy = actor.evaluate_actions(mb.obs, mb.act)
            new_value         = critic(mb.obs)

            ratio     = exp(new_logp - mb.old_logp)
            adv       = normalize(mb.adv)                      # per-minibatch
            pg_loss   = -min(ratio*adv, clip(ratio, 1±ε)*adv).mean()
            v_loss    = 0.5 * (new_value - mb.ret).pow(2).mean()   # optionally clipped
            ent_loss  = entropy.mean()

            loss = pg_loss + c1 * v_loss - c2 * ent_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params, max_grad_norm)   # typically 0.5
            optimizer.step()

    log metrics; maybe checkpoint; maybe anneal LR
```

Total gradient steps per iteration: `K * (N*T / mb_size)`.

---

## 9. Hyperparameters (Ant-v5 defaults, see `configs/ant_v5.yaml`)

| Hyperparameter        | Value      | Comment |
|-----------------------|------------|---------|
| `num_envs`            | 4          | Parallel rollouts (`SyncVectorEnv`) |
| `n_steps`             | 2048       | Steps per env per iter → 8192 samples |
| `total_timesteps`     | 3_000_000  | Enough to see stable locomotion |
| `learning_rate`       | 3e-4       | Anneal linearly to 0 |
| `gamma`               | 0.99       | Discount |
| `gae_lambda`          | 0.95       | GAE tradeoff |
| `clip_coef` (ε)       | 0.2        | Policy clip |
| `clip_vloss`          | true       | Clip value loss with same ε |
| `ent_coef` (c₂)       | 0.0        | Gaussian is already stochastic |
| `vf_coef` (c₁)        | 0.5        | Value loss weight |
| `n_epochs` (K)        | 10         | Passes over the buffer |
| `minibatch_size`      | 64         | 128 minibatches per epoch |
| `max_grad_norm`       | 0.5        | Global-norm grad clipping |
| `target_kl`           | 0.03       | Optional early stop |
| `normalize_advantage` | true       | Per-minibatch |
| `norm_obs` / `norm_rew` | true    | Running stats wrappers |

---

## 10. Common bugs (each one is silent and costs days)

1. **Advantage normalization at the wrong scope**: normalize *inside* the
   minibatch loop, not once over the whole buffer. Otherwise the last epochs
   see stale statistics.
2. **Using `terminated | truncated` as the GAE mask**: kills bootstrap on
   time-limit - biases value fitting downward.
3. **Recomputing `log_prob` on the clipped action**: makes $r_t \neq 1$ even at
   $\theta = \theta_{\text{old}}$. Verify $r_t = 1$ on the very first minibatch
   of every iteration.
4. **`Normal` instead of `Independent(Normal, 1)`**: `log_prob` returns a
   per-dim tensor, so `ratio.shape == (B, act_dim)` and broadcasting with
   scalar advantages silently changes semantics.
5. **Sharing an optimizer step across networks but forgetting to gather all
   params**: `optim = Adam(actor.parameters() + critic.parameters())` - must
   be a chain, not two separate lists concatenated as a Python list of
   iterables.
6. **Not resetting the buffer** between iterations.
7. **State-dependent std that collapses**: if you must have one, output
   `log_std` clipped to `[-5, 2]`.
8. **Observation normalization stats not saved with the checkpoint**: eval
   silently uses `mean=0, std=1` and the policy looks broken.
9. **LR anneal that goes negative**: guard with `max(0, ...)`.
10. **Wrong flatten order in the buffer**: rollouts are `(T, N, ...)`. If you
    reshape to `(T*N, ...)` be consistent across obs/act/adv/ret/logp - mixing
    orders scrambles which advantage belongs to which action.

---

## 11. How to verify correctness

Cheap sanity checks *before* running a long experiment:

- **Ratio at step 0**: at the first gradient step of every iteration, the
  probability ratio is exactly 1 (up to fp32 noise). Assert
  `abs(ratio.mean() - 1) < 1e-5` before the first `optimizer.step()`.
- **Approx-KL sign and magnitude**: monitor
  `approx_kl = ((ratio - 1) - logratio).mean()` (Schulman's estimator, always
  ≥ 0). It should grow across the K epochs but stay under ~2×`target_kl`.
- **Clip fraction**: fraction of samples with `|ratio - 1| > ε`. Healthy PPO
  sits around 0.1–0.3.
- **Explained variance** of the value function:
  `1 - Var(returns - values) / Var(returns)`. Should rise into (0.6, 0.95).
  If it goes negative, the critic is worse than predicting the mean.
- **Gradient norm** before clipping - if it's routinely 100× `max_grad_norm`,
  something upstream is wrong (usually reward scale).
- **Deterministic reproducibility** at seed 0 for the first N updates
  (metrics tab in TB).

Longer checks:

- **Ant-v5 learning curve** should start around −20 to 0 return, climb past
  1000 by ~500k steps, and plateau in the 3000–5000 range by 3M steps with
  these defaults.
- **Video at checkpoint 1M**: legs should be in-phase gaits, not
  spasming; the torso should be upright, moving forward.
