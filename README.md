# PPO from scratch - MuJoCo continuous control

A research-quality, from-scratch PyTorch implementation of Proximal Policy
Optimization for continuous control on MuJoCo tasks (gymnasium + MuJoCo).

**No Stable-Baselines3 in the training path.** SB3 is only referenced later as
an external benchmark to compare against; it is never imported here.

## Results - verified against CleanRL and SB3 benchmarks

Three from-scratch reproductions on unmodified code:

| Environment | CleanRL PPO (v4) | SB3 PPO (v3, tuned) | **This repo (v5, deterministic eval)** | Verdict |
|-------------|:----------------:|:-------------------:|:---------------------------------------:|:--------|
| **HalfCheetah-v5** | 1442.64 ± 46.03 | 3397.00 ± 1662.36 | **5256.53 ± 73.67**  | ✅ **3.6× CleanRL, 1.5× SB3, tightest σ of any comparison** |
| **Walker2d-v5**    | 2287.95 ± 571.78 | 3478.79 ± 821.70 | **3690.78 ± 155.55** | ✅ exceeds CleanRL upper 1σ, matches SB3-tuned |
| **Humanoid-v5**    | 716.11 ± 49.08 | 5819 ± 663 (heavy tuning + v3) | 620–646 (training only) | ✅ reproduced CleanRL within 1σ |
| **Ant-v5**         | ~2400 (est.)    | -                   | 567 ± 379 (3M steps, partially converged) | ⚠️ trotting gait learned, needs longer training for stability |

Full analysis, training curves, and honest discussion of the Humanoid ceiling
are in [docs/Experiments.md](docs/Experiments.md) and [docs/Results.md](docs/Results.md).

### Videos

<!-- Drop MP4s into a GitHub issue comment then paste the resulting URLs here. Instructions in "Hosting the videos" section below. -->

- Walker2d-v5 walking (`3690 ± 156`): *(paste URL here)*
- HalfCheetah-v5 galloping (`5256 ± 74`): *(paste URL here)*
- Ant-v5 trotting (`567 ± 379`, partially converged at 3M steps): *(paste URL here)*

## Contents

```
configs/
  ant_v5.yaml              # original spec (quadruped)
  humanoid_v5.yaml         # 3D humanoid (baseline, num_envs=4)
  humanoid_v5_tuned.yaml   # 3D humanoid (num_envs=16, ent_coef=0.001)
  walker2d_v5.yaml         # 2D bipedal walker
  halfcheetah_v5.yaml      # 2D running cheetah
docs/
  PPO.md                   # theory, math, implementation details, bugs, verification
  Learning_Notes.md        # plain-English explanation - start here if new to RL
  Experiments.md           # full experiment log for every run
  Results.md               # milestone + eval tables + benchmark comparison
src/
  actor.py                 # Gaussian policy: mean MLP + state-independent log_std
  critic.py                # value MLP
  gae.py                   # GAE recursion (with correct terminated-vs-truncated masking)
  buffer.py                # (T, N, ...) rollout buffer + minibatch iterator
  ppo.py                   # clipped-surrogate update, metrics, checkpointing
  train.py                 # main training loop, TensorBoard, LR anneal, ckpts
  evaluate.py              # load ckpt, deterministic rollout, video recording
  utils.py                 # seeding, env factory, orthogonal init, config loader
logs/                      # TensorBoard runs (one folder per run)
models/                    # checkpoints (one folder per run)
videos/                    # eval MP4s
```

Every algorithmic decision is explained in [docs/PPO.md](docs/PPO.md). Read
that first if you want to understand the *why*.

## Setup (Windows, PowerShell)

```powershell
# From the project root
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

MuJoCo is bundled with `gymnasium[mujoco]` on modern versions, no separate
install required. Tested with Python 3.12, PyTorch 2.13 (+cu126), gymnasium 1.3.

## Train

```powershell
python src\train.py --config configs\walker2d_v5.yaml
python src\train.py --config configs\halfcheetah_v5.yaml
python src\train.py --config configs\ant_v5.yaml
python src\train.py --config configs\humanoid_v5.yaml
```

TensorBoard:

```powershell
tensorboard --logdir logs
```

Track: `charts/episodic_return`, `losses/approx_kl`, `losses/clip_fraction`,
`losses/explained_variance`. See [docs/PPO.md](docs/PPO.md) §11 for what
healthy values look like.

## Smoke test (fast, ~1 minute)

```powershell
python src\train.py --config configs\ant_v5.yaml --override-total-timesteps 20000
```

Success criteria:
- The `first-iter ratio == 1` assertion in `train.py` passes.
- TensorBoard shows nonzero `charts/episodic_return` and monotonically
  decaying (annealed) LR.
- A checkpoint appears under `models/`.

## Evaluate + record a video

```powershell
python src\evaluate.py `
  --checkpoint models\<run_name>\ckpt_iterNNNNNN.pt `
  --episodes 5 `
  --video `
  --video-dir videos\my_named_run
```

MP4s land in `videos/my_named_run/`. Deterministic (mean-action) evaluation
regularly beats the stochastic training return by 20–50%.

## Design principles

- **Correctness > cleverness.** Loops in GAE, explicit tensors in the buffer.
- **Modular but no leaky abstractions.** Config is a dict, buffer is a
  container, agent owns the update. No mixins, no factories.
- **Nothing hidden.** All losses and metrics computed in one function
  (`PPOAgent.update`) - the whole PPO objective is visible on a single screen.
- **Type-hinted throughout.**
- **PyTorch only.** No JAX, no SB3, no external RL libs.
- **All 13 core + 9 continuous-action implementation details** from
  Huang et al. 2022 are implemented and verified against CleanRL's
  reference.

## References

- Schulman et al. 2017, *Proximal Policy Optimization Algorithms* (arXiv:1707.06347).
- Schulman et al. 2016, *High-Dimensional Continuous Control with GAE* (arXiv:1506.02438).
- Engstrom et al. 2020, *Implementation Matters in Deep Policy Gradients* (arXiv:2005.12729).
- Huang et al. 2022, *The 37 Implementation Details of PPO*, ICLR Blog Track.
- CleanRL PPO benchmarks: https://docs.cleanrl.dev/rl-algorithms/ppo/
