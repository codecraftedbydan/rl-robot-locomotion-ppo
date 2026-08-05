# Learning Notes — Understanding this RL project

This is a plain-English explanation of what's happening under the hood.
No maths jargon. Real-world analogies throughout.

---

## 1. What are we actually building?

We're teaching a simulated robot to **move forward using trial and error**.

The robot takes an action (move its joints), gets a reward (how far did it move?),
and slowly learns which actions lead to more reward. We never write a single rule
like "bend knee 30 degrees" — the robot figures that out entirely on its own.

**The algorithm doing this is called PPO** (Proximal Policy Optimization).
It's the same family of algorithm that trained OpenAI's Dota 2 bot and
fine-tunes ChatGPT to follow instructions.

---

## 2. The three core concepts

### The Policy (actor.py)
Think of this as **the brain**. It takes in what the robot can sense
(joint angles, velocity, contacts) and outputs which way to move each joint.

Technically: it's a small neural network (two layers of 64 neurons) that maps
*observations → actions*.

The policy is stochastic — it doesn't always pick the same action. It samples
from a probability distribution (a Gaussian / bell curve) centred on its best
guess. This randomness is *intentional* — it's how the robot explores.

### The Value Function (critic.py)
Think of this as **an internal rating of how good the current situation is**.

"If I'm in this position right now, how much total reward can I expect to
collect from here?" That's what the critic answers.

It's a second neural network that outputs a single number: the expected future reward.
We use it to figure out whether an action was *surprisingly good or bad* vs
what we expected.

### The Environment (gymnasium + MuJoCo)
The *world the robot lives in*. It runs the physics simulation, gives the robot
its sensor readings, and hands out rewards.

- **Ant-v5**: 4-legged spider-robot. 8 joints. Falls over = episode ends.
- **Walker2d-v5**: 2D stick-figure biped. 6 joints. Constrained to a flat plane.
- **HalfCheetah-v5**: 2D cheetah. 6 joints. Never falls (no termination).
- **Humanoid-v5**: 3D human-shaped robot. 17 joints. Hardest by far.

---

## 3. How PPO learns — the feedback loop

```
COLLECT DATA (rollout)
    Robot runs in the environment for ~2000 steps
    Every step: store (observation, action taken, reward received, value estimate)

COMPUTE HOW GOOD EACH ACTION WAS (GAE)
    For each step: was this action better or worse than expected?
    This difference is called the "advantage"

UPDATE THE BRAIN (PPO update)
    Nudge the policy toward actions that had positive advantage
    Nudge it away from actions that had negative advantage
    Don't change it too dramatically in one go (that's the "proximal" in PPO)

REPEAT ~1000 times
```

That's the whole training loop. Everything else is implementation detail.

---

## 4. Why "proximal"? The trust region intuition

Imagine you're training a dog.

If you let the dog try lots of different behaviours and only reward the ones
that work, eventually it learns. But if after one reward you immediately
*shout at it* every time it does the old behaviour, it gets confused.

PPO prevents this by saying: **only change the policy a little bit per update**.
If an update would make an action 50% more likely (ratio = 1.5 when we want
max 1.2), we clip the gradient so it stops being rewarded past that limit.

This is the "clipped surrogate objective". It's the core mathematical
innovation of PPO.

---

## 5. The Advantage — what the value function is actually for

After each action we ask: **did things go better or worse than expected?**

```
advantage = (actual reward + expected future reward) - (what we expected from this state)
```

Positive advantage = "that action was better than I thought, do more of it"
Negative advantage = "that action was worse than I thought, do less of it"
Near-zero advantage = "exactly what I expected, no update needed"

This is why we need both an actor *and* a critic. The critic provides the
"what I expected" baseline. Without it, we'd just reward every action that
happened to precede a good outcome — even if it had nothing to do with it.

---

## 6. Generalized Advantage Estimation (GAE)

The advantage formula above is simple but noisy. A slight refinement
blends **short-term accuracy** (TD error: just one step ahead) with
**long-term completeness** (Monte Carlo: sum all future rewards).

GAE mixes them using a parameter λ = 0.95.
- λ = 0: only look one step ahead (low variance, high bias)
- λ = 1: look all the way to the end of the episode (high variance, low bias)
- λ = 0.95: mix that works well in practice

The backward recursion in `src/gae.py` is:
```
advantage[t] = (reward[t] + future_value) - current_value
             + gamma * lambda * advantage[t+1]
```
Starting from the last timestep and going backward. That's it.

---

## 7. Observation normalization — why it matters

MuJoCo observations contain joint angles (around ±3), velocities (could be ±50),
and contact forces (could be ±500). If you feed these raw into a neural network,
the large numbers dominate and training is unstable.

**Solution**: keep a running average of the mean and standard deviation of
every observation dimension. Subtract the mean, divide by the std.
Now everything lives roughly in the range [-5, 5].

This is `NormalizeObservation` in the env factory. The stats are saved inside
the checkpoint — without them, the loaded policy would see "wrong" inputs at
evaluation time and behave randomly.

---

## 8. Reward normalization — why it matters

Similar problem for rewards. Ant's reward is ~3 per step. HalfCheetah's can
be ~30 per step. If you train on both with the same learning rate, the
value function for HalfCheetah gets wild gradients.

**Solution**: divide the reward by a running estimate of the return's
standard deviation (not the reward's — the discounted cumulative reward's).
Now the value function always fits numbers near [-5, 5] regardless of env.

This is `NormalizeReward` in the env factory.

---

## 9. What "terminated" vs "truncated" means (a subtle bug source)

When an episode ends, there are two distinct reasons:

- **Terminated**: the MDP genuinely ended. The Ant fell over → health = 0 → game over.
  Next state has *no value*. Bootstrap = 0.

- **Truncated**: we ran out of time. The episode was cut at 1000 steps but the
  robot was still standing. The game *would have continued*.
  Next state has value = V(s). Bootstrap using the critic.

If you confuse these (using `terminated | truncated` as the mask), you
artificially make the value function think every episode-boundary state is
worthless. This biases training downward and is a classic silent bug.

`src/gae.py` uses `terminated` only. `truncated` is handled separately by
bootstrapping the last value regardless.

---

## 10. Why training return ≠ evaluation return

During training, the policy adds **random noise** to every action (it samples
from the Gaussian). This is intentional — it explores.

But during evaluation (`src/evaluate.py`), we use the **mean action** directly.
No noise. The policy plays its "best guess" every step.

This is why:
- Walker2d training return: ~1800
- Walker2d evaluation return: **3690** (2× higher)

The policy *knows* a much better strategy than the noisy version suggests.
The evaluation number is the real capability. That's what all serious RL papers
report.

---

## 11. Our results in plain English

| Env | What happened | Eval return | Benchmark |
|-----|--------------|-------------|-----------|
| **HalfCheetah-v5** | Robot runs fast, smooth gallop | 5257 | CleanRL: 1443 (we're 3.6× higher) |
| **Walker2d-v5** | Robot hops both legs together, never falls | 3691 | CleanRL: 2288 (we're above the top of their range) |
| **Humanoid-v5** | Robot stands, barely moves | ~640 | CleanRL: 716 (we match) |
| **Ant-v5** | Robot trots but falls ~35% of eval episodes | 567 ± 379 (high variance) | CleanRL: ~2400 (needs more training) |

**Why is Humanoid so much worse?**

Humanoid has 17 joints and can fall in 3D. The reward landscape has a
"standing still" local optimum that earns ~5 reward/step safely. Any attempt
to walk risks falling → 0 reward. PPO with a small number of parallel
environments gets stuck here. Breaking out requires either:
- A different algorithm (SAC — off-policy, replay buffer)
- Massively more parallel environments (8000+ in Isaac Gym)
- Aggressive hyperparameter tuning

This is a known limitation of vanilla PPO on hard continuous control,
not a bug in our code.

---

## 12. The files, in one sentence each

| File | What it does |
|------|-------------|
| `src/actor.py` | Neural network: obs → action distribution |
| `src/critic.py` | Neural network: obs → expected future reward |
| `src/gae.py` | Computes how good each past action was |
| `src/buffer.py` | Stores one rollout's data until the update |
| `src/ppo.py` | Does the gradient update (the actual learning) |
| `src/train.py` | The main loop: collect data → compute advantages → update |
| `src/evaluate.py` | Runs the trained policy without noise, records video |
| `src/utils.py` | Seeds, env wrappers, weight initialisation helpers |
| `configs/*.yaml` | All hyperparameters (one file per environment) |
| `docs/PPO.md` | Full mathematical derivation and implementation details |
| `docs/Experiments.md` | Log of every training run with results and analysis |
| `docs/Results.md` | Summary tables comparing against published benchmarks |

---

## 13. Things that surprised us during this project

1. **Training return ≠ eval return by 2×.** The deterministic policy is much
   better than the stochastic one we measure during training.

2. **Batch size matters more than total compute.** Walker2d trained faster
   when we matched the batch size to what CleanRL uses (2048 samples/update),
   even though we had the same total steps.

3. **Humanoid is not a "bigger ant".** It's a fundamentally harder problem —
   not just a matter of more training or a bigger GPU.

4. **HalfCheetah is suspiciously easy.** 5256 return far exceeded our
   expectations. Probably because it never terminates — the reward signal is
   always flowing, and PPO can always improve without worrying about falling.

5. **The "first-iter ratio == 1" check.** A single assert in the training loop
   catches the most common implementation bug (computing log-probabilities
   wrong) before it silently corrupts a full training run.

6. **Ant is harder than it looks.** HalfCheetah never falls so PPO can always
   improve. Ant falls → episode ends → you get zero reward for the rest of
   the episode. This makes the reward landscape much harder to navigate — a
   partially-learned gait that stumbles occasionally scores catastrophically
   low, making the policy very sensitive to small changes.

---

## 14. Explanation of what I did

I built software that teaches a robot to walk, starting from knowing nothing.
It tries random movements, gets a score based on how far it moved, and gradually
learns which movements lead to higher scores. After a few million attempts, it
figures out a working gait entirely on its own — I never told it what walking
looks like. The same algorithm is used by OpenAI to make ChatGPT respond the
way it does, just with human feedback as the reward signal instead of forward
velocity.
