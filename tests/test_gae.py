"""Unit tests for Generalized Advantage Estimation (GAE)."""
import torch

from gae import compute_gae


def _reference_gae(rewards, values, terminated, last_value, last_terminated, gamma, lam):
    """Straightforward single-environment reference implementation."""
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nonterm, next_v = 1.0 - float(last_terminated), last_value
        else:
            nonterm, next_v = 1.0 - float(terminated[t + 1]), values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv


def test_single_step_is_td_error():
    r = torch.tensor([[1.0]])
    v = torch.tensor([[0.5]])
    adv, ret = compute_gae(r, v, torch.tensor([[False]]), torch.tensor([2.0]),
                           torch.tensor([False]), gamma=0.9, gae_lambda=0.95)
    expected = 1.0 + 0.9 * 2.0 - 0.5
    assert torch.allclose(adv, torch.tensor([[expected]]))
    assert torch.allclose(ret, adv + v)


def test_terminal_last_step_does_not_bootstrap():
    adv, _ = compute_gae(torch.tensor([[1.0]]), torch.tensor([[0.5]]), torch.tensor([[False]]),
                         torch.tensor([100.0]), torch.tensor([True]), gamma=0.99, gae_lambda=0.95)
    assert torch.allclose(adv, torch.tensor([[0.5]]))


def test_lambda_one_gives_discounted_return_minus_value():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.1], [0.2], [0.3]])
    term = torch.zeros(3, 1, dtype=torch.bool)
    gamma = 0.9
    adv, ret = compute_gae(rewards, values, term, torch.tensor([0.0]), torch.tensor([True]),
                           gamma=gamma, gae_lambda=1.0)
    mc = torch.tensor([[1 + gamma * 2 + gamma**2 * 3], [2 + gamma * 3], [3.0]])
    assert torch.allclose(ret, mc, atol=1e-6)
    assert torch.allclose(adv, mc - values, atol=1e-6)


def test_matches_reference_across_parallel_envs_with_episode_boundaries():
    g = torch.Generator().manual_seed(0)
    T, N, gamma, lam = 16, 3, 0.99, 0.95
    rewards = torch.randn(T, N, generator=g)
    values = torch.randn(T, N, generator=g)
    terminated = torch.rand(T, N, generator=g) < 0.2
    last_value = torch.randn(N, generator=g)
    last_terminated = torch.tensor([False, True, False])
    adv, ret = compute_gae(rewards, values, terminated, last_value, last_terminated, gamma, lam)
    assert adv.shape == ret.shape == (T, N)
    for n in range(N):
        ref = _reference_gae(rewards[:, n].tolist(), values[:, n].tolist(), terminated[:, n].tolist(),
                             last_value[n].item(), last_terminated[n].item(), gamma, lam)
        assert torch.allclose(adv[:, n], torch.tensor(ref), atol=1e-5)
