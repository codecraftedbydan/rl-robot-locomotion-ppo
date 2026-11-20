"""Shape and initialisation checks for the diagonal Gaussian actor."""
import torch

from actor import Actor


def test_log_prob_and_entropy_sum_over_action_dims():
    torch.manual_seed(0)
    actor = Actor(obs_dim=5, act_dim=3, hidden_sizes=[16, 16])
    obs = torch.randn(7, 5)
    dist = actor._distribution(obs)
    actions = dist.sample()
    assert actions.shape == (7, 3)
    assert dist.log_prob(actions).shape == (7,)
    assert dist.entropy().shape == (7,)


def test_initial_mean_is_near_zero_and_std_matches_init():
    torch.manual_seed(0)
    actor = Actor(obs_dim=5, act_dim=2, hidden_sizes=[32], log_std_init=-0.5)
    mean = actor.act_deterministic(torch.randn(64, 5))
    assert mean.abs().max() < 0.1  # small last-layer gain (0.01)
    assert torch.allclose(actor.log_std.exp(), torch.full((2,), torch.tensor(-0.5).exp().item()))
