import torch
from tensordict import TensorDict

from rozart_residual.residual import BASE_ACTION_KEY, ResidualEnv, ResidualPolicy


class EchoEnv:
    """Observes the last action it was given."""

    num_envs = 2
    num_actions = 3
    max_episode_length = 10
    device = "cpu"
    cfg = None

    def __init__(self):
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.last = torch.zeros(2, 3)

    def get_observations(self):
        return TensorDict({"last": self.last.clone()}, batch_size=2)

    def step(self, actions):
        self.last = actions.clone()
        return self.get_observations(), torch.zeros(2), torch.zeros(2), {}


class Constant(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, x):
        return torch.full((x.shape[0], 3), float(self.value))


def test_zero_residual_reproduces_base():
    env = ResidualEnv(EchoEnv(), Constant(0.5), ["last"], residual_scale=0.1)
    assert torch.equal(env.get_observations()[BASE_ACTION_KEY], torch.full((2, 3), 0.5))
    obs, *_ = env.step(torch.zeros(2, 3))
    assert torch.allclose(obs["last"], torch.full((2, 3), 0.5))


def test_residual_is_scaled_and_clamped():
    env = ResidualEnv(EchoEnv(), Constant(0.5), ["last"], residual_scale=0.1, action_limit=0.55)
    obs, *_ = env.step(torch.ones(2, 3))
    assert torch.allclose(obs["last"], torch.full((2, 3), 0.55))
    obs, *_ = env.step(-torch.ones(2, 3))
    assert torch.allclose(obs["last"], torch.full((2, 3), 0.4))


def test_policy_matches_env_composition():
    obs = TensorDict({"last": torch.zeros(2, 3)}, batch_size=2)
    policy = ResidualPolicy(
        Constant(0.5), Constant(1.0), ["last"], ["last", BASE_ACTION_KEY], 0.1
    )
    assert torch.allclose(policy(obs), torch.full((2, 3), 0.6))
