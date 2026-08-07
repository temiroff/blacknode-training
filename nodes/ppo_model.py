"""Small continuous-control actor/critic used by Blacknode PPO jobs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - package diagnostics report optional dependency
    torch = None
    nn = None


@dataclass(frozen=True)
class PPOModelConfig:
    observation_dim: int
    action_dim: int
    hidden_dim: int = 128

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PPOModelConfig":
        return cls(
            observation_dim=int(value["observation_dim"]),
            action_dim=int(value["action_dim"]),
            hidden_dim=int(value.get("hidden_dim") or 128),
        )


def _network(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class PPOActorCritic(nn.Module if nn is not None else object):
    """Tanh-Gaussian actor plus a scalar critic."""

    def __init__(self, config: PPOModelConfig) -> None:
        if nn is None or torch is None:
            raise RuntimeError("torch is required for PPO training")
        super().__init__()
        self.config = config
        self.actor = _network(config.observation_dim, config.hidden_dim, config.action_dim)
        self.critic = _network(config.observation_dim, config.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((config.action_dim,), -0.5))

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(observation)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @staticmethod
    def _log_probability(distribution: Any, latent: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(latent)
        correction = torch.log(torch.clamp(1.0 - action.square(), min=1e-6))
        return (distribution.log_prob(latent) - correction).sum(dim=-1)

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        latent = distribution.sample()
        action = torch.tanh(latent)
        log_probability = self._log_probability(distribution, latent)
        value = self.critic(observation).squeeze(-1)
        return action, latent, log_probability, value

    def evaluate(self, observation: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        log_probability = self._log_probability(distribution, latent)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observation).squeeze(-1)
        return log_probability, entropy, value

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observation))
