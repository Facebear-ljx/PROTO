from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False
    dropout_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1), layer_norm=self.layer_norm, dropout_rate=self.dropout_rate)(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False
    dropout_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations,
                     layer_norm=self.layer_norm)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False
    dropout_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations,
                         layer_norm=self.layer_norm)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations,
                         layer_norm=self.layer_norm)(observations, actions)
        return critic1, critic2
