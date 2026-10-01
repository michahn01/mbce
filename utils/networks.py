from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs):
    """Create independently initialized ensemble members."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=0,
        axis_size=num_qs,
    )


class MLP(nn.Module):
    """GELU layers with optional layer normalization after each activation."""

    hidden_dims: Sequence[int]
    activate_final: bool = False
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.gelu(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class Value(nn.Module):
    """Scalar state value or action value, optionally with independent critics."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 1

    def setup(self):
        mlp = ensemblize(MLP, self.num_ensembles) if self.num_ensembles > 1 else MLP
        self.value_net = mlp((*self.hidden_dims, 1), layer_norm=self.layer_norm)

    def __call__(self, observations, actions=None):
        inputs = observations if actions is None else jnp.concatenate([observations, actions], axis=-1)
        return self.value_net(inputs).squeeze(-1)


class ActorVectorField(nn.Module):
    """Flow velocity or one-step action prediction."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True

    def setup(self):
        self.mlp = MLP((*self.hidden_dims, self.action_dim), layer_norm=self.layer_norm)

    def __call__(self, observations, actions, times=None):
        inputs = [observations, actions]
        if times is not None:
            inputs.append(times)
        return self.mlp(jnp.concatenate(inputs, axis=-1))


class EditActor(nn.Module):
    """Squashed diagonal Gaussian corrections conditioned on state and proposal."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))
        self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))

    def __call__(self, inputs):
        features = self.actor_net(inputs)
        means = self.mean_net(features)
        log_stds = jnp.clip(self.log_std_net(features), -5, 2)
        normal = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distrax.Transformed(normal, distrax.Block(distrax.Tanh(), ndims=1))


class Dynamics(nn.Module):
    """Predict the state displacement over an action chunk."""

    hidden_dims: Sequence[int]
    output_dim: int
    layer_norm: bool = True

    def setup(self):
        self.dynamics_net = MLP((*self.hidden_dims, self.output_dim), layer_norm=self.layer_norm)

    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        return observations + self.dynamics_net(inputs)
