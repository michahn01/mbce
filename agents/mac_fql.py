from functools import partial

import jax
import jax.numpy as jnp

from agents.mac import MACAgent
from utils.networks import Value


class MACFQLAgent(MACAgent):
    """MAC with a flow-distilled actor optimized against the model critic."""

    def bc_actor_loss(self, batch, grad_params, rng):
        """Compute the BC flow actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_bc_flow')(
            batch['observations'], actions=x_t, times=t, params=grad_params
        )
        flow_loss = jnp.mean((pred - vel) ** 2)
        return (flow_loss, {'flow_loss': flow_loss, 'vel': jnp.linalg.norm(vel, axis=-1).mean()})

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, batch['rollout_actions'].shape)
        target_flow_actions = self.sample_flow_actions(batch['rollout_observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['rollout_observations'], actions=noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['rollout_observations'], actions=actor_actions)
        # Average across rollout time before computing the normalization scale.
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        q_loss = lam * q_loss
        actor_loss = self.config['alpha'] * distill_loss + q_loss
        return (
            actor_loss,
            {'actor_loss': actor_loss, 'distill_loss': distill_loss, 'q_loss': q_loss, 'q': q.mean()},
        )

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        _, dynamics_rng, bc_rng, success_rng, rollout_rng, critic_rng, value_rng, actor_rng = (
            jax.random.split(rng, 8)
        )
        losses = dict(
            dynamics=self.dynamics_loss(batch, grad_params, dynamics_rng),
            success=self.success_loss(batch, grad_params, success_rng),
            reward=self.reward_loss(batch, grad_params, success_rng),
            bc_actor=self.bc_actor_loss(batch, grad_params, bc_rng),
        )
        self.rollout(batch, grad_params, rollout_rng)
        info = {f'gae/{k}': v for k, v in self.compute_gae(batch).items()}
        losses.update(
            critic=self.critic_loss(batch, grad_params, critic_rng),
            value=self.value_loss(batch, grad_params, value_rng),
            actor=self.actor_loss(batch, grad_params, actor_rng),
        )
        return self.combine_losses(losses, info)

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(self, observations, seed=None, mode='test'):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        actions = jax.random.normal(
            action_seed, (*observations.shape[: -len(self.config['ob_dims'])], self.config['action_dim'])
        )
        actions = self.network.select('actor_onestep_flow')(observations=observations, actions=actions)
        actions = jnp.clip(actions, -1, 1)
        if mode == 'test':
            action_dim = actions.shape[-1] // self.config['action_chunking']
            actions = actions[..., : action_dim * self.config['action_chunking_test']]
        return (actions, {})

    @jax.jit
    def sample_flow_actions(self, observations, noises):
        """Sample actions from the BC flow policy."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions=actions, times=t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def network_info(cls, observations, actions, config):
        info = super().network_info(observations, actions, config)
        critic = Value(config['value_hidden_dims'], layer_norm=config['layer_norm'])
        info['critic'] = critic, (observations, actions)
        info['actor_bc_flow'] = info.pop('actor_flow')
        return info
