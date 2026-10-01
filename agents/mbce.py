import copy
from functools import partial

import jax
import jax.numpy as jnp

from agents.mac import MACAgent, sg
from utils.networks import EditActor, LogParam, Value


class MBCEAgent(MACAgent):
    """Bounded TD-guided edits before model-based chunk selection."""

    def apply_edit(self, observations, base_actions, rng):
        inputs = jnp.concatenate([observations, base_actions], axis=-1)
        edit = self.network.select('edit_actor')(inputs).sample(seed=rng)
        return jnp.clip(base_actions + self.config['edit_scale'] * edit, -1, 1)

    def select_actions(self, observations, rng, num_samples):
        """Edit independent behavior proposals and rank them with the model critic."""
        _, noise_rng, edit_rng = jax.random.split(rng, 3)
        states = jnp.repeat(observations[None], num_samples, axis=0)
        noises = jax.random.normal(
            noise_rng, (num_samples, *observations.shape[:-1], self.config['action_dim'])
        )
        proposals = self.network.select('actor_onestep_flow')(states, noises)
        proposals = jnp.clip(proposals, -1, 1)
        actions = self.apply_edit(states, proposals, edit_rng)
        values = self.network.select('critic')(states, actions)
        if observations.ndim == 2:
            return actions[jnp.argmax(values, axis=0), jnp.arange(observations.shape[0])]
        return actions[jnp.argmax(values)]

    def deploy_actions(self, observations, rng):
        """Return the full deployment chunk for TD bootstrapping."""
        _, select_rng, _ = jax.random.split(rng, 3)
        return self.select_actions(observations, select_rng, self.config['num_samples_test'])

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(self, observations, seed=None, mode='test'):
        if mode == 'train':
            actions = self.select_actions(observations, seed, self.config['num_samples_train'])
        elif mode == 'test':
            actions = self.deploy_actions(observations, seed)
            action_dim = actions.shape[-1] // self.config['action_chunking']
            actions = actions[..., : action_dim * self.config['action_chunking_test']]
        else:
            raise ValueError(f'Unknown sampling mode: {mode}')
        return actions, {}

    def td_critic_loss(self, batch, grad_params, rng):
        """Clipped double-Q regression on recorded, success-masked chunk returns."""
        q = self.network.select('td_critic')(batch['observations'], batch['actions'], params=grad_params)
        _, action_rng = jax.random.split(rng)
        actions = sg(self.deploy_actions(batch['next_observations'], action_rng))
        next_q = self.network.select('target_td_critic')(batch['next_observations'], actions).min(axis=0)
        discount = self.config['discount'] ** self.config['action_chunking']
        masks = batch['td_masks']
        target = sg(batch['td_rewards'] + discount * jnp.where(masks > 0, masks * next_q, 0.0))
        loss = jnp.square(q - target).mean()
        return loss, {'td_critic_loss': loss, 'td_q_mean': q.mean(), 'td_target_mean': target.mean()}

    def edit_actor_loss(self, batch, grad_params, rng):
        """Improve unselected proposals with fixed critics and learned entropy temperature."""
        _, noise_rng, edit_rng = jax.random.split(rng, 3)
        noises = jax.random.normal(noise_rng, batch['actions'].shape)
        base_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises)
        base_actions = sg(jnp.clip(base_actions, -1, 1))
        inputs = jnp.concatenate([batch['observations'], base_actions], axis=-1)
        distribution = self.network.select('edit_actor')(inputs, params=grad_params)
        # Joint sampling avoids an unstable inverse tanh near the action bounds.
        edit, log_probs = distribution.sample_and_log_prob(seed=edit_rng)
        actions = jnp.clip(base_actions + self.config['edit_scale'] * edit, -1, 1)
        q = self.network.select('td_critic')(batch['observations'], actions).mean(axis=0)
        q_loss = -q.mean()
        entropy_loss = (log_probs * self.network.select('edit_alpha')()).mean()
        alpha = self.network.select('edit_alpha')(params=grad_params)
        entropy = -sg(log_probs).mean()
        target_entropy = -self.config['edit_target_entropy_multiplier'] * self.config['action_dim']
        alpha_loss = alpha * (entropy - target_entropy)
        loss = q_loss + entropy_loss + alpha_loss
        return loss, dict(
            edit_q_loss=q_loss,
            edit_entropy_loss=entropy_loss,
            edit_alpha_loss=alpha_loss,
            edit_entropy=entropy,
            edit_alpha=alpha,
            edit_abs_mean=jnp.abs(actions - base_actions).mean(),
        )

    def auxiliary_losses(self, batch, grad_params, td_rng, edit_rng):
        return {
            'td_critic': self.td_critic_loss(batch, grad_params, td_rng),
            'edit': self.edit_actor_loss(batch, grad_params, edit_rng),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        (_, dynamics_rng, bc_rng, rollout_rng, value_rng, critic_rng, success_rng, td_rng, edit_rng) = (
            jax.random.split(rng, 9)
        )
        losses = {
            'dynamics': self.dynamics_loss(batch, grad_params, dynamics_rng),
            'bc_actor': self.bc_actor_loss(batch, grad_params, bc_rng),
        }
        self.rollout(batch, grad_params, rollout_rng)
        info = {f'gae/{k}': v for k, v in self.compute_gae(batch).items()}
        losses.update(
            value=self.value_loss(batch, grad_params, value_rng),
            critic=self.critic_loss(batch, grad_params, critic_rng),
            success=self.success_loss(batch, grad_params, success_rng),
            reward=self.reward_loss(batch, grad_params, success_rng),
        )
        losses.update(self.auxiliary_losses(batch, grad_params, td_rng, edit_rng))
        return self.combine_losses(losses, info)

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        network, info = self.network.apply_loss_fn(lambda params: self.total_loss(batch, params, rng))
        self.target_update(network, 'value')
        self.target_update(network, 'td_critic')
        return self.replace(network=network, rng=new_rng), info

    @classmethod
    def network_info(cls, observations, actions, config):
        info = super().network_info(observations, actions, config)
        critic = Value(config['value_hidden_dims'], layer_norm=config['layer_norm'], num_ensembles=2)
        actor = EditActor(config['actor_hidden_dims'], actions.shape[-1], layer_norm=config['layer_norm'])
        info.update(
            edit_actor=(actor, (jnp.concatenate([observations, actions], axis=-1),)),
            edit_alpha=(LogParam(config['edit_initial_temperature']), ()),
            td_critic=(critic, (observations, actions)),
            target_td_critic=(copy.deepcopy(critic), (observations, actions)),
        )
        return info
