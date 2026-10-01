import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Dynamics, Value, ActorVectorField

sg = jax.lax.stop_gradient


class MACAgent(flax.struct.PyTreeNode):
    """Model-based action chunk selection from distilled behavior proposals."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        """Compute the BCE loss."""
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

    def dynamics_loss(self, batch, grad_params, rng):
        """Compute the dynamics loss."""
        pred_next_ob = self.network.select('dynamics')(
            batch['observations'], batch['actions'], params=grad_params
        )
        dynamics_loss = jnp.square(pred_next_ob - batch['next_observations']).mean()
        return (dynamics_loss, {'dynamics_loss': dynamics_loss})

    def success_loss(self, batch, grad_params, rng):
        """Fit the chunk continuation predictor."""
        logits = self.network.select('success')(batch['observations'], batch['actions'], params=grad_params)
        target = 1 - batch['masks']
        loss = self.bce_loss(logits, target).mean()
        return loss, {'success_loss': loss, 'success_mean': jax.nn.sigmoid(logits).mean()}

    def reward_loss(self, batch, grad_params, rng):
        """Compute the reward prediction loss."""
        rewards = self.network.select('reward')(batch['observations'], batch['actions'], params=grad_params)
        rewards_target = batch['rewards']
        reward_loss = jnp.square(rewards_target - rewards).mean()
        return (
            reward_loss,
            {
                'reward_loss': reward_loss,
                'reward_mean': rewards.mean(),
                'reward_max': rewards.max(),
                'reward_min': rewards.min(),
                'data_reward_mean': batch['rewards'].mean(),
            },
        )

    def bc_actor_loss(self, batch, grad_params, rng):
        """Compute the BC flow actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_flow')(
            batch['observations'], actions=x_t, times=t, params=grad_params
        )
        flow_loss = jnp.mean((pred - vel) ** 2)
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.sample_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], actions=noises, params=grad_params
        )
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
        actor_loss = flow_loss + distill_loss
        actor_info = {
            'flow_loss': flow_loss,
            'vel': jnp.linalg.norm(vel, axis=-1).mean(),
            'distill_loss': distill_loss,
            'actor_loss': actor_loss,
        }
        return (actor_loss, actor_info)

    def rollout(self, batch, grad_params, rng):
        """Perform model-based rollouts."""
        init_ob = batch['observations'][: self.config['rollout_batch_size']]

        def rollout_step(carry, _):
            ob, rng = carry
            rng, action_rng = jax.random.split(rng, 2)
            action, _ = self.sample_actions(ob, action_rng, mode='train')
            next_ob = self.network.select('dynamics')(ob, action)
            return ((next_ob, rng), {'observations': ob, 'actions': action, 'next_observations': next_ob})

        rng, rollout_rng = jax.random.split(rng)
        init_carry = (init_ob, rollout_rng)
        final_carry, rollout_data = jax.lax.scan(
            rollout_step, init_carry, None, length=self.config['rollout_steps']
        )
        batch['rollout_observations'] = rollout_data['observations']
        batch['rollout_next_observations'] = rollout_data['next_observations']
        batch['rollout_actions'] = rollout_data['actions']
        success = jax.nn.sigmoid(
            self.network.select('success')(batch['rollout_observations'], batch['rollout_actions'])
        )
        success = jnp.where(success > 0.5, jnp.ones_like(success), jnp.zeros_like(success))
        rewards = self.network.select('reward')(batch['rollout_observations'], batch['rollout_actions'])
        batch['rollout_rewards'] = rewards
        batch['rollout_masks'] = 1.0 - success

    def compute_gae(self, batch):
        """Compute bootstrapped model returns with lambda = 1."""
        value_module = 'target_value'
        values = self.network.select(value_module)(batch['rollout_observations'])
        next_values = self.network.select(value_module)(batch['rollout_next_observations'])

        def scan_fn(lastgaelam, inputs):
            discount = self.config['discount'] ** self.config['action_chunking']
            reward, mask, value, next_value = inputs
            delta = reward + mask * discount * next_value - value
            advantage = delta + mask * discount * lastgaelam
            return (advantage, advantage)

        zeros = jnp.zeros(batch['rollout_rewards'].shape[1])
        _, advs = jax.lax.scan(
            scan_fn,
            zeros,
            (batch['rollout_rewards'], batch['rollout_masks'], values, next_values),
            reverse=True,
        )
        returns = values + advs
        batch['rollout_returns'] = returns
        return {
            'return_mean': returns.mean(),
            'return_min': returns.min(),
            'return_max': returns.max(),
            'return_std': returns.std(),
        }

    def value_loss(self, batch, grad_params, rng):
        """Compute the value loss."""
        v = self.network.select('value')(batch['rollout_observations'], params=grad_params)
        value_loss = ((v - batch['rollout_returns']) ** 2).mean()
        return (
            value_loss,
            {'value_loss': value_loss, 'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min()},
        )

    def critic_loss(self, batch, grad_params, rng):
        """Compute the critic loss."""
        q = self.network.select('critic')(
            batch['rollout_observations'][0], batch['rollout_actions'][0], params=grad_params
        )
        discount = self.config['discount'] ** self.config['action_chunking']
        reward = batch['rollout_rewards'][0]
        mask = batch['rollout_masks'][0]
        next_v = self.network.select('value')(batch['rollout_next_observations'][0])
        target = sg(reward + discount * mask * next_v)
        critic_loss = ((q - target) ** 2).mean()
        return (
            critic_loss,
            {'critic_loss': critic_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min()},
        )

    @staticmethod
    def combine_losses(losses, info):
        """Combine named objectives and their scalar diagnostics."""
        for name, (_, metrics) in losses.items():
            info.update({f'{name}/{key}': value for key, value in metrics.items()})
        loss = sum(value for value, _ in losses.values())
        return loss, dict(info, total_loss=loss)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        _, dynamics_rng, bc_rng, rollout_rng, value_rng, _, success_rng = jax.random.split(rng, 7)
        losses = dict(
            dynamics=self.dynamics_loss(batch, grad_params, dynamics_rng),
            bc_actor=self.bc_actor_loss(batch, grad_params, bc_rng),
        )
        self.rollout(batch, grad_params, rollout_rng)
        info = {f'gae/{k}': v for k, v in self.compute_gae(batch).items()}
        losses.update(
            value=self.value_loss(batch, grad_params, value_rng),
            critic=self.critic_loss(batch, grad_params, success_rng),
            success=self.success_loss(batch, grad_params, success_rng),
            reward=self.reward_loss(batch, grad_params, success_rng),
        )
        return self.combine_losses(losses, info)

    def target_update(self, network, module_name):
        """Update the target using the online parameters preceding this Adam step."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        return (self.replace(network=new_network, rng=new_rng), info)

    @partial(jax.jit, static_argnames=['mode'])
    def sample_actions(self, observations, seed=None, mode='test'):
        """Sample actions from the actor."""
        num_samples = self.config['num_samples_train'] if mode == 'train' else self.config['num_samples_test']
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), num_samples, axis=0)
        n_noises = jax.random.normal(seed, (num_samples, *observations.shape[:-1], self.config['action_dim']))
        n_actions = self.network.select('actor_onestep_flow')(n_observations, actions=n_noises)
        n_actions = jnp.clip(n_actions, -1, 1)
        q = self.network.select('critic')(n_observations, n_actions)
        if len(observations.shape) == 2:
            actions = n_actions[jnp.argmax(q, axis=0), jnp.arange(observations.shape[0])]
        else:
            actions = n_actions[jnp.argmax(q)]
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
            vels = self.network.select('actor_flow')(observations, actions=actions, times=t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def network_info(cls, observations, actions, config):
        """Network definitions and initialization inputs."""
        value = Value(config['value_hidden_dims'], layer_norm=config['layer_norm'])
        critic = Value(config['dynamics_hidden_dims'], layer_norm=config['layer_norm'])
        flow = ActorVectorField(
            config['actor_hidden_dims'], actions.shape[-1], layer_norm=config['layer_norm']
        )
        dynamics = Dynamics(
            config['dynamics_hidden_dims'], observations.shape[-1], layer_norm=config['layer_norm']
        )
        return dict(
            dynamics=(dynamics, (observations, actions)),
            value=(value, (observations,)),
            critic=(critic, (observations, actions)),
            target_value=(copy.deepcopy(value), (observations,)),
            actor_flow=(flow, (observations, actions, actions[..., :1])),
            actor_onestep_flow=(copy.deepcopy(flow), (observations, actions, None)),
            success=(copy.deepcopy(critic), (observations, actions)),
            reward=(copy.deepcopy(critic), (observations, actions)),
        )

    @classmethod
    def create(cls, seed, example_batch, config):
        """Initialize the networks and Adam optimizer."""
        config = dict(config)
        for name in ('actor_hidden_dims', 'value_hidden_dims', 'dynamics_hidden_dims'):
            config[name] = tuple(config[name])
        observations, actions = example_batch['observations'], example_batch['actions']
        config['action_dim'] = actions.shape[-1]
        config['ob_dims'] = observations.shape[1:]
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        info = cls.network_info(observations, actions, config)
        model = ModuleDict({name: definition for name, (definition, _) in info.items()})
        params = model.init(init_rng, **{name: args for name, (_, args) in info.items()})['params']
        for name in info:
            if name.startswith('target_'):
                params[f'modules_{name}'] = params[f'modules_{name[7:]}']
        network = TrainState.create(model, params, tx=optax.adam(config['lr']))
        return cls(rng=rng, network=network, config=flax.core.freeze(config))
