import jax
import jax.numpy as jnp

from agents.mac import MACAgent
from agents.mbce import MBCEAgent


class MBCERandomAgent(MBCEAgent):
    """Model-based selection with bounded, tanh-squashed standard-normal edits."""

    def apply_edit(self, observations, base_actions, rng):
        edit = jnp.tanh(jax.random.normal(rng, base_actions.shape, dtype=base_actions.dtype))
        return jnp.clip(base_actions + self.config['edit_scale'] * edit, -1, 1)

    def auxiliary_losses(self, batch, grad_params, td_rng, edit_rng):
        return {}

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        network, info = self.network.apply_loss_fn(lambda params: self.total_loss(batch, params, rng))
        self.target_update(network, 'value')
        return self.replace(network=network, rng=new_rng), info

    @classmethod
    def network_info(cls, observations, actions, config):
        return MACAgent.network_info(observations, actions, config)
