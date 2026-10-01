"""Named Flax modules with an explicit parameter-gradient boundary."""

import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """Initialize a collection of networks and apply them by name."""

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError('Initialization inputs must match the network names')
            return {key: self.modules[key](*inputs) for key, inputs in kwargs.items()}
        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Network parameters and Adam state shared by an agent's joint objective."""

    step: int
    apply_fn: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model, params, tx):
        return cls(step=1, apply_fn=model.apply, params=params, tx=tx, opt_state=tx.init(params))

    def __call__(self, *args, params=None, **kwargs):
        # Stored parameters stay fixed when differentiating an explicit params argument.
        return self.apply_fn({'params': self.params if params is None else params}, *args, **kwargs)

    def select(self, name):
        return functools.partial(self, name=name)

    def apply_loss_fn(self, loss_fn):
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=params, opt_state=opt_state), info
