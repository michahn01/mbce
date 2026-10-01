"""Experiment configuration with explicit shared, method, and task defaults."""

import json
import math
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'


def read_config(name):
    return json.loads((CONFIG_DIR / f'{name}.json').read_text())


DEFAULTS = read_config('defaults')
ALGORITHMS = read_config('algorithms')
TASKS = read_config('tasks')
SWEEPS = read_config('sweeps')
ABLATION = read_config('ablation')


def resolve(algorithm, task, overrides=()):
    config = {
        'training': dict(DEFAULTS['training']),
        'agent': {**DEFAULTS['agent'], **ALGORITHMS[algorithm], **TASKS[task]['agents'].get(algorithm, {})},
    }
    for override in overrides:
        key, separator, raw = override.partition('=')
        section, dot, name = key.partition('.')
        if not separator or not dot or section not in config or name not in config[section]:
            raise ValueError(f'Unknown setting: {key}; use training.NAME=JSON or agent.NAME=JSON')
        config[section][name] = json.loads(raw)
    validate(config)
    return config


def validate(config):
    """Reject invalid values before launching a worker."""
    training, agent = config['training'], config['agent']
    integers = [
        *training.items(),
        *(
            (k, agent[k])
            for k in (
                'rollout_batch_size',
                'rollout_steps',
                'flow_steps',
                'action_chunking',
                'action_chunking_test',
            )
        ),
    ]
    integers += [(k, agent[k]) for k in ('num_samples_train', 'num_samples_test') if k in agent]
    for name, value in integers:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for key in ('actor_hidden_dims', 'value_hidden_dims', 'dynamics_hidden_dims'):
        dims = agent[key]
        if not isinstance(dims, (list, tuple)) or not dims or any(type(d) is not int or d < 1 for d in dims):
            raise ValueError(f'{key} must contain positive integer widths')
    for name in (
        'lr',
        'discount',
        'tau',
        'edit_scale',
        'alpha',
        'edit_initial_temperature',
        'edit_target_entropy_multiplier',
    ):
        if name in agent and (
            type(agent[name]) not in (int, float) or not math.isfinite(agent[name]) or agent[name] <= 0
        ):
            raise ValueError(f'{name} must be finite and positive')
    if not 0 < agent['discount'] <= 1 or not 0 < agent['tau'] <= 1:
        raise ValueError('discount and tau must lie in (0, 1]')
    if type(agent['layer_norm']) is not bool:
        raise ValueError('layer_norm must be boolean')
    if agent['action_chunking_test'] > agent['action_chunking']:
        raise ValueError('Execution length cannot exceed chunk length')
    if agent['rollout_batch_size'] > training['batch_size']:
        raise ValueError('Rollout batch size cannot exceed the training batch size')
