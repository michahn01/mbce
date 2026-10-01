"""Train one configured offline RL run."""

import argparse
import csv
import json
from pathlib import Path
import time

import flax.serialization
import jax
import numpy as np
from tqdm import trange

from agents.mac import MACAgent
from agents.mac_fql import MACFQLAgent
from agents.mbce import MBCEAgent
from agents.mbce_random import MBCERandomAgent
from envs.tasks import make_env_and_datasets
from utils.datasets import ChunkDataset
from utils.evaluation import evaluate

AGENTS = {'mac': MACAgent, 'mbce': MBCEAgent, 'mbce-random': MBCERandomAgent, 'mac-fql': MACFQLAgent}


def train(directory, dataset_dir, wandb_mode):
    job = json.loads((directory / 'job.json').read_text())
    config, seed = job['config'], job['seed']
    settings, agent_config = config['training'], config['agent']
    env, data, validation, scale = make_env_and_datasets(job['task'], dataset_dir)
    np.savez(directory / 'normalization.npz', **scale)
    td_rewards = job['algorithm'] == 'mbce'
    dataset = ChunkDataset(data, agent_config, scale, seed, td_rewards)
    validation = ChunkDataset(validation, agent_config, scale, seed + 1, td_rewards)
    agent = AGENTS[job['algorithm']].create(seed, dataset.sample(1), agent_config)
    run = None
    if wandb_mode != 'disabled':
        import wandb

        run = wandb.init(
            project='mbce',
            name=directory.name,
            config=job,
            mode=wandb_mode,
            dir=str(directory),
            group=job['algorithm'],
        )
    start = time.monotonic()
    try:
        with (
            (directory / 'train.jsonl').open('w') as log,
            (directory / 'eval.csv').open('w', newline='') as evaluations,
        ):
            writer = csv.DictWriter(evaluations, fieldnames=['step', 'success', 'return'])
            writer.writeheader()
            for step in trange(1, settings['steps'] + 1, desc=directory.name):
                batch = jax.device_put(dataset.sample(settings['batch_size']))
                agent, info = agent.update(batch)
                if step == 1 or step % settings['log_interval'] == 0:
                    metrics = {f'training/{k}': float(v) for k, v in jax.device_get(info).items()}
                    _, val_info = agent.total_loss(
                        validation.sample(settings['batch_size']), grad_params=None
                    )
                    metrics.update({f'validation/{k}': float(v) for k, v in jax.device_get(val_info).items()})
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError(f'Nonfinite training metrics at step {step}')
                    metrics['elapsed_seconds'] = time.monotonic() - start
                    log.write(json.dumps(dict(step=step, **metrics)) + '\n')
                    log.flush()
                    if run:
                        run.log(metrics, step=step)
                if step % settings['eval_interval'] == 0 or step == settings['steps']:
                    metrics = evaluate(agent, env, scale, settings['eval_episodes'], (seed + step) % 2**32)
                    writer.writerow(dict(step=step, **metrics))
                    evaluations.flush()
                    if run:
                        run.log({f'evaluation/{k}': v for k, v in metrics.items()}, step=step)
            checkpoint = directory / 'checkpoint.msgpack'
            temporary = checkpoint.with_suffix('.tmp')
            temporary.write_bytes(flax.serialization.to_bytes(agent))
            temporary.replace(checkpoint)
            (directory / 'done.json').write_text(json.dumps({'step': step, **metrics}) + '\n')
    finally:
        env.close()
        if run:
            run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--wandb', choices=['disabled', 'online', 'offline'], default='disabled')
    args = parser.parse_args()
    train(args.run_dir, args.dataset_dir, args.wandb)
