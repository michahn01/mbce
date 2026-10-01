"""Recorded chunks with distinct model and episodic TD reward targets."""

import numpy as np


def normalize(observations, scale):
    midpoint = (scale['max'] + scale['min']) / 2
    radius = (scale['max'] - scale['min']) / 2
    return (observations - midpoint) / (radius + 1e-3)


class ChunkDataset:
    """Sample complete chunks within compact OGBench trajectory boundaries."""

    def __init__(self, data, config, scale, seed, td_rewards=False):
        self.data = {key: data[key] for key in ('observations', 'actions', 'rewards', 'masks', 'terminals')}
        self.data['observations'] = normalize(self.data['observations'], scale)
        self.data['actions'] = np.clip(self.data['actions'], -1 + 1e-5, 1 - 1e-5)
        self.chunk_length = config['action_chunking']
        self.discounts = config['discount'] ** np.arange(self.chunk_length)
        self.td_rewards = td_rewards
        self.rng = np.random.RandomState(seed)
        ends = np.flatnonzero(self.data['terminals'] > 0)
        if not len(ends) or ends[-1] != len(self.data['observations']) - 1:
            raise ValueError('Dataset must end at a trajectory boundary')
        starts = np.r_[0, ends[:-1] + 1]
        self.valid_starts = np.concatenate(
            [np.arange(start, end - self.chunk_length) for start, end in zip(starts, ends)]
        )
        if not len(self.valid_starts):
            raise ValueError('Dataset has no complete action chunks')

    def sample(self, batch_size):
        starts = self.rng.choice(self.valid_starts, size=batch_size)
        indices = starts[:, None] + np.arange(self.chunk_length)
        rewards = self.data['rewards'][indices]
        batch = {
            'observations': self.data['observations'][starts],
            'actions': self.data['actions'][indices].reshape(batch_size, -1),
            'next_observations': self.data['observations'][starts + self.chunk_length],
            # The model backbone learns discounted mean rewards with continuing masks.
            'rewards': np.mean(rewards * self.discounts, axis=1),
            'masks': np.ones(batch_size, dtype=np.float32),
        }
        if self.td_rewards:
            masks = self.data['masks'][indices]
            prefix = np.concatenate([np.ones((batch_size, 1)), np.cumprod(masks[:, :-1], axis=1)], axis=1)
            batch['td_rewards'] = np.sum(rewards * prefix * self.discounts, axis=1).astype(np.float32)
            batch['td_masks'] = np.prod(masks, axis=1).astype(np.float32)
        return batch
