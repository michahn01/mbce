"""Evaluate the action-chunk policy with replanning after each execution prefix."""

import jax
import numpy as np

from utils.datasets import normalize


def evaluate(agent, env, scale, episodes, seed):
    successes, returns = [], []
    rng = jax.random.PRNGKey(seed)
    for episode in range(episodes):
        observation, _ = env.reset(seed=(seed + episode) % 2**32)
        done, total_reward, success = False, 0.0, 0.0
        while not done:
            rng, key = jax.random.split(rng)
            actions, _ = agent.sample_actions(normalize(observation, scale), seed=key)
            actions = np.asarray(actions).reshape(-1, env.action_space.shape[-1])
            for action in actions:
                observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
                total_reward += float(reward)
                success = float(info['success'])
                done = terminated or truncated
                if done:
                    break
        successes.append(success)
        returns.append(total_reward)
    return {'success': float(np.mean(successes)), 'return': float(np.mean(returns))}
