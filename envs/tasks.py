"""Single-task OGBench environments and their recorded datasets."""

import os
from pathlib import Path
import shutil
import urllib.request

from config import TASKS

DATASET_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'


def ensure_datasets(tasks, directory):
    """Download each training/validation pair atomically before workers start."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        name = TASKS[task]['dataset']
        for suffix in ('', '-val'):
            path = directory / f'{name}{suffix}.npz'
            if path.is_file() and path.stat().st_size:
                continue
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            url = f'{DATASET_URL}/{path.name}'
            print(f'Downloading {path.name}', flush=True)
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
                if not temporary.stat().st_size:
                    raise RuntimeError(f'Empty download: {url}')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def make_env_and_datasets(task, directory):
    import ogbench
    import numpy as np

    spec = TASKS[task]
    path = str(Path(directory).expanduser() / f'{spec["dataset"]}.npz')
    env, train, validation = ogbench.make_env_and_datasets(
        spec['environment'], dataset_path=path, compact_dataset=True
    )
    # Use validation-set observation bounds consistently for training and execution.
    scale = {key: getattr(np, key)(validation['observations'], axis=0) for key in ('min', 'max')}
    return env, train, validation, scale
