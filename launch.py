"""Run the four-task suite, hyperparameter tuning, or random-edit ablation."""

import argparse
from collections import deque
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from config import ABLATION, ALGORITHMS, DEFAULTS, SWEEPS, TASKS, resolve
from envs.tasks import ensure_datasets

ROOT = Path(__file__).resolve().parent


def interrupt(signum, frame):
    """Let the scheduler close workers when the launcher receives SIGTERM."""
    raise KeyboardInterrupt


def make_jobs(args):
    if args.command == 'ablation':
        tasks, algorithms = [ABLATION['task']], ['mbce-random']
    else:
        tasks = args.tasks
        algorithms = args.algorithms or list(ALGORITHMS)
    if args.command == 'sweep' and len(algorithms) != 1:
        raise ValueError('Choose one sweep algorithm: mbce or mac-fql')
    seeds = SWEEPS['seeds'] if args.command == 'sweep' else (args.seeds or DEFAULTS['seeds'])
    if len(seeds) != len(set(seeds)) or any(not 0 <= s < 2**32 - 1 for s in seeds):
        raise ValueError('Seeds must be unique integers in [0, 2**32 - 2]')
    if len(tasks) != len(set(tasks)) or len(algorithms) != len(set(algorithms)):
        raise ValueError('Tasks and algorithms must be unique')
    selection = json.loads(args.selection.read_text()) if getattr(args, 'selection', None) else None
    jobs = []
    for algorithm in algorithms:
        if args.command == 'sweep' and algorithm not in SWEEPS['algorithms']:
            raise ValueError('Tuning supports mbce and mac-fql')
        grid = (
            SWEEPS['algorithms'][algorithm]
            if args.command == 'sweep'
            else (ABLATION if args.command == 'ablation' else None)
        )
        for task in tasks:
            for value in grid['values'] if grid else [None]:
                overrides = list(args.set)
                if selection and algorithm in SWEEPS['algorithms']:
                    parameter = SWEEPS['algorithms'][algorithm]['parameter']
                    selected = selection[algorithm][task][parameter]
                    if selected not in SWEEPS['algorithms'][algorithm]['values']:
                        raise ValueError('Selected value is outside the tuning grid')
                    overrides.insert(0, f'agent.{parameter}={selected}')
                if grid:
                    overrides.append(f'agent.{grid["parameter"]}={value}')
                config = resolve(algorithm, task, overrides)
                for seed in seeds:
                    name = f'{algorithm}-{task}-seed{seed}'
                    if grid:
                        name += f'-{grid["parameter"]}{value:g}'
                    jobs.append(dict(name=name, algorithm=algorithm, task=task, seed=seed, config=config))
    return jobs


def completed(directory, job):
    path = directory / job['name'] / 'done.json'
    if not path.is_file():
        return False
    done = json.loads(path.read_text())
    return (
        done['step'] == job['config']['training']['steps']
        and math.isfinite(done['success'])
        and 0 <= done['success'] <= 1
        and (path.parent / 'checkpoint.msgpack').is_file()
    )


def summarize(directory):
    """Rank complete tuning configurations by mean final-checkpoint success."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['mode'] != 'sweep':
        raise ValueError('Selection requires a tuning run')
    jobs = manifest['jobs']
    if any(not completed(directory, job) for job in jobs):
        raise ValueError('Every tuning run must complete before selection, so resume the sweep first')
    selected, rows = {}, []
    for algorithm in dict.fromkeys(job['algorithm'] for job in jobs):
        grid = SWEEPS['algorithms'][algorithm]
        selected[algorithm] = {}
        for task in dict.fromkeys(job['task'] for job in jobs if job['algorithm'] == algorithm):
            means = {}
            for value in grid['values']:
                group = [
                    job
                    for job in jobs
                    if job['algorithm'] == algorithm
                    and job['task'] == task
                    and job['config']['agent'][grid['parameter']] == value
                ]
                if sorted(job['seed'] for job in group) != SWEEPS['seeds']:
                    raise ValueError('Each candidate must contain tuning seeds 0, 1, and 2 exactly once')
                scores = []
                for job in group:
                    with (directory / job['name'] / 'eval.csv').open() as stream:
                        evaluations = [
                            row
                            for row in csv.DictReader(stream)
                            if int(row['step']) == DEFAULTS['training']['steps']
                        ]
                    if len(evaluations) != 1:
                        raise ValueError(f'Missing unique final evaluation: {job["name"]}')
                    score = float(evaluations[0]['success'])
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f'Invalid success rate: {job["name"]}')
                    scores.append(score)
                means[value] = statistics.mean(scores)
                rows.append(
                    dict(
                        algorithm=algorithm,
                        task=task,
                        parameter=grid['parameter'],
                        value=value,
                        mean_success=means[value],
                        seed_success=scores,
                    )
                )
            best = max(means.values())
            tied = [
                value for value, mean in means.items() if math.isclose(mean, best, abs_tol=1e-12, rel_tol=0)
            ]
            preferred = resolve(algorithm, task)['agent'][grid['parameter']]
            winner = preferred if preferred in tied else min(tied)
            selected[algorithm][task] = {grid['parameter']: winner}
            print(f'{algorithm:8} {task:12} {grid["parameter"]}={winner:g}  success={best:.1%}  tied={tied}')
    (directory / 'sweep_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    (directory / 'selected.json').write_text(json.dumps(selected, indent=2) + '\n')
    return selected


def execute(jobs, directory, dataset_dir, devices, wandb):
    """Keep one worker per explicitly selected GPU, MIG instance, or CPU slot."""
    ensure_datasets(list(dict.fromkeys(job['task'] for job in jobs)), dataset_dir)
    pending = deque(job for job in jobs if not completed(directory, job))
    active, failures = {}, []
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending:
                    continue
                job = pending.popleft()
                run_dir = directory / job['name']
                run_dir.mkdir(exist_ok=True)
                (run_dir / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
                environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')
                environment['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda'
                environment['CUDA_VISIBLE_DEVICES'] = '' if device == 'cpu' else device
                log = (run_dir / 'console.log').open('w')
                command = [
                    sys.executable,
                    str(ROOT / 'train.py'),
                    '--run-dir',
                    str(run_dir),
                    '--dataset-dir',
                    str(dataset_dir),
                    '--wandb',
                    wandb,
                ]
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                active[device] = process, log, job
                print(f'Started {job["name"]} on {device}', flush=True)
            for device, (process, log, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[device]
                success = code == 0 and completed(directory, job)
                if not success:
                    failures.append(job['name'])
                print(f'{"Finished" if success else "FAILED"} {job["name"]}', flush=True)
            if active:
                time.sleep(0.25)
    finally:
        for process, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
    if failures:
        raise RuntimeError(f'{len(failures)} runs failed; inspect console.log and rerun with --resume')


def main():
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in [
        ('run', 'Run selected algorithms and tasks'),
        ('sweep', 'Tune one algorithm on seeds 0–2'),
        ('ablation', 'Run the six-scale Puzzle-3x3 random-edit ablation'),
    ]:
        sub = commands.add_parser(name, help=help_text)
        if name != 'ablation':
            sub.add_argument(
                '--algorithms',
                nargs=1 if name == 'sweep' else '+',
                choices=list(SWEEPS['algorithms']) if name == 'sweep' else list(ALGORITHMS),
                required=name == 'sweep',
                help='Choose exactly one algorithm' if name == 'sweep' else 'Algorithms to run',
            )
            sub.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
        if name != 'sweep':
            sub.add_argument('--seeds', nargs='+', type=int)
            sub.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=JSON')
        else:
            sub.set_defaults(seeds=None, set=[])
        if name == 'run':
            sub.add_argument('--selection', type=Path, help='Use selected.json from a completed sweep')
        sub.add_argument(
            '--devices',
            nargs='+',
            default=['0'],
            help='GPU indices/UUIDs, MIG UUIDs, or cpu; one run per device',
        )
        sub.add_argument('--output', type=Path, help='Run directory; defaults to runs/<command>-<timestamp>')
        sub.add_argument('--dataset-dir', type=Path, default=Path('~/.cache/mbce/datasets'))
        sub.add_argument('--wandb', choices=['disabled', 'online', 'offline'], default='disabled')
        sub.add_argument(
            '--dry-run', action='store_true', help='Print resolved jobs without downloads or training'
        )
        sub.add_argument(
            '--resume',
            action='store_true',
            help='Skip completed runs; restart unfinished runs from their seed',
        )
    sub = commands.add_parser('select', help='Summarize a completed tuning run')
    sub.add_argument('directory', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'select':
            summarize(args.directory.expanduser().resolve())
            return
        jobs = make_jobs(args)
        if len(set(args.devices)) != len(args.devices) or ('cpu' in args.devices and len(args.devices) != 1):
            raise ValueError('Use distinct GPU identifiers, or a single cpu slot')
        if any(not (d == 'cpu' or d.isdigit() or d.startswith(('GPU-', 'MIG-'))) for d in args.devices):
            raise ValueError('Invalid device identifier')
        if args.dry_run:
            print(json.dumps(jobs, indent=2))
            print(f'{len(jobs)} jobs', file=sys.stderr)
            return
        if args.resume and not args.output:
            raise ValueError('--resume requires --output')
        directory = (
            (args.output or ROOT / 'runs' / f'{args.command}-{datetime.now():%Y%m%d-%H%M%S-%f}')
            .expanduser()
            .resolve()
        )
        manifest = dict(mode=args.command, jobs=jobs)
        if args.resume:
            if json.loads((directory / 'manifest.json').read_text()) != manifest:
                raise ValueError('Resume arguments must match the saved experiment configuration')
        else:
            directory.mkdir(parents=True, exist_ok=False)
            (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{len(jobs)} jobs; logs and checkpoints: {directory}', flush=True)
        execute(jobs, directory, args.dataset_dir.expanduser().resolve(), args.devices, args.wandb)
        if args.command == 'sweep':
            summarize(directory)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted. Use the same arguments with --resume to restart unfinished runs.\n')


if __name__ == '__main__':
    main()
