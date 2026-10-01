# MBCE

Model-Based Chunk Editing for offline reinforcement learning. MBCE learns bounded corrections to behavior-cloned action chunks using TD-learning, then selects among the edited chunks using a model-based critic.

| Algorithm | Policy extraction |
| --- | --- |
| `mac` | Model-based best-of-N selection |
| `mbce` | Learned edits before selection and imagined rollouts |
| `mbce-random` | Tanh-squashed standard-normal edits before selection |
| `mac-fql` | Flow distillation and model-critic policy gradients |

The task suite contains `puzzle-3x3`, `puzzle-4x4`, `cube-double`, and `antsoccer`.

## Setup

On an NVIDIA Linux VM with a driver installed:

```bash
./setup_vm.sh
```

## Run

```bash
# Inspect the full plan without downloading data or starting workers
./launch.sh run --dry-run

# Four algorithms, four tasks, six seeds
./launch.sh run --devices 0 1 --output runs/suite

# A single algorithm and task
./launch.sh run --algorithms mbce --tasks puzzle-3x3 --devices 0

# The six-scale random-edit ablation, six seeds per scale
./launch.sh ablation --devices 0 1 --output runs/random-ablation
```

Datasets get downloaded to `~/.cache/mbce/datasets`.

## Tune

```bash
# MBCE: three edit scales, four tasks, three tuning seeds = 36 runs
./launch.sh sweep --algorithms mbce --devices 0 1 --output runs/tuning-mbce

# MAC-FQL: six BC coefficients, four tasks, three tuning seeds = 72 runs
./launch.sh sweep --algorithms mac-fql --devices 0 1 --output runs/tuning-mac-fql

# Evaluate each algorithm with its selected hyperparameters
./launch.sh run --algorithms mbce --selection runs/tuning-mbce/selected.json --devices 0 1
./launch.sh run --algorithms mac-fql --selection runs/tuning-mac-fql/selected.json --devices 0 1
```

MBCE sweeps edit scales `[0.03, 0.065, 0.1]`. MAC-FQL sweeps BC coefficients `[0.03, 0.1, 0.3, 1, 3, 10]`. Both use tuning seeds `0–2` and one million updates.

Chosen values go into `selected.json`. Selection runs automatically after a successful sweep and can be repeated with `./launch.sh select runs/tuning-mbce`.

## Configuration

- [defaults.json](configs/defaults.json): shared training and network settings, plus evaluation seeds `10000–10005`.
- [algorithms.json](configs/algorithms.json): method-specific settings.
- [tasks.json](configs/tasks.json): environment identifiers and per-task overrides.
- [sweeps.json](configs/sweeps.json): tuning grids and tuning seeds.
- [ablation.json](configs/ablation.json): random-edit ablation task and scales; seeds use the shared evaluation defaults.

Ordinary runs accept explicit overrides, applied after task defaults:

```bash
./launch.sh run --algorithms mbce --tasks cube-double --seeds 10000 \
  --set agent.edit_scale=0.065 --dry-run
```

## Implementation

[agents/](agents/) contains the four policies.

## Attribution

The implementation for MAC, MAC-FQL, networks, environment utilities, and dataset utilities are adapted from the MAC implementation accompanying **Scalable Offline Model-Based RL with Action Chunks**, by Kwanyoung Park, Seohong Park, Youngwoon Lee, and Sergey Levine ([arXiv:2512.08108](https://arxiv.org/abs/2512.08108)). The code is distributed under the [MIT license](LICENSE), retaining MAC's copyright notice. The editor follows QAM-E; applicable [QAM](licenses/qam-LICENSE) and [FQL](licenses/fql-LICENSE) notices are included.
