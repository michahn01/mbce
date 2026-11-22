# MBCE

### [[Workshop Page]](https://uncertainwm2026.github.io/) [[Paper]](paper/uncertain_wm_workshop_version.pdf)

## What is MBCE ("Model-Based Chunk Editing" for offline RL)?

MBCE learns bounded corrections to behavior-cloned action chunks using TD-learning, then selects
among the edited chunks using a model-based critic.

## Intuition

In offline MBRL, in general you don't want to veer too far from the offline dataset's support,
because you can't make corrections with live interaction data. So, suppose you train a
*behavior-cloned* actor, and just perform best-of-N planning on it (e.g., using a model-based
critic). This is an idea proposed by Park et al. in [Scalable Offline Model-Based RL with Action
Chunks](https://arxiv.org/abs/2512.08108).

However, if an action required to solve a task lies *outside* that dataset support, the BC actor may
never produce it. In practice, if that action lies outside the dataset support, but not *too* far
out where the model-based planner becomes severely unreliable, it may be beneficial to lend a little
slack to the BC actor and allow it to reach that action slightly beyond the dataset support.

MBCE introduces an edit policy (as in [EXPO](https://arxiv.org/abs/2507.07986) by Dong et al., and
[QAM-E](https://arxiv.org/abs/2601.14234) by Li et al.) right after the BC actor, right before
model-based best-of-N selection. An edit policy inserts a small, bounded edit to a base action, and
it's well-suited for this use case, because it hard-bounds the magnitude of the edit it produces by
construction. So, it's very easy to tune/control the magnitude of edits we introduce to the
behavior-cloned proposals.

## Contained in this repo

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

The implementation for MAC and MAC-FQL baselines, networks, environment utilities, and dataset utilities are adapted from the MAC implementation accompanying **Scalable Offline Model-Based RL with Action Chunks**, by Kwanyoung Park, Seohong Park, Youngwoon Lee, and Sergey Levine ([arXiv:2512.08108](https://arxiv.org/abs/2512.08108)). The code is distributed under the [MIT license](LICENSE), retaining MAC's copyright notice. The edit policy implementation is adapted from QAM-E.
