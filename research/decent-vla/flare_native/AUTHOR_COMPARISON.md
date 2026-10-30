# Comparing the original and native FLARE implementations

Use the original contribution as the reference and `flare_native/` as the
candidate. First establish numerical agreement on a short deterministic run,
then compare the complete training curves with the paper's settings. Both jobs
run through FLARE; the comparison does not introduce another simulator.

## 1. Install once and start with the CPU check

The expected layout during validation is:

```text
research/decent-vla/                 # original reference, revision 41bd965
  src/, examples/, configs/, ...
  flare_native/                     # independent native project
    src/, examples/, configs/, scripts/, ...
```

Keep the original source unchanged. Use a separate environment for each policy
family, following the native project's [setup instructions](README.md#setup).
Install the **native** package in that environment. The comparison runner selects
the reference or native source with a separate subprocess and `PYTHONPATH`; do
not alternately install both packages, since they have the same distribution
and command name. NVFlare must be **2.8.1** for these instructions.

From `flare_native/`, with that environment active:

```bash
python -m pip install -e '.[dev]'
python scripts/compare_training.py \
  --reference-dir .. \
  --config examples/mock_fl_nvflare.yaml \
  --output-dir /absolute/path/to/comparisons/mock-fedavg-01 \
  --strategy fedavg --wire-dtype fp32 --rounds 2
```

Choose a **new output directory** for each training comparison. The runner
starts the reference job, waits for it to finish, then starts the native job;
the two implementations do not compete for GPUs. Their clients can still run
concurrently *within* each job. After unfolding the native project, use a
separate original checkout as `--reference-dir` instead of `..`.

## 2. Compare a real policy

Run both implementations on the same machine, GPU types, installed dependency
versions, data snapshot and pretrained weights. Download/cache the data and
weights first. Keep policy architecture, trainable scope/LoRA initialization,
batch size, local optimizer, learning-rate schedule, preprocessing, local steps
and partition seed identical. Reconstructing each site's non-overlapping share
from the same cached dataset is supported; separate physical dataset copies are
unnecessary.

For example, with two GPUs that each fit one full π0.5 client:

```bash
python scripts/compare_training.py \
  --reference-dir .. \
  --config examples/pi05_libero_fullft_fedadam_nvflare_smoke.yaml \
  --output-dir /absolute/path/to/comparisons/pi05-fedadam-fp32-01 \
  --rounds 2 --strategy fedadam --wire-dtype fp32 \
  --num-clients 2 --threads 2 --gpu '0,1' \
  --num-workers 0 --checkpoint-every 1 \
  --atol 1e-6 --rtol 1e-5
```

The mock config also supports `--strategy fedprox` or `--strategy fedadam` for
quick algorithm checks. For policy validation, use each algorithm's own example
YAML so its server LR/betas/epsilon or proximal coefficient is preserved. FedProx
needs both `fl.strategy.extra.normalize: true` and `false` comparisons, using
otherwise identical YAML copies. `--strategy` changes the name only; it does not
replace those hyperparameters. Likewise, changing `--num-clients` changes the
partitioning experiment; compare both implementations at the same chosen count.
An `explicit_repos` setup also needs one configured repo list per client.

The runner automatically applies these controls to **both** jobs:

| Control | Applied value |
| --- | --- |
| Execution | FLARE simulator, with fresh separate outputs/workspaces |
| Participation | Every configured client must contribute every round |
| Randomness | Same `seed`; Python, NumPy, Torch CPU and CUDA seeds |
| GPU determinism | Strict deterministic algorithms, cuDNN benchmarking off, TF32 off |
| Process environment | Fixed `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one BLAS/OpenMP thread |
| Data workers | `0` by default; change explicitly with `--num-workers` |
| Resume | Disabled, to compare complete uninterrupted runs |
| W&B | Disabled; JSON and CSV curves are written locally |
| Checkpoints | Every round by default; configurable with `--checkpoint-every` |

Other training settings stay as configured unless overridden on the command
line. Inspect `reference.yaml` and `native.yaml` in the output directory to see
the exact inputs. The saved `comparison_meta.json` identifies source folders,
Python/Torch/CUDA/FLARE versions, expected rounds, clients and checkpoint cadence.
It also records the latest successful check's tolerances and weight-check rounds.
Also retain `python -m pip freeze` and `nvidia-smi` output with GPU results.

### How the original is made deterministic

No manual patch to the original is required. Only reference subprocesses load
`scripts/reference_probe/sitecustomize.py` with `DVLA_REFERENCE_PROBE=1`. It
wraps the original `build_backend_factory` so that each server/client backend
starts from the same seeded RNGs and GPU settings as native `seed_training()`.
It records the original round history as `training_curve.json` and resolves
the original strategy classes from the reference source. It leaves the
original training loop, aggregation formulas and optimizer updates unchanged.
The runner checks that this probe loaded before starting expensive training.

These controls establish a controlled **new** reference run; they do not claim
to reproduce historical runs that used different randomness or math settings.
If a CUDA operator has no deterministic implementation, let the run fail and
choose the same supported implementation in both policy setups. Do not hide
the error with `warn_only=True`. Libraries with their own RNG generators or
custom CUDA kernels may need additional backend-specific controls. Start with
zero data workers, then check the intended worker/prefetch settings separately.
See [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.7/notes/randomness.html)
for the limits across releases/hardware and worker seeding.

## 3. Read the results and extend to full curves

A successful command exits zero and writes:

| Output | Purpose |
| --- | --- |
| `comparison.json` | Maximum per-client loss error each round and maximum shared-weight error at checked rounds |
| `training_curves.csv` | Reference/native per-client and unweighted mean losses, ready for plotting |
| `reference/` and `native/` | Partition manifests, full training curves, checkpoints and simulator workspaces |
| `reference.log` and `native.log` | Complete stdout/stderr for each implementation |

The comparator requires matching partition manifests, the full expected round
sequence, every expected client, finite losses, and matching shared parameter
names. It checks each client's loss, not just an average that could conceal
offsetting differences. It then compares every shared tensor at the configured
checkpoint rounds. This checks the actual global update, in addition to the
training curve. For FedProx, the reported local loss includes the proximal term.
The mean-loss curve is an unweighted mean over clients, as in the original;
model aggregation retains the configured dataset-frame/sample weighting.

JSON round indices are zero-based; CSV rounds and checkpoint filenames are
one-based. `max_weight_error: null` means that round's weights were not checked.
It does **not** mean that the weight error was zero. The final weights are
always checked. Do not average away a failing client's discrepancy.

Use `atol=1e-6, rtol=1e-5` as a starting diagnostic threshold, not a guarantee
for every GPU policy and full-length run. Native streaming reduction can use a
different arrival order, and native Adam can round differently. If a check
fails, first repeat the **same** implementation with the same inputs to measure
its own repeatability, then inspect the first divergent round. Compare fp32
wire first, then the original fp16/per-direction wire setting; changing wire
precision changes the numerical experiment in both jobs.

For a full run, use the full example config and omit `--rounds`. Preserve the
client count/GPU layout and local steps of that experiment. Checkpoints can be
very large: a full π0.5 FedAdam checkpoint contains weights and two moments.
Saving one every round can dominate disk use. Instead use:

```bash
python scripts/compare_training.py \
  --reference-dir .. \
  --config examples/pi05_libero_fullft_fedadam_nvflare.yaml \
  --output-dir /absolute/path/to/comparisons/pi05-fedadam-full-01 \
  --checkpoint-every 10 --num-workers 0
```

This compares losses at every round and weights every ten rounds plus the
final round. `--checkpoint-every 0` compares all losses and **only final
weights**. Final model exports and any enabled client checkpoints still need
disk space; this option does not remove those. Weight comparison memory-maps
the saved tensors rather than eagerly loading unused Adam moments. Allow ample
host RAM for training, server aggregation and final export too.

Recheck existing outputs with different tolerances without retraining:

```bash
python scripts/compare_training.py \
  --output-dir /absolute/path/to/comparisons/pi05-fedadam-full-01 \
  --compare-only --atol 1e-6 --rtol 1e-5
```

This uses the original run's saved round/client/checkpoint expectations. Review
both client and mean curves, then repeat with the original worker settings,
wire precision and several seeds. Finish with the same checkpoint evaluation
protocol for both versions. Record which algorithm, policy, full-tuning/LoRA,
precision, client count and worker settings actually passed. CPU mock agreement
alone does not establish GPU VLA convergence alignment.

## 4. Client count, threads and GPU placement

The FL client count and the number of GPUs are separate settings. Choose the
logical count for the experiment, then choose a scheduling layout that fits
the available hardware. The behavior below was checked against **FLARE 2.8.1**.

| Setting | Meaning in this example |
| --- | --- |
| `data.num_clients` or `data.partitioning.num_clients` | Logical clients/shards; sites are `c0` through `cN-1` |
| `fl.nvflare.threads` | Simulator worker concurrency; default here is `num_clients` |
| `-t` / `--threads` | The same setting in `nvflare simulator` |
| `fl.nvflare.gpu` / CLI `-gpu` | GPU **groups**, allocated round-robin to logical clients |
| `data.num_workers` | DataLoader workers per client; unrelated to simulator threads |
| `fl.min_clients` | Required updates for a round; unrelated to concurrent worker count |

The YAML key and `FedJob.simulator_run` argument are **`threads`**, not
`num_threads`. Despite its name, it controls concurrent simulated client worker
processes, not Torch CPU thread count. This example defaults it to `N` to keep
clients resident; the bare simulator CLI can default to one, so specify it.

**Multiple GPU groups override the thread setting.** In 2.8.1, `gpu: '0,1'`
creates two groups, each with one worker, even with `threads: 8`. Eight clients
then rotate through those two groups. Repeated GPU IDs create additional
groups/slots on the same physical GPU: `gpu: '0,1,0,1,0,1,0,1'` creates eight
slots, four on each card. Setting `threads: 1` with this eight-entry map does
**not** serialize the eight clients; there are still eight group workers.

For training-curve alignment with the current warm-client implementation:

| Clients | Available GPUs | Settings | Effect |
| ---: | ---: | --- | --- |
| 2 | 2 | `threads: 2`, `gpu: '0,1'` | One resident client per GPU |
| 8 | 4 | `threads: 8`, `gpu: '0,0,1,1,2,2,3,3'` | Two resident clients per GPU, if memory fits |
| 8 | 2 | `threads: 8`, `gpu: '0,1,0,1,0,1,0,1'` | Four resident clients per GPU, if memory fits |
| 8 | 1 | `threads: 8`, `gpu: '0'` | Eight resident clients sharing one GPU, if memory fits |
| N | CPU | `device: cpu`, `threads: N`, no GPU map | N resident CPU clients, if host RAM fits |

For example, eight logical clients on two sufficiently large cards:

```yaml
device: auto
data:
  num_clients: 8
  num_workers: 0
fl:
  min_clients: 8
  nvflare:
    threads: 8
    gpu: '0,1,0,1,0,1,0,1'
```

Merge these fields into the policy's config; this is not a complete training
config. Repeated IDs permit sharing, but do not reserve, partition or enlarge
VRAM. Every resident client retains its model, optimizer and data pipeline, so
the combined memory must fit. Server/final-export memory and preprocessing
buffers also matter. In particular, repeating IDs is not a remedy for an OOM.

`gpu: '[0,1]'` is **one group** exposing two cards to a worker. It does not
automatically shard a VLA model or enable DDP/model parallelism. A backend that
selects `cuda:0` still uses the first visible card. The worker sets its own
`CUDA_VISIBLE_DEVICES`; use GPU IDs available inside the job/container and
normally `device: auto` or the worker-local `cuda:0`, not a physical device
index as the Torch device string.

## 5. Sequential execution and the disk-checkpoint requirement

FLARE itself supports `T < N` worker slots and sequential execution. With one
GPU group, `gpu: '0', threads: 1` rotates clients on one GPU. With eight clients
and `gpu: '0,1'`, the stock simulator rotates clients across two groups, with
one executing client per GPU. The server can still collect all eight updates
before aggregation; do not reduce `min_clients` to the worker count to save
memory, because that changes the FL experiment.

**Those layouts are not currently equivalent to the resident VLA run.** The
simulator swaps client processes out, so process-local optimizer, RNG and
data-iterator state disappear. The native launcher rejects short multi-group
maps to prevent this; the strict comparison runner also rejects all layouts
that require swapping. Exporting a job and using the bare simulator CLI can
bypass that validation, but does not restore the missing state.

For large models that cannot coexist in GPU memory, disk-backed swap state is
the appropriate next step. FLARE provides `SWAP_OUT` and `SWAP_IN` lifecycle
events where an application can save and restore state. A curve-preserving
implementation must cover:

- All client parameters, including personal/unshared weights and relevant buffers.
- Optimizer moments, parameter-group settings, scheduler/global step and AMP scaler where used.
- Python, NumPy, Torch CPU/CUDA RNG state and any backend-owned RNG generators.
- Sampler order, epoch and batch/iterator position, including the cycling loader;
  worker/prefetch state needs a checkpointable or reproducibly replayable design.
- Completed-round identity and atomic checkpoint publication before acknowledging
  completion, so a repeated task cannot spend the training budget twice.

The current `fl.persist_client_state: true` plus
`fl.save_every_n_rounds: 1` saves weights, optimizer and scheduler state. It does
**not** save RNG or data-iterator position. It therefore helps warm resume but
is insufficient for equivalent sequential simulation. Also, strict comparison
runs deliberately set `DECENT_VLA_NO_RESUME=1`; changing that variable is not a
substitute for implementing and validating complete swap state. Server FedAdam
checkpoints are separate and cannot recover a client's missing local state.

Implement swap persistence through FLARE's lifecycle and the backend's local
state interfaces, while keeping server rounds/aggregation in native workflows.
Then validate a resident versus swapped run of the **same implementation** for
multiple rounds, including stochastic sampling and client Adam continuity,
before comparing original versus native. Expect model/optimizer reload I/O and
substantial local disk use. This feature is a prerequisite for claiming that
an arbitrary number of large VLA clients can share a small GPU pool without
changing their training process; it is not implemented by the current flags.

### More than 100 logical clients

The stock simulator defaults to a maximum of 100 clients. The current
`FedJob.simulator_run()` wrapper does not expose that maximum. For a larger
experiment, export a job with matching `data.num_clients` and explicit
`fl.nvflare.env: export`, then use the CLI's `-m` setting, for example:

```bash
# The exported config must already describe 128 clients; this CPU example
# needs enough host memory for 128 resident clients.
CLIENTS=$(python -c "print(','.join(f'c{i}' for i in range(128)))")
nvflare simulator /absolute/path/to/exported/job \
  -w /absolute/path/to/fresh/workspace -c "$CLIENTS" -m 128 -t 128
```

Use the same GPU-group rules if running on GPUs. Raising `-m` changes the
allowed logical count; it does not provide GPU memory or make swapping
state-preserving. The comparison wrapper's automatic launch retains the
100-client limit; a larger comparison needs a launch extension or separately
managed exported jobs plus explicit result validation.

The [FLARE simulator guide](https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html)
explains worker slots, GPU groups and swap lifecycle events. Its online main
version may describe a newer release; recheck the installed simulator when
changing the pinned FLARE version.


## Local verification of this guide

The CPU mock quick start passed with two clients/two rounds, fp32 wire and
per-round weight checks (maximum shared-weight error `2.98e-8`, zero loss
error). A three-client/three-round FedAdam run also passed with fp16 wire and
`--checkpoint-every 0` (final shared-weight error `3.58e-7`, zero loss error).
The saved-output recheck path passed. The package's 200 unit tests passed,
including incomplete-curve and missing-checkpoint rejection tests; scoped
Black/isort/flake8 checks passed.

These checks used FLARE 2.8.1, Python 3.12 and an Intel macOS Torch 2.2.2
interpreter, below the package's supported Torch >=2.4 minimum. Two optional
LeRobot statistics cases and one newer-Torch autocast case were not validated
locally. GPU policy training, full-length curve alignment and complete
disk-backed swap persistence remain to be validated/implemented as described
above. The GPU scheduling rules were checked against the installed 2.8.1
simulator source, not against a local CUDA run.
