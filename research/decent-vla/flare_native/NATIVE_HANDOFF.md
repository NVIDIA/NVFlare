# Native FLARE migration handoff

Reference: NVIDIA/NVFlare PR #5566, revision `41bd9659658b89f5184711584349959f8d54e59a`.

## Architecture

The server inherits FLARE 2.8.1 `FedAvg.run()` or `FedOpt.run()` unchanged. The
native weighted aggregation helper combines standard `FLModel` DIFF replies;
FedAdam uses FLARE FedOpt with `torch.optim.Adam` and an identity LR scheduler.
There is no local simulator, custom round scheduler, Communicator, Strategy
protocol, RoundPlan, ClientUpdate or private message codec in production.

Small adapters configure wire precision, sample weights, model persistence and
trace recording. `VLAPersistor` implements FLARE's `LearnablePersistor` API and
keeps the existing checkpoint names so the VLA evaluators can still load them.
The CPU FedAdam model contains only the shared parameter tree, not a second
full VLA policy. Its optimizer state is included in native checkpoints.

Clients use FLARE's Client API and retain their policy, optimizer, scheduler
and cycling data iterator throughout a job. `runtime/training.py` is only the
policy's local training loop, not a federation runtime.

## Training semantics retained

- Dataset-frame weighting when a backend exposes `num_train_frames`, otherwise
  actual samples consumed. The standard FLARE `NUM_STEPS_CURRENT_ROUND` weight
  deliberately carries that sample count rather than optimizer iterations.
- Shared/personal scopes, full tuning/LoRA and backend data partitions.
- Optimizer/scheduler continuity and forward, proximal penalty, backward,
  gradient clipping, optimizer step, zero-grad, scheduler-step order.
- Configured uplink/downlink dtype, fp32 master and fp32 accumulation. Wire
  deltas are upcast before multiplying by dataset counts to avoid fp16 overflow.
- FLARE `PTFedProxLoss` on lightweight shared-parameter views, including the
  original mean/sum option and arithmetic order without cloning a full VLA.
- Bias-corrected FedAdam with the configured LR, betas and epsilon.

## Run and compare

See [AUTHOR_COMPARISON.md](AUTHOR_COMPARISON.md) for the complete author guide,
storage-aware full-curve comparisons and GPU/thread/sequential execution settings.

Install this package and the policy dependencies on every site using the
existing setup instructions. NVFlare 2.8.1 is now a required dependency.

```bash
python -m pip install -e '.[dev]'
decent-vla train --config examples/mock_fl_nvflare.yaml
python -m pytest tests/unit_test -v
RUN_NATIVE_SIMULATOR_TESTS=1 python -m pytest tests/integration_test/fast -v
```

Keep an unchanged reference checkout in a separate directory. Both versions
must use the same Python/Torch/policy dependencies, data snapshot, pretrained
weights, GPU map, worker count and preprocessing. Run from this project folder:

```bash
python scripts/compare_training.py \
  --reference-dir /absolute/path/to/reference/research/decent-vla \
  --config examples/pi05_libero_fullft_fedadam_nvflare_smoke.yaml \
  --output-dir /absolute/path/to/fresh/comparison \
  --rounds 2 --atol 1e-6 --rtol 1e-5
```

The runner executes both versions through FLARE, requires every client, records
per-client loss each round and compares each shared tensor at checkpoint rounds.
By default it saves/checks every round; `--checkpoint-every` can reduce disk use. It checks identical partition manifests and writes `comparison.json`
plus both complete logs and checkpoints. The reference probe changes seeding
and records the existing round history; it does not change its algorithms.
It supplies deterministic Torch, NumPy and Python seeds in both versions and
sets deterministic CUDA algorithms/CUBLAS configuration. Unsupported
deterministic GPU operations fail rather than silently using a different mode.
For strict comparisons begin with `data.num_workers: 0`, then validate the
paper's worker settings separately. Use fresh output directories every time.

Repeat for FedAvg, FedProx (both normalization settings), FedAdam, full tuning
and LoRA on each policy. Inspect training curves as well as numerical tests;
CPU mock parity does not establish VLA/GPU convergence parity. Native streaming
aggregation sums in arrival order, so floating-point equality is tolerance
based rather than bitwise. Start with fp32 wire, then test the original dtype.

## Persistence and lifecycle

The existing auto-resume rule selects the latest periodic native checkpoint in
an absolute output directory. FedAdam requires its native optimizer state;
old reference FedAdam checkpoints are rejected rather than cold-restarted.
Warm client resume still requires `persist_client_state` and available client
checkpoints. The data iterator and RNG are not checkpointed, so interrupted
runs are not claimed to resume bitwise identically.

FLARE owns errors, timeouts and quorum. This version removes the custom
one-retry server loop and private error/reply protocol. A failed quorum
aborts; it does not silently average too few clients. Do not use fault or
partial-participation runs as the deterministic training-parity comparison.
The old `client_gpu_spread` and `lean_host_memory` switches are removed; use
FLARE's simulator GPU/thread placement and native streaming aggregation.

## Validation record

CPU checks on FLARE 2.8.1 / Python 3.12 / Torch 2.2.2 (Intel macOS;
Torch is below the package's supported minimum, so repeat in the authors'
supported GPU environment): two clients, two rounds, mock backend.

| Comparison | Max shared-weight error | Max client-loss error |
|---|---:|---:|
| FedAvg, fp16 wire | 0 | 0 |
| FedProx sum, fp32 wire | 2.98e-8 | 5.96e-8 |
| FedAdam, fp32 wire | 3.13e-7 | 2.98e-8 |
| FedAdam, fp16 wire | 2.98e-8 | 0 |

These comparisons passed with atol=1e-6 and rtol=1e-5. Numerical errors reflect
native reduction/optimizer arithmetic rather than a changed training budget.
The comparison directories contain the original full-precision results. GPU policy and
full-length training must be verified by the authors before replacement of the
original contribution. The original simulator and its tests stay in the parent
reference project during validation. They are not included in this independent
folder and are not imported by its production code or tests.


## Folder handoff and final contribution

During author validation, the repository contains both versions:

```text
research/decent-vla/                # unchanged reference at 41bd965
  flare_native/                    # complete native contribution
    pyproject.toml
    README.md
    LICENSE
    src/decent_vla/
    configs/
    examples/
    scripts/
    tests/
```

Install from `research/decent-vla/flare_native`, preferably in a separate virtual
environment from the original package: both versions use the `decent-vla`
distribution and CLI name. The parent is needed only for reference comparisons.
For example, from this folder use `--reference-dir ..` in the comparison command
above. Production training, exported jobs, evaluation and tests do not use it.

After performance alignment is confirmed, retain the **entire contents** of
`flare_native/` (including `.gitignore`), remove the other original files under
`research/decent-vla/`, and move the retained contents one level up. No source,
configuration or script path changes are needed. The resulting project root is
the directory containing `pyproject.toml`. The comparison utility can still use
a separately archived original checkout via `--reference-dir`.

This validation candidate is included in [PR #5566](https://github.com/NVIDIA/NVFlare/pull/5566)
on `K-XZY/NVFlare:research/decent-vla`. It adds `flare_native/` beside the
original implementation so the authors can compare both before unfolding the
native contribution. GPU and full-length policy training remain for the
authors to validate.

Standalone validation after relocation: the folder was copied outside the
repository, built as a wheel and installed without an editable source path.
With `PYTHONPATH` unset, 186 unit tests and both real simulator deployment tests
passed using the installed wheel. A fresh two-round deterministic FedAdam
comparison against the unchanged parent also passed (fp32 wire, max shared-weight
error 3.13e-7 and max client-loss error 2.98e-8). Three adapter tests were not
validated locally: two require LeRobot and one requires a newer Torch autocast
API. Original runtime tests belong to the reference version and are not copied
into the native project. See `tests/README.md` for the standalone test commands.


The subsequent author-comparison tooling update passed 200 unit tests and two
fresh real-simulator comparisons: two-client FedAvg fp32 with per-round checks,
and three-client FedAdam fp16 with final-only weight checks. See
`AUTHOR_COMPARISON.md` for commands, results and the same local environment
limitations. Sequential large-model simulation still needs complete disk-backed
client swap state; this documentation/tooling update does not implement it.
