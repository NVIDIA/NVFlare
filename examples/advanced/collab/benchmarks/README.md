# Standard Simulator versus Collab Simulator

This benchmark compares two NVFlare simulation schemes on the same
full-parameter Hugging Face SFT workload:

| Scheme | Programming and exchange path |
|---|---|
| `standard` | `FedAvgRecipe`, the NVFlare Client API, `FLModel`, and the standard simulator path used by the `llm_hf` example |
| `collab` | `CollabRecipe`, published Python functions, native PyTorch state dictionaries, and Collab simulator direct calls |

Both schemes use the same model checkpoint, four site shards, client training
class, optimizer steps, precision, full-model states, aggregation frequency,
and GPU assignment. The default schedule is one epoch divided into five global
synchronizations. The intended variable is the simulator/API path, not the
learning workload.

The older native-versus-NumPy transition controls remain available in the
individual workload modules for diagnostic use, but they are not the primary
benchmark comparison.

## Recorded cluster results

The following runs were performed on July 29-30, 2026. They are recorded here
for reproducibility, including unsuccessful submissions and runs made before
the final one-GPU protocol was frozen.

### Workload and reusable assets

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, revision
  `fe8a4ea1ffedaf415f4da2f062534de366a451e6`.
- Dataset: `databricks/databricks-dolly-15k`.
- Data split: four clients, each with 50 training and 10 validation examples.
- Dolly source fingerprint: `2537fb912ac88184`.
- Selected-data fingerprint: `ad577d13f1c87fc6`.
- Prepared-data root:
  `/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/data/pt_llm_sft`.
- Environment:
  `/lustre/fsw/portfolios/coreai/users/ziyuex/miniconda3/envs/dfkd_async`;
  the environment was reused without package changes.
- GPU class: NVIDIA A100-SXM4-80GB with 81,920 MiB, driver `535.129.03`.

The initial preparation completed at commit
`a26c3554989469aa1467f4cf4a89e0a43fcb1cd3`. All per-file SHA-256 checks
passed. At the final benchmark commit
`f8ad5a7349d1384dc21325dfb6f46c020d72c299`, the preparation command took the
reuse path without downloading Dolly again and revalidated the manifest and
all shards. The validated manifest SHA-256 was
`d2d545d8e7ba062f417c5828a46278a1cb4e5be00f8f12d8844bfa9fdb6e642a`.

### Initial submission and smoke runs

The first standard-smoke submission requested one GPU, 32 CPUs, 128 GB, and
one hour. Slurm rejected it before creating a job because the cluster permits
at most 30 CPUs for a one-GPU request. Reducing the request to eight CPUs
allowed both initial smoke jobs to complete:

| Scheme | Job | Allocation | Elapsed | Runner process | Training/sample time | MaxRSS |
|---|---:|---|---:|---:|---:|---:|
| Standard | `31075151` | 1 GPU, 8 CPUs, 128 GB | 2m 12s | 95.03s | 90.89s execution | 8,800,724K |
| Collab | `31075589` | 1 GPU, 8 CPUs, 128 GB | 2m 10s | 91.15s | 33.93s sample total | 11,453,420K |

Both jobs completed with exit code `0:0` at commit
`a26c3554989469aa1467f4cf4a89e0a43fcb1cd3`. BF16 model loading and training
succeeded. The Collab smoke moved a 2,200,096,768-byte full-model payload.
These smoke runs predate the final resource scripts and did not collect
per-second GPU samples.

### Preliminary four-GPU paired run

Job `31076308` completed one standard-then-Collab pair:

| Measurement | Standard | Collab | Difference |
|---|---:|---:|---:|
| Runner process time | 300.32s | 247.14s | Collab was 53.19s (17.71%) faster |
| Scheme-specific time | 295.68s execution | 38.28s mean per sync | Not directly comparable |

The job completed with exit code `0:0` in 12m 08s. It used four
A100-SXM4-80GB GPUs, 16 CPUs, and 256 GB of host memory; batch MaxRSS was
134,296,984K. The Collab full-model payload was 2,200,096,768 bytes.
Artifacts are under:

```text
/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/results/paired_31076308
```

This is a preliminary functional result, not the final cluster-efficient
measurement. It ran commit
`a26c3554989469aa1467f4cf4a89e0a43fcb1cd3` with the superseded four-GPU
request. The job was submitted after newer hold and one-GPU messages had been
written, but before the experiment agent polled those messages. The old script
captured only a static GPU inventory, so peak GPU memory and mean/peak GPU
utilization cannot be reconstructed. No additional repetitions were performed
under this allocation.

### Final-commit one-GPU validation

The final protocol packs all four logical clients onto one non-exclusive GPU.
The single-client and four-client capacity gates produced:

| Gate | Job | Clients | Elapsed | Runner process | Peak GPU memory | Mean/peak GPU utilization | MaxRSS | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Standard single | `31143768` | 1 | 2m 25s | 100.50s | 11,523 MiB | 1.34% / 58% | 14,685,900K | Passed |
| Collab single | `31144006` | 1 | 2m 41s | 97.37s | 11,523 MiB | 0.82% / 42% | 11,139,604K | Passed |
| Standard capacity | `31144300` | 4 | 2m 25s | 101.32s | 46,075 MiB | 5.12% / 100% | 51,086,876K | Passed |
| Collab capacity | `31144623` | 4 | 2m 23s | 100.56s | 46,075 MiB | 4.85% / 100% | 44,132,116K | Passed |

All four jobs used commit
`f8ad5a7349d1384dc21325dfb6f46c020d72c299`, one A100-SXM4-80GB, eight CPUs,
64 GB of host memory, and exit code `0:0`. Their logs passed the error scan.
The standard single execution time was 95.70s; the Collab single mean total
sample time was 28.64s and its full-model payload was 2,200,096,768 bytes. The
standard capacity execution time was 97.05s. The Collab capacity mean total
sample time was 37.30s, again with a 2,200,096,768-byte payload.

The single-sync smoke timings are launch and capacity checks, not the primary
performance comparison. Both four-client schemes fit on one GPU with identical
46,075 MiB measured peaks, so the capacity gates authorized the paired run.

### Final-commit one-GPU paired run

Job `31144947` ran the full five-sync standard-then-Collab pair:

| Measurement | Standard | Collab | Difference |
|---|---:|---:|---:|
| Runner process time | 270.45s | 228.60s | Collab was 41.86s (15.48%) faster |
| Scheme-specific time | 266.52s execution | 33.73s mean per sync | Not directly comparable |

The job completed with exit code `0:0` in 9m 04s on commit
`f8ad5a7349d1384dc21325dfb6f46c020d72c299`. It requested one non-exclusive
A100-SXM4-80GB, 16 CPUs, and 128 GB of host memory. Batch MaxRSS was
123,076,640K. Across 509 one-second samples, peak GPU memory was 46,075 MiB,
mean utilization was 6.87%, and peak utilization was 100%. The Collab median
per-sync total was 33.10s and its full-model payload was 2,200,096,768 bytes.
The result summary, both scheme metrics, and GPU samples are present; stderr
was empty, and the run had no OOM, timeout, package, or error-scan failure.

Compared with the superseded four-GPU pair, the raw allocation footprint fell
from 48.53 to 9.07 GPU-minutes, an 81.3% reduction. This cross-run allocation
comparison is descriptive rather than controlled because the runs used
different commits and resource scripts.

Both completed paired runs favored Collab in runner-process time: 17.71% on
the preliminary four-GPU run and 15.48% on the final-commit one-GPU run. Only
one pair was run under each allocation, always in standard-then-Collab order.
A publishable performance claim still requires repeated pairs and alternating
scheme order as described in the measurement protocol.

### Result artifacts

All successful job artifacts use this base directory:

```text
/lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/results
```

| Job | Result directory |
|---:|---|
| `31075151` | `smoke_standard_31075151` |
| `31075589` | `smoke_collab_31075589` |
| `31076308` | `paired_31076308` |
| `31143768` | `smoke_single_standard_31143768` |
| `31144006` | `smoke_single_collab_31144006` |
| `31144300` | `smoke_capacity_standard_31144300` |
| `31144623` | `smoke_capacity_collab_31144623` |
| `31144947` | `paired_31144947` |

Each result directory preserves scheme metrics and a combined summary. Jobs
from the final commit also preserve per-second GPU samples under
`environment/gpu_samples.csv`. Slurm stdout and stderr are under the sibling
`logs` directory in the project root. The rejected 32-CPU submission has no
job ID or result directory because Slurm refused it before job creation.

## Qwen3-8B four-GPU scale-up

The next benchmark scales only the model and GPU mapping while preserving the
full-parameter BF16 SFT workload, AdamW optimizer, batch size of one, sequence
length of 64, four Dolly sites, and five synchronizations. It uses the public
text-only checkpoint:

```text
Qwen/Qwen3-8B
revision b968826d9c46dd6066d109eabc6255188de91218
8,190,735,360 BF16 parameters
```

Qwen3 is a commonly used model family with public, Apache-2.0 weights and no
gated-access credential requirement, making the benchmark reproducible for
repository users. Qwen3-8B is the largest plausible dense Qwen3 size for this
unchanged single-A100 training method. Its parameters, gradients, and two BF16
Adam moment tensors have a lower bound of about 61.0 GiB before activations and
framework overhead. The next dense family size, Qwen3-14B, exceeds 80 GiB on
those four tensors alone. The one-client gates determine the actual fit rather
than assuming the estimate is sufficient.

The three immutable configs are:

- `configs/pt_llm_sft_slurm_qwen3_8b_single.json`: one site, one sync, GPU 0;
- `configs/pt_llm_sft_slurm_qwen3_8b_capacity.json`: four sites, one sync,
  GPUs `0,1,2,3`;
- `configs/pt_llm_sft_slurm_qwen3_8b.json`: four sites, five syncs, GPUs
  `0,1,2,3`.

### Prepare only in the user-owned cache

The model must not use a system, default-home, shared-global, or node-local
cache. `prepare_model.py` and every Qwen3 Slurm launcher pin all cache paths
beneath:

```text
/lustre/fsw/portfolios/coreai/users/ziyuex/huggingface_cache
```

Preparation passes that directory explicitly to the Hugging Face downloader,
pins the model revision, verifies all weight shards, checks that the resolved
snapshot remains beneath the configured cache root, and writes a model
manifest under `nvflare_manifests/`. Measured jobs run offline and repeat the
same validation with `--local-files-only` before timing.

After exporting `NVFLARE_SOURCE_ROOT` and `EXPECTED_COMMIT` for a clean,
immutable checkout, prepare the model once and reuse the existing Dolly data:

```bash
bash collab/benchmarks/slurm/prepare_model.sh
CONFIG=collab/benchmarks/configs/pt_llm_sft_slurm_qwen3_8b.json \
  bash collab/benchmarks/slurm/prepare_data.sh
```

### Run capacity gates before the pair

Submit and monitor each job sequentially:

```bash
SCHEME=standard sbatch --export=ALL,SCHEME=standard collab/benchmarks/slurm/qwen3_8b_single.sbatch
SCHEME=collab sbatch --export=ALL,SCHEME=collab collab/benchmarks/slurm/qwen3_8b_single.sbatch

SCHEME=standard sbatch --export=ALL,SCHEME=standard collab/benchmarks/slurm/qwen3_8b_capacity.sbatch
SCHEME=collab sbatch --export=ALL,SCHEME=collab collab/benchmarks/slurm/qwen3_8b_capacity.sbatch

SCHEME_ORDER="standard collab" \
  sbatch --export=ALL,SCHEME_ORDER="standard collab" collab/benchmarks/slurm/qwen3_8b_paired.sbatch
```

The single gate requests one GPU, eight CPUs, and 160 GB for 90 minutes. The
capacity gate requests four GPUs, 32 CPUs, and 512 GB for two hours. Only if
both schemes pass those gates should the paired job request four GPUs, 32 CPUs,
and 1 TB for four hours. All jobs are non-exclusive. Each site receives one
A100; no GPU is shared by two sites. Stop on an OOM instead of changing the
model or training method inside a matched pair.

## Configure cluster assets

`configs/pt_llm_sft.json` is the lightweight local smoke configuration.
`configs/pt_llm_sft_slurm.json` records the current cluster configuration and
reuses this immutable cached checkpoint:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
revision fe8a4ea1ffedaf415f4da2f062534de366a451e6
```

Before a cluster run, verify the paths in the Slurm config and set:

- `model_name_or_path` to the selected shared checkpoint snapshot path;
- `data_root` to the prepared four-site dataset root;
- `trainer_output_root` to writable scratch storage;
- `force_cpu` to `false`;
- `gpu_config` to the allocation used by both schemes. The checked-in Slurm
  config uses `"0"`, placing all four clients in one simulator GPU group so
  the benchmark requests only one GPU.
- `precision` explicitly to `bfloat16` or `float32` for both schemes. The
  checked-in Slurm config uses BF16 and must fail during the smoke check if the
  allocated GPU does not support it; do not let the schemes choose precision
  independently.

Prefer a compatible model snapshot and prepared dataset already available on
the cluster. If either asset is unavailable, download and prepare it before the
timed jobs. Never allow a model or dataset download to occur during a measured
run. Record the model revision or snapshot path and a dataset manifest with
the results.

The lightweight model in the local configuration remains a smoke-test default.
TinyLlama 1.1B is the recorded baseline; Qwen3-8B is the four-GPU scale-up.

## Install dependencies

Starting from `examples/advanced`:

```bash
python -m pip install -r collab/benchmarks/requirements.txt
```

NVFlare itself is supplied by the repository installation described by the
advanced-examples README.

## 1. Prepare data and model

The local smoke config creates deterministic synthetic instruction shards. The
Slurm config uses `databricks/databricks-dolly-15k`, the same text-only SFT
dataset featured by the NVFlare `llm_hf` example. Preparation deterministically
selects 240 Dolly rows and creates four client shards with 50 training and 10
validation examples each. It records dataset fingerprints and a SHA-256 digest
for every shard. MNIST preparation remains available only for the supplemental
split-learning microbenchmark.

```bash
python collab/benchmarks/prepare_data.py
```

On Slurm, compatible prepared Dolly shards are reused only when their manifest
matches the configured dataset, seed, and counts and every file digest still
matches. Otherwise preparation refuses to overwrite a nonempty directory.
Resolve the model to the shared cache during this phase so both timed jobs run
from the same immutable files.

For the current cluster config, prepare or reuse the deterministic Dolly
shards; the selected model is already cached. Dolly is downloaded only when it
is missing:

```bash
python collab/benchmarks/prepare_data.py \
    --workload pt_llm_sft \
    --config collab/benchmarks/configs/pt_llm_sft_slurm.json
```

The equivalent remote preparation entrypoint is
`collab/benchmarks/slurm/prepare_data.sh`. It requires `NVFLARE_SOURCE_ROOT`
and `EXPECTED_COMMIT` and refuses a dirty or mismatched checkout.

## 2. Run the matched comparison

```bash
python -m collab.benchmarks.run_benchmarks
```

Use the immutable cluster assets and persistent result root on Slurm:

```bash
python -m collab.benchmarks.run_benchmarks \
    --config collab/benchmarks/configs/pt_llm_sft_slurm.json \
    --output-root /lustre/fsw/portfolios/coreai/users/ziyuex/projects/collab_project/results/paired_run
```

Ready-to-review Slurm entrypoints are under `collab/benchmarks/slurm/`:

- `smoke.sbatch` with `SMOKE_KIND=single`: one client, one synchronization,
  and one selected scheme;
- `smoke.sbatch` with `SMOKE_KIND=capacity`: all four clients and one
  synchronization on one GPU, verifying that the minimum allocation fits;
- `paired.sbatch`: all four clients and five synchronizations on one GPU, with
  both schemes in a configurable order.

Both jobs verify the source commit, require a clean checkout, force offline
Hugging Face operation, and capture GPU, package, host, Git, and
`/usr/bin/time` metadata alongside persistent results.

The capacity smoke must pass for both schemes before the paired job is
submitted. Request more than one GPU only after a reproducible OOM demonstrates
that the one-GPU allocation cannot satisfy the workload.

The jobs are non-exclusive so Slurm can pack other workloads onto the same
eight-GPU node. The single smoke requests 8 CPUs and 64 GB of host memory; the
paired job requests 16 CPUs and 128 GB. GPU samples and `/usr/bin/time` RSS are
used to adjust later requests instead of reserving an entire node.

Run one scheme while validating a launch script:

```bash
python -m collab.benchmarks.run_benchmarks --scheme standard
python -m collab.benchmarks.run_benchmarks --scheme collab
```

The runner creates a fresh process and workspace per scheme, records its
end-to-end process time, preserves scheme-specific metrics, and writes a
combined summary under `/tmp/nvflare/collab/benchmarks/results` by default.
Use a shared scratch path through `--output-root` on Slurm.

## Measurement protocol

- Run a smoke test first, then at least three paired measured repetitions.
- Alternate scheme order across pairs to reduce cache and thermal bias.
- Use the same Slurm allocation and placement policy for a pair.
- Warm the model and dataset caches before timing and enable offline Hugging
  Face mode for measured jobs.
- Record job IDs, commit SHA, config, model snapshot, dataset manifest,
  environment, GPU type, peak host/GPU memory, and wall-clock time.
- Treat model initialization and simulator startup as part of the primary
  end-to-end result. Use the Collab per-sync samples only as supporting detail.

The standard scheme uses PyTorch as its server exchange format. This avoids
changing model precision merely to manufacture a larger difference: both
schemes carry the same tensor values, while the standard path still exercises
normal NVFlare task/model transport and the Collab path exercises its direct
function-call programming model.

## Supplemental split-learning microbenchmark

`simple_split_learning/benchmark.py` is retained as a targeted activation and
gradient transition microbenchmark. It is separate from the primary
standard-versus-Collab LLM comparison and is not launched by
`run_benchmarks.py`.
