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
- `gpu_config` to the allocation used by both schemes, for example
  `"[0],[1],[2],[3]"` for one GPU per client.
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
The 1.1B TinyLlama snapshot is the current primary Slurm candidate because it
is already complete, text-only, and compatible with `AutoModelForCausalLM`.

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

- `smoke.sbatch`: one client, one synchronization, and one selected scheme;
- `paired.sbatch`: four clients, five synchronizations, and both schemes in a
  configurable order.

Both jobs verify the source commit, require a clean checkout, force offline
Hugging Face operation, and capture GPU, package, host, Git, and
`/usr/bin/time` metadata alongside persistent results.

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
- Use the same Slurm allocation and exclusive-node policy for a pair.
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
