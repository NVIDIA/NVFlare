# TCGA-BRCA BreastG-FCL

This implementation combines disease-aware TCIA spatial/temporal graph
construction with the GFedCL encoder, predictor, latent replay generator,
and server discriminator roles. Graph attention runs forward without a
separate training objective; `E/F/G/D` are trained. The TCGA prediction
labels and clinical-stage task definitions are retained. Train/test
assignment now separates patients globally before constructing tasks and
client partitions. This implementation uses the NVFlare 2.9.0 Collab API
for federated execution.

## Data roles

| Source | Workflow role |
| --- | --- |
| TCGA-BRCA expression and clinical loaders | Prediction samples and labels, with patient-separated train/test assignments followed by task/client partitioning |
| TCIA lesion morphology radiomics | Spatial client/task summary `s_i^k` |
| TCIA DCE kinetic radiomics | Temporal client/task summary `r_i^k` |

TCIA features are aggregated from the training patients already assigned to
each client/task and are used only to generate the graph. They are not
concatenated to the TCGA input and do not replace the existing GFedCL training
labels or determine train/test assignment.

The public TCIA analysis result provides 91 lesion-radiomics rows, of which 84
match its PAM50 and clinical workbooks. The downloader separates 25 morphology
fields and 11 kinetic fields and records source checksums:

```bash
python TCGA-BRCA/scripts/download_tcia_official_radiogenomics.py
```

## Patient split and audit

The loader first groups expression records by `case_submitter_id` across
the dataset. It assigns approximately 80% of patients to training and 20%
to evaluation, stratifying by whether each patient has Normal records only,
Tumor records only, or both. Every record from one patient stays on the same
side, including multiple files for the same sample and paired Normal/Tumor
samples. Clinical-stage tasks and client assignments are constructed
separately within each side. The `random` task strategy follows the same
global patient split.

Partition validation rejects overlapping train/test patient or sample IDs
and duplicate record-index assignments. It also requires nonempty train
and evaluation data for every client/task; insufficient data raises an
error without duplicating samples.

`OUTPUT_DIR/tcga_brca_partitioning_seed<seed>.pkl` records
`split_unit='patient'`, `split_protocol_version=2`, and patient/sample/file
IDs for each dataset index. Its `identity_split` audit lists the patients
and samples assigned to training and evaluation. Historical file-level
partitions use a different protocol.

## Training-fitted normalization

Expression cache schema v3 stores `log1p` values before standardization.
The fingerprint covers the expression column, requested gene count, ordered
file/patient/sample/label identities, and the SHA-256 of each source file's
actual contents. Each reuse recomputes those hashes, detecting content edits
even when file names, sizes, and modification times are unchanged.

The cache saves those identities, source hashes, and configuration alongside
the arrays. Reuse validates metadata, array shapes and dtypes, and finite
values. Arrays use non-object dtypes and load with `allow_pickle=False`.
Older v1/v2 caches are rebuilt automatically. Corrupt v3 data or mismatched
metadata produces an explicit error; remove the indicated cache file and
rerun to rebuild it. Building a cache rejects inconsistent gene order across
source files and publishes the completed cache with an atomic rename, so
an interrupted write cannot leave a partial cache at the final path.

After the global patient split and task assignment, expression mean and
scale are fitted only on the union of the final eligible training record
indices. Evaluation records and excluded unknown-stage tumors do not
participate in fitting. Training and evaluation data use the same fitted
transform.

Spatial and temporal TCIA tables each fit their own transform using only
the unique rows matched by the final training records. Multiple expression
files that match one TCIA row count once when fitting that table. The same
statistics transform all rows in that table. If a table has no matched
training rows, preparation raises an error without fitting on the full
table as a fallback.

The partition pickle stores `normalization` with `schema_version=1` and
three entries: `expression`, `tcia_spatial`, and `tcia_temporal`. Each records
`mean`, `scale`, and `n_samples_seen`; expression also records `fit_indices`,
and each TCIA entry records `fit_ids` for the unique rows used in fitting.

These statistics are fitted once across all training tasks during local
simulator preparation. This preprocessing uses future training tasks and
does not claim a strictly online fit per task. It also does not implement
distributed or privacy-preserving estimation of the statistics.

## Learning and replay

| Component | Input and output | Update |
| --- | --- | --- |
| Encoder `E` | RNA-seq vector + graph row → latent | Trained locally on current real samples; no label input |
| Generator `G` | Gaussian noise + label + graph row → synthetic latent | Trained locally on current and enabled replay tasks |
| Predictor `F` | Real or synthetic latent → class prediction | Trained locally together with `E/G` |
| Discriminator `D` | Real or synthetic latent → graph row | Trained on the server by minimizing graph-row MSE |
| Graph attention | TCIA spatial/DCE summaries → relational graph | Forward computation only; no attention optimizer |

Both `E` and `G` output `nh` dimensions (800 by default). The graph row and
`D` output have one dimension per client (4 by default). `G` uses a label
embedding and graph projection, concatenates them with noise, and maps the
result through an MLP to the latent space. Its default noise dimension of
100 (`--noise-dim`) is a public implementation choice; the papers and local
GFedCL reference do not provide an original author value.

For task `k`, clients prepare current real latents and synthetic latents for
tasks `j <= k` when replay is enabled. Before returning them through Collab,
the client's `generate_encodings` adds independent Laplace noise with scale
`b = sensitivity / epsilon` to every current real `E` encoding and every
current or historical `G` encoding (`epsilon = 0` retains the existing
no-noise mode, `b = 0`). The server updates `D` directly on the
received perturbed latents and their matching graph rows, without adding
latent noise a second time, then sends `D` back to clients.
Local training minimizes prediction NLL minus `lambda_gan` times graph-row
MSE, using current real latents from `E` and the sum of synthetic losses
from `G`. The local copy of `D` stays in evaluation mode with frozen
parameters; autograd through its input remains enabled, so its adversarial
loss reaches both `E` and `G`.

`--replay true` covers every prior task as well as the current task.
`--replay false` removes historical synthetic losses but still trains `G`
on the current task. Clients retain per-task label counts and batch sizes;
historical labels are sampled from those counts and combined with fresh
Gaussian noise and the corresponding task's graph row. Replay does not
read historical raw expression samples. Current-task labels can condition
`G`, but never enter `E`, including during evaluation.

Each client training operation computes current and replay updates on the
same client instance. The coordinator averages all clients' `E/F/G` weights
with equal weight and redistributes them, retaining each client's returned
Adam state, scheduler state, and task metadata for subsequent rounds. The
server takes one `D` optimizer step per communication round. The joint Adam optimizer
has separate parameter groups controlled by `--lr-e`, `--lr-f`, and
`--lr-g`; the server uses `--lr-d`. All four default to `1e-4`.

The new encoder has no label embedding, and `GNet` now generates latents
instead of graph embeddings. Old `E/G` checkpoints therefore cannot be
loaded directly into these architectures; start a new run or provide an
explicit migration. Attention checkpoints described below have a separate
format and purpose.

## Relational graph

For task `k`, the graph generator standardizes each summary feature across
clients (as in the previous scorer), then computes

```text
alpha_ij^k = mean_h softmax_j a_s,h(s_i^k, s_j^k)
R_i^k      = r_i^{max(1,k-m+1)} || ... || r_i^k
q_i^k      = W_Q pad_left(R_i^k)
key_j^k    = W_K pad_left(R_j^k)
beta_ij^k  = softmax_j ((q_i^k)^T key_j^k / sqrt(64))
G_ij^k     = alpha_ij^k beta_ij^k /
             (sum_l alpha_il^k beta_il^k + epsilon)
```

The temporal window and multiplicative fusion follow the paper directly.
Spatial attention uses the local GFedCL reference network structure: a
client-summary encoder with hidden/output dimensions `128 -> 64`, followed
by four additive GAT heads, each with a 32-dimensional projection. A head
scores a client pair with `LeakyReLU(a^T [W h_i || W h_j])` and applies
row-wise softmax; the four attention matrices are averaged. The encoder uses
LayerNorm in place of the reference's BatchNorm to support small client
counts. Configurable dropout defaults to `0.2`.

Temporal attention uses trainable query/key projections and the paper's scaled
dot-product score. It concatenates the latest `m` DCE summaries, including the
current task, and left-pads early windows with zeros to a fixed width of
`m * temporal_feature_dimension`. The same projection layers produce
64-dimensional queries and keys for every task. The attention temperature
divides the spatial and temporal scores before their respective softmaxes;
the equations above show the default temperature of `1.0`.

Each task writes its spatial attention, temporal attention, temporal-window
input, and fused graph to `OUTPUT_DIR/relational_graphs/*.npy` for auditing.
It also saves `task_<k>_attention.pt` in that directory, containing the
attention `state_dict` (including DCE history in extra state) and
`network_config` (dimensions, window, temperature, epsilon, and related
settings).

To restore a checkpoint on CPU, construct `BreastGraphGenerator` with
`SimpleNamespace(**checkpoint["network_config"])` and load
`checkpoint["state_dict"]`. The stored DCE history allows graph generation
to continue with the next task.

Both networks expose a differentiable `forward` path. By default, no separate
attention training objective or optimizer is applied: the experiment seed
fixes initialization, and `learn()` is a compatibility inference entry point
whose `epochs` argument is unused. It temporarily switches to evaluation
mode to disable dropout and restores the previous mode afterward.
Consequently, a saved checkpoint records the reference network state, not
evidence of attention training. The authors' exact BreastG-FCL network
parameterization, trained weights, supervision, and optimization remain
unpublished; this implementation does not claim to recover them.

## NVFlare execution and data packaging

Install the [project environment](../README.md#setup), which pins
`nvflare==2.9.0`. `python TCGA-BRCA/job.py` creates a `CollabRecipe`, exports
its job, and runs it with the local `SimEnv` simulator in a fresh Python
process. The simulator process loads the prepared bundles, avoiding inherited
PyTorch autograd threads and CUDA state from data preparation. `main.py`
remains a compatibility entry point that calls `job.main`. The integration
follows the client/server/job organization of the [synchronous Collab examples in
NVFlare PR #5036](https://github.com/NVIDIA/NVFlare/pull/5036).

| Module | Responsibility |
| --- | --- |
| `job.py` | Build the `CollabRecipe`, package site data, export the job, and launch `SimEnv` |
| `prepare_data.py` | Prepare patient-separated train/test splits and the server coordination bundle |
| `federated/client.py` | Load site data with `@collab.init`; expose `encode`, `train`, and `test` with `@collab.publish` |
| `federated/server.py` | Enter the task/round workflow through `@collab.main` and save final state |
| `federated/transport.py` | Adapt coordinator operations to Collab client calls and returned results |
| `federated/runtime.py` | Execute the shared client encoding, training, and evaluation operations |
| `federated/state.py` | Reconstruct workflow state and capture checkpoints |

The existing `breastgfcl` coordinator owns task/round order, graph
construction, discriminator updates, and aggregation. Published methods
exchange their arguments and results through Collab, preserving the shared
`federated.runtime.execute_client` operations.

This launcher targets local simulation. Preparation calls
`setup_tcga_brca_loaders` once for a real-data run. It packages the resulting
patient-separated splits by dataset index, without further repartitioning:

| Exported artifact | Contents |
| --- | --- |
| `nvflare_job/breastg_fcl/app_site-<n>/config/data/site.pt` | Only that site's assigned train/test tensors and configuration |
| `nvflare_job/breastg_fcl/app_server/config/data/server.pt` | Client summaries, initial model state, task label counts/batch sizes, configuration, and RNG state; no raw expression inputs |
| Each application's `custom/` directory | Shared Python code, without the source `data/` or `dump/` directories |

The server reconstructs its coordination state from the server bundle and
sends operations to sites; it does not reload or repartition TCGA data.
The coordination loop preserves the model, loss, replay, optimizer,
equal-weight E/F/G aggregation, and once-per-round D update described above.
Latent perturbation occurs on clients before the upload; relational-graph
Laplace noise remains on the server.

Each operation receives a seed derived from the experiment seed, task,
round, client ID, and operation. The runtime saves and restores Python,
NumPy, Torch CPU, and the client's CUDA RNG state, so the transport does not
advance server randomness. These seeds provide deterministic simulation;
the implementation does not establish formal differential privacy or a
production privacy guarantee.

## Run and verify

Run all commands from `NVFlare/research/breastg-fcl/`. Follow the project
README's [data preparation](../README.md#data-preparation) first. The expression
downloader validates existing files by size and MD5 and downloads missing
or damaged files again, so it can be rerun without resetting its local
download ledger. Then run:

```bash
python TCGA-BRCA/job.py --seed 42 --output-dir TCGA-BRCA/dump/nvflare_seed42
```

The real-data defaults are 4 clients, 3 tasks, 10 rounds per task, 20 local
epochs per round, latent width 800, noise width 100, and replay enabled.
Choose a fresh output directory: an existing `nvflare_job/` is never replaced
by the exporter.

The [selected development configuration](../README.md#selected-development-configuration)
uses `lr_e = lr_g = 3e-5`, `lr_f = 3e-4`, `lr_d = 1e-3`,
`lambda_gan = 0.1`, and 3 local epochs per round. It retains 4 clients,
3 tasks, 10 rounds per task, latent width 800, noise width 100, batch size 32,
and replay. Its other overrides are `p = 0.1` and `shuffle = false`, with
`no_bn = true` and `sensitivity = epsilon = 1.0`. These experiment settings
do not change the runtime defaults.

The [recorded NVFlare experiment](../README.md#recorded-nvflare-experiment)
used NVFlare 2.9.0 Collab with the patient/sample separation and
training-fitted normalization described above. It froze the original
seed-42 partition of 894 training records and 225 outer-evaluation records.
A patient-level split with seed 2026 divided the 894 records into 674 inner
training records and 220 validation records. During search, normalization
was fitted and graph summaries were computed using only the 674 training
records; final refitting used all 894 training records for these steps.

The search evaluated 24 configurations with training seed 42, confirmed the
top three with seeds 43 and 44 while reusing their seed-42 runs, and then
trained the locked configuration afresh on all 894 records with seeds
42, 43, and 44. These 33 successful runs kept their respective prepared
data partitions fixed across training seeds. The 225 outer-evaluation
records and their metrics were not used for selection during this search.
This outer split had been inspected in earlier experiments, so the final
results are not a new, previously unseen blind test.

Across the three final refits, pooled accuracy was **97.33%**, balanced
accuracy **88.88%**, Normal recall **78.26%**, Tumor recall **99.50%**, and
macro F1 **92.12%**. The simulator's unweighted mean accuracy over the
12 client/task cells was **96.90%**; it is a different aggregation from
pooled accuracy. See the recorded experiment for per-seed results and
limitations.

The standard launcher uses `--seed` for both data preparation and training.
Changing it on each CLI run also changes the patient/task/client partition,
so a simple seed sweep does not reproduce this fixed-partition study.
The separate search and evaluation tooling used for the recorded study is
not included in this directory.

To exercise the NVFlare workflow without downloading data:

```bash
python TCGA-BRCA/job.py \
  --device cpu \
  --smoke \
  --output-dir /tmp/unique_directory
```

This uses a deterministic **synthetic fixture**, not the TCGA cohort. Its
default topology is 4 clients and 3 tasks, with 2 rounds per task and 1 local
epoch. Input, latent, and noise widths are 8, 16, and 5; batch size is 4.
The fixture also reduces attention dimensions for a short integration run.
Its metrics are not comparable with the paper's reported results.

A completed smoke test writes the model and graph checkpoints, metric files,
and simulator logs described below. Inspect these artifacts to verify that
the synthetic task sequence finished in your environment. The current run
with client-side latent noise completed all six rounds with replay, finite
checkpoint and graph values, and both metric CSV files. Before latent
perturbation moved to clients, the recorded Collab smoke run matched the
previous Controller implementation's model and training state, graph
artifacts, and metric CSV files exactly; see the historical
[migration validation](../README.md#expected-results). That comparison
covers the transport migration before the client-side noise change and
does not measure TCGA accuracy.

| Option | Behavior |
| --- | --- |
| `--export-only` | Write the NVFlare job without starting training |
| `--nvflare-timeout 300` | Client-operation timeout in seconds; increase for longer training calls |
| `--max-in-flight N` | Controls concurrent client operations; defaults to `min(num_clients, 8)` |
| `--nvflare-workspace PATH` | Simulator workspace; defaults to `OUTPUT_DIR/nvflare_workspace` |
| `--nvflare-threads N` | Defaults to the client count and must equal it, keeping every Collab client process resident |
| `--nvflare-gpu IDS` | One GPU ID, e.g. `0`, or one shared GPU group, e.g. `'[0,1]'`; CPU runs need no GPU setting |

Use `--max-in-flight` to adjust training concurrency. All clients share the
selected GPU or GPU group; separate groups such as `--nvflare-gpu 0,1` are
rejected because NVFlare 2.9 would rotate client workers.

Graph controls remain available through the same default entry point:

```bash
python TCGA-BRCA/job.py \
  --temporal-window 2 \
  --attention-temperature 1.0 \
  --gat-dropout 0.2 \
  --graph-epsilon 1e-8
python -m unittest discover -s TCGA-BRCA/tests -v
```

After training, the output directory contains `final_state.pt`,
`round_accuracy.csv`, `all_tasks_accuracy.csv`, `relational_graphs/`,
`nvflare_job/breastg_fcl/`, and the default
`nvflare_workspace/breastg_fcl/`. The final-state artifact
records E/F/G/D parameters, attention state, client and discriminator
optimizer/scheduler state, replay metadata, and metrics. Simulator logs are
in the job directory under the selected workspace root, with console output
in `simulator.log`.
Export-only runs produce the job without training artifacts.
