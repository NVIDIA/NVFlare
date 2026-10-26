# BreastG-FCL: Graph-Conditioned Federated Continual Learning for Breast Cancer Radiogenomics

This directory contains the research implementation for our IEEE HealthCom 2026 paper:<br>
**BreastG-FCL: Graph-Conditioned Federated Continual Learning for Breast Cancer Radiogenomics**<br>
Qingyang Yu, Jingyi Wang, Xinyue Zhang, Miao Pan, Ziyue Xu, Hao Wang<br>
*In Proceedings of IEEE HealthCom, 2026.*

## Abstract

BreastG-FCL addresses learning across institutions whose data evolve over
successive tasks. It builds client relations from breast morphology and
DCE-MRI summaries, then uses the resulting graphs to condition representation
learning and generative replay. This implementation applies the method to
Normal versus Tumor classification on TCGA-BRCA RNA-seq using NVIDIA FLARE.
TCIA features supply graph context. The NVFlare Collab API connects a server
workflow with published client methods, and `CollabRecipe` runs the experiment
through the local `SimEnv` simulator.

## Papers and Links

- BreastG-FCL: [paper](https://ryougish1k1.github.io/assets/pdf/qiangyang2026healthcom.pdf).
- NVFlare code: [BreastG-FCL implementation](TCGA-BRCA/).

## Objective

Demonstrate how the NVFlare Collab API can run graph-conditioned federated
continual learning while preserving the reference training operations.
Readers can prepare the public data, run a
four-client experiment, inspect graph and model checkpoints, and exercise
the complete NVFlare workflow with a synthetic CPU smoke test.

## Method Summary

This is a horizontal federated learning simulation: clients have different
records with the same expression features and two-class target. Clinical
stage defines the task sequence, while client heterogeneity arises from
the assigned records and their TCIA summaries. TCIA features are used only
for the graph, not concatenated to the prediction input.

The `--num-classes` option defaults to 2 and accepts larger output widths,
which are preserved during real-data preparation. Values below 2 cannot
represent the fixed Normal=0/Tumor=1 labels and are rejected. Extra outputs
do not add labels to the TCGA dataset.
`--nc` is an alias of `--num-classes`, and `--ni` is an alias of `--nh`.
Each pair sets the same value; if both forms are supplied, the last value wins.


| Component | Operation | Training |
| --- | --- | --- |
| Encoder `E` | Expression vector + client graph row → latent | Client |
| Predictor `F` | Real or generated latent → class prediction | Client |
| Generator `G` | Gaussian noise + label + graph row → replay latent | Client |
| Discriminator `D` | Latent → reconstructed graph row | Server |
| Spatial/temporal attention | Morphology/DCE summaries → client relations | Forward computation only; no attention optimizer |

For each task, spatial multi-head additive attention and temporal
query/key attention over the latest two tasks are multiplied and normalized
by row. Each communication round uploads perturbed real and synthetic
latents, updates `D` once on the server, trains `E/F/G` on clients, and
averages their weights with equal client weights. Client optimizer and
scheduler states persist between rounds.

`G` generates latent features, not expression records or images. With replay
enabled, synthetic training covers all tasks up to and including the current
task, using saved label counts and graph rows without reading historical raw
inputs. `E` never receives labels. During client training, `D` has frozen
parameters but passes gradients to `E/G`. The local objective combines
prediction NLL with a negative, weighted graph-reconstruction MSE; the server
minimizes that MSE.

Before a Collab upload, each client's `generate_encodings` adds independent
Laplace noise with scale `b = sensitivity / epsilon` to every latent it
returns: current real encodings from `E` and current or historical synthetic
encodings from `G`. The server trains `D` directly on these received latents
without perturbing them again. Relational-graph noise remains on the server.
The existing `epsilon = 0` setting disables both perturbations (`b = 0`).
Operation-derived seeds support deterministic simulation; this mechanism
does not establish a formal differential-privacy or production privacy
guarantee.

See the [implementation notes](TCGA-BRCA/README.md) for attention equations,
network dimensions, replay details, and transport contracts.

## Repository Layout

Project files live under `research/breastg-fcl/`:

```text
TCGA-BRCA/
├── job.py                  # CollabRecipe, job export and SimEnv launcher
├── main.py                 # Compatibility entry point delegating to job.main
├── prepare_data.py         # Site splits and server coordination bundle
├── breastgfcl.py           # Task/round coordinator
├── configs/                # Defaults and paper-reproduction protocol
├── federated/
│   ├── client.py           # @collab.init and published encode/train/test methods
│   ├── server.py           # @collab.main server workflow
│   ├── transport.py        # Coordinator-to-Collab operation adapter
│   ├── runtime.py          # Shared client training operations
│   └── state.py            # Workflow and checkpoint state helpers
├── model/                  # E, F, G, D and attention networks
├── utils/                  # Data loading, partitioning and evaluation
├── scripts/                # Data downloads and source audit
├── tests/                  # Training and transport regression tests
└── metadata/               # GDC manifest and clinical metadata snapshot
```

## Setup

Use a Linux environment with Python 3.10 and **NVFlare 2.9.0**. The commands
below create an isolated Conda environment. CPU execution is supported for
the synthetic smoke test and real-data workflow. CUDA training needs
a compatible NVIDIA driver and PyTorch build.

The 2026-10-10 and 2026-10-13 real-data validations used the `breastgfcl` Miniconda environment with
Python 3.10.22 and PyTorch 2.5.1 (CUDA 12.4). Each real-data run used one
NVIDIA GeForce RTX 3090 (24 GB), shared by the server and four clients.
The earlier parameter search used an older environment; both are listed
under [Requirements](#requirements). Real-data execution also needs the
downloaded expression files, clinical metadata, and TCIA feature tables
below.

```bash
git clone https://github.com/NVIDIA/NVFlare.git
cd NVFlare/research/breastg-fcl
conda env create -f environment.yml
conda activate breastgfcl
```

Alternatively, in an existing Python 3.10 environment:

```bash
# Install NVFlare first (stable shown; use nvflare-nightly, or "pip install -e ." from a source checkout)
python -m pip install "nvflare==2.9.0"
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA environment when using a
GPU. Run all subsequent commands from `NVFlare/research/breastg-fcl/`.

## Data Preparation

### TCGA-BRCA expression and clinical data

Source: the NCI GDC [TCGA-BRCA project](https://portal.gdc.cancer.gov/projects/TCGA-BRCA).
The downloader selects only open-access STAR-Counts expression files;
[GDC open data access](https://gdc.cancer.gov/access-data/data-access-processes-and-tools)
does not require authentication. Data use remains subject to
[GDC policies](https://gdc.cancer.gov/about-gdc/gdc-policies), including the
source acknowledgement requirements.

This project includes a GDC clinical metadata snapshot. The downloader
refreshes the expression manifest and downloads STAR-Counts files; it does
not refresh clinical metadata. Expression files are downloaded separately
and are not included in this repository.

The clinical TSV is derived from the bundled GDC JSON snapshot. Rebuild or
verify it without querying GDC:

```bash
python TCGA-BRCA/scripts/build_clinical_metadata.py
python TCGA-BRCA/scripts/build_clinical_metadata.py --check
```

The exporter selects a diagnosis classified exactly as `primary` with a
breast tissue of origin before reading its stage. It does not use diagnosis
array order, the first nonempty stage, or the highest stage. A missing or
unknown stage on that primary diagnosis stays unknown. If no explicit primary
breast diagnosis exists, the only fallback is a unique breast diagnosis at
`days_to_diagnosis == 0` with missing or `not reported` classification.
Distinct competing candidates raise an error. Prior, synchronous, subsequent,
recurrent, metastatic, and other-organ diagnoses are not fallback candidates.
The output retains one row per case, sorted by patient ID, with diagnosis
fields taken together from the selected record.

Download the expression files with:

```bash
python TCGA-BRCA/scripts/download_tcga_brca.py
```

Rerun the same command to resume. The downloader checks local files against
the manifest's file sizes and MD5 checksums, skips verified files, and
downloads missing or damaged files again. The local download ledger,
`TCGA-BRCA/metadata/downloaded_files.json`, is generated automatically and
excluded from version control; no manual reset is needed.
`--metadata-only` refreshes the manifest without downloading expression
files. Preserve the manifest, clinical snapshot, and raw data
used by an experiment: a later GDC query can return a different cohort.

Expected inputs:

```text
TCGA-BRCA/metadata/gdc_files_manifest.tsv
TCGA-BRCA/metadata/gdc_clinical_cases.tsv
TCGA-BRCA/data/raw/<file_id>/*.rna_seq.augmented_star_gene_counts.tsv
```

### TCIA spatial and temporal features

Source: the TCIA [TCGA-Breast-Radiogenomics analysis result](https://www.cancerimagingarchive.net/analysis-result/tcga-breast-radiogenomics/),
data DOI [10.7937/K9/TCIA.2014.8SIPIY6G](https://doi.org/10.7937/K9/TCIA.2014.8SIPIY6G).
The five source artifacts used by this downloader are listed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Follow the dataset's
citation instructions and [TCIA usage policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).
This path uses the public analysis files and does not require downloading
the original DICOM collection.

Download those artifacts with checksum verification:

```bash
python TCGA-BRCA/scripts/download_tcia_official_radiogenomics.py
```

The downloader produces `official_spatial_patient_features.csv`,
`official_temporal_patient_features.csv`, and `official_source_manifest.json`
in `TCGA-BRCA/data/tcia_official_radiogenomics/`. The public tables contain
84 matched patients, with 25 morphology fields and 11 kinetic fields.
Client/task graph summaries aggregate matching training patients only.

### Continual tasks and client partition

The default `clinical_stage` strategy defines three successive tasks. Each
has the same two labels: Normal (`0`) and Tumor (`1`).

| Task | Tumor population | Controls |
| --- | --- | --- |
| 0: early | AJCC Stage 0/I | Assigned Normal records |
| 1: intermediate | AJCC Stage II | Assigned Normal records |
| 2: advanced | AJCC Stage III/IV or metastatic | Assigned Normal records |

Before assigning tasks or clients, the loader groups records globally by
`case_submitter_id` and splits patients approximately 80/20 into train/test.
The split is stratified by each patient's label combination: Normal only,
Tumor only, or both. All expression files from a patient stay on the same
side, including multiple files for one sample and paired Normal/Tumor
samples. The fraction of expression records need not be exactly 80/20.
Set `--train-split` to change the training patient fraction; evaluation uses
the remaining patients. Fractions are approximate after stratification and
rounding. The unused `--test-split` option has been removed.

Within each side, records are assigned to clinical-stage tasks and then
four clients; Normal records are distributed across tasks. The `random`
task strategy uses the same global patient split before task assignment.
Unknown-stage tumors are excluded by default under `clinical_stage`.
These are stage-defined tasks, not longitudinal visits of the same patient.
The loader checks that train/test patient and sample IDs are disjoint and
that record indices are not duplicated across assignments. Every client/task
must have training and evaluation records; insufficient data raises an error
instead of copying records to fill empty splits.

The loader saves assignments in
`OUTPUT_DIR/tcga_brca_partitioning_seed<seed>.pkl`, with `split_unit='patient'`,
`split_protocol_version=2`, patient/sample/file IDs for each record index,
and an `identity_split` audit containing train/test patient and sample lists.
Expression cache schema v3 stores `log1p` values without standardization.
Its fingerprint includes the expression column, gene count, ordered
file/patient/sample/label identities, and SHA-256 hashes of the source file
contents. Source hashes are recomputed before every reuse, so content changes
are detected even if names, sizes, and modification times stay the same.
Older v1/v2 caches are rebuilt automatically. A corrupt v3 cache or mismatched
metadata raises an error; delete the indicated cache file to rebuild it.

The expression input defaults to 4,096 `tpm_unstranded` gene values with
`log1p` and standardization. After patient and task assignment, expression
mean and scale are fitted on the union of the final eligible training
records. Evaluation records and excluded unknown-stage tumors do not
contribute to those statistics. The same transform is applied to training
and evaluation data.

TCIA spatial and temporal normalization are fitted separately on the unique
TCIA rows matched by those training records. Multiple expression files
matching the same TCIA row do not give it extra weight. Each fitted transform
is applied to all rows in its table; no matching training rows raises an
error instead of falling back to the full table.

The partition pickle also records `normalization` with `schema_version=1`
and `expression`, `tcia_spatial`, and `tcia_temporal` entries. These store
`mean`, `scale`, `n_samples_seen`, and the fitting record indices
(`fit_indices`) or TCIA row IDs (`fit_ids`). Statistics are fitted once
across all training tasks during local simulator preparation. This is not
strictly online preprocessing per task or a distributed computation of
private statistics.

## Run Instructions

### Default NVFlare simulation

After preparing the data:

```bash
python TCGA-BRCA/job.py \
  --seed 42 \
  --output-dir TCGA-BRCA/dump/nvflare_default_seed42
```

The launcher creates a `CollabRecipe`, exports the NVFlare job, and runs it
with `SimEnv` in a fresh Python process. This workflow targets local
simulation, with a separate application bundle for each site. Choose a
fresh output directory for each run: an existing `nvflare_job/` will not
be overwritten.

### Selected development configuration

The following applies the configuration selected by the patient-separated
internal-validation search described in the
[recorded NVFlare experiment](#recorded-nvflare-experiment). It is an explicit
override of the project defaults:

```bash
python TCGA-BRCA/job.py \
  --seed 42 --num-clients 4 --num-task 3 \
  --num-rounds 10 --num-local-epochs 3 \
  --nh 800 --noise-dim 100 --batch-size 32 \
  --replay true --lambda-gan 0.1 \
  --lr-e 3e-5 --lr-f 3e-4 --lr-g 3e-5 --lr-d 1e-3 \
  --weight-decay 0.0 \
  --p 0.1 --no-bn true --shuffle false \
  --sensitivity 1.0 --epsilon 1.0 \
  --output-dir TCGA-BRCA/dump/nvflare_selected_seed42
```

| Setting | Repository default | Selected configuration |
| --- | ---: | ---: |
| Clients / tasks | 4 / 3 | 4 / 3 |
| Rounds per task | 10 | 10 |
| Local epochs per round | 20 | 3 |
| Latent / noise dimension | 800 / 100 | 800 / 100 |
| Graph row / discriminator output dimension | 4 (client count) | 4 |
| Batch size | 32 | 32 |
| Learning rates `E / F / G / D` | `1e-4 / 1e-4 / 1e-4 / 1e-4` | `3e-5 / 3e-4 / 3e-5 / 1e-3` |
| Adam weight decay, `E/F/G/D` | 0 | 0 |
| `lambda_gan` | 0.5 | 0.1 |
| Dropout `p` | 0.2 | 0.1 |
| `no_bn` | `true` | `true` |
| Training DataLoader shuffle | `true` | `false` |
| Replay | Enabled | Enabled |

Despite its name, `--no-bn true` disables the encoder's **LayerNorm**.
`--shuffle false` changes training batch order, not the data partition.
`--weight-decay` now applies to every client `E/F/G` Adam parameter group
and the server `D` optimizer. Its default is `0`, preserving the previous
effective behavior: the old CLI displayed `1e-4`, but did not pass that value
to Adam. A positive value explicitly enables weight decay and changes the
training updates. Encoder and generator graph conditioning is required;
the unused `--use-g-encode` switch has been removed.
Graph dimensions follow `--num-clients`; `--nt` and `--nd-out` are
compatibility options. All defaults are in
[TCGA_BRCA.py](TCGA-BRCA/configs/TCGA_BRCA.py).

This command prepares the seed-42 patient partition and runs the selected
configuration. Changing `--seed` also changes data preparation. A plain CLI
seed sweep therefore does not reproduce the fixed-partition, multiple-seed
study below, which reused prepared data and changed only training seeds.

### Simulator options

| Option | Purpose |
| --- | --- |
| `--device cpu` | Run without CUDA; CUDA is selected by default when available |
| `--nvflare-gpu 0` | Physical GPU ID or one shared GPU group, e.g. `'[0,1]'`; defaults to inherited CUDA visibility |
| `--nvflare-threads N` | Defaults to the client count and must equal it, keeping every Collab client process resident |
| `--nvflare-timeout 300` | Client-operation timeout in seconds; increase for longer local training |
| `--max-in-flight N` | Controls concurrent client operations; defaults to `min(num_clients, 8)` |
| `--nvflare-workspace PATH` | Override the simulator workspace |
| `--export-only` | Export the job without training |

Use `--max-in-flight` to adjust training concurrency. All clients share the
selected GPU or GPU group; separate groups such as `--nvflare-gpu 0,1` are
rejected because NVFlare 2.9 would rotate client workers.

`--device cuda --nvflare-gpu 1` selects the same physical GPU for data/model
preparation, the server, and all clients, including job export. IDs follow
`nvidia-smi` ordering (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). Without `--nvflare-gpu`,
the launcher and workers preserve the inherited CUDA mask and ordering.
If a mask is already set, an explicit selection must be a subset of a numeric
mask with PCI bus ordering; otherwise omit `--nvflare-gpu` to use that mask
unchanged (including GPU UUID/MIG masks). CPU mode rejects `--nvflare-gpu`.
Launch the command in a fresh Python process so selection precedes CUDA queries.

The exported job lives under `OUTPUT_DIR/nvflare_job/breastg_fcl/`. Each
`app_site-<n>/config/data/site.pt` contains only that site's train/test partition.
`app_server/config/data/server.pt` holds coordination state and summaries, without
raw expression inputs. Application `custom/` directories contain code only.

### Outputs and verification

A completed NVFlare run writes:

```text
OUTPUT_DIR/
├── tcga_brca_partitioning_seed<seed>.pkl  # Real-data runs
├── final_state.pt
├── round_accuracy.csv
├── all_tasks_accuracy.csv
├── relational_graphs/
│   ├── task_<k>_spatial.npy
│   ├── task_<k>_temporal.npy
│   ├── task_<k>_temporal_window.npy
│   ├── task_<k>_fused.npy
│   ├── task_<k>_private.npy
│   └── task_<k>_attention.pt
├── simulator.log
├── nvflare_job/
│   └── breastg_fcl/            # Exported Collab job and per-site bundles
└── nvflare_workspace/
    └── breastg_fcl/            # SimEnv job workspace
```

`final_state.pt` includes model and attention state, client/server optimizer
and scheduler state, replay metadata, metrics, and the task-indexed
`relational_graphs` used by training and evaluation. Each `task_<k>_private.npy`
contains the same post-noise graph; `task_<k>_fused.npy` retains the clean graph
for auditing. With the same configuration and evaluation data, restore model
weights and use the saved `relational_graphs` to reproduce predictions without
sampling noise again. Simulator/site logs live in the job workspace;
`simulator.log` captures the simulator console. Export-only runs produce the
job without training results.

To plot the round-accuracy CSV, including smoke runs with two rounds per task:

```bash
python TCGA-BRCA/visualize.py --csv_path OUTPUT_DIR/round_accuracy.csv --output_path curve.png
```

Task endpoints are labeled by actual cumulative round counts, including
unequal task lengths. A basename such as `curve.png` saves in the current
directory; parent directories are created when a nested path is supplied.

#### CPU smoke test

To check the complete NVFlare workflow without downloading data:

```bash
python TCGA-BRCA/job.py \
  --device cpu --smoke \
  --output-dir TCGA-BRCA/dump/nvflare_smoke
```

The synthetic fixture uses four clients, three tasks, two rounds per task,
and one local epoch, with reduced model dimensions. Its scores are not
TCGA results. A completed smoke test writes model and attention checkpoints,
optimizer/scheduler state, replay metadata, and metrics in the output
directory. Omit `--smoke` to run the real-data workflow on CPU after preparing
the data.

Run the regression suite with:

```bash
python -m unittest discover -s TCGA-BRCA/tests -v
```

## Expected Results

A successful real-data simulation produces the checkpoints, graph matrices,
and metric files listed above. The CPU smoke test produces the same kinds
of training artifacts from its synthetic fixture. Smoke-test accuracy is
only a diagnostic of the synthetic workflow, not an expected TCGA score.

Author validation reported on 2026-10-11 passed all **183 regression tests** in the
`breastgfcl` environment. Scoped Black, isort, and flake8 checks, the example's
license-header check, and `pip check` also passed.

Maintenance validation on 2026-10-12 passed all **203 regression tests**,
including offline TCIA checksum/workbook fixtures, unsupported-option rejection,
positive experiment counts, visualization failure handling, and manifest UUID
matching. Noise parameters must be finite and nonnegative; `epsilon = 0` keeps
the documented no-noise behavior. Input width is derived from `--max-genes`,
and importing configuration helpers does not create directories. This CPU check
used Python 3.13.5, PyTorch 2.14.1, NumPy 2.5.4, and NVFlare 2.9.0.
The multiprocess CPU smoke also completed four clients, three tasks and six
rounds, with both metric CSVs and private graphs matching the final checkpoint.
The project style check passed across the whole example using the repository
pins: Black 25.9.0, isort 5.13.2, and Flake8 7.1.1. A separate CPU smoke
with `--num-classes 3` completed all six rounds with three-wide predictor
outputs and replay counts on all four clients.

CLI alias validation on 2026-10-13 passed all **206 regression tests** and
the whole-example pinned style check, including effective alias values,
last-value precedence, invalid class widths, and prepared model dimensions.

Clinical metadata validation on 2026-10-13 passed all **224 regression tests**,
including 18 tests for primary-breast diagnosis selection, ambiguous candidates,
unknown-stage preservation, case identity validation, order independence, and
exact TSV regeneration. Whole-example pinned style and license checks and
`pip check` passed. The corrected real-data rerun is reported below.

The NVFlare 2.9.0 Collab CPU smoke completed all six rounds across four
client processes and three tasks, with replay and client-side latent
perturbation enabled. It exited successfully with no errors or warnings.
Model, optimizer, scheduler, replay, and graph artifacts were finite, and
both metric CSV files contained six rounds. Each saved private graph
matched its checkpoint entry; restoring the models and graphs reproduced
all 12 client/task accuracies. Plotting this smoke CSV with a basename-only
output path succeeded and labeled task endpoints as 2, 4, and 6.

Before latent perturbation moved to clients, an NVFlare 2.9.0 Collab CPU
smoke run completed four clients, three tasks, two rounds per task, and one
local epoch with replay enabled. Its model, optimizer, scheduler, attention,
and replay checkpoint contents matched the previous NVFlare 2.7.1 Controller
implementation exactly, as did all 15 graph
artifacts and both accuracy CSV files. This historical check validates the
transport migration before the client-side noise change on a synthetic
fixture; it does not establish TCGA accuracy.

### Recorded NVFlare experiment

The **2026-10-10 historical evaluation** below used the inherited clinical
TSV, whose diagnosis fields came from the first diagnosis in each case's
JSON. Its cohort and results are retained to document the earlier parameter
search and code validation. The [clinical metadata correction and rerun](#clinical-metadata-correction-and-rerun-2026-10-13)
reports results from the corrected cohort.

The historical rerun included stable prediction log probabilities, graph
normalization, and persistence of the actual noisy training graphs. Its
three final training seeds used the previously selected configuration;
the parameter search was not repeated.

These measurements use the NVFlare 2.9.0 Collab implementation with
patient-separated partitions, training-fitted expression/TCIA normalization,
client-side latent perturbation, and expression-cache schema v3. Each run
completed four clients, three tasks, ten rounds per task, and three local
epochs per round with replay enabled. Evaluation uses the final
Task 3 / Round 10 checkpoint.

#### Historical parameter selection and fixed data

After excluding 112 unknown-stage tumors from the 1,231 expression records,
the original seed-42 partition contains 894 training records from 792 patients
and 225 evaluation records from 201 patients. Patient and sample identities
are disjoint across the two sides. The 225 evaluation records contain
23 Normal and 202 Tumor records.

The historical parameter selection used only the original training side:

1. Split its patients again with split seed 2026, retaining the original
   task/client assignments: 674 internal-training records and 220 validation
   records (22 Normal, 198 Tumor). Internal training, validation, and outer
   evaluation have pairwise-disjoint patient and sample identities. Fit
   expression and TCIA statistics and construct graph summaries using only
   internal training. The outer evaluation tensors are excluded from these
   search jobs.
2. Screen 24 predefined learning-rate, local-epoch, and adversarial-weight
   configurations with training seed 42. Prefer candidates meeting both
   validation recall targets (Normal ≥70%, Tumor ≥85%), then rank by pooled
   balanced accuracy, macro F1, lower maximum logged discriminator loss,
   and candidate ID, in that order.
3. Repeat the top three configurations with training seeds 43 and 44 on the
   same prepared data. Rank by attainment of both mean recall targets, mean
   balanced accuracy, minimum class recall across seeds, mean macro F1,
   lower maximum discriminator loss, and candidate ID. Lock the selected
   parameters before evaluating them on the outer split.
4. Refit on all 894 original training records, fitting preprocessing and
   graph summaries again on that training pool. Use fresh models and
   optimizers for each of the predefined seeds 42, 43, and 44, with identical
   prepared inputs and the original 225-record evaluation side. Report all
   three seeds without further parameter or checkpoint selection.

That study produced 33 completed runs: 24 screening runs, six additional
confirmation runs, and three final refits. The selected configuration uses
`lr_e = lr_g = 3e-5`; the complete settings and launch command are
[above](#selected-development-configuration). It tied another candidate on
the confirmation metrics and was selected by the predefined candidate-ID
tie-breaker. All three selected validation runs achieved pooled accuracy
99.09%, balanced accuracy 95.45%, Normal recall 90.91%, Tumor recall 100.00%,
and macro F1 97.37%.

#### Historical final evaluation results

For the 2026-10-10 rerun, all 1,231 expression files were downloaded against
the repository's frozen manifest and verified by size and MD5; all five
TCIA source artifacts passed SHA-256 checks. The seed-42 preparation
reproduced the documented cohort counts above. Its normalized tensors,
patient/task/client assignments, and TCIA summaries were then frozen and
reused for training seeds 42, 43, and 44. Each seed started with fresh
models, attention parameters, optimizers, and random state. Input and
partition hashes matched across all three runs. No parameters or
checkpoints were selected using the outer evaluation results.

The rerun pools predictions across the same 225-record evaluation population
for all three seeds. The five classification metrics match the historical
final table at the displayed precision:

| Training seed | Accuracy (%) | Balanced accuracy (%) | Normal recall (%) | Tumor recall (%) | Macro F1 (%) | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 97.33 | 88.88 | 78.26 | 99.50 | 92.12 | 0.9759 |
| 43 | 97.33 | 88.88 | 78.26 | 99.50 | 92.12 | 0.9707 |
| 44 | 97.33 | 88.88 | 78.26 | 99.50 | 92.12 | 0.9856 |
| Mean ± sample standard deviation | 97.33 ± 0.00 | 88.88 ± 0.00 | 78.26 ± 0.00 | 99.50 ± 0.00 | 92.12 ± 0.00 | 0.9774 ± 0.0075 |

AUROC is on a 0–1 scale and uses the Tumor log odds, with average ranks for
tied scores.

For each seed, the confusion matrix is `[[18, 5], [1, 201]]`, with rows as
true labels, columns as predicted labels, and class order Normal/Tumor.
Balanced accuracy is the mean of the two pooled class recalls:
`(18/23 + 201/202) / 2 = 88.88%`. Macro F1 is the mean of the two class F1
scores. The simulator's `overall_avg_acc` instead averages the 12
client/task cell accuracies equally and is **96.90%** for each final run;
it is a different aggregation from the pooled **97.33%** above.

Post-training evaluation restored each saved encoder/predictor and the
actual perturbed graph rows on CPU. Recomputed accuracies matched all
**36 saved client/task results** across the three seeds. All 90 training
rounds completed with finite checkpoint states and logged losses, and no
errors or warnings appeared in the simulator or site logs. Training-only
expression and TCIA normalization statistics were independently recomputed
and matched the saved statistics and prepared tensors.

Each client's Adam state recorded 240 updates and its scheduler 90 steps;
the server optimizer and scheduler each recorded 30 steps. Every run saved
18 graph artifacts and 30 rows in each metric CSV. The maximum logged `D`
loss across the three runs was 0.2807; final values for seeds 42/43/44 were
0.1417/0.1273/0.1104. Loss magnitude alone does not establish convergence.

The three runs have distinct trained model weights and AUROCs despite their
identical hard-classification metrics. Sample standard deviation describes
training-seed variation on this fixed cohort, not a patient-level confidence
interval.
There are only 23 Normal evaluation records. Multiple expression files
remain separate records within their assigned patient side.

The outer evaluation split had been inspected in earlier experiments, so
these are not results from a previously unseen blind test. Its data and
metrics were excluded from this parameter search. Training-fitted
preprocessing still spans all training tasks, as documented
[above](#data-preparation); it is not strictly online per-task preprocessing.
These results also do not reproduce the paper's original experimental
protocol. Historical results using overlapping file-level partitions and
full-cohort normalization are superseded here and are not comparable.

The search and fixed-partition rerun/evaluation drivers, frozen prepared
tensors, per-record predictions, environment records, and full experiment
bundles are local experiment artifacts and are not included in this
contribution. The public launch command runs the selected configuration;
exact replay of the fixed-partition search and multiple-seed table requires
those additional artifacts. The extra pooled metrics were computed by the
local evaluator, not by the standard accuracy CSV output.

#### Clinical metadata correction and rerun (2026-10-13)

The clinical exporter now selects the primary breast diagnosis from the
unchanged 1,098-case JSON snapshot using the rules in [data preparation](#tcga-brca-expression-and-clinical-data).
This changes 106 TSV rows, including 101 stage values. For example,
`TCGA-A2-A04P` retains its primary breast Stage IIIC; `TCGA-B6-A0I8` retains
its primary Stage X and is excluded instead of using a prior tumor's Stage II.

Of the same 1,231 expression records, 1,206 are eligible and 25 unknown-stage
tumors are excluded. The cohort contains 113 Normal records and 184/631/278
Tumor records in the early/intermediate/advanced tasks. Relative to the
historical TSV, 88 records are recovered, 10 retained records change task,
and one record becomes ineligible. These are expression-record counts;
the three changes involve 84, 8, and 1 distinct patients, respectively.

Preparing the corrected data with seed 42 gives **966 training records from
857 patients** and **240 evaluation records from 214 patients**. The evaluation
side contains 23 Normal and 217 Tumor records. Previously eligible records
that remain eligible stay on their original train/evaluation side; patient
and sample identities are disjoint. Expression normalization is fitted on
the 966 training records, and spatial/temporal normalization each uses 68
unique matched training TCIA rows. Statistics, normalized tensors, and
training-only graph summaries were recomputed and independently checked.

All expression size/MD5 checks and the five TCIA source SHA-256 checks passed.
Fresh preparation used a separate cache and froze the corrected tensors,
assignments, and summaries. Three fresh refits used the previously locked
configuration and training seeds 42, 43, and 44 on GPUs 0, 2, and 4,
respectively. The parameter search was not repeated, and no parameters or
checkpoints were selected using these evaluation results.

| Training seed | Accuracy (%) | Balanced accuracy (%) | Normal recall (%) | Tumor recall (%) | Macro F1 (%) | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 97.50 | 88.90 | 78.26 | 99.54 | 92.17 | 0.9611 |
| 43 | 97.92 | 91.07 | 82.61 | 99.54 | 93.61 | 0.9661 |
| 44 | 97.50 | 88.90 | 78.26 | 99.54 | 92.17 | 0.9711 |
| Mean ± sample standard deviation | 97.64 ± 0.24 | 89.62 ± 1.26 | 79.71 ± 2.51 | 99.54 ± 0.00 | 92.65 ± 0.83 | 0.9661 ± 0.0050 |

The confusion matrices are `[[18, 5], [1, 216]]` for seeds 42 and 44, and
`[[19, 4], [1, 216]]` for seed 43, using the same class order and pooled
record-level metric definitions as above. The unweighted client/task mean
accuracy is 97.05%, 97.74%, and 97.05%, respectively.

All **90 rounds** completed. CPU restoration of the final encoders,
predictors, and saved noisy graphs reproduced all **36 client/task accuracy
cells**. Prepared data and partition hashes match across seeds; fresh
initialization and final trained weights differ. Checkpoint states and
logged losses are finite, and simulator/site logs contain no errors or
warnings. Each run contains 18 graph artifacts and 30 rows in each metric CSV.

These measurements use the corrected cohort, so differences from the
historical table do not establish a performance improvement. The outer split
was previously inspected, only 23 evaluation records are Normal, and seed
variation is not patient-level uncertainty. The historical search and paper
protocol limitations still apply. Source snapshots, frozen inputs, drivers,
per-record predictions, and full run artifacts are retained locally; this
rerun does not alter the historical experiment artifacts.

## License

This contribution is released under [Apache-2.0](LICENSE), consistent with
the surrounding [NVIDIA FLARE repository](../../LICENSE). It incorporates
code from the IntelliSys-Lab BreastG-FCL project; the original MIT copyright
and permission notices are preserved in [NOTICE](NOTICE) and the source files.

External code, data, and dependencies retain their own terms:

| Source | License or data terms |
| --- | --- |
| GFedCL reference implementation | [MIT](https://github.com/IntelliSys-Lab/GFedCL/blob/main/LICENSE); retain upstream notices when reusing its code |
| NVIDIA FLARE | [Apache-2.0](https://github.com/NVIDIA/NVFlare/blob/main/LICENSE) |
| TCGA-BRCA expression and clinical data | [GDC policies](https://gdc.cancer.gov/about-gdc/gdc-policies) |
| TCIA radiogenomic analysis artifacts | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), with the dataset's citation requirements |

The contribution's Apache-2.0 license does not relicense these datasets or dependencies.
Data source links and access instructions are in [Data Preparation](#data-preparation).
No pretrained author checkpoints are distributed with this implementation.

## Requirements

The project dependency specifications are [requirements.txt](requirements.txt)
and [environment.yml](environment.yml). NVFlare is pinned to **2.9.0** for
the Collab API and `CollabRecipe`/`SimEnv` integration.

The completed validations used these installed versions:

| Dependency | Historical search | 2026-10-10 and 2026-10-13 validation and final reruns |
| --- | --- | --- |
| Python | 3.10.12 | 3.10.22 |
| NVFlare | 2.9.0 | 2.9.0 |
| PyTorch | 2.2.1+cu118 | 2.5.1 (CUDA 12.4) |
| NumPy | 1.26.4 | 2.2.6 |

The new run validates the corrected implementation in the recorded
Miniconda environment; it does not establish bitwise equivalence with
historical models trained under the older dependencies.
The requirements files allow broader versions for several packages; they
are not exact lockfiles for these environments. Spreadsheet readers
`openpyxl==3.1.5` and `xlrd==2.0.2` support the TCIA source workbooks.

## Citation

```bibtex
@inproceedings{yu2026breastgfcl,
  author    = {Yu, Qingyang and Wang, Jingyi and Zhang, Xinyue and Pan, Miao and Xu, Ziyue and Wang, Hao},
  title     = {{BreastG-FCL}: Graph-Conditioned Federated Continual Learning for Breast Cancer Radiogenomics},
  booktitle = {2026 IEEE International Conference on E-health Networking, Application \& Services (HealthCom)},
  year      = {2026}
}
```
