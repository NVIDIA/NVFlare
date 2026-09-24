# <img align="right" src="assets/logo_uc.jpg" alt="The University of Chicago" width="80" height="80"> FABRIC: A General Framework for Federated Computational Pathology with NVIDIA FLARE

**Contributors**

- Anbang Liu — [anbangliu2027@u.northwestern.edu](mailto:anbangliu2027@u.northwestern.edu)
- Junhan Zhao — [junhanzv@uchicago.edu](mailto:junhanzv@uchicago.edu)
- Ziyue Xu — [ziyuex@nvidia.com](mailto:ziyuex@nvidia.com)

**Sponsored by NVIDIA Academic Award.**

A general framework combining pathology foundation models with NVIDIA FLARE. Pediatric brain tumor recurrence prediction is the first use case.

## Abstract

FABRIC is a general framework for federated learning in computational pathology that combines pathology foundation models with NVIDIA FLARE. It connects pre-extracted patch representations, multiple-instance learning (MIL), client training, and server aggregation in a reusable workflow. Attention pooling and Top-k feature selection provide two alternatives for aggregating patch features, while NVIDIA FLARE manages the federated training process.

Pediatric brain tumor recurrence prediction is the framework's first use case. We evaluated patient-level pediatric glioma recurrence prediction using diagnostic hematoxylin and eosin whole-slide images from 804 patients across four cohorts, with UNI or Virchow2 features, three federated clients, and patient-count-weighted FedAvg. This contribution provides the implementation and results for that use case, together with a CPU demonstration using synthetic inputs.

## Papers and Links

- [FABRIC project and upstream implementation](https://github.com/AndrewLiu666/FABRIC)
- [Data format](docs/DATA_FORMAT.md)

The first use case is described in *Privacy-Preserving Federated Distillation of Foundation Models for Multi-Institutional Pediatric Glioma Recurrence Prediction*. The name FABRIC expands to *Federated Assessment of Brain tumor Recurrence In Children*, reflecting this initial application. A public manuscript link and formal citation will be added when available.

## Objective

FABRIC provides a reusable pipeline for federated learning with pathology foundation models. The framework separates feature representations, MIL prediction, and federated orchestration so that this structure can be adapted to other pathology datasets and prediction tasks. Pediatric glioma recurrence provides the first evaluated use case; readers can explore the workflow with synthetic data or reproduce that study with authorized inputs.

## Method Summary

### General workflow

- **Foundation-model representations:** frozen pathology foundation models encode tissue patches into feature vectors. Feature extraction takes place upstream; this repository consumes existing features and does not train or exchange foundation-model weights.
- **MIL prediction:** attention pooling or Top-k feature selection aggregates patch representations into a bag-level prediction. The first use case groups all slides from a patient into one bag.
- **Federated optimization:** NVIDIA FLARE distributes the shared MIL model, runs client training, exchanges model updates, and coordinates weighted server aggregation over communication rounds. The final shared model is used for prediction and evaluation.

### NVIDIA FLARE integration

| Component | Role in FABRIC |
| --- | --- |
| `FedAvgRecipe` | Configures the federated training job and its clients |
| FedAvg workflow | Distributes global models and coordinates communication rounds |
| `ScriptRunner` and Client API | Launch local PyTorch training and exchange full model updates, training loss, and metadata |
| FABRIC `ModelAggregator` implementation | Applies training-patient-count-weighted averaging with FABRIC's aggregation function |
| `PTFileModelPersistor` | Loads the initial model and saves the shared model |

The first use case uses three federated clients. The Client API exchanges model parameters, training loss, counts, and operational metadata, not feature bags, slides, or patient identifiers. Local manifests and predictions contain identifiers and must remain private. Differential privacy and secure aggregation are not configured.

### First use case: pediatric brain tumor recurrence

The evaluated clinical task is patient-level recurrence prediction in pediatric glioma. Its study protocol is:

1. Consume pre-extracted patch features: 1,024 values per UNI patch or 2,560 per Virchow2 patch. All slides from one patient form one bag.
2. Select `pooling` or `topk`. Pooling uses attention-weighted aggregation. Top-k partitions each bag into up to eight nonempty pseudo-bags, supervises their pooled vectors, ranks attention-weighted patches by recurrence score, and selects the K highest and K lowest per group (default K = 1, without duplicates). A second MIL tier aggregates the selected embeddings during both training and prediction. Its loss combines patient and pseudo-bag cross-entropy with weight 1.0.
3. Train all three clients from the same global model each round. Aggregate full updates in the order `CBTN_CQU`, `Harvard`, `EBRAINS`, weighted by training-patient counts. Recreate the Adam optimizer and AMP scaler for every local task.
4. Use five-fold cross-validation, five communication rounds, and five local epochs per round. Evaluate the final round without early stopping or best-round selection. AUC uses recurrence probabilities; class predictions use a 0.5 threshold.

Both encoders use the same labels, folds, client assignments, and training settings, with 1,284 feature files each. Settings are recorded in [UNI](configs/uni.json) and [Virchow2](configs/virchow2.json). Initialization uses seed `42 + fold*1000`; local training uses `42 + fold*1000 + round*100 + client_index`, with rounds 1–5 and client indices 0–2 in aggregation order. Top-k prediction uses a separate grouping generator seeded with 42.

The MIL design is informed by [DTFD-MIL](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_Slide_CVPR_2022_paper.html).

### Adapting to other computational pathology tasks

The framework can be extended to other digital pathology applications by adapting the input data and labels, prediction target, model settings, and evaluation metrics. The supplied study runner fixes the pediatric glioma cohort and client configuration for reproduction, so another use case also requires updating those code checks and validating the adapted pipeline. The results reported here are from the first use case.

## Repository Layout

All paths below are relative to `research/fabric/`.

```text
fabric/
|-- README.md
|-- LICENSE
|-- NOTICE.md
|-- environment.yml
|-- assets/                  README image assets
|-- configs/                 Study settings and example local data configuration
|-- fabric/                  Study runner, FLARE adapters, and core MIL implementation
|-- demo/                    Synthetic CPU demonstration and its dependencies
|-- tests/                   Automated CPU tests using synthetic inputs
`-- docs/DATA_FORMAT.md       Private input schema
```

## Setup

Run the following commands from this example directory, not the NVFlare repository root:

```bash
cd research/fabric
```

The supplied launchers run all three clients sequentially on a single host. Input paths must be accessible on that host.

### Synthetic CPU demonstration

Use Python 3.10 and an isolated environment. No GPU, slide files, foundation-model downloads, or clinical data are needed.

```bash
python3.10 -m venv .demo/venv
source .demo/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r demo/requirements.txt
```

The study environment below can also run this demonstration; the demo disables CUDA and limits CPU threads.

### Study environment

The study requires Linux, Python 3.10, NVIDIA FLARE 2.7.2, PyTorch 2.11.0 with CUDA 12.8, and a compatible NVIDIA GPU and driver. Create the pinned environment:

```bash
conda env create -f environment.yml
conda activate fabric-nvflare
```

Keep this pinned environment separate from the parent NVFlare development installation; the study runner checks package versions against [the environment record](configs/environment-lock.json).

## Data Preparation

### Synthetic data

The demo generates random arrays with UNI- or Virchow2-compatible dimensions, fictional labels, and `synthetic-` identifiers. It does not extract foundation-model features or read clinical data. Clients use 4, 6, and 8 synthetic cases, with 6 separate test cases and 17–23 patches per bag. Generated inputs and outputs stay under the Git-ignored `.demo/` directory.

### Pediatric glioma study inputs

Study data are not redistributed. Obtain authorized access through the relevant cohort or institution. Reproduction requires the original labels, patient-level folds, client assignments, slide ordering, and feature tensors; this code does not preprocess slides or extract features.

Create a local configuration and edit it to point to your authorized files:

```bash
cp configs/data_paths.example.json configs/data_paths.local.json
```

Set the manifest root and optional feature-path prefix mapping in that local copy. Inputs may be under `private_data/` or in secure external storage. The local config and generated outputs are ignored by Git. The [data schema](docs/DATA_FORMAT.md) specifies the 40 required manifests. The runner checks the 804-patient cohort without regenerating or rebalancing its splits.

## Run Instructions

### Run the synthetic demonstration

Export a job and generate its fictional inputs without starting training:

```bash
python -B demo/run_demo.py --encoder uni --variant pooling --run-name inspect_demo
```

Run two communication rounds through NVIDIA FLARE on CPU:

```bash
python -B -u demo/run_demo.py --encoder uni --variant pooling --run-name demo_uni_pooling --execute
python -B -u demo/run_demo.py --encoder virchow2 --variant topk --run-name demo_virchow2_topk --execute
```

Both encoders support both variants. The demo reuses the study's core model, training, evaluation, and averaging functions with two rounds, one local epoch, batch size 2, and Top-k K = 1. It uses separate FLARE adapters without changing study settings. Each run needs a new name; demo runs do not resume or overwrite outputs.

Results are under `.demo/<run-name>/`: `summary.json` records completion and synthetic metrics, `simulator.log` captures execution, and `nvflare_workspace/` contains the FLARE checkpoint. Generated inputs, job exports, round records, and predictions stay in the same directory.

### Reproduce the first use case

Validate authorized inputs before starting training:

```bash
python -B fabric/run_fabric.py --experiment both
python -B fabric/run_fabric.py --experiment both --variant topk --top-k 1
```

Run each variant with both encoders:

```bash
python -B -u fabric/run_fabric.py \
  --experiment both --run-name pooling_01 --execute

python -B -u fabric/run_fabric.py \
  --experiment both --variant topk --top-k 1 \
  --run-name topk_01 --execute
```

Resume an interrupted study run with the same code, inputs, settings, and run name:

```bash
python -B -u fabric/run_fabric.py \
  --experiment both --variant topk --top-k 1 \
  --run-name topk_01 --resume --execute
```

For pooling, omit `--variant topk --top-k 1` and use `--run-name pooling_01`. Resume requires the original code version, configuration, and saved manifests. Completed folds are skipped; an unfinished round restarts from the latest verified global checkpoint with the original round number and seeds. If only evaluation was interrupted, it is rerun without training. Recovery does not resume inside a local epoch. Omit `--execute` to inspect a recovery plan.

Study outputs stay under `results/<run-name>/<encoder>/`: each `fold_<n>/` contains checkpoints, predictions, metrics, and FLARE logs; `mean_std_summary.tsv` summarizes completed folds. Resume logs and incomplete records stay under each fold's `resume_attempts/`. Temporary IPC files remain under `.tmp/`. These outputs can contain patient identifiers and paths and must not be published.

Before evaluation, each fold must have five aggregation records, 15 client updates, and matching final FABRIC and FLARE checkpoint hashes. For failures, inspect `simulator.log` or the latest resume log. Use a short repository path for local sockets and allow subprocess/socket creation. Missing features require correcting the local path mapping; changed code or model variants require the matching original revision or a new run, not bypassing recovery checks.

### Automated tests

In either environment described above, install the test dependency and run from `research/fabric/`:

```bash
python -m pip install -r tests/requirements.txt
python -B -m pytest --confcutdir=tests -q -p no:cacheprovider tests
```

The tests generate small CPU tensors and fictional manifests in temporary directories. They check both MIL variants, class-guided Top-k selection, training and prediction, weighted aggregation, FLARE message serialization and job export, client settings, and round-boundary recovery. No clinical data, feature downloads, GPU, or running FLARE deployment is required; client transport is mocked for the adapter tests.

The `--confcutdir` option keeps the parent repository's pytest setup from substituting its development source for the installed NVIDIA FLARE 2.7.2 dependency. These tests must be invoked explicitly: the parent repository's default unit-test target is `tests/unit_test/`. Passing this project suite does not replace the parent NVFlare checks or a complete federated run.

## Expected Results

The synthetic demonstration is successful when all six client tasks and both server aggregations complete and FLARE's saved model matches the audited final aggregate. Synthetic AUC and accuracy are not clinical results or estimates of model quality.

The following results are for the first use case, pediatric glioma recurrence prediction, from completed NVIDIA FLARE runs with the original study data and environment. Values are mean ± sample standard deviation across five patient-level test folds. Top-k uses K = 1 per high/low set.

| Encoder | FABRIC variant | AUC | Balanced accuracy | Accuracy |
| --- | --- | ---: | ---: | ---: |
| UNI | `pooling` | 0.644 ± 0.038 | 0.629 ± 0.057 | 0.715 ± 0.032 |
| UNI | `topk` | 0.617 ± 0.053 | 0.572 ± 0.068 | 0.718 ± 0.025 |
| Virchow2 | `pooling` | 0.649 ± 0.018 | 0.622 ± 0.043 | 0.718 ± 0.012 |
| Virchow2 | `topk` | 0.649 ± 0.049 | 0.591 ± 0.059 | 0.709 ± 0.038 |

These results are not expected from synthetic or substituted data. Exact numerical agreement also depends on feature ordering, software versions, and compatible GPU/CUDA/cuDNN behavior.

## License

This contribution is provided under the [Apache License 2.0](LICENSE). See [NOTICE.md](NOTICE.md) for source attribution, the original license notice, and separate terms for external models and data.

## Requirements

- Synthetic demonstration: Python 3.10 and [demo/requirements.txt](demo/requirements.txt), with CPU PyTorch as installed above.
- Study reproduction: [environment.yml](environment.yml); runtime version checks use [configs/environment-lock.json](configs/environment-lock.json).
- NVIDIA FLARE version: **2.7.2** for both paths.

## Citation

The formal publication citation will be added when available. Until then, refer to the [FABRIC project](https://github.com/AndrewLiu666/FABRIC) and the study title above.
