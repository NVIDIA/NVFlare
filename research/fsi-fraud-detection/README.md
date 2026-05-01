# Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE

Implementation for the paper [Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE](https://arxiv.org/abs/2603.13617).

Holger R. Roth, Sarthak Tickoo, Mayank Kumar, Isaac Yang, Andrew Liu, Amit Varshney, Sayani Kundu, Iustina Vintila, Peter Madsgaard, Juraj Milcak, Chester Chen, Yan Cheng, Andrew Feng, Jeff Savio, Vikram Singh, Craig Stancill, Gloria Wan, Evan Powell, Anwar Ul Haq, Sudhir Upadhyay, Jisoo Lee

> Fraud-related financial losses continue to rise, while regulatory, privacy, and data-sovereignty constraints increasingly limit the feasibility of centralized fraud detection systems. Federated Learning (FL) has emerged as a promising paradigm for enabling collaborative model training across institutions without sharing raw transaction data. Yet, its practical effectiveness under realistic, non-IID financial data distributions remains insufficiently validated.
>
> In this work, we present a multi-institution, industry-oriented proof-of-concept study evaluating federated anomaly detection for payment transactions using the NVIDIA FLARE framework. We simulate a realistic federation of heterogeneous financial institutions, each observing distinct fraud typologies and operating under strict data isolation. Using a deep neural network trained via federated averaging (FedAvg), we demonstrate that federated models achieve a mean F1-score of 0.903—substantially outperforming locally trained models (0.643) and closely approaching centralized training performance (0.925), while preserving full data sovereignty.
>
> We further analyze convergence behavior, showing that strong performance is achieved within 10 federated communication rounds, highlighting the operational viability of FL in latency- and cost-sensitive financial environments. To support deployment in regulated settings, we evaluate model interpretability using Shapley-based feature attribution and confirm that federated models rely on semantically coherent, domain-relevant decision signals. Finally, we incorporate sample-level differential privacy via DP-SGD and demonstrate favorable privacy–utility trade-offs, achieving effective privacy budgets below $\epsilon = 10.0$ with moderate degradation in fraud detection performance. Collectively, these results provide empirical evidence that FL can enable effective cross-institution fraud detection, delivering near-centralized performance while maintaining strict data isolation and supporting formal privacy guarantees.

## Paper

- [Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE](https://arxiv.org/abs/2603.13617)

## Highlights

- Studies payment fraud detection across heterogeneous financial institutions with non-IID transaction distributions.
- Evaluates federated anomaly detection with NVIDIA FLARE using federated averaging.
- Reports strong performance relative to local training while preserving data sovereignty.
- Explores interpretability with Shapley-based feature attribution and privacy-utility trade-offs with DP-SGD.

# Code

A repository of examples, tools, and reference implementations for running
**federated learning** experiments in financial services using
[NVFlare](https://nvidia.github.io/NVFlare/). Because real payment data is
rarely available due to regulatory constraints, this project includes a
synthetic data generation toolkit that produces realistic payment transaction
datasets with configurable, rule-based anomaly injection -- enabling
reproducible FL experimentation without sensitive data.

## Table of contents <!-- omit in toc -->

- [Paper](#paper)
- [Highlights](#highlights)
- [Code](#code)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
- [Synthetic Data Generation](#synthetic-data-generation)
  - [Components](#components)
  - [Exploration Notebook](#exploration-notebook)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Step 1: Generate Datasets](#step-1-generate-datasets)
- [Step 2: Federated Data Analytics](#step-2-federated-data-analytics)
- [Step 3: Federated Learning](#step-3-federated-learning)
- [Starting Jupyter Lab](#starting-jupyter-lab)
- [Central Training Baseline](#central-training-baseline)
- [Data Generation Documentation](#data-generation-documentation)
- [Development](#development)

### Overview

This repository is organized around two goals:

1. **Synthetic data generation** -- produce realistic-looking payment records (debtor/creditor identities, geo-coordinates, timestamps, currencies,
   amounts) with controllable anomalies that simulate fraud patterns. Each "site" in a federated learning setup receives its own configuration
   (distribution parameters, anomaly types, dataset sizes), enabling experiments with heterogeneous, non-IID data partitions.
2. **Federated learning examples** -- end-to-end NVFlare workflows for federated statistics and training across the generated sites ([Step 2](#step-2-federated-data-analytics), [Step 3](#step-3-federated-learning)).

### Repository Structure

| Directory / File   | Purpose                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| `data_generation/` | Synthetic payment data toolkit (see [below](#components))                                        |
| `config/`          | Per-site YAML configuration files                                                                |
| `notebooks/`       | Interactive exploration notebooks                                                                |
| `docs/`            | Technical documentation                                                                          |
| `tests/`           | Test suite                                                                                       |
| `main.py`          | CLI entry point for dataset generation, checksum writing, and universal scaling dataset assembly |

## Synthetic Data Generation

The data generation toolkit produces realistic payment records and injects controllable anomalies that simulate fraud patterns. Each federated learning site receives its own configuration, enabling experiments with heterogeneous, non-IID data partitions.

### Components

| Component                                  | Description                                                                                                                                                                                                      |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                                  | CLI entry point for bulk dataset generation. Reads per-site YAML configs, orchestrates the pipeline, writes CSV files, optionally computes per-site SHA256 checksums, and assembles a universal scaling dataset. |
| `data_generation/attributes.py`            | Declares all dataset columns as attribute descriptors with typed provider requirements and inter-column dependencies.                                                                                            |
| `data_generation/dataset.py`               | Resolves attribute dependencies and generates a complete DataFrame column-by-column via pluggable providers.                                                                                                     |
| `data_generation/dataset_attribute/`       | `PaymentDatasetAttribute` and `PaymentDatasetAttributeGroup` descriptors that bind column names to provider callables.                                                                                           |
| `data_generation/attribute_data_provider/` | `AttributeDataProviderProtocol` -- the callable interface every column generator implements.                                                                                                                     |
| `data_generation/synthetic_data_provider/` | Provider classes wrapping Faker, RNG samplers, and vectorised helper functions.                                                                                                                                  |
| `data_generation/rng/`                     | Seedable RNG wrappers for uniform, normal, log-normal, gamma, and random-choice distributions.                                                                                                                   |
| `data_generation/anomaly_transformers/`    | Anomaly injection framework: four fraud types, row sampling with overlap control, and probability thinning.                                                                                                      |
| `data_generation/static_data/`             | Country-currency mappings, exchange rates (CurrencyConverter API, cached), and field constants.                                                                                                                  |
| `data_generation/commons/`                 | Shared type aliases (`ColumnValueType`, `MultiColumnValueType`).                                                                                                                                                 |
| `config/`                                  | Per-site YAML configuration files defining distribution parameters and dataset generation specs.                                                                                                                 |
| `tests/`                                   | Comprehensive test suite (126 tests) covering RNG, static data, anomaly transformers, and end-to-end dataset generation.                                                                                         |

### Exploration Notebook

The interactive notebook at `notebooks/data_generation_exploration.ipynb`
provides a guided walkthrough of the entire pipeline -- from configuration
loading through data generation, anomaly injection, and fraud probability
thinning. It serves as both documentation and a development sandbox for
understanding how each layer composes into a complete dataset. See
[docs/exploration-notebook.md](docs/exploration-notebook.md) for details.

To run the notebook locally, follow [Starting Jupyter Lab](#starting-jupyter-lab) below.

## Quick Start

### Prerequisites

This project requires Python 3.12 and uses
[`uv`](https://docs.astral.sh/uv/) for dependency management.

To quickly install `uv`, run:

```bash
pip install uv
```

For other ways to install `uv`, see the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/#installing-uv).

### Setup

```bash
uv venv --python 3.12 --seed
uv sync --frozen
```

## Step 1: Generate Datasets

```bash
# All sites found in config/ (checksums and universal scaling dataset produced by default)
uv run main.py -o datasets/
```

Optionally, you can customize the command as follows.
```bash
# Specific sites with a custom seed
uv run main.py -s siteA -s siteB -o datasets/ -S 42

# Skip checksum generation and universal scaling dataset
uv run main.py -o datasets/ --no-checksum --no-generate-universal-set

# Custom universal scaling dataset output path and sample size
uv run main.py -o datasets/ -F datasets/scaling_combined.csv -C 5000
```

After generation each site directory contains its CSV datasets plus a timestamped
`checksum_YYYYMMDD_HHMMSS.csv` file. The combined scaling dataset (one row per
scaling file across all sites, with a leading `SITE` column) is written to
`datasets/universal_scaling_datasets_all_banks.csv` by default.

See [docs/pipeline.md](docs/pipeline.md) for full pipeline documentation and
CLI reference.

## Step 2: Federated Data Analytics

With per-site datasets in place, the next step is to characterize those partitions in a privacy-preserving way before any model training. Federated data analytics lets each institution contribute aggregate statistics only, so you can compare distributions and spot drift across sites without centralizing raw transactions.

Run [**Federated Statistics**](notebooks/compute_fed_stats.ipynb) under [`notebooks/`](notebooks/) to compute distributed statistics across client datasets without exposing raw data.

- **Measures:** count, mean, sum, standard deviation, histogram, quantiles
- **Exploratory analysis:** interactive visualization of aggregated results


## Step 3: Federated Learning

Once you have a read on cross-site data heterogeneity, you can move from descriptive aggregates to collaborative model training. Federated learning exchanges only model updates (not raw rows), so institutions can jointly improve a fraud classifier while keeping transaction data on their own systems.

Continue in the same folder with [**Training a deep learning model for fraud detection**](notebooks/train_pytorch_model.ipynb). It uses NVFlare’s FedAvg (Federated Averaging) recipe to train a `SimpleNetwork` model for binary fraud classification, and can be configured for simulation (local prototyping) or production (multi-client deployment).

- **Experiment tracking:** training metrics are integrated with MLflow (see the notebook).

## Starting Jupyter Lab

After [`uv sync`](#setup) (and any extra packages the notebooks require), point Jupyter Lab at this example folder:

```shell
export PYTHONPATH="$(pwd)"
NB_DIR="$(pwd)/notebooks"
uv run --with jupyter jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --notebook-dir="${NB_DIR}" --NotebookApp.allow_origin='*'
```

## Central Training Baseline

Simply run `./run_central_train.sh` for a central training baseline. This will combine all the generated site data and treat as one continuous dataset.

## Data Generation Documentation

Detailed technical documentation is available in the `docs/` directory:

| Document                                                     | Contents                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| [docs/pipeline.md](docs/pipeline.md)                         | End-to-end generation pipeline, CLI reference, and output conventions           |
| [docs/data-generation.md](docs/data-generation.md)           | Dependency graph, topological sort, providers, and attribute system             |
| [docs/anomaly-injection.md](docs/anomaly-injection.md)       | Fraud types 1-4, injection framework, overlap control, and probability thinning |
| [docs/configuration.md](docs/configuration.md)               | Per-site YAML schema, distribution parameters, and dataset generation specs     |
| [docs/rng.md](docs/rng.md)                                   | RNG architecture, distribution samplers, and reproducibility model              |
| [docs/exploration-notebook.md](docs/exploration-notebook.md) | Guide to the interactive exploration notebook                                   |

## Development

For development environment setup, running tests, linting, and custom package
index configuration, see [docs/development.md](docs/development.md).
