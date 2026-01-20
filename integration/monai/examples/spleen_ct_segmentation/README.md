# 3D Spleen CT Segmentation - Simulation

This example demonstrates federated learning with [MONAI Bundles](https://monai.readthedocs.io/en/1.3.1/bundle_intro.html) using NVFlare's Client API and Recipes.

## Directory Structure

```
spleen_ct_segmentation/
├── job_fedavg/              # Federated training with FedAvg
│   ├── job.py              # Job configuration and launcher
│   └── client.py           # Client training script
├── job_stats/              # Federated statistics collection
│   ├── job.py              # Statistics job configuration
│   └── client.py           # Client statistics script
├── bundles/                # MONAI bundles (downloaded)
├── model.py                # Model definition (FLUNet)
├── client_stats.py         # Statistics generator implementation
└── download_spleen_dataset.py  # Data download script
```

## Setup

```bash
pip install -r requirements.txt
```

### 1. Download the Spleen Bundle

```bash
python3 -m monai.bundle download --name "spleen_ct_segmentation" --version "0.5.4" --bundle_dir ./bundles
```

### 2. Download Data

```bash
python3 download_spleen_dataset.py --output_dir /tmp/MONAI/data
```

Update the data path in the downloaded bundle
```bash
sed -i 's|/workspace/data/Task09_Spleen|/tmp/MONAI/data/Task09_Spleen|g' ./bundles/spleen_ct_segmentation/configs/*.json
```


## Run Federated Training

### Basic Training with FedAvg

**Note:** Full training might take several hours. For quick testing, we reduce `--num_rounds` to 5.

```bash
python job_fedavg/job.py --n_clients 2 --num_rounds 3 --local_epochs 1 --tracking "tensorboard"
```

Optional arguments:
- `--n_clients`: Number of federated clients (default: 2)
- `--num_rounds`: Number of FL rounds (default: 10)
- `--local_epochs`: Local training epochs per round (default: 1)
- `--threads`: Parallel threads for simulation (default: 2)
- `--workspace`: NVFlare workspace directory (default: /tmp/nvflare/simulation)
- `--send_weight_diff`: Send weight differences instead of full weights
- `--tracking`: Experiment tracking type (tensorboard, mlflow, both, none)

### TensorBoard Monitoring

```bash
tensorboard --logdir /tmp/nvflare/simulation/spleen_bundle_fedavg
```

## Collect Federated Statistics

To compute dataset statistics before training:

### Prepare data and bundle for statistics job

Download the bundle (if you haven't already downloaded above)
```bash
python3 -m monai.bundle download --name "spleen_ct_segmentation" --version "0.5.4" --bundle_dir ./bundles
```

Update the data path in the downloaded bundle
```bash
sed -i 's|/workspace/data/Task09_Spleen|/tmp/MONAI/data/Task09_Spleen|g' ./bundles/spleen_ct_segmentation/configs/*.json
```

### Run statistics job

```bash
python job_stats/job.py --n_clients 2
```

Results are saved in the workspace under `/tmp/nvflare/simulation/spleen_bundle_stats/server/simulate_job/statistics/image_statistics.json`.

For visualization of the results, see [stats_demo.ipynb](./stats_demo.ipynb).

## Architecture

This example uses:
- **Client API**: Simple Python API for FL training (see `job_fedavg/client.py`)
- **FedAvgRecipe**: Pre-configured FedAvg workflow (see `job_fedavg/job.py`)
- **MONAI Bundle**: Standard MONAI model and configuration
- **MonaiAlgo**: MONAI's federated learning client algorithm for bundle integration
- **FedStatsRecipe**: Federated statistics collection (see `job_stats/job.py`)

### FedAvg Training Workflow

The `job_fedavg/` directory contains:
- `job.py`: Configures and launches the FedAvg experiment using `FedAvgRecipe`
- `client.py`: Client-side training logic using MONAI's `MonaiAlgo` with NVFlare's Client API

### Statistics Collection Workflow

The `job_stats/` directory contains:
- `job.py`: Configures and launches the statistics collection using `FedStatsRecipe`
- `client.py`: Client-side statistics collection

The training script (`job_fedavg/client.py`) uses NVFlare's Client API with MONAI's `MonaiAlgo` for seamless bundle integration, eliminating the need for custom executors or the deprecated `monai_nvflare` package.
