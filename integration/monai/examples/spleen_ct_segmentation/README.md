# 3D Spleen CT Segmentation - Simulation

This example demonstrates federated learning with [MONAI Bundles](https://monai.readthedocs.io/en/1.3.1/bundle_intro.html) using NVFlare's Client API and FedAvgRecipe.

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

**Note:** Full training takes several hours. For quick testing, reduce `--num_rounds` to 5-10.

```bash
python job.py --n_clients 2 --num_rounds 10 --local_epochs 10 --tracking "tensorboard"
```

Optional arguments:
- `--n_clients`: Number of federated clients (default: 2)
- `--num_rounds`: Number of FL rounds (default: 100)
- `--local_epochs`: Local training epochs per round (default: 10)
- `--threads`: Parallel threads for simulation (default: 2)
- `--workspace`: NVFlare workspace directory (default: /tmp/nvflare/simulation)

### TensorBoard Monitoring

```bash
tensorboard --logdir /tmp/nvflare/simulation/spleen_bundle_fedavg
```

## Collect Federated Statistics

To compute dataset statistics before training:

# Prepare data and bundle for statistics job

Download the bundle (if you haven't already downloaded above)
```bash
python3 -m monai.bundle download --name "spleen_ct_segmentation" --version "0.5.4" --bundle_dir ./bundles
```

Update the data path in the downloaded bundle
```bash
sed -i 's|/workspace/data/Task09_Spleen|/tmp/MONAI/data/Task09_Spleen|g' ./bundles/spleen_ct_segmentation/configs/*.json
```

Run statistics job
```bash
python job_stats.py --n_clients 2
```

Results are saved in the workspace under `/tmp/nvflare/simulation/spleen_bundle_stats/server/simulate_job/statistics/image_statistics.json`.

For visualization the results, see [stats_demo.ipynb](./stats_demo.ipynb).

## Architecture

This example uses:
- **Client API**: Simple Python API for FL training (see `client.py`)
- **FedAvgRecipe**: Pre-configured FedAvg workflow (see `job.py`)
- **MONAI Bundle**: Standard MONAI model and configuration
- **FedStatsRecipe**: Federated statistics collection (see `job_stats.py`)

The training script (`client.py`) uses NVFlare's Client API with MONAI's bundle configuration, eliminating the need for custom executors or the deprecated `monai_nvflare` package.
