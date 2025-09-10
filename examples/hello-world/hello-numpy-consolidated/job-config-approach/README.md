# Hello NumPy - Job Config

This example demonstrates federated learning with NumPy using traditional FLARE job configuration files.

> **_NOTE:_** This example uses a Numpy-based trainer and will generate its data within the code.

## Quick Start

```bash
pip install -r requirements.txt
nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 jobs/hello-numpy-sag
```

## What This Example Does

This is a simple federated learning example using NumPy that demonstrates how multiple clients can collaboratively train a model without sharing their data. The model starts with weights `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` and each client adds 1 to each weight during training, showing how federated averaging works.

## How It Works

"[Scatter and Gather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html)" is the standard workflow to implement Federated Averaging ([FedAvg](https://arxiv.org/abs/1602.05629)). This workflow follows the hub and spoke model for communicating the global model to each client for local training (i.e., "scattering") and aggregates the result to perform the global model update (i.e., "gathering").

### Application Structure

Custom FL applications can contain the folders:
- **`jobs/`**: Contains the job configuration with client and server configs
- **`config/`**: Contains client and server configurations (`config_fed_client.json`, `config_fed_server.json`)

### Client Implementation

The code for the client and server components has been implemented in the `nvflare/app-common/np` folder of the NVFlare code tree. These files can be copied into a `custom` folder in the application and modified to perform additional tasks.

### Server Configuration

The server configuration (`config_fed_server.json`) includes:
- **Controller**: ScatterAndGather workflow
- **Model Persistor**: NPModelPersistor for saving/loading NumPy models
- **Aggregator**: InTimeAccumulateWeightedAggregator for combining client updates

### Client Configuration

The client configuration (`config_fed_client.json`) includes:
- **Executor**: NPTrainer for running the training script
- **Tasks**: Configured for the "train" task

## Files

- **`jobs/hello-numpy-sag/`**: Traditional FLARE job configuration
- **`hello_numpy_sag.ipynb`**: Interactive notebook example

## Installation

Follow the [Installation](../../getting_started/README.md) instructions:

```bash
pip install nvflare
pip install -r requirements.txt
```

## Expected Output

You can find the running logs and results inside the simulator's workspace:

```bash
ls /tmp/nvflare/hello-numpy-sag/simulate_job/
# app_server  app_site-1  app_site-2  log.txt  model  models
```

## Job Configuration Approach

This approach uses traditional FLARE job configuration files (JSON) to define the federated learning job. It's useful for:
- Understanding how FLARE jobs are structured and configured
- Learning traditional FLARE patterns and workflows
- Reference for custom job configuration

## Next Steps

For new development, use the recipe API or job API approaches instead. For learning FLARE job configuration, see the [notebook](hello_numpy_sag.ipynb).
