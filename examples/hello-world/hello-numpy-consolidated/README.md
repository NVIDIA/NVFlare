# Hello FedAvg with NumPy

This example demonstrates federated learning with NumPy using NVIDIA FLARE. Multiple clients collaboratively train a model without sharing their data.

## Quick Start

**Recommended for new users:**
```bash
cd recipe-api-approach
pip install -r requirements.txt
python job.py
```

**Interactive learning with notebooks:**
```bash
cd job-api-approach
pip install -r requirements.txt
jupyter lab
```

## What This Example Does

- **Model**: Simple NumPy array with weights `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`
- **Training**: Each client adds 1 to each weight (simulating local training)
- **Aggregation**: Server averages the client updates using FedAvg
- **Result**: You'll see weights increase by 1 each round

## Directory Structure

```
hello-fedavg-numpy/
├── recipe-api-approach/       # Modern approach (recommended)
│   ├── client.py              # Client training script
│   ├── model.py               # NumPy model definition
│   ├── job.py                 # Job configuration
│   └── requirements.txt
├── job-api-approach/          # Detailed learning approach
│   ├── fedavg_script_runner_hello-numpy.py
│   ├── hello-fedavg-numpy_getting_started.ipynb
│   ├── hello-fedavg-numpy_flare_api.ipynb
│   └── requirements.txt
└── job-config-approach/       # Job configuration files (reference)
    ├── jobs/hello-numpy-sag/
    └── hello_numpy_sag.ipynb
```

## Choose Your Approach

### Recipe API (Recommended)
- **Best for**: New users, modern development
- **Files**: `client.py`, `model.py`, `job.py`
- **Why**: Clean, simple, follows FLARE best practices

### Job API (Learning)
- **Best for**: Understanding FLARE internals
- **Files**: Scripts + Jupyter notebooks
- **Why**: Interactive tutorials, detailed explanations

### Job Config (Reference)
- **Best for**: Understanding job configuration, legacy migration
- **Files**: JSON configuration files + job structure
- **Why**: Shows traditional FLARE job configuration patterns with `config_fed_client.json` and `config_fed_server.json`

## Installation

```bash
pip install nvflare numpy
```

## What Each Approach Provides

### Recipe API Approach
- **Modern Python code**: Clean `client.py`, `model.py`, `job.py` structure
- **Custom NumPy recipe**: `NumpyFedAvgRecipe` designed specifically for NumPy models
- **Easy to understand**: Follows the same pattern as other hello-world examples
- **Best for**: New users, modern development, learning FLARE concepts

### Job API Approach  
- **Interactive learning**: Jupyter notebooks with step-by-step tutorials
- **Detailed explanations**: Shows how to build jobs programmatically
- **FLARE internals**: Understanding of `FedJob`, `ScriptRunner`, and client API
- **Best for**: Learning FLARE internals, understanding job configuration

### Job Config Approach
- **Traditional structure**: Complete FLARE job with JSON configuration files
- **Configuration examples**: `config_fed_client.json` and `config_fed_server.json`
- **Legacy patterns**: Shows how FLARE jobs were traditionally configured
- **Best for**: Understanding job configuration, legacy migration, reference

## Next Steps

1. Run the recipe API example
2. Explore the notebooks for deeper understanding
3. Try other hello-world examples (hello-pt, hello-tf)
4. Move to advanced examples when ready
