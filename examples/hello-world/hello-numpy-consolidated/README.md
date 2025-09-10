# Hello NumPy - Consolidated Examples

This directory contains consolidated NumPy examples for NVIDIA FLARE, showcasing the evolution of FLARE APIs and providing multiple approaches for different use cases. Whether you're new to federated learning or migrating from older FLARE versions, this collection has something for you.

## What is This Example?

This is a simple federated learning example using NumPy that demonstrates how multiple clients can collaboratively train a model without sharing their data. The model starts with weights `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` and each client adds 1 to each weight during training, showing how federated averaging works.

## Directory Structure

```
hello-numpy-consolidated/
├── recipe-api-approach/       # NEW: Modern recipe API (recommended for new users)
│   ├── client.py              # Client training script
│   ├── model.py               # NumPy model definition
│   ├── job.py                 # Job recipe using NumpyFedAvgRecipe
│   ├── numpy_fedavg_recipe.py # Custom NumPy-specific recipe
│   ├── requirements.txt
│   └── README.md
├── job-api-approach/          # Modern job API approach
│   ├── fedavg_script_runner_hello-numpy.py
│   ├── hello-fedavg-numpy_flare_api.ipynb
│   ├── hello-fedavg-numpy_getting_started.ipynb
│   ├── README.md
│   ├── requirements.txt
│   └── src/
│       └── hello-numpy_fl.py
└── sag-approach/              # Legacy scatter-and-gather approach (for reference)
    ├── jobs/
    │   └── hello-numpy-sag/
    │       └── app/
    │           └── config/
    ├── hello_numpy_sag.ipynb
    ├── README.md
    └── requirements.txt
```

## Which Approach Should I Use?

### 🆕 Recipe API Approach (Recommended for New Users)
**Location**: `recipe-api-approach/`

This is the **newest and most modern** approach using FLARE's recipe API. It provides:
- **Clean, simple structure** following the same pattern as `hello-pt`
- **Custom NumPy recipe** (`NumpyFedAvgRecipe`) specifically designed for NumPy models
- **Minimal code** with clear separation of concerns
- **Best practices** for current FLARE development
- **Easy to understand** for beginners

**Best for**: New users, modern development, learning FLARE concepts

### Job API Approach (Good for Learning)
**Location**: `job-api-approach/`

This approach uses the job API with more explicit configuration:
- **Jupyter notebooks** for interactive learning
- **Detailed examples** of job configuration
- **Step-by-step tutorials** for understanding FLARE concepts
- **More verbose** but educational

**Best for**: Learning FLARE internals, understanding job configuration, interactive tutorials

### Legacy SAG Approach (Reference Only)
**Location**: `sag-approach/`

This is the original scatter-and-gather approach:
- **Historical reference** for understanding FLARE evolution
- **Traditional job config files** (JSON-based)
- **Legacy workflow patterns**
- **For reference** and migration understanding

**Best for**: Understanding FLARE history, migrating from older versions, reference

## Quick Start Guide

### Option 1: Recipe API (Recommended)
```bash
cd recipe-api-approach
pip install -r requirements.txt
python job.py
```

### Option 2: Job API with Notebooks
```bash
cd job-api-approach
pip install -r requirements.txt
# Open Jupyter and run the notebooks, or:
python fedavg_script_runner_hello-numpy.py
```

### Option 3: Legacy SAG (Reference)
```bash
cd sag-approach
pip install -r requirements.txt
nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 jobs/hello-numpy-sag
```

## Understanding the Evolution

### Recipe API (Latest)
- **Files**: `job.py` + `numpy_fedavg_recipe.py` - Clean separation
- **API**: `NumpyFedAvgRecipe` - High-level, declarative, NumPy-specific
- **Structure**: `client.py`, `model.py`, `job.py` - Clean separation
- **Learning curve**: Gentle, follows familiar patterns

### Job API (Modern)
- **Files**: Multiple files with explicit configuration
- **API**: `FedJob` - More control, more verbose
- **Structure**: Scripts + notebooks + configuration
- **Learning curve**: Steeper, more educational

### SAG (Legacy)
- **Files**: JSON configuration files + custom code
- **API**: Direct workflow components
- **Structure**: Traditional FLARE app structure
- **Learning curve**: Steepest, most complex

## What You'll Learn

Regardless of which approach you choose, you'll learn:
- How federated learning works in practice
- How clients train models locally
- How servers aggregate model updates
- How FLARE coordinates the process
- How to monitor and debug FL training

## Next Steps

1. **Start with Recipe API** if you're new to FLARE
2. **Try Job API** if you want to understand more details
3. **Reference SAG** if you're migrating from older versions
4. **Move to advanced examples** once you understand the basics

## Migration Path

- **From SAG → Job API**: Use the job API approach to understand modern patterns
- **From Job API → Recipe API**: Use the recipe API for cleaner, more maintainable code
- **New to FLARE**: Start with recipe API, then explore others for deeper understanding

This consolidated structure preserves the evolution of FLARE APIs while providing clear guidance for users at different stages of their FLARE journey.
