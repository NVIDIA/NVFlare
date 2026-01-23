# Hello FedAvg with NumPy

This example demonstrates federated learning with NumPy using NVIDIA FLARE's Recipe API. Multiple clients collaboratively train a model without sharing their data.

## NVIDIA FLARE Installation

For complete installation instructions, visit [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

```bash
  pip install nvflare
```

Clone the example code from GitHub:

```bash
  git clone https://github.com/NVIDIA/NVFlare.git
```

Navigate to the hello-numpy directory:

```bash
    git switch <release branch>
    cd examples/hello-world/hello-numpy
```

Install the dependencies:

```bash
  pip install -r requirements.txt
```

## Code Structure

```
    hello-numpy
    |
    |-- client.py         # client local training script
    |-- job.py            # creates the FL recipe and executes it using SimEnv
    |-- requirements.txt  # dependencies
```

## What This Example Does

- **Model**: Simple NumPy array with weights `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`
- **Training**: Each client adds 1 to each weight (simulating local training)
- **Aggregation**: Server averages the client updates using FedAvg
- **Result**: You'll see weights increase by 1 each round

## Files

- **`client.py`** - Client-side training script that receives models, trains locally, and sends updates
- **`job.py`** - Script that creates and runs the federated learning job using the Recipe API
- **`requirements.txt`** - Python dependencies

## How It Works

### Recipe API Approach

This example uses NVIDIA FLARE's **Recipe API**, which provides a high-level, declarative way to define federated learning jobs. The Recipe API:

1. **Hides Complexity**: No need to manually configure job components, workflows, or communication patterns
2. **Framework-Specific**: Uses `NumpyFedAvgRecipe` designed specifically for NumPy models
3. **Simple Configuration**: Just specify the model, training script, and parameters
4. **Easy to Understand**: Follows the same pattern as other hello-world examples

### The Federated Learning Process

1. **Initialization**: Server creates initial model and distributes it to clients
2. **Local Training**: Each client receives the model, trains it locally, and sends updates back
3. **Aggregation**: Server averages all client updates using weighted FedAvg
4. **Iteration**: Process repeats for the specified number of rounds

### Key Components

- **`NumpyFedAvgRecipe`**: Custom recipe for NumPy models (located in `nvflare.app_common.np.recipes`)
- **`SimEnv`**: Simulation environment for running FL jobs locally


## Customization

You can modify the example by:

- **Adjusting training**: Modify the `train()` method in `client.py`
- **Adding more clients**: Use `--n_clients` parameter
- **Changing rounds**: Use `--num_rounds` parameter


## Run the Experiment

Execute the script using the recipe API to create the job and run it with the simulator:

```bash
python job.py
```

## Access the Logs and Results

Find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/simulation/hello-numpy/server/simulate_job
```

## Next Steps

1. Run this example to understand the basics
2. Try other hello-world examples (hello-pt, hello-tf)
3. Explore advanced examples when ready
4. Learn about custom recipes and workflows
