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

Navigate to the hello-fedavg-numpy directory:

```bash
    git switch <release branch>
    cd examples/hello-world/hello-fedavg-numpy
```

Install the dependencies:

```bash
  pip install -r requirements.txt
```

## Code Structure

```
    hello-fedavg-numpy
    |
    |-- client.py         # client local training script
    |-- model.py          # model definition
    |-- server.py         # server recipe that defines client and server configurations
    |-- requirements.txt  # dependencies
```

## What This Example Does

- **Model**: Simple NumPy array with weights `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`
- **Training**: Each client adds 1 to each weight (simulating local training)
- **Aggregation**: Server averages the client updates using FedAvg
- **Result**: You'll see weights increase by 1 each round

## Files

- **`client.py`** - Client-side training script that receives models, trains locally, and sends updates
- **`model.py`** - Simple NumPy model definition with training and evaluation methods
- **`server.py`** - Server-side script that creates and runs the federated learning job using the Recipe API
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
- **`SimpleNumpyModel`**: Basic NumPy model with trainable weights
- **`SimEnv`**: Simulation environment for running FL jobs locally
- **`ScriptRunner`**: Executes client training scripts with proper framework configuration

## Customization

You can modify the example by:

- **Changing the model**: Edit `model.py` to use your own NumPy model
- **Adjusting training**: Modify the `train_step()` method in `SimpleNumpyModel`
- **Adding more clients**: Use `--n_clients` parameter
- **Changing rounds**: Use `--num_rounds` parameter
- **Adjusting learning rate**: Use `--learning_rate` parameter

## Example Output

```
Client site-1 initialized
Client site-2 initialized
Client site-1, current_round=1
Received weights: {'numpy_key': array([[1., 2., 3.],
       [4., 5., 6.],
       [7., 8., 9.]], dtype=float32)}
Client site-1 starting training...
Client site-1 evaluation metrics: {'accuracy': 5.0}
Client site-1 finished training for round 1
Sending weights: [[2. 3. 4.]
 [5. 6. 7.]
 [8. 9. 10.]]
```

## Run the Experiment

Execute the script using the recipe API to create the job and run it with the simulator:

```bash
python server.py
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
