# Hello NumPy - Job API

This example showcases Federated Averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) with NumPy using NVIDIA FLARE's job API with interactive notebooks.

> **_NOTE:_** This example uses a NumPy-based trainer and will generate its data within the code.

## Quick Start

```bash
pip install -r requirements.txt
python fedavg_script_runner_hello-numpy.py
```

## Interactive Learning

Open Jupyter Lab for detailed tutorials:

```bash
jupyter lab
```

Then open:
- `hello-fedavg-numpy_getting_started.ipynb` - Step-by-step tutorial
- `hello-fedavg-numpy_flare_api.ipynb` - FLARE API examples

## What This Example Does

This is a simple federated learning example using NumPy that demonstrates how multiple clients can collaboratively train a model without sharing their data. The model starts with weights `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` and each client adds 1 to each weight during training, showing how federated averaging works.

## How It Works

### NVIDIA FLARE Job API

The `fedavg_script_runner_hello-numpy.py` script builds the job with the Job API. Key components:

**Define a FedJob:**
```python
job = FedJob(name="hello-fedavg-numpy")
```

**Define the Controller Workflow:**
```python
controller = FedAvg(
    num_clients=n_clients,
    num_rounds=num_rounds,
    persistor_id=persistor_id,
)
job.to(controller, "server")
```

**Add Clients:**
```python
for i in range(n_clients):
    executor = ScriptRunner(script=train_script, script_args="", framework=FrameworkType.NUMPY)
    job.to(executor, f"site-{i + 1}")
```

### Client Training Script

The training script `src/hello-numpy_fl.py` contains the NumPy-specific logic for training:

1. **Receive model** from the FL server
2. **Perform training** on the received global model and calculate metrics
3. **Send the new model** back to the FL server

Using NVFlare's Client API, there are three essential methods:
- `init()`: Initializes NVFlare Client API environment
- `receive()`: Receives model from the FL server
- `send()`: Sends the model to the FL server

## Files

- **`fedavg_script_runner_hello-numpy.py`**: Main script to run the example
- **`src/hello-numpy_fl.py`**: Client training code
- **`hello-fedavg-numpy_getting_started.ipynb`**: Interactive tutorial
- **`hello-fedavg-numpy_flare_api.ipynb`**: FLARE API examples

## Installation

Follow the [Installation](../../getting_started/README.md) instructions:

```bash
pip install nvflare
pip install -r requirements.txt
```

## Expected Output

You'll see weights increasing by 1 each round, demonstrating federated learning in action. The results are saved in the simulator's workspace:

```bash
ls /tmp/nvflare/jobs/workdir/
```

## Next Steps

1. Work through the notebooks for detailed understanding
2. Try the recipe API approach for simpler usage
3. Explore other hello-world examples
