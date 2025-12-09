# ML to FL with NumPy

This example demonstrates how to convert a simple NumPy-based ML training script to federated learning using NVFlare's Client API and Recipe pattern.

## Software Requirements

Please install the requirements first (recommended to install inside a virtual environment):

```bash
pip install -r requirements.txt
```

## Minimum Hardware Requirements

1 CPU

## Quick Start

Run the default FedAvg training with 2 clients:

```bash
python job.py
```

## Project Structure

```
np/
├── job.py         # Job configuration using NumpyFedAvgRecipe
├── client.py      # Client training script (supports full/diff modes and metrics tracking)
├── README.md
└── requirements.txt
```

## Job Configuration

The `job.py` uses the `NumpyFedAvgRecipe` to configure a complete federated learning workflow:

```python
recipe = NumpyFedAvgRecipe(
    name="np_client_api",
    initial_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # Initial model as list
    min_clients=n_clients,
    num_rounds=num_rounds,
    train_script="client.py",
    train_args=train_args,
)
```

### Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_clients` | Number of clients | 2 |
| `--num_rounds` | Number of training rounds | 5 |
| `--update_type` | Parameter transfer mode: `full` or `diff` | `full` |
| `--metrics_tracking` | Enable MLflow metrics streaming | False |
| `--launch_process` | Launch client in external process | False |
| `--export_config` | Export job config instead of running | False |

## Training Modes

### Full Model Transfer (Default)

Send complete model parameters back to the server:

```bash
python job.py --update_type full
```

### Diff Model Transfer

Send only the parameter differences (delta) back to the server.

```bash
python job.py --update_type diff
```

## Metrics Streaming

Enable MLflow metrics tracking to monitor training progress:

```bash
python job.py --metrics_tracking
```

This uses the `MLflowWriter` in the client script to log metrics during training:

```python
if args.metrics_tracking:
    from nvflare.client.tracking import MLflowWriter
    writer = MLflowWriter()
    # ... in training loop:
    writer.log_metric(key="global_step", value=global_step, step=global_step)
```

After the experiment finishes, view results with:

```bash
mlflow ui --port 5000
```

Navigate to the MLflow tracking directory shown in the output.

### Other Tracking Options

NVFlare supports multiple tracking backends:
- `SummaryWriter` - TensorBoard compatible
- `WandBWriter` - Weights & Biases compatible
- `MLflowWriter` - MLflow compatible

## In-Process vs Sub-Process Execution

### In-Process (Default)

The client script runs within the same process as the NVFlare job. Best for development and single-GPU scenarios:

```bash
python job.py
```

### Sub-Process (External Process)

Launch the client script as a separate process. Ideal for multi-GPU or distributed training:

```bash
python job.py --launch_process
```

## Export Job Configuration

Export the job configuration to a folder instead of running it (useful for deployment):

```bash
python job.py --export_config
```

The configuration will be saved to `/tmp/nvflare/jobs/job_config`.

## Example Commands

```bash
# Basic run with defaults
python job.py

# Run with "diff" and 3 clients for 10 rounds
python job.py --update_type diff --n_clients 3 --num_rounds 10

# Run with metrics tracking enabled
python job.py --metrics_tracking

# Run in external process mode with metrics
python job.py --launch_process --metrics_tracking

# Export configuration only
python job.py --export_config
```

## Client Script Details

The `client.py` implements the standard NVFlare Client API pattern:

1. **Initialize**: `flare.init()`
2. **Training Loop**: `while flare.is_running()`
3. **Receive Model**: `input_model = flare.receive()`
4. **Train Locally**: Update model with local data
5. **Send Results**: `flare.send(flare.FLModel(...))`

The script supports both `full` and `diff` update type via the `--update_type` argument, and optional metrics tracking via `--metrics_tracking`.
