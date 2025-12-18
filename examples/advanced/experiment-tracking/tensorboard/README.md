# Hello PyTorch with TensorBoard Streaming

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework with **TensorBoard experiment tracking**.

This example demonstrates the **Recipe API** for easily adding TensorBoard streaming to FL training jobs.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

## What's New: Recipe API + Experiment Tracking

This example uses the new `FedAvgRecipe` combined with the `add_experiment_tracking()` utility:

```python
from nvflare.app_opt.pt.recipes import FedAvgRecipe
from nvflare.recipe.utils import add_experiment_tracking

# Create training recipe
recipe = FedAvgRecipe(
    name="fedavg_tensorboard",
    min_clients=2,
    num_rounds=5,
    initial_model=SimpleNetwork(),
    train_script="src/train_script.py",
)

# Add TensorBoard tracking with one line!
add_experiment_tracking(recipe, "tensorboard", tracking_config={"tb_folder": "tb_events"})

# Run
recipe.run()
```

Benefits:
- **70% less code** compared to manual FedJob configuration
- **Cleaner separation** between training workflow and experiment tracking
- **Easy to switch** tracking backends (just change "tensorboard" to "mlflow" or "wandb")

### 1. Install requirements

Install additional requirements:

Assuming the current directory is `examples/advanced/experiment-tracking/tensorboard`, run the following command to install the requirements:

```
python -m pip install -r requirements.txt
```

### 2. Download data
Here we just use the same data for each site. It's better to pre-download the data to avoid multiple sites concurrently downloading the same data.

again, we are assuming the current directory is `examples/advanced/experiment-tracking/tensorboard`,

```bash
examples/advanced/experiment-tracking/prepare_data.sh
```
### 3. Run the experiment

Navigate to the job directory and run:

```bash
cd ./jobs/tensorboard-streaming/code
python3 job.py
```


### 4. Access the logs and results

You can find the running logs and results inside the simulator's workspace/`<server name>`/simulate_job

The workspace in `job.py` is defined as `/tmp/nvflare/jobs/workdir`:

```
The results will be at:

```bash
$ tree /tmp/nvflare/jobs/workdir/server/simulate_job/

/tmp/nvflare/jobs/workdir/server/simulate_job/
├── app_server
 <... skip ...>
└── tb_events
    ├── site-1
    │ └── events.out.tfevents.1744857479.rtx.30497.0
    └── site-2
      └── events.out.tfevents.1744857479.rtx.30497.1

```


### 5. View TensorBoard Results

To view training metrics that are being streamed to the server, run:


```bash
tensorboard --logdir=/tmp/nvflare/jobs/workdir/server/simulate_job/tb_events
```

Then open your browser to `http://localhost:6006` to view the metrics.

**Note**: If the server is running on a remote machine, use port forwarding:
```bash
ssh -L 6006:127.0.0.1:6006 user@server_ip
```

---

## How It Works

### Client-Side Tracking

In `train_script.py`, the client code uses the NVFlare tracking API:

```python
from nvflare.client.tracking import SummaryWriter

# Create writer
summary_writer = SummaryWriter()

# Log metrics
summary_writer.add_scalar("train_loss", loss, global_step=epoch)
summary_writer.add_scalar("train_accuracy", accuracy, global_step=epoch)
```

This generates NVIDIA FLARE events of type `analytix_log_stats`.

### Server-Side Tracking

The `add_experiment_tracking()` utility automatically configures:
1. **`TBAnalyticsReceiver`** on the server - receives and writes metrics to TensorBoard files
2. **`ConvertToFedEvent`** on clients - converts local events to federated events for streaming

All metrics from all clients are aggregated into a single TensorBoard view on the server!

---

## Add TensorBoard to Your Own Recipe

Adding TensorBoard to any Recipe is simple:

```python
from nvflare.recipe.utils import add_experiment_tracking

# After creating your recipe
add_experiment_tracking(recipe, "tensorboard")

# Optional: customize the folder
add_experiment_tracking(
    recipe,
    "tensorboard",
    tracking_config={"tb_folder": "my_custom_folder"}
)
```

You can also switch to other tracking systems by changing the tracking type:
- `"tensorboard"` - TensorBoard streaming
- `"mlflow"` - MLflow tracking
- `"wandb"` - Weights & Biases tracking

---

## Additional Resources

- [Experiment Tracking Guide](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
- [TensorBoard Streaming Details](https://nvflare.readthedocs.io/en/main/examples/tensorboard_streaming.html)
- [Recipe API Documentation](https://nvflare.readthedocs.io/en/main/programming_guide/job_recipes.html)
