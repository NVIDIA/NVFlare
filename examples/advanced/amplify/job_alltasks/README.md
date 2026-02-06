# AMPLIFY All-Tasks Federated Fine-tuning

This example demonstrates how to use NVIDIA FLARE with the AMPLIFY protein language model for federated all-tasks fine-tuning. In this scenario, each client trains on **all downstream tasks** with heterogeneous data distributions. Clients can choose to either share regression heads jointly or keep them private.

## Installation

For the complete NVIDIA FLARE installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)

```bash
pip install nvflare
```

Install AMPLIFY and dependencies. Note that AMPLIFY was only tested with Python 3.11:

```bash
git clone https://github.com/chandar-lab/AMPLIFY
python3.11 -m venv env && \
source env/bin/activate && \
python3 -m pip install --upgrade pip && \
python3 -m pip install --editable AMPLIFY[dev]
```

Install the example dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH="${PWD}/src"
```

## Code Structure

Navigate to the job_alltasks directory:

```bash
cd examples/advanced/amplify/job_alltasks
```

```
job_alltasks/
|
|-- client.py             # client local training script for all tasks
|-- job.py                # job recipe that defines client and server configurations
|-- README.md             # this file
|
../src/
|-- model.py              # AmplifyRegressor model definition
|-- filters.py            # ExcludeParamsFilter to keep regressors private (optional)
|-- utils.py              # data loading utilities
|-- combine_data.py       # data preparation script with Dirichlet splitting
```

## Data

This example uses antibody fitness datasets from [FLAb](https://github.com/Graylab/FLAb) for six downstream tasks:
- Aggregation
- Binding affinity
- Expression
- Immunogenicity
- Polyreactivity
- Thermostability

### Data Preparation

First, clone the FLAb repository:

```bash
cd examples/advanced/amplify
git clone https://github.com/Graylab/FLAb.git
```

Then prepare and split the data for all tasks using Dirichlet distribution:

```bash
for task in "aggregation" "binding" "expression" "immunogenicity" "polyreactivity" "thermostability" 
do
    echo "Combining $task CSV data"
    python src/combine_data.py \
        --input_dir ./FLAb/data/${task} \
        --output_dir ./FLAb/data_fl/${task} \
        --num_clients 6 \
        --alpha 1.0
done
```

This will:
1. Combine the 'heavy' and 'light' antibody sequences with a '|' separator
2. Split the data into training (80%) and test (20%) sets
3. **Distribute training data heterogeneously across 6 clients** using Dirichlet distribution (alpha=1.0)
4. Save processed data to `./FLAb/data_fl/{task}/`

Each task directory will contain:
- `client1_train_data.csv`, `client2_train_data.csv`, ..., `client6_train_data.csv`: Client-specific training data
- `test_data.csv`: Test data (shared across all clients for evaluation)

The Dirichlet distribution with `alpha=1.0` creates heterogeneous data splits where each client receives different amounts of data for each task, simulating real-world federated scenarios with data imbalance.

## Model

The `AmplifyRegressor` model consists of two main components:

1. **AMPLIFY Trunk**: Pre-trained protein language model from [chandar-lab/AMPLIFY](https://huggingface.co/chandar-lab/AMPLIFY_120M)
2. **Regression Heads**: Multiple task-specific MLP layers (one per task)

In this all-tasks scenario:
- Each client has **6 regression heads** (one for each task)
- The AMPLIFY trunk is **shared and jointly fine-tuned** across all clients
- Regression heads can be:
  - **Shared**: Jointly trained and aggregated (default behavior)
  - **Private**: Kept local to each client using `--private_regressors` flag

The model implementation can be found in [`../src/model.py`](../src/model.py).

## Client Code

The client code [`client.py`](client.py) handles local training for all tasks simultaneously. The key steps are:

1. **Initialize NVFlare Client API**: `flare.init()`
2. **Determine client ID**: Extract from client name (e.g., "site-1" → client1)
3. **Load data for all tasks**: Load client-specific training data for each of the 6 tasks
4. **Training loop**:
   - Receive global model from server
   - Perform local training on all tasks
   - Evaluate on all test sets
   - Send updated model back to server

The client uses the NVFlare Client API:

```python
import nvflare.client as flare

# Initialize NVFlare client
flare.init()
client_name = flare.get_site_name()  # e.g., "site-1"

# Load data for all 6 tasks
for task in TASKS:
    train_csv = f"./FLAb/data_fl/{task}/client{client_id}_train_data.csv"
    test_csv = f"./FLAb/data_fl/{task}/test_data.csv"
    # Load datasets...

# Main federated learning loop
while flare.is_running():
    # Receive global model from server
    input_model = flare.receive()
    
    # Load global model parameters
    model.load_state_dict(input_model.params, strict=False)
    
    # Local training on all tasks...
    for task_idx, task in enumerate(TASKS):
        for batch in dataloader_train[task_idx]:
            # Train with specific regressor for this task
            output = model(input_ids, attention_mask, regressor_idx=task_idx)
            # ...
    
    # Send updated model back to server
    output_model = flare.FLModel(
        params=model.cpu().state_dict(),
        metrics={"RMSE": avg_rmse, "Pearson": avg_pearson}
    )
    flare.send(output_model)
```

## Server-Side Workflow

This example uses the [`FedAvgRecipe`](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html), which implements the [FedAvg](https://arxiv.org/abs/1602.05629) algorithm. The Recipe API handles all server-side logic automatically:

1. Initialize the global model (AMPLIFY trunk + regression heads)
2. For each training round:
   - Send the global model to all clients
   - Wait for client updates
   - **Optionally filter out private regression heads** (if `--private_regressors` flag is used)
   - Aggregate model parameters across clients
   - Update the global model

With the Recipe API, **there is no need to write custom server code**. The federated averaging workflow is provided by NVFlare.

### Sharing Modes

**1. Shared Regressors (Default)**
- All regression heads are jointly trained and aggregated
- Best when all clients jointly own the model IP
- Typically achieves better performance

**2. Private Regressors (`--private_regressors`)**
- Regression heads remain private to each client
- Only the AMPLIFY trunk is aggregated
- Best when clients want personalized task models
- Uses [`ExcludeParamsFilter`](../src/filters.py):

```python
# In job.py
if args.private_regressors:
    recipe.add_client_output_filter(
        ExcludeParamsFilter(exclude_vars="regressor"), 
        tasks=["train", "validate"]
    )
```

## Job Recipe Code

The [`job.py`](job.py) script configures the federated learning job:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking
from src.filters import ExcludeParamsFilter
from src.model import AmplifyRegressor

# Six tasks
TASKS = ["aggregation", "binding", "expression", "immunogenicity", 
         "polyreactivity", "thermostability"]

# Build initial model with multiple regressors
model = AmplifyRegressor(
    pretrained_model_name_or_path="chandar-lab/AMPLIFY_120M",
    layer_sizes=[128, 64, 32],
    num_targets=len(TASKS)  # 6 regression heads
)

# Create FedAvg recipe
recipe = FedAvgRecipe(
    name="amplify_alltasks_fedavg",
    min_clients=num_clients,
    num_rounds=300,
    # Model can be class instance or dict config
    # For pre-trained weights: initial_ckpt="/server/path/to/pretrained.pt"
    initial_model=model,
    train_script="client.py",
    train_args="--data_root ./FLAb/data_fl --tasks <all tasks> ..."
)

# Add TensorBoard tracking
add_experiment_tracking(recipe, tracking_type="tensorboard")

# Optionally add filter for private regressors
if args.private_regressors:
    recipe.add_client_output_filter(
        ExcludeParamsFilter(exclude_vars="regressor"), 
        tasks=["train", "validate"]
    )

# Run simulation
env = SimEnv(
    num_clients=num_clients,
    workspace_root="/tmp/nvflare/AMPLIFY/alltasks",
    gpu_config="0,1,2,0,1,2"  # 6 clients on 3 GPUs
)
recipe.execute(env)
```

## Run Job

### Local Training (Baseline)

Run 1 round with 600 local epochs to establish a baseline:

```bash
cd job_alltasks
python job.py \
    --num_clients 6 \
    --num_rounds 1 \
    --local_epochs 600 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "local_alltasks" \
    --sim_gpus "0,1,2,0,1,2" \
    --max_samples 1000
```

### Federated Learning with Shared Regressors

Run 300 rounds with 2 local epochs per round, sharing all model components:

```bash
cd job_alltasks
python job.py \
    --num_clients 6 \
    --num_rounds 300 \
    --local_epochs 2 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "fedavg_alltasks" \
    --sim_gpus "0,1,2,0,1,2" \
    --max_samples 1000
```

### Federated Learning with Private Regressors

Run 300 rounds with private regression heads (only AMPLIFY trunk is shared):

```bash
cd job_alltasks
python job.py \
    --num_clients 6 \
    --num_rounds 300 \
    --local_epochs 2 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "fedavg_alltasks_private" \
    --private_regressors \
    --sim_gpus "0,1,2,0,1,2" \
    --max_samples 1000
```

**Arguments:**
- `--num_clients`: Number of clients to simulate (default: 6)
- `--num_rounds`: Number of federated learning rounds
- `--local_epochs`: Number of local training epochs per round
- `--pretrained_model`: AMPLIFY model to use (120M or 350M)
- `--layer_sizes`: Regression MLP architecture (comma-separated)
- `--exp_name`: Experiment name for organizing results
- `--private_regressors`: Keep regression heads private (don't aggregate)
- `--sim_gpus`: GPU allocation for simulated clients
- `--max_samples`: Limit samples per client (for quick testing; omit for full training)

> **Note:** The `--max_samples` parameter is useful for quick testing. Remove it for full training runs.

## Results

### Visualize Training Curves

The experiment results are saved in `/tmp/nvflare/AMPLIFY/alltasks`. You can visualize the training metrics with TensorBoard:

```bash
tensorboard --logdir /tmp/nvflare/AMPLIFY/alltasks
```

Or generate plots comparing different approaches:

```bash
# Plot Pearson correlation for a specific task
python ../figs/plot_training_curves.py \
    --log_dir /tmp/nvflare/AMPLIFY/alltasks \
    --output_dir ../figs/tb_figs_pearson_alltasks \
    --tag "Pearson/local_test_expression" \
    --out_metric "Pearson expression"
```

### Expected Outcomes

**Comparison across three scenarios:**

1. **Local Training**: Each client trains only on their local data
2. **FedAvg with Shared Regressors**: Both trunk and regression heads are jointly trained
3. **FedAvg with Private Regressors**: Only trunk is shared, regression heads stay private

**Typical observations:**
- **Shared regressors** achieve the best performance by leveraging all client data for both trunk and regression heads
- **Private regressors** still benefit from the jointly trained trunk while maintaining personalized regression heads
- **Both FedAvg approaches** outperform local-only training due to collaborative learning
- **Data heterogeneity** (from Dirichlet split) makes federated learning more challenging but realistic

## Output Summary

#### Initialization
* **TensorBoard**: Logs available at `/tmp/nvflare/AMPLIFY/alltasks/amplify_alltasks_*/server/simulate_job/tb_events`
* **Workflow**: FedAvg controller initialized with specified number of clients
* **Model**: AMPLIFY trunk + 6 regression heads initialized

#### Each Round
* **Clients Sampled**: All clients participate (e.g., site-1, site-2, ..., site-6)
* **Training**:
  * Global model sent to all clients
  * Each client performs local training on all 6 tasks with heterogeneous data
  * Clients evaluate on all test sets and report per-task metrics (RMSE, Pearson correlation)
* **Filtering**: If `--private_regressors`, regression head parameters excluded via `ExcludeParamsFilter`
* **Aggregation**: 
  * Shared regressors mode: All parameters aggregated
  * Private regressors mode: Only AMPLIFY trunk aggregated
* **Persistence**: Updated global model persisted on server

#### Completion
* **Final Model**: 
  * Shared mode: Globally fine-tuned AMPLIFY trunk + regression heads
  * Private mode: Globally fine-tuned AMPLIFY trunk; each client retains their own regression heads
* **Performance**: Per-client, per-task metrics available in TensorBoard logs

## References

- AMPLIFY: [Protein language models: is scaling necessary?](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
- FLAb: [Benchmarking deep learning methods for antibody fitness prediction](https://www.biorxiv.org/content/10.1101/2024.01.13.575504v1)
- FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- Dirichlet Distribution for data heterogeneity: [Federated Learning on Non-IID Data](https://arxiv.org/abs/1806.00582)
