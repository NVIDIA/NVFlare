# AMPLIFY Multi-Task Federated Fine-tuning

This example demonstrates how to use NVIDIA FLARE with the AMPLIFY protein language model for federated multi-task fine-tuning. In this scenario, each client trains on a different downstream task (e.g., aggregation, binding, expression) while jointly fine-tuning the shared AMPLIFY trunk using federated averaging (FedAvg). Each client maintains their own private regression head that is not shared with the server.

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

Navigate to the job_multitask directory:

```bash
cd examples/advanced/amplify/job_multitask
```

```
job_multitask/
|
|-- client.py             # client local training script for multi-task
|-- job.py                # job recipe that defines client and server configurations
|-- README.md             # this file
|
../src/
|-- model.py              # AmplifyRegressor model definition
|-- filters.py            # ExcludeParamsFilter to keep regressors private
|-- utils.py              # data loading utilities
|-- combine_data.py       # data preparation script
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

Then prepare the data for each task by combining the 'heavy' and 'light' antibody sequences:

```bash
for task in "aggregation" "binding" "expression" "immunogenicity" "polyreactivity" "thermostability" 
do
    echo "Combining $task CSV data"
    python src/combine_data.py --input_dir ./FLAb/data/${task} --output_dir ./FLAb/data_fl/${task}
done
```

This will:
1. Combine the 'heavy' and 'light' columns with a '|' separator
2. Split the data into training (80%) and test (20%) sets
3. Save processed data to `./FLAb/data_fl/{task}/`

Each task directory will contain:
- `train_data.csv`: Training data
- `test_data.csv`: Test data (shared across all clients for evaluation)

## Model

The `AmplifyRegressor` model consists of two main components:

1. **AMPLIFY Trunk**: Pre-trained protein language model from [chandar-lab/AMPLIFY](https://huggingface.co/chandar-lab/AMPLIFY_120M)
2. **Regression Head**: Task-specific MLP layers for fitness prediction

In this multi-task scenario:
- Each client has a **single regression head** for their specific task
- The AMPLIFY trunk is **shared and jointly fine-tuned** across all clients
- Regression heads remain **private** to each client (not aggregated on the server)

The model implementation can be found in [`../src/model.py`](../src/model.py).

## Client Code

The client code [`client.py`](client.py) handles local training for a single task. The key steps are:

1. **Initialize NVFlare Client API**: `flare.init()`
2. **Determine task**: Client name matches the task name (e.g., "aggregation")
3. **Load data**: Load the task-specific training and test datasets
4. **Training loop**:
   - Receive global model from server
   - Perform local training
   - Evaluate on test set
   - Send updated model back to server

The client uses the NVFlare Client API:

```python
import nvflare.client as flare

# Initialize NVFlare client
flare.init()
client_name = flare.get_site_name()  # e.g., "aggregation"

# Main federated learning loop
while flare.is_running():
    # Receive global model from server
    input_model = flare.receive()
    
    # Load global model parameters
    model.load_state_dict(input_model.params, strict=False)
    
    # Local training...
    
    # Send updated model back to server
    output_model = flare.FLModel(
        params=model.cpu().state_dict(),
        metrics={"RMSE": rmse, "Pearson": pearson_corr}
    )
    flare.send(output_model)
```

## Server-Side Workflow

This example uses the [`FedAvgRecipe`](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html), which implements the [FedAvg](https://arxiv.org/abs/1602.05629) algorithm. The Recipe API handles all server-side logic automatically:

1. Initialize the global model (AMPLIFY trunk)
2. For each training round:
   - Send the global model to all 6 clients
   - Wait for client updates
   - **Filter out private regression heads** using `ExcludeParamsFilter`
   - Aggregate only the AMPLIFY trunk parameters
   - Update the global model

With the Recipe API, **there is no need to write custom server code**. The federated averaging workflow is provided by NVFlare.

### Private Regression Heads

To keep regression heads private, we use the [`ExcludeParamsFilter`](../src/filters.py):

```python
# In job.py
recipe.add_client_output_filter(
    ExcludeParamsFilter(exclude_vars="regressor"), 
    tasks=["train", "validate"]
)
```

This filter removes the regression head parameters from the model before sending updates to the server, ensuring only the AMPLIFY trunk is aggregated.

## Job Recipe Code

The [`job.py`](job.py) script configures the federated learning job:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking
from src.filters import ExcludeParamsFilter
from src.model import AmplifyRegressor

# Six tasks, one per client
TASKS = ["aggregation", "binding", "expression", "immunogenicity", 
         "polyreactivity", "thermostability"]

# Build initial model
model = AmplifyRegressor(
    pretrained_model_name_or_path="chandar-lab/AMPLIFY_120M",
    layer_sizes=[128, 64, 32]
)

# Create FedAvg recipe
recipe = FedAvgRecipe(
    name="amplify_seqregression_fedavg_multitask",
    min_clients=len(TASKS),
    num_rounds=600,
    initial_model=model,
    train_script="client.py",
    train_args="--data_root ./FLAb/data_fl --n_epochs 1 ..."
)

# Add filter to keep regressors private
recipe.add_client_output_filter(
    ExcludeParamsFilter(exclude_vars="regressor"), 
    tasks=["train", "validate"]
)

# Add TensorBoard tracking
add_experiment_tracking(recipe, tracking_type="tensorboard")

# Run simulation
env = SimEnv(
    clients=TASKS,  # Client names = task names
    workspace_root="/tmp/nvflare/AMPLIFY/multitask",
    gpu_config="0,1,2,0,1,2"  # 6 clients on 3 GPUs
)
recipe.execute(env)
```

## Run Job

### Local Training (Baseline)

Run 1 round with 600 local epochs to establish a baseline:

```bash
cd job_multitask
python job.py \
    --num_rounds 1 \
    --local_epochs 600 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "local_singletask" \
    --sim_gpus "0,1,2,0,1,2" \
    --max_samples 1000
```

### Federated Learning (FedAvg)

Run 600 rounds with 1 local epoch per round:

```bash
cd job_multitask
python job.py \
    --num_rounds 600 \
    --local_epochs 1 \
    --pretrained_model "chandar-lab/AMPLIFY_120M" \
    --layer_sizes "128,64,32" \
    --exp_name "fedavg_multitask" \
    --sim_gpus "0,1,2,0,1,2" \
    --max_samples 1000
```

**Arguments:**
- `--num_rounds`: Number of federated learning rounds
- `--local_epochs`: Number of local training epochs per round
- `--pretrained_model`: AMPLIFY model to use (120M or 350M)
- `--layer_sizes`: Regression MLP architecture (comma-separated)
- `--exp_name`: Experiment name for organizing results
- `--sim_gpus`: GPU allocation for simulated clients
- `--max_samples`: Limit samples per client (for quick testing; omit for full training)

> **Note:** The `--max_samples` parameter is useful for quick testing. Remove it for full training runs.

## Results

### Visualize Training Curves

The experiment results are saved in `/tmp/nvflare/AMPLIFY/multitask`. You can visualize the training metrics with TensorBoard:

```bash
tensorboard --logdir /tmp/nvflare/AMPLIFY/multitask
```

Or generate plots comparing local vs. federated training:

```bash
# Plot RMSE metrics
python ../figs/plot_training_curves.py \
    --log_dir /tmp/nvflare/AMPLIFY/multitask \
    --output_dir ../figs/tb_figs_rmse \
    --tag "RMSE/local_test" \
    --out_metric "RMSE"

# Plot Pearson correlation
python ../figs/plot_training_curves.py \
    --log_dir /tmp/nvflare/AMPLIFY/multitask \
    --output_dir ../figs/tb_figs_pearson \
    --tag "Pearson/local_test" \
    --out_metric "Pearson"
```

### Expected Outcomes

- **RMSE (lower is better)**: FedAvg typically achieves lower RMSE values compared to local-only training across multiple tasks
- **Pearson Correlation (closer to 1.0 is better)**: Models trained with FedAvg show improved correlation between predicted and actual values for several downstream tasks

The benefits of federated learning come from:
1. **Shared representations**: All clients benefit from a jointly trained AMPLIFY trunk
2. **Task diversity**: Different tasks provide complementary learning signals
3. **Privacy preservation**: Regression heads remain private to each client

## Output Summary

#### Initialization
* **TensorBoard**: Logs available at `/tmp/nvflare/AMPLIFY/multitask/amplify_seqregression_*/server/simulate_job/tb_events`
* **Workflow**: FedAvg controller initialized with 6 clients

#### Each Round
* **Clients Sampled**: All 6 task-specific clients (aggregation, binding, expression, immunogenicity, polyreactivity, thermostability)
* **Training**:
  * Global model sent to all clients
  * Each client performs local training on their task
  * Clients evaluate on test sets and report metrics (RMSE, Pearson correlation)
* **Filtering**: Regression head parameters excluded via `ExcludeParamsFilter`
* **Aggregation**: Only AMPLIFY trunk parameters aggregated and persisted on server

#### Completion
* **Final Model**: Globally fine-tuned AMPLIFY trunk available
* **Client Models**: Each client has their own fine-tuned regression head (not shared)

## References

- AMPLIFY: [Protein language models: is scaling necessary?](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
- FLAb: [Benchmarking deep learning methods for antibody fitness prediction](https://www.biorxiv.org/content/10.1101/2024.01.13.575504v1)
- FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
