# Hello NumPy - Recipe API

This example demonstrates how to use NVIDIA FLARE with NumPy to train a simple model using federated averaging (FedAvg) with the modern recipe API. It is recommended to create a virtual environment and run everything within a virtualenv.

## NVIDIA FLARE Installation

For the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)
```
pip install nvflare
```

Install the dependency

```
pip install -r requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python job.py
```

## Code Structure

``` bash
recipe-api-approach/
|
|-- client.py              # client local training script
|-- model.py               # model definition
|-- job.py                 # job recipe that defines client and server configurations
|-- numpy_fedavg_recipe.py # custom NumPy recipe
|-- requirements.txt       # dependencies
```

## Data

This example uses synthetic data generated within the code for demonstration purposes.

In a real FL experiment, each client would have their own dataset used for their local training. 
For this simple example, we generate synthetic data within the model to demonstrate the federated learning concepts.

## Model

The `SimpleNumpyModel` class implements a basic neural network model using NumPy arrays. 
The model's architecture consists of a simple weight matrix that can be trained through federated learning.
The implementation of this model can be found in `model.py`.

## Client Code

The client code `client.py` is responsible for:
- Receiving the global model from the server
- Performing local training on the received model
- Evaluating the model and computing metrics
- Sending the updated model back to the server

Notice the training code is almost identical to standard NumPy training code. 
The only difference is that we added a few lines to receive and send data to the server.

## Server Code

In federated averaging, the server code is responsible for aggregating model updates from clients, the workflow pattern is similar to scatter-gather. In this example, we will directly use the default federated averaging algorithm provided by NVFlare. 
The FedAvg class is defined in `nvflare.app_common.workflows.fedavg.FedAvg`
There is no need to define a customized server code for this example.

## Job Recipe Code

Job Recipe contains the client.py and built-in Fed average algorithm using our custom `NumpyFedAvgRecipe`:

```python
recipe = NumpyFedAvgRecipe(
    name="hello-numpy",
    min_clients=n_clients,
    num_rounds=num_rounds,
    initial_model=SimpleNumpyModel(),
    train_script="client.py",
    train_args=f"--learning_rate {learning_rate}",
)

env = SimEnv(num_clients=n_clients)
recipe.execute(env=env)
```

The `NumpyFedAvgRecipe` is specifically designed for NumPy models and uses `FrameworkType.NUMPY` for proper parameter exchange.
 
## Run Job

From terminal try to run the code:

```bash
# Default: 2 clients, 3 rounds
python job.py

# Custom parameters
python job.py --n_clients 3 --num_rounds 5 --learning_rate 2.0
```

## Output summary

### Initialization
* **TensorBoard**: Logs available at /tmp/nvflare/simulation/hello-numpy/server/simulate_job/tb_events.
* **Workflow**: BaseModelController initialized.

### Round 0
* **Model Loading**: Initial model loaded from persistor.
* **Clients Sampled**: site-1, site-2.
* **Training**:
  * Tasks sent to both sites.
  * Local training completed with metrics reported.
* **Aggregation**: Models aggregated and persisted on the server.

### Round 1
* **Clients Sampled**: site-1, site-2.
* **Training**:
  * Similar process as Round 0.
  * **Aggregation**: Models aggregated and persisted.

### Completion
* **FedAvg Process**: Successfully finished with the final model persisted.

## Understanding the Results

The model starts with weights `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` and each client adds 1 to each weight during training.
After aggregation, you should see the weights increase by 1 each round, demonstrating the federated learning process.
