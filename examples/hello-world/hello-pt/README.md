
# Hello PyTorch
This example demonstrates how to use NVIDIA FLARE with PyTorch to train an image classifier using federated averaging (FedAvg). The complete example code can be found in the `hello-pt directory <examples/hello-world/hello-pt/>`. It is recommended to create a virtual environment and run everything within a virtualenv.

## NVIDIA FLARE Installation

For complete installation instructions, visit [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

```bash
  pip install nvflare
```

Get the example code from github:

```
    git clone https://github.com/NVIDIA/NVFlare.git
```
Then navigate to the hello-pt directory:

```
    git switch <release branch>
    cd examples/hello-world/hello-pt
```

Install the dependencies:

```
  pip install -r requirements.txt
```

## Code Structure

``` bash
    hello-pt
    |
    |-- client.py             # client local training script
    |-- model.py              # model definition
    |-- job.py                # job recipe that defines client and server configurations
    |-- requirements.txt      # dependencies
```

## NVIDIA FLARE Installation
Here, we install nvflare with the PT extensions. For the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)
```
pip install nvflare[PT]

```
Install all dependencies

```
pip install -r requirements.txt
```

## Data
This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. You can download the CIFAR10 dataset from the Internet via torchvision’s datasets module.

In a real FL experiment, each client would have their own dataset used for their local training. 
You could split the datasets for different clients, so that each client has its own dataset. 
Here for simplicity's sake, we will be using the same dataset on each client.

## Model
In PyTorch, neural networks are implemented by defining a class (e.g., `SimpleNetwork`) that extends `nn.Module`. 
The network’s architecture is set up in the __init__ method, while the forward method determines how input data flows
through the layers. For faster computations, the model is transferred to a hardware accelerator (such as NVIDIA GPUs) if available; otherwise, it runs on the CPU. The implementation of this model can be found in [model.py](model.py).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

 
## Client Code
On the client side, the training workflow is as follows:
1. Receive the model from the FL server.
2. Perform local training on the received global model and/or evaluate the received global model for model selection.
3. Send the new model back to the FL server.

The client code ([client.py](./client.py)) is responsible for implementing this training workflow. Notice the training code is almost identical to a standard training PyTorch code. 
The only difference is that we added a few lines to receive and send data to the server.

Using NVFlare's client API, we can easily adapt machine learning code that was written for centralized training and apply it in a federated scenario.
For a general use case, there are three essential methods to achieve this using the Client API :
- `init()`: Initializes NVFlare Client API environment.
- `receive()`: Receives model from the FL server.
- `send()`: Sends the model to the FL server.
With these simple methods, the developers can use the Client API
to change their centralized training code to an FL scenario with
five lines of code changes as shown below.

```
import nvflare.client as flare
    
flare.init() # 1. Initializes NVFlare Client API environment.
input_model = flare.receive() # 2. Receives model from the FL server.
params = input_model.params # 3. Obtain the required information from the received model.
    
# original local training code
new_params = local_train(params)
    
output_model = flare.FLModel(params=new_params) # 4. Put the results in a new `FLModel`
flare.send(output_model) # 5. Sends the model to the FL server.  
```

## Server Code
In federated averaging, the server code is responsible for distributing the global model and aggregating model updates from clients. 

First, we provide a robust implementation of the [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com) algorithm with NVFlare. 

The server implements these main steps:
1. FL server initializes an initial model.
2. For each round (global iteration):
    - FL server samples available clients.
    - FL server sends the global model to clients and waits for their updates.
    - FL server aggregates all the `results` and produces a new global model.

In this example, we will directly use the default federated averaging algorithm provided by NVFlare utilizing the [FedAvgRecipe](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html#nvflare.app_opt.pt.recipes.fedavg.FedAvgRecipe) for PyTorch. 

There is no need to define a customized server code for this example.

## Job Recipe Code
The Job Recipe specifies the `client.py` and selects the built-in federated averaging algorithm.
```
    recipe = FedAvgRecipe(
        name="hello-pt",
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=SimpleNetwork(),
        train_script="client.py",
        train_args=f"--batch_size {batch_size}",
    )

    env = SimEnv(num_clients=n_clients, num_threads=n_clients)
    recipe.execute(env=env)
```

## Run Job
From terminal simply run the job script to execute the job in a simulation environment.

```
    python job.py
```

> Note, as part of the job script, use `add_experiment_tracking(recipe, tracking_type="tensorboard")` to stream training metrics to the server using NVIDIA FLARE's [SummaryWriter](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.client.tracking.html#nvflare.client.tracking.SummaryWriter) in [client.py](client.py).

## Notebook

For an interactive version of this example, see this [notebook](./hello-pt.ipynb), which can be executed in Google Colab.

## Output summary

#### Initialization
* **TensorBoard**: Logs available at /tmp/nvflare/simulation/hello-pt/server/simulate_job/tb_events.
* **Workflow**: BaseModelController initialized.
#### Round 0
* **Model Loading**: Initial model loaded from persistor.
* **Clients Sampled**: site-1, site-2.
* **Training**:
  * Tasks sent to both sites.
  * Two epochs completed with loss reported.
* **Aggregation**: Models aggregated and persisted on the server.

#### Round 1
* **Clients Sampled**: site-1, site-2.
* **Training**:
  * Similar process as Round 0.
  * **Aggregation**: Models aggregated and persisted.
#### Completion
* **FedAvg Process**: Successfully finished with the final model persisted.