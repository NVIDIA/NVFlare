
# Hello Pytorch
This example demonstrates how to use NVIDIA FLARE with PyTorch to train an image classifier using federated averaging (FedAvg).The complete example code can be found in the`hello-pt directory <examples/hello-world/hello-pt/>`. It is recommended to create a virtual environment and run everything within a virtualenv.

## NVIDIA FLARE Installation
for the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)
```
pip install nvflare

```
Install the dependency

```
pip install -r requirements.txt
```
## Code Structure
first get the example code from github:

```
git clone https://github.com/NVIDIA/NVFlare.git
```
then navigate to the hello-pt directory:

```
git switch <release branch>
cd examples/hello-world/hello-pt
```
``` bash
hello-pt
|
|-- client.py             # client local training script
|-- client_with_eval.py   # alternative client local training script with both traiing and evaluation
|-- model.py              # model definition
|-- job.py                # job recipe that defines client and server configurations
|-- requirements.txt      # dependencies
```

## Data
This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset

In a real FL experiment, each client would have their own dataset used for their local training. 
You can download the CIFAR10 dataset from the Internet via torchvision’s datasets module, 
You can split the datasets for different clients, so that each client has its own dataset. 
Here for simplicity’s sake, the same dataset we will be using on each client.

## Model
In PyTorch, neural networks are implemented by defining a class (e.g., SimpleNetwork) that extends nn.Module. 
The network’s architecture is set up in the __init__ method,# while the forward method determines how input data flows
through the layers. For faster computations, the model is transferred to a hardware accelerator (such as CUDA GPUs) if available; otherwise, it runs on the CPU. The implementation of this model can be found in model.py.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
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
The client code ```client.py``` is responsible for Notice the training code is almost identical to the pytorch standard training code. 
The only difference is that we added a few lines to receive and send data to the server.

Now, we need to adapt this centralized training code to something that can run in a federated setting.

On the client side, the training workflow is as follows:
1. Receive the model from the FL server.
2. Perform local training on the received global model and/or evaluate the received global model for model selection.
3. Send the new model back to the FL server.

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
In federated averaging, the server code is responsible for aggregating model updates from clients, the workflow pattern is similar to scatter-gather. 

First, we provide a simple implementation of the [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com) algorithm with NVFlare. 
The `run()` routine implements the main algorithmic logic. 
Subroutines, like `sample_clients()` and `scatter_and_gather_model()` utilize the communicator object, native to each Controller to get the list of available clients,
distribute the current global model to the clients, and collect their results.

The FedAvg controller implements these main steps:
1. FL server initializes an initial model using `self.load_model()`.
2. For each round (global iteration):
    - FL server samples available clients using `self.sample_clients()`.
    - FL server sends the global model to clients and waits for their updates using `self.send_model_and_wait()`.
    - FL server aggregates all the `results` and produces a new global model using `self.update_model()`.

```python
class FedAvg(BaseFedAvg):
    def run(self) -> None:
        self.info("Start FedAvg.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            results = self.send_model_and_wait(targets=clients, data=model)

            aggregate_results = self.aggregate(results)

            model = self.update_model(model, aggregate_results)

            self.save_model(model)

        self.info("Finished FedAvg.")
```

In this example, we will directly use the default federated averaging algorithm provided by NVFlare. The FedAvg class is defined in nvflare.app_common.workflows.fedavg.FedAvg
There is no need to defined a customized server code for this example.

## Job Recipe Code
Job Recipe contains the client.py and built-in Fed average algorithm.
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

To include both training and evaluation, you can change the recipe's training script 

```python

train_script="client_with_eval.py",

```
or simply overwrite client.py with client_with_eval.py

## Run Job
from terminal try to run the code


```
    python job.py
```
> Note: 
>> depends on the number of clients, you might run into error due to several client try to download the data at the same time. 
>> suggest to pre-download the data to avoid such errors. 

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