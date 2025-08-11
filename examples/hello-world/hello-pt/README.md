
# Hello Pytorch
This example demonstrates how to use NVIDIA FLARE with PyTorch to train an image classifier using federated averaging (FedAvg).The complete example code can be found in the`hello-pt directory <examples/hello-world/hello-pt/>`. It is recommended to create a virtual environment and run everything within a virtualenv.

## NVIDIA FLARE Installation
for the complete installation instructions, see <../../installation.html>
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
|-- client.py         # client local training script
|-- model.py          # model definition
|-- job.py            # job recipe that defines client and server configurations
|-- requirements.txt  # dependencies
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
 
## Client Code
The client code ```client.py``` is responsible for Notice the training code is almost identical to the pytorch standard training code. 
The only difference is that we added a few lines to receive and send data to the server.

## Server Code
In federated averaging, the server code is responsible for aggregating model updates from clients, the workflow pattern is similar to scatter-gather. In this example, we will directly use the default federated averaging algorithm provided by NVFlare. 
The FedAvg class is defined in nvflare.app_common.workflows.fedavg.FedAvg
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
 
## Run Job
from terminal try to run the code


```
    python job.py
```
## Output summary

### Initialization
* **TensorBoard**: Logs available at /tmp/nvflare/simulation/hello-pt/server/simulate_job/tb_events.
* **Workflow**: BaseModelController initialized.
### Round 0
* **Model Loading**: Initial model loaded from persistor.
* **Clients Sampled**: site-1, site-2.
* **Training**:
  * Tasks sent to both sites.
  * Two epochs completed with loss reported.
* **Aggregation**: Models aggregated and persisted on the server.

### Round 1
* **Clients Sampled**: site-1, site-2.
* **Training**:
  * Similar process as Round 0.
  * **Aggregation**: Models aggregated and persisted.
### Completion
* **FedAvg Process**: Successfully finished with the final model persisted.