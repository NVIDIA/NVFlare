# Hello TensorFlow

This example demonstrates how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) with TensorFlow to train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)). TensorFlow serves as the deep learning training framework in this example.

For detailed documentation, see the [Hello TensorFlow](https://www.tensorflow.org/datasets/catalog/mnist) example page.

We recommend using the [NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) for GPU support. If GPU is not required, a Python virtual environment is sufficient.

To run this example with the FLARE API, refer to the [hello_world notebook](../hello_world.ipynb).

## Run NVIDIA TensorFlow Container

Ensure the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed. Then execute the following command:

```bash
  docker run --gpus=all -it --rm -v [path_to_NVFlare]:/NVFlare nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3
```

## NVIDIA FLARE Installation

For complete installation instructions, visit [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

```bash
  pip install nvflare
```
Clone the example code from GitHub:

```bash
  git clone https://github.com/NVIDIA/NVFlare.git
```
Navigate to the hello-tf directory:

```bash
  git switch <release branch>
  cd examples/hello-world/hello-tf
```

Install the dependencies:

```bash
  pip install -r requirements.txt
```
## Code Structure


```text
hello-pt
|
|-- client.py         # client local training script
|-- model.py          # model definition
|-- job.py            # job recipe that defines client and server configurations
|-- requirements.txt  # dependencies
```

## Data

This example uses the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) handwritten digits dataset, which is loaded within the trainer code.

## Model

```
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

The client code `client.py` is responsible for training. The training code closely resembles standard PyTorch training code, with additional lines to handle data exchange with the server.

## Server Code

In federated averaging, the server code aggregates model updates from clients, following a scatter-gather workflow pattern. This example uses the default federated averaging algorithm provided by NVFlare, eliminating the need for custom server code.

## Job Recipe Code

The job recipe includes `client.py` and the built-in FedAvg algorithm.

```python
n_clients = 2
num_rounds = 3
train_script = "client.py"

recipe = FedAvgRecipe(
    name="hello-tf_fedavg",
    num_rounds=num_rounds,
    initial_model=Net(),
    min_clients=n_clients,
    train_script=train_script,
)
add_experiment_tracking(recipe, tracking_type="tensorboard")

env = SimEnv(num_clients=n_clients)
run = recipe.execute(env=env)
print()
print("Result can be found in :", run.get_result())
print("Job Status is:", run.get_status())
print()
```

## Run the Experiment

Execute the script using the job API to create the job and run it with the simulator:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python3 job.py
```

## Access the Logs and Results

Find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir
```

## Notes on Running with GPUs

When using GPUs, TensorFlow attempts to allocate all available GPU memory at startup. To prevent this in multi-client scenarios, set the following flags:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async
```

If you have more GPUs than clients, consider running one client per GPU using the `--gpu` argument during simulation, e.g., `nvflare simulator -n 2 --gpu 0,1 [job]`.
