# Hello Cyclic Weight Transfer

["Cyclic Weight Transfer"](https://pubmed.ncbi.nlm.nih.gov/29617797/
) (CWT) is an alternative to [FedAvg](https://arxiv.org/abs/1602.05629). CWT uses the [Cyclic Controller](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic.html) to pass the model weights from one site to the next for repeated fine-tuning.

> **_NOTE:_** This example uses the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset and will load its data within the trainer code.

## Running Tensorflow with GPU

We recommend to use [NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) if you want to use GPU.
If you don't need to run using GPU, you can just use python virtual environment.

### Run NVIDIA TensorFlow container
Please install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first.
Then run the following command:

```bash
docker run --gpus=all -it --rm -v [path_to_NVFlare]:/NVFlare nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3
```

### Notes on running with GPUs

If you choose to run the example using GPUs, it is important to note that,
by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have to prevent TensorFlow from allocating all GPU memory
by setting the following flags.
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async
```

## Install NVFlare

for the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)
```
    pip install nvflare

```
get the example code from github:
```
    git clone https://github.com/NVIDIA/NVFlare.git
```

then navigate to the hello-cyclic directory:

```bash

    git switch <release branch>
    cd examples/hello-world/hello-cyclic
    
```

Install the dependency

```
    pip install -r requirements.txt
```

## Code Structure

```
hello-cyclic
|
|-- client.py           # client local training script
|-- model.py            # model definition
|-- job.py              # job recipe that defines client and server configurations
|-- prepare_data.sh     # scripts to download the data
|-- requirements.txt    # dependencies
```

## Data

In this example, We will use Mnist datasets, which is provided by 
Tensorflow Keras API 

## Model

In Tensorflow, we define a simple model via Keras API

```
class Net(models.Sequential):
    def __init__(self, input_shape=(None, 28, 28)):
    super().__init__()
    self._input_shape = input_shape
    self.add(layers.Flatten())
    self.add(layers.Dense(128, activation="relu"))
    self.add(layers.Dropout(0.2))
    self.add(layers.Dense(10))
```

## Client Code

The client code ```client.py``` is responsible for Notice the training code is almost identical to the pytorch standard training code.
The only difference is that we added a few lines to receive and send data to the server.

## Server Code

In cyclic transfer, the server code is responsible for
replay model updates from client to another.We will directly use 
the default federated cyclic algorithm provided by NVFlare.

## Job Recipe

```
    n_clients = 2
    num_rounds = 3
    train_script = "client.py"

    recipe = CyclicRecipe(
        num_rounds=num_rounds,
        initial_model=Net(),
        train_script=train_script,
    )

    env = SimEnv(num_clients=n_clients)
    run = recipe.execute(env=env)
    print()
    print("Result can be found in :", run.get_result())
    print("Job Status is:", run.get_status())
    print()


```

## Run the experiment

Prepare the data first:

```
bash ./prepare_data.sh

python job.py
```

## Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls "/tmp/nvflare/simulation/cyclic"
```
