:orphan:

# NVFlare hello-world examples

This folder contains hello-world examples for NVFlare.

Please make sure you set up a virtual environment and install JupyterLab following the [example root readme](../README.md).

Please also install "./requirements.txt" in each example folder.

## Hello World Notebook
### Prerequisites
Before you run the notebook, the following preparation work must be done:

  1. Install a virtual environment following the instructions in [example root readme](../README.md)
  2. Install Jupyter Lab and install a new kernel for the virtualenv called `nvflare_example`
  3. Run [hw_pre_start.sh](./hw_pre_start.sh) in the terminal before running the notebook
  4. Run [hw_post_cleanup.sh](./hw_post_cleanup.sh) in the terminal after running the notebook 

* [Hello world notebook](./hello_world.ipynb)

## Hello World Examples
### Easier ML/DL to FL transition
* [ML to FL](./ml-to-fl/README.md): Showcases how to convert existing ML/DL code to an NVFlare job.

### Step by step examples
* [Step by step examples](./step-by-step/README.md): Shows specific techniques and workflows and what needs to be changed for each.

### Workflows
* [Hello Scatter and Gather](./hello-numpy-sag/README.md)
    * Example using [ScatterAndGather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html) controller workflow.
* [Hello Cross-Site Validation](./hello-numpy-cross-val/README.md)
    * Example using [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.
* [Hello Cyclic Weight Transfer](./hello-cyclic/README.md)
    * Example using [CyclicController](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic_ctl.html) controller workflow to implement [Cyclic Weight Transfer](https://pubmed.ncbi.nlm.nih.gov/29617797/).
* [Hello Client Controlled Workflows](./hello-ccwf/README.md)
    * Example using [Client Controlled Workflows](https://nvflare.readthedocs.io/en/main/programming_guide/controllers/client_controlled_workflows.html).

### Deep Learning
* [Hello PyTorch](./hello-pt/README.md)
  * Example using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) an image classifier using ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework.
* [Hello TensorFlow](./hello-tf/README.md)
  * Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) an image classifier using ([FedAvg](https://arxiv.org/abs/1602.05629)) and [TensorFlow](https://tensorflow.org/) as the deep learning training framework.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello Client Controlled Workflow (CCWF)

[Client Controlled Workflows](https://nvflare.readthedocs.io/en/main/programming_guide/controllers/client_controlled_workflows.html) are managed
by logic from clients. This example shows the components used in a job for a client controlled workflow.

### 1. Install NVIDIA FLARE

Follow the [Installation](../../getting_started/README.md) instructions.

### 2. Run the experiment

Run the script using the job API to create the job and run it with the simulator:

```
python3 swarm_script_runner_np.py
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the swarm script runner for numpy-based federated learning.">

.. only:: html

  .. image:: /hello-world/hello-ccwf/images/thumb/sphx_glr_swarm_script_runner_np_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-ccwf_swarm_script_runner_np.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the swarm script runner for numpy-based federated learning.</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello Cyclic Weight Transfer

["Cyclic Weight Transfer"](https://pubmed.ncbi.nlm.nih.gov/29617797/
) (CWT) is an alternative to [FedAvg](https://arxiv.org/abs/1602.05629). CWT uses the [Cyclic Controller](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic.html) to pass the model weights from one site to the next for repeated fine-tuning.

> **_NOTE:_** This example uses the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset and will load its data within the trainer code.

We recommend to use [NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) if you want to use GPU.
If you don't need to run using GPU, you can just use python virtual environment.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb).

## Run NVIDIA TensorFlow container

Please install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first.
Then run the following command:

```bash
docker run --gpus=all -it --rm -v [path_to_NVFlare]:/NVFlare nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3
```

## Install NVFlare

```bash
pip3 install nvflare
```

## Run the experiment

Prepare the data first:

```
bash ./prepare_data.sh
```

Run the script using the job API to create the job and run it with the simulator:

```
python3 cyclic_script_runner.py
```

## Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir
```

### Notes on running with GPUs

If you choose to run the example using GPUs, it is important to note that,
by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have to prevent TensorFlow from allocating all GPU memory 
by setting the following flags.
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async
```

If you possess more GPUs than clients, a good strategy is to run one client on each GPU.
This can be achieved by using the `--gpu` argument during simulation, e.g., `nvflare simulator -n 2 --gpu 0,1 [job]`.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the cyclic script runner for federated learning.">

.. only:: html

  .. image:: /hello-world/hello-cyclic/images/thumb/sphx_glr_cyclic_script_runner_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-cyclic_cyclic_script_runner.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the cyclic script runner for federated learning.</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" Hello Pytorch ===================">

.. only:: html

  .. image:: /hello-world/hello-cyclic/images/thumb/sphx_glr_doc_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-cyclic_doc.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hello Pytorch ||</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello FedAvg

In this example we highlight the flexibility of the ModelController API, and show how to write a Federated Averaging workflow with early stopping, model selection, and saving and loading. Follow along in the [hello-fedavg.ipynb](hello-fedavg.ipynb) notebook for more details.

### 1. Setup

```
pip install nvflare~=2.5.0rc torch torchvision tensorboard
```

Download the dataset:
```
./prepare_data.sh
```

### 2. PTFedAvgEarlyStopping using ModelController API

The ModelController API enables the option to easily customize a workflow with Python code.

- FedAvg: We subclass the BaseFedAvg class to leverage the predefined aggregation functions.
- Early Stopping: We add a `stop_condition` argument (eg. `"accuracy >= 80"`) and end the workflow early if the corresponding global model metric meets the condition.
- Patience: If set to a value greater than 1, the FL experiment will stop if the defined `stop_condition` does not improve over X consecutive FL rounds.
- Task to optimize: Allows the user to specify which task to apply the `early stopping` mechanism to (e.g., the validation phase)
- Model Selection: As and alternative to using a `IntimeModelSelector` componenet for model selection, we instead compare the metrics of the models in the workflow to select the best model each round.
- Saving/Loading: Rather than configuring a persistor such as `PTFileModelPersistor` component, we choose to utilize PyTorch's save and load functions and save the metadata of the FLModel separately.

### 3. Run the script

Use the Job API to define and run the example with the simulator:

```
python3 pt_fedavg_early_stopping_script.py
```

View the results in the job workspace: `/tmp/nvflare/jobs/workdir`.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the fedavg script runner">

.. only:: html

  .. image:: /hello-world/hello-fedavg/images/thumb/sphx_glr_pt_fedavg_early_stopping_script_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-fedavg_pt_fedavg_early_stopping_script.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the fedavg script runner</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello FedAvg NumPy
 
This example showcases Federated Averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) with NumPy.  

> **_NOTE:_** This example uses a NumPy-based trainer and will generate its data within the code.

You can follow the [Getting Started with NVFlare (NumPy)](hello-fedavg-numpy_getting_started.ipynb)
for a detailed walkthrough of the basic concepts.

See the [Hello FedAvg with NumPy](https://nvflare.readthedocs.io/en/main/examples/hello_fedavg_numpy.html) example documentation page for details on this
example.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb), or you can quickly get
started with the following:

### 1. Install NVIDIA FLARE

Follow the [Installation](../../getting_started/README.md) instructions.

### 2. Run the experiment

Run the script using the job API to create the job and run it with the simulator:

```
python3 fedavg_script_runner_hello-numpy.py
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir/
```

For how to use the FLARE API to run this app, see [this notebook](hello-fedavg-numpy_flare_api.ipynb).



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the fedavg script runner">

.. only:: html

  .. image:: /hello-world/hello-fedavg-numpy/images/thumb/sphx_glr_fedavg_script_runner_hello-numpy_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-fedavg-numpy_fedavg_script_runner_hello-numpy.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the fedavg script runner</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Flower App (PyTorch) in NVIDIA FLARE

In this example, we run 2 Flower clients and Flower Server in parallel using NVFlare's simulator.

## Preconditions

To run Flower code in NVFlare, we created a job, including an app with the following custom folder content 
```bash
$ tree jobs/hello-flwr-pt/app/custom

├── flwr_pt
│   ├── client.py   # <-- contains `ClientApp`
│   ├── __init__.py # <-- to register the python module
│   ├── server.py   # <-- contains `ServerApp`
│   └── task.py     # <-- task-specific code (model, data)
└── pyproject.toml  # <-- Flower project file
```
Note, this code is adapted from Flower's [app-pytorch](https://github.com/adap/flower/tree/main/examples/app-pytorch) example.

## 1. Install dependencies
If you haven't already, we recommend creating a virtual environment.
```bash
python3 -m venv nvflare_flwr
source nvflare_flwr/bin/activate
```
We recommend installing an older version of NumPy as torch/torchvision doesn't support NumPy 2 at this time.
```bash
pip install numpy==1.26.4
```
## 2.1 Run a simulation

To run flwr-pt job with NVFlare, we first need to install its dependencies.
```bash
pip install ./flwr-pt/
```

Next, we run 2 Flower clients and Flower Server in parallel using NVFlare's simulator.
```bash
python job.py --job_name "flwr-pt" --content_dir "./flwr-pt"
```

## 2.2 Run a simulation with TensorBoard streaming

To run flwr-pt_tb_streaming job with NVFlare, we first need to install its dependencies.
```bash
pip install ./flwr-pt-tb/
```

Next, we run 2 Flower clients and Flower Server in parallel using NVFlare while streaming 
the TensorBoard metrics to the server at each iteration using NVFlare's metric streaming.

```bash
python job.py --job_name "flwr-pt-tb" --content_dir "./flwr-pt-tb" --stream_metrics
```

You can visualize the metrics streamed to the server using TensorBoard.
```bash
tensorboard --logdir /tmp/nvflare/hello-flower
```
![tensorboard training curve](./train.png)

## Notes
Make sure your `pyproject.toml` files in the Flower apps contain an "address" field. This needs to be present as the `--federation-config` option of the `flwr run` command tries to override the `“address”` field.
Your `pyproject.toml` should include a section similar to this:
```
[tool.flwr.federations]
default = "xxx"

[tool.flwr.federations.xxx]
options.num-supernodes = 2
address = "127.0.0.1:9093"
insecure = false
```
The number `options.num-supernodes` should match the number of NVFlare clients defined in [job.py](./job.py), e.g., `job.simulator_run(args.workdir, gpu="0", n_clients=2)`.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the flower script">

.. only:: html

  .. image:: /hello-world/hello-flower/images/thumb/sphx_glr_job_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-flower_job.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the flower script</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" Hello Pytorch ===================">

.. only:: html

  .. image:: /hello-world/hello-flower/images/thumb/sphx_glr_doc_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-flower_doc.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hello Pytorch ||</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello Numpy Cross-Site Validation

The cross-site model evaluation workflow uses the data from clients to run evaluation with the models of other clients. Data is not shared. Rather the collection of models is distributed to each client site to run local validation. The server collects the results of local validation to construct an all-to-all matrix of model performance vs. client dataset. It uses the [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.


## Installation

Follow the [Installation](../../getting_started/README.md) instructions.

# Run training and cross site validation right after training

This example uses a NumPy-based trainer to simulate the training steps.

We first perform FedAvg training and then conduct cross-site validation.

So you will see two workflows (ScatterAndGather and CrossSiteModelEval) are configured.

## 1. Prepare the job and run the experiment using simulator

We use Job API to generate the job and run the job using simulator:

```bash
python3 job_train_and_cse.py
```

## 2. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir/
server/  site-1/  site-2/  startup/
```

The cross-site validation results:

```bash
$ cat /tmp/nvflare/jobs/workdir/server/simulate_job/cross_site_val/cross_val_results.json
```

# Run cross site evaluation using the previous trained results

We can also run cross-site evaluation without the training workflow, making use of the previous results or just want to evaluate on the pretrained models.

You can provide / use your own pretrained models for the cross-site evaluation.

## 1. Generate the pretrained model

In reality, users would use any training workflows to obtain these pretrained models

To mimic that, run the following command to generate the pre-trained models:

```bash
python3 generate_pretrain_models.py
```

## 2. Prepare the job and run the experiment using simulator

Note that our pre-trained models are generated under:

```python
SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"
```

In our job_cse.py we also specify that.

Then we can use Job API to generate the job and run it using simulator:

```bash
python3 job_cse.py
```

## 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir/
server/  site-1/  site-2/  startup/
```

The cross-site validation results:

```bash
$ cat /tmp/nvflare/jobs/workdir/server/simulate_job/cross_site_val/cross_val_results.json
```


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the generate pretrain models">

.. only:: html

  .. image:: /hello-world/hello-numpy-cross-val/images/thumb/sphx_glr_generate_pretrain_models_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-numpy-cross-val_generate_pretrain_models.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the generate pretrain models</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the job client side evaluation for federated learning">

.. only:: html

  .. image:: /hello-world/hello-numpy-cross-val/images/thumb/sphx_glr_job_cse_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-numpy-cross-val_job_cse.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the job client side evaluation for federated learning</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the crsoss-site evaluation script runner">

.. only:: html

  .. image:: /hello-world/hello-numpy-cross-val/images/thumb/sphx_glr_job_train_and_cse_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-numpy-cross-val_job_train_and_cse.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the crsoss-site evaluation script runner</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" Hello Pytorch =================== This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.">

.. only:: html

  .. image:: /hello-world/hello-numpy-cross-val/images/thumb/sphx_glr_tutorial_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-numpy-cross-val_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hello Pytorch ||</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello Numpy Scatter and Gather

"[Scatter and Gather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html)" is the standard workflow to implement Federated Averaging ([FedAvg](https://arxiv.org/abs/1602.05629)). 
This workflow follows the hub and spoke model for communicating the global model to each client for local training (i.e., "scattering") and aggregates the result to perform the global model update (i.e., "gathering").  

> **_NOTE:_** This example uses a Numpy-based trainer and will generate its data within the code.

You can follow the [hello_world notebook](../hello_world.ipynb) or the following:

### 1. Install NVIDIA FLARE

Follow the [Installation](../../getting_started/README.md) instructions.

### 2. Run the experiment

Use nvflare simulator to run the hello-examples:

```
nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 hello-world/hello-numpy-sag/jobs/hello-numpy-sag
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/hello-numpy-sag/simulate_job/
app_server  app_site-1  app_site-2  log.txt  model  models

```

For how to use the FLARE API to run this app, see [this notebook](hello_numpy_sag.ipynb).



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello PyTorch

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [PyTorch](https://pytorch.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the client train code.

You can follow the [Getting Started with NVFlare (PyTorch) notebook](../../getting_started/pt/nvflare_pt_getting_started.ipynb)
for a detailed walkthrough of the basic concepts.

See the [Hello PyTorch](https://nvflare.readthedocs.io/en/main/examples/hello_pt_job_api.html#hello-pt-job-api) example documentation page for details on this
example.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb), or you can quickly get
started with the following:


### 1. Install NVIDIA FLARE

Follow the [Installation](../../getting_started/README.md) instructions to install NVFlare.

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):

```
pip3 install -r requirements.txt
```

### 2. Run the experiment

Run the script using the job API to create the job and run it with the simulator:

```
python3 fedavg_script_runner_pt.py
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir/
```

By default, the Tensorboard event logs can be found in the directory for each client on the server job's tb_events folder,
for example:

```bash
$ ls /tmp/nvflare/jobs/workdir/server/simulate_job/tb_events/site-1
events.out.tfevents.1728070846.machinename.15928.1
```



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This code show to use NVIDIA FLARE Job Recipe to connect both Federated learning client and server algorithm">

.. only:: html

  .. image:: /hello-world/hello-pt/images/thumb/sphx_glr_job_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-pt_job.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This code show to use NVIDIA FLARE Job Recipe to connect both Federated learning client and server algorithm</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="pytorch model">

.. only:: html

  .. image:: /hello-world/hello-pt/images/thumb/sphx_glr_model_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-pt_model.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">pytorch model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="client side training scripts">

.. only:: html

  .. image:: /hello-world/hello-pt/images/thumb/sphx_glr_client_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-pt_client.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">client side training scripts</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" Hello Pytorch ===================">

.. only:: html

  .. image:: /hello-world/hello-pt/images/thumb/sphx_glr_doc_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-pt_doc.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hello Pytorch ||</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello PyTorch ResNet

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [PyTorch](https://pytorch.org/) as the deep learning training framework. Comparing with the Hello PyTorch example, it uses the torchvision ResNet, 
instead of the SimpleNetwork.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the client train code.

The Job API only supports the object instance created directly out of the Python Class. It does not support 
the object instance created through using the Python function. Comparing with the hello-pt example, 
if we replace the SimpleNetwork() object with the resnet18(num_classes=10), 
the "resnet18(num_classes=10)" creates an torchvision "ResNet" object instance out of the "resnet18" function. 
As shown in the [torchvision reset](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L684-L705), 
the resnet18 is a Python function, which creates and returns a ResNet object. The job API can 
only use the "ResNet" object instance for generating the job config. It can not detect the object creating function logic in the "resnet18".

This example demonstrates how to wrap up the resnet18 Python function into a Resnet18 Python class. Then uses the Resnet18(num_classes=10)
object instance in the job API. After replacing the SimpleNetwork() with the Resnet18(num_classes=10),
you can follow the exact same steps in the hello-pt example to run the fedavg_script_runner_pt.py.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the resent with pytorch script runner">

.. only:: html

  .. image:: /hello-world/hello-pt-resnet/images/thumb/sphx_glr_fedavg_script_runner_pt_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-pt-resnet_fedavg_script_runner_pt.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the resent with pytorch script runner</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Hello TensorFlow

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [TensorFlow](https://tensorflow.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) handwritten digits dataset and will load its data within the trainer code.

See the [Hello TensorFlow](https://nvflare.readthedocs.io/en/main/examples/hello_tf_job_api.html#hello-tf-job-api) example documentation page for details on this
example.

We recommend to use [NVIDIA TensorFlow docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) if you want to use GPU.
If you don't need to run using GPU, you can just use python virtual environment.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb).

## Run NVIDIA TensorFlow container

Please install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first.
Then run the following command:

```bash
docker run --gpus=all -it --rm -v [path_to_NVFlare]:/NVFlare nvcr.io/nvidia/tensorflow:xx.xx-tf2-py3
```

## Install NVFlare

```bash
pip3 install nvflare
```

## Run the experiment

Run the script using the job API to create the job and run it with the simulator:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async python3 fedavg_script_runner_tf.py
```

## Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir
```

### Notes on running with GPUs

If you choose to run the example using GPUs, it is important to note that,
by default, TensorFlow will attempt to allocate all available GPU memory at the start.
In scenarios where multiple clients are involved, you have to prevent TensorFlow from allocating all GPU memory 
by setting the following flags.
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_GPU_ALLOCATOR=cuda_malloc_async
```

If you possess more GPUs than clients, a good strategy is to run one client on each GPU.
This can be achieved by using the `--gpu` argument during simulation, e.g., `nvflare simulator -n 2 --gpu 0,1 [job]`.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates how to run the fedavg with tensorflow script runner">

.. only:: html

  .. image:: /hello-world/hello-tf/images/thumb/sphx_glr_fedavg_script_runner_tf_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-tf_fedavg_script_runner_tf.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">This script demonstrates how to run the fedavg with tensorflow script runner</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" Hello Pytorch =================== This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.">

.. only:: html

  .. image:: /hello-world/hello-tf/images/thumb/sphx_glr_doc_thumb.png
    :alt:

  :ref:`sphx_glr_hello-world_hello-tf_doc.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hello Pytorch ||</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

# Simple ML/DL to FL transition with NVFlare


Converting Deep Learning (DL) models to Federated Learning (FL) entails several key steps:

Formulating the algorithm: This involves determining how to adapt a DL model into an FL framework, including specifying the information exchange protocol between the Client and Server.

Code conversion: Adapting existing standalone DL code into FL-compatible code. This typically involves minimal changes, often just a few lines of code, thanks to tools like NVFlare.

Workflow configuration: Once the code is modified, configuring the workflow to integrate the newly adapted FL code seamlessly.

NVFlare simplifies the process of transitioning from traditional Machine Learning (ML) or DL algorithms to FL. With NVFlare, the conversion process requires only minor code adjustments.

In our examples, we assume that algorithm formulation follows NVFlare's predefined workflow algorithms (such as FedAvg). Detailed tutorials on converting traditional ML to FL, particularly with tabular datasets, are available in our step-by-step guides.

We offer various techniques tailored to different code structures and user preferences. Configuration guidance and documentation are provided to facilitate workflow setup.

Our coverage includes:

Configurations for NVFlare Client API: [np](./np/README.md)
Integration with PyTorch and PyTorch Lightning frameworks:[pt](./pt/README.md)
Support for TensorFlow implementations: [tf](./tf/README.md)

For detailed instructions on configuring the workflow, refer to our provided examples and documentation.
If you're solely interested in converting DL to FL code, feel free to skip ahead to the examples without delving further into this readme.

For those eager to explore various implementations and use cases, read on.

## Advanced User Options: Client API with Different Implementations

Within the Client API, we offer multiple implementations tailored to diverse requirements:

* In-process Client API: In this setup, the client training script operates within the same process as the NVFlare Client job.
This configuration, utilizing the ```InProcessClientAPIExecutor```, offers shared the memory usage, is efficient and with simple configuration. 
Use this configuration for development or single GPU

* Sub-process Client API: Here, the client training script runs in a separate subprocess.
Utilizing the ```ClientAPILauncherExecutor```, this option offers flexibility in communication mechanisms:
  * Communication via CellPipe (default)
  * Communication via FilePipe ( no capability to stream experiment track log metrics) 
This configuration is ideal for scenarios requiring multi-GPU or distributed PyTorch training.

Choose the option best suited to your specific requirements and workflow preferences.

These implementations can be easily configured using the JobAPI's ScriptRunner.
By default, the ```InProcessClientAPIExecutor``` is used, however setting `launch_external_process=True` uses the ```ClientAPILauncherExecutor```
with pre-configured CellPipes for communication and metrics streaming.

Note: Avoid installing TensorFlow and PyTorch in the same virtual environment due to library conflicts.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

#  Step-by-Step Examples

To run the notebooks in each example, please make sure you first set up a virtual environment and install "./requirements.txt" and JupyterLab following the [example root readme](../README.md).

* [cifar10](cifar10) - Multi-class classification with image data using CIFAR10 dataset
* [higgs](higgs) - Binary classification with tabular data using HIGGS dataset

These step-by-step example series are aimed to help users quickly get started and learn about FLARE.
For consistency, each example in the series uses the same dataset- CIFAR10 for image data and the HIGGS dataset for tabular data.
The examples will build upon previous ones to showcase different features, workflows, or APIs, allowing users to gain a comprehensive understanding of FLARE functionalities (Note: each example is self-contained, so going through them in order is not required, but recommended). See the README in each directory for more details about each series.

## Common Questions

Here are some common questions we aim to cover in these examples series when formulating a federated learning problem:

* What does the data look like?
* How do we compare global statistics with the site's local data statistics? 
* How to formulate the [federated algorithms](https://developer.download.nvidia.com/healthcare/clara/docs/federated_traditional_machine_learning_algorithms.pdf)?
* How do we convert the existing machine learning or deep learning code to federated learning code? [ML to FL examples](../../../examples/hello-world/ml-to-fl/README.md)
* How do we use different types of federated learning workflows (e.g. Scatter and Gather, Cyclic Weight Transfer, Swarming learning,
Vertical learning) and what do we need to change?
* How can we capture the experiment log, so all sites' metrics and global metrics can be viewed in experiment tracking tools such as Weights & Biases MLfLow, or Tensorboard



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /hello-world/hello-ccwf/index.rst
   /hello-world/hello-cyclic/index.rst
   /hello-world/hello-fedavg/index.rst
   /hello-world/hello-fedavg-numpy/index.rst
   /hello-world/hello-flower/index.rst
   /hello-world/hello-numpy-cross-val/index.rst
   /hello-world/hello-numpy-sag/index.rst
   /hello-world/hello-pt/index.rst
   /hello-world/hello-pt-resnet/index.rst
   /hello-world/hello-tf/index.rst
   /hello-world/ml-to-fl/index.rst
   /hello-world/step-by-step/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: hello-world_python.zip </hello-world/hello-world_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: hello-world_jupyter.zip </hello-world/hello-world_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
