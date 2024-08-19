# Hello PyTorch

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [PyTorch](https://pytorch.org/) as the deep learning training framework.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the client train code.

You can follow the [Getting Started with NVFlare (PyTorch) notebook](../../getting_started/pt/nvflare_pt_getting_started.ipynb)
for a detailed walkthrough of the basic concepts.

See the [Hello PyTorch](https://nvflare.readthedocs.io/en/main/examples/hello_pt.html) example documentation page for details on this
example.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb), or you can quickly get
started with the following:


### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to install NVFlare.

Install additional requirements:

```
pip3 install -r requirements.txt
```

### 2. Run the experiment

Run the script using the job API to create the job and run it with the simulator:

```
python3 fedavg_script_executor_hello-pt.py
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir/
```
