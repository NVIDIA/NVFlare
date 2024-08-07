# Hello Numpy Scatter and Gather

"[Scatter and Gather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html)" is the standard workflow to implement Federated Averaging ([FedAvg](https://arxiv.org/abs/1602.05629)). 
This workflow follows the hub and spoke model for communicating the global model to each client for local training (i.e., "scattering") and aggregates the result to perform the global model update (i.e., "gathering").  

> **_NOTE:_** This example uses a Numpy-based trainer and will generate its data within the code.

You can follow the [Getting Started with NVFlare (Numpy)](hello-fedavg-numpy_getting_started.ipynb)
for a detailed walkthrough of the basic concepts.

See the [Hello FedAvg with NumPy](https://nvflare.readthedocs.io/en/main/examples/hello_fedavg_numpy.html) example documentation page for details on this
example.

To run this example with the FLARE API, you can follow the [hello_world notebook](../hello_world.ipynb), or you can quickly get
started with the following:

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.

### 2. Run the experiment

Run the script using the job API to create the job and run it with the simulator:

```
python3 fedavg_script_executor_hello-numpy.py
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir/
```

For how to use the FLARE API to run this app, see [this notebook](hello-fedavg-numpy_flare_api.ipynb).
