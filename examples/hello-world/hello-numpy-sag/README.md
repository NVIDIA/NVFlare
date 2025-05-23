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
