# Hello Numpy Scatter and Gather

"[Scatter and Gather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html)" is the standard workflow to implement Federated Averaging ([FedAvg](https://arxiv.org/abs/1602.05629)). 
This workflow follows the hub and spoke model for communicating the global model to each client for local training (i.e., "scattering") and aggregates the result to perform the global model update (i.e., "gathering").  

> **_NOTE:_** This example uses a Numpy-based trainer and will generate its data within the code.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
submit_job hello-numpy-sag
```

### 4. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvflare.readthedocs.io/en/main/user_guide/operation.html).
