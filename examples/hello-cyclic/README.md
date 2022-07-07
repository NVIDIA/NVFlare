# Hello Cyclic Weight Transfer

["Cyclic Weight Transfer"](https://pubmed.ncbi.nlm.nih.gov/29617797/
) (CWT) is an alternative to the scatter-and-gather approach used in [FedAvg](https://arxiv.org/abs/1602.05629). CWT uses the [CyclicController](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic_ctl.html) to pass the model weights from one site to the next for repeated fine-tuning.

> **_NOTE:_** This example uses the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits dataset and will load its data within the trainer code.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
Install additional requirements:

```
pip3 install tensorflow
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
submit_job hello-cyclic
```

### 4. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvflare.readthedocs.io/en/main/user_guide/operation.html).
