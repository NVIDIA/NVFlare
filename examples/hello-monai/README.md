# Hello MONAI

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train a medical image analysis model using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [MONAI](https://monai.io/), the "Medical Open Network for Artificial Intelligence", as the deep learning training framework.

See this [Tutorial](https://github.com/Project-MONAI/tutorials/tree/master/federated_learning/nvflare/nvflare_spleen_example) for an example of how to use this trainer for 3D spleen segmentation in computed tomography.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
Install additional requirements:

```
pip3 install monai
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
submit_job hello-monai
```


### 4. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvflare.readthedocs.io/en/main/user_guide/operation.html).
