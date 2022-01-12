# Hello MONAI

Example of using [NVIDIA FLARE](https://nvidia.github.io/NVFlare) to train a medical image analysis model using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [MONAI](https://monai.io/), the "Medical Open Network for Artificial Intelligence", as the deep learning training framework.

See this [Tutorial](https://github.com/Project-MONAI/tutorials/tree/master/federated_learning/nvflare/nvflare_spleen_example) for an example of how to use this trainer for 3D spleen segmentation in computed tomography.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvidia.github.io/NVFlare/installation.html) instructions.
Install additional requirements:

```
pip3 install monai
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvidia.github.io/NVFlare/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
set_run_number 1
upload_app hello-monai
deploy_app hello-monai all
start_app all
```

### 4. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvidia.github.io/NVFlare/user_guide/admin_commands.html).
