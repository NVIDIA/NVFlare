# Hello MONAI

Example of using [NVIDIA FLARE](https://nvidia.github.io/NVFlare) to train a medical image analysis model using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [MONAI Bundle](https://docs.monai.io/en/latest/mb_specification.html).


### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvidia.github.io/NVFlare/installation.html) instructions.
Install additional requirements:

```
pip3 install monai==0.9.0
pip3 install fire
```

### 2. Download Spleen Bundle

(the following command will be valid after the model-zoo repo is public)

```
cd configs
python -m monai.bundle download --name "spleen_ct_segmentation_v0.1.0"
unzip spleen_ct_segmentation_v0.1.0.zip
```

### 3. Set up your FL workspace

Follow the [Quickstart](https://nvidia.github.io/NVFlare/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 4. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
submit_job hello-monai-bundle
```

### 5. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvidia.github.io/NVFlare/user_guide/admin_commands.html).
