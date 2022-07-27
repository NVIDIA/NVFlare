# Hello MONAI

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train a medical image analysis model using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [MONAI Bundle](https://docs.monai.io/en/latest/mb_specification.html).


### 1. Install NVIDIA FLARE

Follow the [Installation](https://github.com/NVIDIA/NVFlare#installation) instructions.
Install additional requirements:

```
python3 -m pip install monai[nibabel]==0.9.0
python3 -m pip install tqdm
python3 -m pip install pytorch-ignite
python3 -m pip install fire
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to set up your POC ("proof of concept") workspace.
The folder structure is like:
```
current_path/
	poc/
	hello-monai-bundle/
```

### 3. Download Spleen Bundle

```
python -m monai.bundle download --name "spleen_ct_segmentation_v0.1.0" --bundle_dir hello-monai-bundle/app/config/
mkdir -p poc/admin/transfer
cp -rf hello-monai-bundle poc/admin/transfer
```

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

> **_NOTE:_** For more information about the Admin client, see [here](https://nvflare.readthedocs.io/en/main/user_guide/operation.html).
