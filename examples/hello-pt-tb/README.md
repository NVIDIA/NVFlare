# Hello PyTorch with Tensorboard Streaming

Example of using [NVIDIA FLARE](https://nvidia.github.io/NVFlare) to train an image classifier using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [PyTorch](https://pytorch.org/) as the deep learning training framework. This example also highlights the TensorBoard streaming capability from the clients to the server.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvidia.github.io/NVFlare/installation.html) instructions.
Install additional requirements:

```
pip3 install torch torchvision tensorboard
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvidia.github.io/NVFlare/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
set_run_number 1
upload_app hello-pt-tb
deploy_app hello-pt-tb all
start_app all
```

### 4. Tensorboard Streaming

With `AnalyticsSender` and `ConvertToFedEvent` configured on the client, and `TBAnalyticsReceiver` configured on the server, the clients will stream TensorBoard events to the server in the folder `server/run_1/tb_events`.

To view training metrics that are being streamed to the server, run:

```
tensorboard --logdir=poc/server/run_1/tb_events
```

Note: if the server is running on a remote machine, use port forwarding to view the TensorBoard dashboard in a browser. For example:
```
ssh -L {local_machine_port}:127.0.0.1:6006 user@server_ip)
```

### 5. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvidia.github.io/NVFlare/user_guide/admin_commands.html).
