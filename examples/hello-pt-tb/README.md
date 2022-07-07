# Hello PyTorch with Tensorboard Streaming

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) and [PyTorch](https://pytorch.org/) as the deep learning training framework. This example also highlights the TensorBoard streaming capability from the clients to the server.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.
Install additional requirements:

```
pip3 install torch torchvision tensorboard
```

### 2. Set up your FL workspace

Follow the [Quickstart](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions to set up your POC ("proof of concept") workspace.

### 3. Run the experiment

Log into the Admin client by entering `admin` for both the username and password.
Then, use these Admin commands to run the experiment:

```
submit_job hello-pt-tb
```

After submitting the job, you will see the job ID in the admin client output, for example:

```
Submitted job: 9694041a-1999-4509-8ca8-ac1137cf94ae
```

This job ID will be used in the following section to access the server's TB files.

### 4. Tensorboard Streaming

On the client side, the `AnalyticsSender` works as a TensorBoard SummaryWriter. Instead of writing to TB files, it actually generates NVFLARE events of type `analytix_log_stats`.
The `ConvertToFedEvent` widget will turn the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`, which will be delivered to the server side.

On the server side, the `TBAnalyticsReceiver` is configured to process `fed.analytix_log_stats` events, which writes received TB data into appropriate TB files on the server
(defaults to `server/[job ID]/tb_events`).

To view training metrics that are being streamed to the server, run:

```
tensorboard --logdir=poc/server/[job ID]/tb_events
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

> **_NOTE:_** For a more in-depth guide about the TensorBoard streaming feature, see [Quickstart (PyTorch with TensorBoard)](https://nvflare.readthedocs.io/en/main/examples/hello_pt_tb.html).
