# Hello PyTorch with Tensorboard Streaming

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework.

This example also highlights the TensorBoard streaming capability from the clients to the server.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

### 1. Install requirements and configure PYTHONPATH

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):

```
python -m pip install -r requirements.txt
```

Set `PYTHONPATH` to include custom files of this example:

```
export PYTHONPATH=${PWD}/..
```

### 2. Run the experiment

Use nvflare simulator to run the example:

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 ./jobs/tensorboard-streaming
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt tb_events

```

### 4. Tensorboard Streaming

On the client side, `TBWriter` works as a TensorBoard SummaryWriter.
Instead of writing to TB files, it actually generates NVFLARE events of type `analytix_log_stats`.
The `ConvertToFedEvent` widget will turn the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`,
which will be delivered to the server side.

On the server side, the `TBAnalyticsReceiver` is configured to process `fed.analytix_log_stats` events,
which writes received TB data into appropriate TB files on the server.

To view training metrics that are being streamed to the server, run:

```
tensorboard --logdir=/tmp/nvflare/simulate_job/tb_events
```

Note: If the server is running on a remote machine, use port forwarding to view the TensorBoard dashboard in a browser.
For example:
```
ssh -L {local_machine_port}:127.0.0.1:6006 user@server_ip
```

> **_NOTE:_** For a more in-depth guide about the TensorBoard streaming feature, see [PyTorch with TensorBoard](https://nvflare.readthedocs.io/en/main/examples/tensorboard_streaming.html).
