# Hello PyTorch with Tensorboard Streaming

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework.

This example also highlights the TensorBoard streaming capability from the clients to the server.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

### 1. Install requirements and configure PYTHONPATH

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare from the requirements to avoid reinstalling it):


```
python -m pip install -r requirements.txt
```

### 2. Download data
Here we just use the same data for each site. It's better to pre-download the data to avoid multiple sites concurrently downloading the same data.

```bash
../../../prepare_data.sh
```
### 3. Run the experiment

Use the nvflare job API to run the example:

In ```./jobs/tensorboard-streaming/code```:

```
python3 fl_job.py
```


### 4. Access the logs and results

You can find the running logs and results inside the simulator's workspace/<server name>/simulate_job

The workspace in fl_job.py is defined as "/tmp/nvflare/jobs/workdir":

```
Therefore, the results will be at: 

```bash
$ tree /tmp/nvflare/jobs/workdir/server/simulate_job/

/tmp/nvflare/jobs/workdir/server/simulate_job/
├── app_server
 <... skip ...>
└── tb_events
    ├── site-1
    │ └── events.out.tfevents.1744857479.rtx.30497.0
    └── site-2
      └── events.out.tfevents.1744857479.rtx.30497.1

```


### 5. Tensorboard Streaming

On the client side:

```
from nvflare.client.tracking import SummaryWriter
```
Instead of writing to TB files, this actually generates NVIDIA FLARE events of type `analytix_log_stats`.
The `ConvertToFedEvent` widget will turn the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`,
which will be delivered to the server side.

On the server side, the `TBAnalyticsReceiver` is configured to process `fed.analytix_log_stats` events,
which writes the received TB data into appropriate TB files on the server.

To view training metrics that are being streamed to the server, run:


```
tensorboard --logdir=/tmp/nvflare/jobs/workdir/server/simulate_job/tb_events
```

Note: If the server is running on a remote machine, use port forwarding to view the TensorBoard dashboard in a browser.
For example:

```
ssh -L {local_machine_port}:127.0.0.1:6006 user@server_ip
```

> **_NOTE:_** For a more in-depth guide about the TensorBoard streaming feature, see [PyTorch with TensorBoard](https://nvflare.readthedocs.io/en/main/examples/tensorboard_streaming.html).
