# Hello PyTorch with Weights and Biases

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework.

This example also highlights the Weights and Biases streaming capability from the clients to the server.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.

### 1. Install requirements

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):

```
python -m pip install -r requirements.txt
```

### 2. Make sure the FL server is logged into Weights and Biases

Import the W&B Python SDK and log in:

```
python3
>>> import wandb
>>> wandb.login()
```

Provide your API key when prompted.

### 3. Run the experiment

Use job api to run the example:

```
python wandb_job.py
```

### 4. Access the logs and results

By default, Weights and Biases will create a directory named "wandb" in the server workspace. With "mode": "online" in the WandBReceiver, the
files will be synced with the Weights and Biases server. You can visit https://wandb.ai/ and log in to see your run data.

### 5. How it works

To enable tracking with Weights & Biases (WandB), you can use the `WandBWriter` utility provided by NVFlare. Here's a basic example of how to integrate it into your training script:

```python
from nvflare.client.tracking import WandBWriter

wandb_writer = WandBWriter()
wandb_writer.log({"train_loss": cost.item()}, current_step)

```

The `WandBWriter` follows a syntax similar to the native WandB API, making it easy to adopt.

Under the hood, `WandBWriter` uses NVFlare’s event system to send tracking data. Specifically, it fires an `analytix_log_stats` event. On the server side, this event needs to be received and processed — which is where the `WandBReceiver` comes in.

By default, we configured `WandBReceiver` and a `ConvertToFedEvent` on the NVFlare server side.

The `ConvertToFedEvent` widget turns the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`, enabling it to be sent from the NVFlare client to the NVFlare server.

The `WandBReceiver` listens for `fed.analytix_log_stats` events on the NVFlare server side and forwards the metric data to the WandB tracking server.

This allows for the server to be the only party that needs to deal with authentication for the WandB tracking server, and the server
can buffer the events from many clients to better manage the load of requests to the tracking server.

### 6. Optional: Stream Metrics Directly from Clients

Alternatively, you can stream metrics **directly from each NVFlare client** to WandB, bypassing the NVFlare server entirely.

To enable this mode, run your training script with the following flags:

```bash
python wandb_job.py --streamed_to_clients --no-streamed_to_server
```

In this mode, the `WandBReceiver` is configured on the NVFlare client side to process the `analytix_log_stats` event.

So each NVFlare client will directly send the metrics to their corresponding WandB server.
