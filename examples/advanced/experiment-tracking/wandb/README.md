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

Internally, `WandBWriter` leverages the NVFlare client API to send metrics and trigger an `analytix_log_stats` event. This event can be received and processed by our `AnalyticsReceiver`, with the `WandBReceiver` being one implementation of it.

In `wandb_job.py`, we configure the following components by default:

  - The `ConvertToFedEvent` widget on the NVFlare client side, which transfroms the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`. This enables the event to be sent from the NVFlare client to the NVFlare server.

  - The `WandBReceiver` on the NVFlare server side, which listens for `fed.analytix_log_stats` events and forwards the metric data to the WandB tracking server.

This setup ensures that the server handles all authentication with the WandB tracking server and buffers events from multiple clients, effectively managing the load of requests to the server.

### 6. Optional: Stream Metrics Directly from Clients

Alternatively, you can stream metrics **directly from each NVFlare client** to WandB, bypassing the NVFlare server entirely.

To enable this mode, run your training script with the following flags:

```bash
python wandb_job.py --streamed_to_clients --no-streamed_to_server
```

In this configuration, the `WandBReceiver` is set up on the NVFlare client side to process the `analytix_log_stats` event.

As a result, each NVFlare client sends its metrics directly to its corresponding WandB server.
