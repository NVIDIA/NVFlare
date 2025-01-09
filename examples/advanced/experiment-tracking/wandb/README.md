# Hello PyTorch with Weights and Biases

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework.

This example also highlights the Weights and Biases streaming capability from the clients to the server.

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

### 2. Make sure the FL server is logged into Weights and Biases

Import the W&B Python SDK and log in:

```
python3
>>> import wandb
>>> wandb.login()
```

Provide your API key when prompted.

### 3. Run the experiment

Use nvflare simulator to run the example:

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 ./jobs/hello-pt-wandb
```

### 3. Access the logs and results

By default, Weights and Biases will create a directory named "wandb" in the server workspace. With "mode": "online" in the WandBReceiver, the
files will be synced with the Weights and Biases server. You can visit https://wandb.ai/ and log in to see your run data.

### 4. Weights and Biases tracking

For the job `hello-pt-wandb`, on the client side, the client code in `PTLearner` uses the syntax for Weights and Biases:

```
self.writer.log({"train_loss": cost.item()}, current_step)

self.writer.log({"validation_accuracy": metric}, epoch)
```

The `WandBWriter` mimics the syntax from Weights and Biases to send the information in events to the server through NVFlare events
of type `analytix_log_stats` for the server to write the data for the WandB tracking server.

The `ConvertToFedEvent` widget turns the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`,
which will be delivered to the server side.

On the server side, the `WandBReceiver` is configured to process `fed.analytix_log_stats` events,
which writes received data from these events.

This allows for the server to be the only party that needs to deal with authentication for the WandB tracking server, and the server
can buffer the events from many clients to better manage the load of requests to the tracking server.
