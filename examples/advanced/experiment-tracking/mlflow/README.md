# Hello PyTorch with MLflow

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework.

This example also highlights the MLflow streaming capability from the clients to the server.

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
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 ./jobs/hello-pt-mlflow
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace in a directory named "simulate_job".

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt tb_events

```

By default, MLflow will create an experiment log directory under a directory named "mlruns" in the simulator's workspace. If you ran the simulator with "/tmp/nvflare" as the workspace, then you can launch the MLflow UI with:

```
mlflow ui --backend-store-uri /tmp/nvflare/server/simulate_job/mlruns/
```

### 4. MLflow Streaming

For the job `hello-pt-mlflow`, on the client side, the client code in `PTLearner` uses the syntax for mlflow (to make it easier to use code already using tracking with MLflow):

```
self.writer.log_metrics({"train_loss": cost.item(), "running_loss": running_loss}, current_step)

self.writer.log_metric("validation_accuracy", metric, epoch)

self.writer.log_text(f"last running_loss reset at '{len(self.train_loader) * epoch + i}' step", "running_loss_reset.txt")
```

The `MLflowWriter` actually mimics the mlflow to send the information in events to the server through NVFlare events
of type `analytix_log_stats` for the server to write the data to the MLflow tracking server.

The `ConvertToFedEvent` widget turns the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`,
which will be delivered to the server side.

On the server side, the `MLflowReceiver` is configured to process `fed.analytix_log_stats` events,
which writes received data from these events to the MLflow tracking server.

This allows for the server to be the only party that needs to deal with authentication for the MLflow tracking server, and the server
can buffer the events from many clients to better manage the load of requests to the tracking server.

Note that the server also has `TBAnalyticsReceiver` configured, which also listens to `fed.analytix_log_stats` events by default,
so the data is also written into TB files on the server.

### 5. Tensorboard Streaming with MLflow

For the job `hello-pt-tb-mlflow`, on the client side, the client code in `PTLearner` uses the syntax for Tensorboard:

```
self.writer.add_scalar("train_loss", cost.item(), current_step)

self.writer.add_scalar("validation_accuracy", metric, epoch)
```

The `TBWriter` mimics Tensorboard SummaryWriter and streams events over to the server side instead.

Note that in this job, the server still has `MLflowReceiver` and `TBAnalyticsReceiver` configured the same as in the job with `MLflowWriter`
on the client side, and the events are converted by the `MLflowReceiver` to write to the MLflow tracking server.
