# Hello PyTorch-Lighting with MLflow

This example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch Lightning](https://lightning.ai/)
as the deep learning training framework.
> Minimum Hardware Requirements: **1 CPU or 1 GPU**

This example also highlights the MLflow streaming capability from the clients to the server.
as well as using FLARE's lightning logger.

### 1. Install requirements

Install additional requirements:

Assuming the current directory is "examples/advanced/experiment-tracking/mlflow/jobs/hello-lightning-mlflow", 
run the following command to install the requirements:

```
python3 -m pip install -r requirements.txt
```
### 2. Download data
Here we just use the same data for each site. It's better to pre-downloaded data to avoid multiple sites to concurrent download the same data.
We are still assuming we are in the directory "examples/advanced/experiment-tracking/mlflow/jobs/hello-lightning-mlflow"

```bash
examples/advanced/experiment-tracking/prepare_data.sh
```

### 3. Run the experiment

Use nvflare job api with simulator to run the example:

```
cd ./jobs/hello-lightning-mlflow/code

python3 fl_job.py
```

### 4. Access the logs and results

You can find the running logs and results inside the server's simulator's workspace in a directory named "simulate_job".

```WORKSPACE = "/tmp/nvflare/jobs/workdir"```

By default, MLflow will create an experiment log directory under a directory named "mlruns" in the simulator's workspace. 
If you ran the simulator with "/tmp/nvflare/jobs/workdir" as the workspace, then you can launch the MLflow UI with:

```bash
$ tree /tmp/nvflare/jobs/workdir/server/simulate_job/mlruns/
```

```
mlflow ui --backend-store-uri /tmp/nvflare/jobs/workdir/server/simulate_job/mlruns/
```

### 5. MLflow Streaming

tracking_uri=f"file://{WORKSPACE}/server/simulate_job/mlruns",

For the job `hello-pt-mlflow`, on the client side, the client code in `client.py`

```
mlflow_writer.log_metric(key="local_accuracy", value=local_accuracy, step=global_step)
```

The `MLflowWriter` actually mimics the mlflow to send the information in events to the server through NVFlare events
of type `analytix_log_stats` for the server to write the data to the MLflow tracking server.

The `ConvertToFedEvent` widget turns the event `analytix_log_stats` into a fed event `fed.analytix_log_stats`,
which will be delivered to the server side.

On the server side, the `MLflowReceiver` is configured to process `fed.analytix_log_stats` events,
which writes received data from these events to the MLflow tracking server.

This allows for the server to be the only party that needs to deal with authentication for the MLflow tracking server, and the server
can buffer the events from many clients to better manage the load of requests to the tracking server.

