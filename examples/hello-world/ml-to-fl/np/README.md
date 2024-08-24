# NVFlare Client API

In this example we use simple numpy scripts to showcase the Client API with the ScriptRunner in both in-process and sub-process settings.

## Software Requirements

Please install the requirements first, it is suggested to install inside a virtual environment:

```bash
pip install -r requirements.txt
```

## Minimum Hardware Requirements

1 CPU


## In-process Client API

The default mode of the `ScriptRunner` with `launch_external_process=False` uses the `InProcessClientAPIExecutor` for in-process script execution.
With the `InProcessClientAPIExecutor`, the client training script operates within the same process as the NVFlare Client job.
This provides benefits with efficient shared the memory usage and a simple configuration useful for development or single GPU use cases.

### Send model parameters back to the NVFlare server

We use the mock training script in [./src/train_full.py](./src/train_full.py)
And we send back the FLModel with "params_type"="FULL" in [./src/train_full.py](./src/train_full.py)

After we modify our training script, we can create a job using the ScriptRunner: [np_client_api_job.py](./np_client_api_job.py).
(Please refer to [FedJob API](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html) for more details on formulating a job)

Then we can run the job using the simulator with the Job API. (This is equivalent to using the CLI command `nvflare simulator <job_folder>`)

```bash
python3 np_client_api_job.py --script src/train_full.py
```

Note: We can instead export the job configuration to use in other modes with the flag `--export_config`.

### Send model parameters differences back to the NVFlare server

We can send model parameter differences back to the NVFlare server by calculating the parameters differences and sending it back: [./src/train_diff.py](./src/train_diff.py)

Note that we set the "params_type" to DIFF when creating flare.FLModel.

Then we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_job.py --script src/train_diff.py
```

### Metrics streaming

Sometimes we want to stream the training progress to the server.

We have several ways of doing that:

  - `SummaryWriter` mimics Tensorboard `SummaryWriter`'s `add_scalar`, `add_scalars` method
  - `WandBWriter` mimics Weights And Biases's `log` method
  - `MLflowWriter` mimics MLflow's tracking api

In this example we use `MLflowWriter` in [./src/train_metrics.py](./src/train_metrics.py) and configure a corresponding `MLflowReceiver` in the job script [np_client_api_job.py](np_client_api_job.py)

Then we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_job.py --script src/train_metrics.py
```

After the experiment is finished, you can view the results by running the the mlflow command: `mlflow ui --port 5000` inside the directory `/tmp/nvflare/jobs/workdir/server/simulate_job/`.

Please refer to MLflow examples and documentation for more information.


## Sub-process Client API

The `ScriptRunner` with `launch_external_process=True` uses the `ClientAPILauncherExecutor` for external process script execution.
This configuration is ideal for scenarios requiring multi-GPU or distributed PyTorch training.

### Launching the script

When launching a script in an external process, it is launched once for the entire job.
We must ensure our training script [./src/train_full.py](./src/train_full.py) is in a loop to support this.

Then we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_job.py --script src/train_full.py --launch_process
```

### Metrics streaming

In this example we use `MLflowWriter` in [./src/train_metrics.py](./src/train_metrics.py) and configure a corresponding `MLflowReceiver` in the job script [np_client_api_job.py](np_client_api_job.py)

Then we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_job.py --script src/train_metrics.py --launch_process
```
