# NVFlare Client API

We will demonstrate how to send back model parameters or model parameters differences in different approaches in the following examples:

  1. [Send model parameters back to the NVFlare server](#send-model-parameters-back-to-the-nvflare-server)
  2. [Send model parameters differences back to the NVFlare server](#send-model-parameters-differences-back-to-the-nvflare-server)


By default, the "SubprocessLauncher" is going to launch the script once for a job.

If your data setup is taking a long time, you don't want to launch the whole training script every round.
(This implies that the dataset will be loaded again every round and all the cache will be lost for each round).

On the other hand, if your system is very resource limited, and you don't want the training process to live throughout the whole
job training, you can use "launch_once=False".

We demonstrate how to launch training script once and have training script keeps exchanging training parameters with NVFlare:

  1. [Launch once for the whole job](#launch-once-for-the-whole-job)

## Software Requirements

Please install the requirements first, it is suggested to install inside a virtual environment:

```bash
pip install -r requirements.txt
```

Please also configure the job templates folder:

```bash
nvflare config -jt ../../../../job_templates/
nvflare job list_templates
```

## Minimum Hardware Requirements

1 CPU


## In-process Client API

With the ```InProcessClientAPIExecutor```, the client training script operates within the same process as the NVFlare Client job.
This provides benefits with efficient shared the memory usage and a simple configuration useful for development or single GPU use cases.


### Send model parameters back to the NVFlare server

We use the mock training script in [./src/train_full.py](./src/train_full.py)
And we send back the FLModel with "params_type"="FULL" in [./src/train_full.py](./src/train_full.py)

To send back the whole model parameters, we need to make sure the "params_transfer_type" is also "FULL".

After we modify our training script, we can create a job using the in-process script executor: [np_client_api_in_process_job.py](./np_client_api_in_process_job.py).
(Please refer to [FedJob API](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html) for more details on formulating a job)

```bash
python3 np_client_api_in_process_job.py --script src/train_full.py --params_transfer_type FULL
```

### Send model parameters differences back to the NVFlare server

There are two ways to send model parameters differences back to the NVFlare server:

1. Send the full parameters in training script, change params_transfer_type to "DIFF"
2. Calculate the parameters differences in training script and send it back via "flare.send"

For the first way, we can reuse the mock training script [./src/train_full.py](./src/train_full.py)

By setting "params_transfer_type=DIFF" we are using the NVFlare built-in parameter difference method to calculate differences.

Then we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_in_process_job.py --script src/train_full.py --params_transfer_type DIFF
```

For the second way, we write a new mock training script that calculate the model difference and send it back: [./src/train_diff.py](./src/train_diff.py)

Note that we set the "params_type" to DIFF when creating flare.FLModel.

The "params_transfer_type" is "FULL", means that we DO NOT calculate the difference again using the NVFlare built-in parameter difference method.

Then we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_in_process_job.py --script src/train_diff.py --params_transfer_type FULL
```

### Metrics streaming

Sometimes we want to stream the training progress to the server.

We have several ways of doing that:

  - `SummaryWriter` mimics Tensorboard `SummaryWriter`'s `add_scalar`, `add_scalars` method
  - `WandBWriter` mimics Weights And Biases's `log` method
  - `MLflowWriter` mimics MLflow's tracking api
  - `flare.log` is the underlying common pattern that can be directly used as well, you need to figure out the
    corresponding `AnalyticsDataType` for your value

We showcase `MLflowWriter` in [./src/train_metrics.py](./src/train_metrics.py) and a `MLflowReceiver` in the job script [np_client_api_in_process_job.py](np_client_api_in_process_job.py)

Once the job is set up with a `MLflowReceiver`, we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_in_process_job.py --script src/train_metrics.py --params_transfer_type DIFF
```


## Sub-process Client API

With the ```ClientAPILauncherExecutor``` and ``SubprocessLauncher`` the client training script runs in a separate subprocess.
Different communication mechanisms with the CellPipe and FilePipe can be used for different scenarios.
This configuration is ideal for scenarios requiring multi-GPU or distributed PyTorch training.

### Launch once for the whole job

In some training scenarios, the data loading is taking a lot of time.
And throughout the whole training job, we only want to load/set up the data once.

In that case, we could use the "launch_once" option of "SubprocessLauncher" and ensure our training script [./src/train_full.py](./src/train_full.py) is in a loop.

Then we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_ex_process_job.py --script src/train_full.py --params_transfer_type FULL --launch_once
```

### Launch for every task

Rather than launching once for the whole training job, we also have the option to launch for each task.
We can use the train script [./src/train_once.py](./src/train_once.py) which does not have a `while flare.is_running():` loop.

Then we can run it using the NVFlare Simulator:

```bash
python3 np_client_api_ex_process_job.py --script src/train_once.py --params_transfer_type FULL --no-launch_once
```


### Data exchange mechanism

The underlying communication between the external process and NVFlare client is facilitated by the `Pipe` class.

Two distinct types of `Pipe` are implemented:

1. FilePipe:
   - The `FilePipe` utilizes the file system for communication, involving read and write operations to a file.
   - Suitable when the NVFlare client and the external system/process share a common file system.
   - Ideal for scenarios where data exchange frequency is not high; however, it may not be efficient for high-frequency exchanges.

2. CellPipe:
    - The `CellPipe` leverages the `Cell` from NVFlare's foundation layer (f3) for communication. 
      This allows it to make use of drivers from the f3 layer, such as TCP, GRPC, HTTP, and any customized drivers.

    - Recommended for scenarios with a high frequency of data exchange (for example metrics logging)
      or when the file system is beyond your control.

You can also implement your own `Pipe`, please refer to https://github.com/NVIDIA/NVFlare/blob/main/nvflare/fuel/utils/pipe/pipe.py

So far, we have demonstrated how to use the `FilePipe`.
The following example illustrates how to use the `CellPipe`.

The CellPipe is currently not support with the Job API, so instead we can use a job template with the CellPipe.

First, let's create the job using the sag_np_cell_pipe template

```bash
nvflare job create -force -j ./jobs/np_loop_cell_pipe -w sag_np_cell_pipe -sd ./src/ \
-f config_fed_client.conf app_script=train_full.py params_transfer_type=FULL launch_once=true
```

Then we can run it using the NVFlare Simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/np_loop_cell_pipe -w np_loop_cell_pipe_workspace
```

### Launch once for the whole job with metrics streaming

Metrics streaming with the sub-process client API requires the use of CellPipe for high frequency data exchange. 

The CellPipe is currently not support with the Job API, so instead we can use a job template.

We use sag_np_metrics template which uses the CellPipe and components such as "metrics_pipe," "metric_relayer," and "event_to_fed." 
to allow values from an external process to be sent back to the server.

Create the job with the [./src/train_metrics.py](./src/train_metrics.py) script:

```bash
nvflare job create -force -j ./jobs/np_metrics -w sag_np_metrics -sd ./src/ \
-f config_fed_client.conf app_script=train_metrics.py params_transfer_type=DIFF launch_once=true \
-f config_fed_server.conf expected_data_kind=WEIGHT_DIFF
```

Once the job is set up, we can run it using the NVFlare Simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/np_metrics -w np_metrics_workspace
```
