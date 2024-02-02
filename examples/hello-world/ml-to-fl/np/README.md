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


## Send model parameters back to the NVFlare server

We use the mock training script in [./code/train_full.py](./code/train_full.py)
And we send back the FLModel with "params_type"="FULL" in [./code/train_full.py](./code/train_full.py)

To send back the whole model parameters, we need to make sure the "params_transfer_type" is also "FULL".

Let reuse the job templates from [sag_np](../../../../job_templates/sag_np/):

```bash
nvflare job create -force -j ./jobs/np_param_full_transfer_full -w sag_np -sd ./code/ \
-f config_fed_client.conf app_script=train_full.py params_transfer_type=FULL launch_once=false
```

Then we can run it using the NVFlare Simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/np_param_full_transfer_full -w np_param_full_transfer_full_workspace
```

## Send model parameters differences back to the NVFlare server

There are two ways to send model parameters differences back to the NVFlare server:

1. Send the full parameters in training script, change params_transfer_type to "DIFF"
2. Calculate the parameters differences in training script and send it back via "flare.send"

For the first way, we can reuse the mock training script [./code/train_full.py](./code/train_full.py)

But we need to pass different parameters when creating job:

```bash
nvflare job create -force -j ./jobs/np_param_full_transfer_diff -w sag_np -sd ./code/ \
-f config_fed_client.conf app_script=train_full.py params_transfer_type=DIFF launch_once=false \
-f config_fed_server.conf expected_data_kind=WEIGHT_DIFF
```

By setting "params_transfer_type=DIFF" we are using the NVFlare built-in parameter difference method to calculate differences.

Then we can run it using the NVFlare Simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/np_param_full_transfer_diff -w np_param_full_transfer_diff_workspace
```

For the second way, we write a new mock training script that calculate the model difference and send it back: [./code/train_diff.py](./code/train_diff.py)

Note that we set the "params_type" to DIFF when creating flare.FLModel.

Then we create the job using the following command:

```bash
nvflare job create -force -j ./jobs/np_param_diff_transfer_full -w sag_np -sd ./code/ \
-f config_fed_client.conf app_script=train_diff.py launch_once=false \
-f config_fed_server.conf expected_data_kind=WEIGHT_DIFF
```

The "params_transfer_type" is "FULL", means that we DO NOT calculate the difference again using the NVFlare built-in parameter difference method.

Then we can run it using the NVFlare Simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/np_param_diff_transfer_full -w np_param_diff_transfer_full_workspace
```

## Launch once for the whole job

In some training scenarios, the data loading is taking a lot of time.
And throughout the whole training job, we only want to load/set up the data once.

In that case, we could use the "launch_once" option of "SubprocessLauncher" and wraps our training script into a loop.

We wrap the [./code/train_full.py](./code/train_full.py) into a loop: [./code/train_loop.py](./code/train_loop.py)

Then we can create the job:

```bash
nvflare job create -force -j ./jobs/np_loop -w sag_np -sd ./code/ \
-f config_fed_client.conf app_script=train_loop.py params_transfer_type=FULL launch_once=true
```

Then we can run it using the NVFlare Simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/np_loop -w np_loop_workspace
```

## Data exchange mechanism

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

First, let's create the job using the sag_np_cell_pipe template

```bash
nvflare job create -force -j ./jobs/np_loop_cell_pipe -w sag_np_cell_pipe -sd ./code/ \
-f config_fed_client.conf app_script=train_loop.py params_transfer_type=FULL launch_once=true
```

Then we can run it using the NVFlare Simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/np_loop_cell_pipe -w np_loop_cell_pipe_workspace
```

## Launch once for the whole job and with metrics streaming

Sometimes we want to stream the training progress to the server.

We have several ways of doing that:

  - `SummaryWriter` mimics Tensorboard `SummaryWriter`'s `add_scalar`, `add_scalars` method
  - `WandBWriter` mimics Weights And Biases's `log` method
  - `MLflowWriter` mimics MLflow's tracking api
  - `flare.log` is the underlying common pattern that can be directly used as well, you need to figure out the
    corresponding `AnalyticsDataType` for your value

We showcase `MLflowWriter` in [./code/train_metrics.py](./code/train_metrics.py)

After that, we can set up the job using the sag_np_metrics template:

```bash
nvflare job create -force -j ./jobs/np_metrics -w sag_np_metrics -sd ./code/ \
-f config_fed_client.conf app_script=train_metrics.py params_transfer_type=DIFF launch_once=true \
-f config_fed_server.conf expected_data_kind=WEIGHT_DIFF
```

Once the job is set up, we can run it using the NVFlare Simulator:

```bash
nvflare simulator -n 2 -t 2 ./jobs/np_metrics -w np_metrics_workspace
```

Keep in mind that the difference between sag_np_cell_pipe and sag_np_metrics is the
addition of components like "metrics_pipe," "metric_relayer," and "event_to_fed."
These components allow values from an external process to be sent back to the server.

