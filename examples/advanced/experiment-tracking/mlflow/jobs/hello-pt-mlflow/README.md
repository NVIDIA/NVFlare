# Hello PyTorch with MLflow

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/2.6/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework.

This example also highlights the MLflow streaming capability from the clients to the server.

### 1. Install requirements

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):

```
python -m pip install -r requirements.txt
```
### 2. Download data
Here we just use the same data for each site. It's better to pre-downloaded data to avoid multiple sites to concurrent download the same data.

```bash
../../../prepare_data.sh
```


### 3. Run the experiment

Use nvflare job api with simulator to run the example:

```
cd ./jobs/hello-pt-mlflow/code

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
mlflow ui --backend-store-uri tmp/nvflare/jobs/workdir/server/simulate_job/mlruns/
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


### 6. Experimental tracking with multi-receivers

NVIDIA FLARE experiment tracking is designed in such a way that, the metrics collector ( such as MLflowWriter, or SummaryWriter) are not directly tie to the metrics receivers (such as MLflowReceiver or TBAnalyticsReceiver)

The metrics collected can also streamed to any number of supported receivers, as long as it has registered receiver component.  By default, the ```BaseFedJob``` always pre-registered TensorBoard receiver for easy of use.  We can take a look at this by generating the job configuration of above job.  


```
cd ./jobs/hello-pt-mlflow/code

python3 fl_job.py -e 
```
The output will be something like this if the default values are used. 

```Exporting job config... /tmp/nvflare/jobs/fedavg```


We can now take a look at the configuration on the server

```
tmp/nvflare/jobs/fedavg
├── app_server
│ ├── config
│ │    └── config_fed_server.json

```



Note that the server also has `TBAnalyticsReceiver` configured, which also listens to `fed.analytix_log_stats` events by default, so the data is also written into TB files on the server.

```
cat /tmp/nvflare/jobs/fedavg/app_server/config/config_fed_server.json 

```
Notice we have two receiver components: TBAnalyticsReceiver, MLflowReceiver, which means we should also have a tensorboard results. 

```

{
 ...
    "components": [
      ....
        {
            "id": "receiver",
            "path": "nvflare.app_opt.tracking.tb.tb_receiver.TBAnalyticsReceiver",
            "args": {
                "events": [
                    "analytix_log_stats",
                    "fed.analytix_log_stats"
                ]
            }
        },
       
        {
            "id": "component",
            "path": "nvflare.app_opt.tracking.mlflow.mlflow_receiver.MLflowReceiver",
            "args": {
                "tracking_uri": "file:///tmp/nvflare/jobs/workdir/server/simulate_job/mlruns",
                "kw_args": {
                    "experiment_name": "nvflare-fedavg-experiment",
                    "run_name": "nvflare-fedavg-with-mlflow",
                    "experiment_tags": {
                        "mlflow.note.content": "## **NVFlare FedAvg experiment with MLflow**"
                    },
                    "run_tags": {
                        "mlflow.note.content": "## Federated Experiment tracking with MLflow.\n"
                    }
                },
                "artifact_location": "artifacts",
                "events": [
                    "fed.analytix_log_stats"
                ]
            }
        }
    ],
...

```

Now, let's take a look at this by directly loading the tensorboard
 
```
tensorboard --logdir=/tmp/nvflare/jobs/workdir/server/simulate_job/tb_events
```

**Note**
If you prefer not receive tensorboard metrics on server, you can simply remove the following 
component from the job configuration  
```json
        {
            "id": "receiver",
            "path": "nvflare.app_opt.tracking.tb.tb_receiver.TBAnalyticsReceiver",
            "args": {
                "events": [
                    "analytix_log_stats",
                    "fed.analytix_log_stats"
                ]
            }
        }
       
```

