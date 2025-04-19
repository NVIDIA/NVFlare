# Hello PyTorch with MLflow: Streaming metrics to client side 

NVIDIA FLARE's metrics streaming capability is quite flexible. If you decide to stream the metrics to the client side instead of passing through the NVFlare server, you can do that as well. 
This example demonstrates how to set up such a configuration. 

Here we use the CIFAR10 example to train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework.

The basic setup and code is almost the same as the ```hello-pt-mlflow``` example. 

The only differences are the following:
1) We will add a ```MLflowReceiver``` component on the client side instead of on the server side
2) We don't need to convert the local event to federated event, i.e., we don't need to register the ```ConvertToFedEvent``` component


In the FLARE job API, 

we add two arguments to indicate that we don't need to use ```convert_to_fed_event``` and don't need to
set a default ```analytics_receiver```

convert_to_fed_event = False,
analytics_receiver =False
```
    job = FedAvgJob(name=job_name, n_clients=n_clients, num_rounds=num_rounds, initial_model=SimpleNetwork(), 
                    convert_to_fed_event = False,
                    analytics_receiver =False)
```

For the MLflowReceiver configuration:
```
        tracking_uri = f"file://{work_dir}/site-{i + 1}/mlruns"
        receiver = MLflowReceiver(
            tracking_uri=tracking_uri,
            events=[ANALYTIC_EVENT_TYPE],
            ...
            )
```
We set the events we listen to with `events=[ANALYTIC_EVENT_TYPE]`. By default, it listens to Fed Events `[f"fed.{ANALYTIC_EVENT_TYPE}"]`

### 1. Install requirements

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

Use nvflare job api with simulator to run the example:

```bash
../prepare_data.sh
```

### 3. Run the experiment

Use nvflare job api with simulator to run the example:

```
cd ./jobs/hello-pt-mlflow-client/code

python3 fl_job.py
```

### 4. Access the logs and results

You can find the running logs and results inside the server simulator's workspace in a directory named "simulate_job".

```WORKSPACE = "/tmp/nvflare/jobs/workdir"```
  
Now, since each site has a receiver, and we have set the tracking URI as:

```tracking_uri = f"file://{work_dir}/site-{i + 1}/mlruns"```

We should be able to access the tracking data at:
```
mlflow ui --backend-store-uri /tmp/nvflare/jobs/workdir/site-2/mlruns
```

