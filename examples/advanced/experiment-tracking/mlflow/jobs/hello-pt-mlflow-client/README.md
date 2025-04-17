# Hello PyTorch with MLflow: Streaming metrics to client side 

NVIDIA FLARE metrics streaming capability is quite flexible. If you deciced to stream the metrics to client side instread of without passing through the NVFlare server, you can do that as well. 
This example demonstrates how to do such setup. 

here we use the CIFAR10 example to train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework.

The normal setup and code is almost the same as ```hello-pt-mlflow``` example. 

The only differences are the followings
1) we will registered a ```MLflowReceiver``` on the Client side instead of on Server Side
2) we don't need to convert the local event to federated event, i.e don't need register ```ConvertToFedEvent``` component


In FLARE job API, 

```
    # create a FedAvg job but not registered the default components for fed_event converter and analytics receiver.
    # As we are going to add ourselves. 

    job = FedAvgJob(name=job_name, n_clients=n_clients, num_rounds=num_rounds, initial_model=SimpleNetwork(), 
                                convert_to_fed_event = False, analytics_receiver =False)
    
    # Add a MLFlow Receiver component to the Client site 
    receiver = MLflowReceiver(...)
    
    # Add clients
    for i in range(n_clients):
        ...
        job.to(receiver, f"site-{i + 1}")

```


### 1. Install requirements

Install additional requirements (if you already have a specific version of nvflare installed in your environment, you may want to remove nvflare in the requirements to avoid reinstalling nvflare):

```
python -m pip install -r requirements.txt
```
### 2. Download data
Here we just use the same data for each site. It's better to pre-downloaded data to avoid multiple sites to concurrent download the same data.

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

Now, since each site has a reciever, and we are have set the tracking URI as 

```tracking_uri = f"file://{work_dir}/site-{i + 1}/mlruns"```

We should be able to look at the 
tracking_uri=f"file://{WORKSPACE}/site-n/mlruns",

```
 mlflow ui --backend-store-uri /tmp/nvflare/jobs/workdir/site-2/mlruns

```

