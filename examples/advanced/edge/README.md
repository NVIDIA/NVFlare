# Running NVFlare Mobile Example


## Setup the NVFlare System

### Prepare the Workspace
```commandline
./setup_nvflare.sh
```
For details, please refer to [setup NVFlare system for Edge](./setup_system.md)
This will create a deployment with 4 leaf nodes, 2 aggregators, 2 relays, and 1 server. 

For edge-device connection, we only needs the information of the leaf nodes, let's check the lcp map:
```commandline
cat /tmp/nvflare/workspaces/edge_example/prod_00/demo/lcp_map.json
```
We can see the address and port of each leaf node, which will be used by the mobile devices to connect to the system.
```
{
    "C11": {
        "host": "localhost",
        "port": 9003
    },
    "C12": {
        "host": "localhost",
        "port": 9004
    },
    "C21": {
        "host": "localhost",
        "port": 9006
    },
    "C22": {
        "host": "localhost",
        "port": 9007
    }
}
```
## Start the NVFlare System

To start the system, run the following command:
```commandline
./start_nvflare.sh
```    

## Start the Mobile App
Install the app from App store and open it.

You will see the following screen:

<img src="./screenshot.png" alt="App Screenshot" width="400" height="800">

You need to configure the server PORT to be the PORT shown in lcp_map.json (for example: 9003).

And you can find out the IP address of your machine and fill it there.

Then click "Start Training". (This will be enhanced in the future by adding resource monitoring to auto start/stop training)

## Prepare and Submit a Job

We have prepared two jobs for you: [xor_mobile_et](./jobs/xor_mobile_et/) and [cifar10_mobile_et](./jobs/cifar10_mobile_et/).
You can easily write your own components to replace any of the pre-configured ones. 

First, copy the jobs to the admin console transfer folder:

```commandline
cp -r ./jobs/* /tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/transfer
```

Start the admin console to interact with the NVFlare system:

```commandline
/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/startup/fl_admin.sh
```

Submit a job:

```
submit_job cifar10_mobile_et
```

You will then see the device start receiving the model from the server and complete local training.
The server will perform aggregation and proceed to the next round.
After the configured rounds have finished, the training is complete!

## [Optional] Local Proof-Of-Concept with Simulated TaskProcessingDevice: an End-to-end Cifar10 Example 
Above we show how to run the mobile example with the NVFlare system on an actual device. For prototyping and testing a cross-device FL pipeline, 
we usually do not start with real devices. Therefore, NVFlare provides flexible mechanisms to simulate devices for testing the FL process.

In the following, we will show how to run a simulated cross-device federated learning with the same NVFlare system we just started. The simulated devices will be 
talking to leaf nodes directly without going through the proxy.

Let's run an end-to-end example with Cifar10 dataset with baseline comparisons.

### Baselines
First, let's run the centralized baseline on the whole dataset, and a 16-client federated baseline via NVFlare's standard single-layer pipeline using [JobAPI](https://nvflare.readthedocs.io/en/main/programming_guide/fed_job_api.html).
1. Run the centralized baseline
```commandline
cd baselines
python cifar_train_central.py
cd ..
```
2. Run the federated baseline under regular single-layer setting
```commandline
cd baselines
python cifar_fl_base_job.py
cd ..
```

### Simulated Cross-Device Federated Learning
Assuming the previous steps are completed, we can now run the end-to-end example with the same already prepared NVFlare system.
#### Step1: Start the NVFlare System
Again, we first start the system, open a new terminal window and run the following command:
```commandline
./start_nvflare.sh
```  

#### Step2: Generate Job Configs using the EdgeJob API
Next, let's generate job configs for cifar10 via EdgeJob API.

```commandline
cd jobs
python cifar10_job.py --job_name cifar10_sync_job --simulation_config_file configs/cifar10_silo_config.json --device_reuse --const_selection
python cifar10_job.py --job_name cifar10_async_job --simulation_config_file configs/cifar10_silo_config.json --min_hole_to_fill 1 --global_lr 0.05 --max_model_aggr 40 --max_model_history 40 --num_updates_for_model 1 --max_model_version 160 --eval_frequency 16 --device_reuse --const_selection

python cifar10_job.py --job_name cifar10_sync_lcp_job --device_reuse --const_selection
python cifar10_job.py --job_name cifar10_async_lcp_job --min_hole_to_fill 1 --global_lr 0.05 --max_model_aggr 40 --max_model_history 40 --num_updates_for_model 1 --max_model_version 160 --eval_frequency 16 --device_reuse --const_selection
cd ..
```

#### Step3: Submit NVFlare Job
First, copy the job to the admin console transfer folder:
```commandline
cp -r /tmp/nvflare/jobs/* /tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/transfer
```

Start the admin console to interact with the NVFlare system:
```commandline
/tmp/nvflare/workspaces/edge_example/prod_00/admin@nvidia.com/startup/fl_admin.sh
```

For simulations performed directly at leaf nodes, simply submit the job:
```
submit_job cifar10_sync_job
```

You will then see the simulated devices start receiving the model from the server and complete local trainings.

### Results
#### Federated Training v.s. Centralized Training
After the configured rounds have finished, the training is complete, now let's check the training results.
```commandline
tensorboard --logdir=/tmp/nvflare/workspaces
```
With the centralized training of 10 epochs, and the federated training of 10 rounds (4 local epoch per round), you should see the following results:
<img src="./figs/cifar10_acc.png" alt="Cifar10 Results" width="800" >

Red curve is the centralized training, blue is the baseline federated training with regular single-layer setting, and green is the simulated cross-device federated training.
The three learning will converge to similar accuracy, note that in this case each client holds partial data that is 1/16 of the whole training set sequentially split.

#### Synchronous v.s. Asynchronous Federated Training
Comparing synchronous (sync) vs. asynchronous (async) training, as configured above, we tested an async scheme that produces a new global model after receiving 1 model update, compared to the sync scheme which requires 16 model updates to generate a new global model. 

Theoretically, the async scheme has a latency of the communication time plus the **average time** of all devices to complete a local training. In comparison, the sync scheme has a latency of the **slowest** device to complete a local training.

Under our current setting where each device is uniformly sampled from three different device types, each modeled as an independent Gaussian distribution, we have the expectation of one-round FL approximately:
- Sync scheme: expectation of the max of the three Gaussian plus the communication mean $40 + (3/2)\pi^{-1/2} \times 4 + 5 = 48.4$
- Async scheme: the average of the three means plus the communication mean $(10+20+40)/3 + 5 = 28.3$

So if we omit other time costs, async scheme should be about 60% of the sync scheme.

Now let's take a look at the results of the two schemes. Note that here we set the global learning rate to 0.05 for the async scheme, and 1.0 for the sync scheme. To match the total number of model updates processed, we let the async scheme run for 160 model versions as compared with 10 rounds of sync training.

The global accuracy curves are shown below, with x-axis representing the relative time of the training process, and y-axis representing the global accuracy:
<img src="./figs/async_comp.png" alt="Cifar10 Async Results" width="800" >

The blue curve represents async training, and the orange curve represents sync training. Under iid data-split with 16 concurrent devices, async scheme 
achieved comparable global accuracy while taking ~60% the time as compared with sync scheme as expected. 