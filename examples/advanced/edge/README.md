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

python cifar10_job.py --job_name cifar10_sync_no_delay_job --simulation_config_file configs/cifar10_silo_no_delay_config.json --device_reuse --const_selection
python cifar10_job.py --job_name cifar10_async_no_delay_job --simulation_config_file configs/cifar10_silo_no_delay_config.json --min_hole_to_fill 1 --global_lr 0.05 --max_model_aggr 40 --max_model_history 40 --num_updates_for_model 1 --max_model_version 160 --eval_frequency 16 --device_reuse --const_selection

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
Comparing synchronous (sync) vs. asynchronous (async) training, we tested an async scheme that produces a new global model once receiving 1 model update, compared to the sync scheme which requires all 16 model updates to generate a new global model. 

We compare the two schemes under two settings:
- No delay in local training by setting both **communication_delay** and **device_speed** to 0. In this case, since all devices are running in parallel and have essentially the same training data size, they are 
expected to finish local training at almost the same time, thus async scheme will not be able to accelerate the training.
- With delay in local training, we set **communication_delay** to 5 seconds, and **device_speed** to a Gaussian distribution with a large mean of 100.0, 200.0, or 400.0 seconds. 
In this case, the devices will finish local training at different times, thus async scheme is expected to accelerate the training.

For async scheme as we cast a new model whenever receiving an update, the overall expectation of additional latency will be the **mean** of all devices' latency

$(400+200+100)/3+5=238.3$

In comparison, the sync scheme has a latency of the **slowest** device to complete a local training, and under our current setting where each device is uniformly sampled from three different device types, each modeled as an independent Gaussian distribution, we have the expectation of the **max** of the three Gaussians plus the communication mean 

$400+(3/2)\pi^{-1/2}\times4+5=408.4$

So running for 10 rounds, comparing with training without delays, the async scheme will take approximately 2383 sec $\approx$ 39 min more, 
while the sync scheme will take approximately 4084 sec $\approx$ 68 min more.

Now let's take a look at the results of the two schemes. Note that here we set the global learning rate to 0.05 for the async scheme, and 1.0 for the sync scheme. To match the total number of model updates processed, we let the async scheme run for 160 model versions as compared with 10 rounds of sync training.

The global accuracy curves are shown below, with x-axis representing the relative time (in hours) of the training process, and y-axis representing the global accuracy:

<img src="./figs/async_comp.png" alt="Cifar10 Async Results" width="800" >

The dark blue curve represents async training without delay, orange for sync training without delay. 

As expected, in this setting, the async scheme does not accelerate the training process, and both schemes converge to similar accuracy at similar time around 10 min.

The light blue curve represents async training with delay, and the red curve represents sync training with delay.

As expected, the async scheme accelerates the training process, taking 45 min, 35 min more than the no-delay scheme.
While the sync scheme takes 82 min, 72 min more than the no-delay scheme. As compared with our theoretical expectation of delays of 39 min and 68 min, the
experimental results align well with our calculation.
