# Running NVFlare Mobile Example


## Setup the NVFlare System

```commandline
bash setup_nvflare.sh
```

For details, please refer to [setup NVFlare system for Edge](./setup_system.md)

This will create a deployment with 2 clients, 4 leaf-clients, 2 relays, and 1 server.

## Start the NVFlare System

To start the system, run the following command:

```commandline
./start_nvflare.sh
```    

Check the lcp_map.json:
```commandline
cat ./edge_example/prod_00/lcp_map.json
```

Web servers will be started at the ports specified in lcp_map.json.
The mobile devices and/or device emulators will communicate through these ports.

## Start the Mobile App
Install the app from App store and open it.

You will see the following screen:

![App Screenshot](./screenshot.png)

You need to configure the server URL and PORT to be the IP address shown in lcp_map.json (using the first one is recommended).

Then click "Start Training". (This will be enhanced in the future by adding resource monitoring to auto start/stop training)

## Prepare and Submit a Job

We have prepared two jobs for you: [xor_mobile](./jobs/xor_mobile/) and [cifar10_mobile](./jobs/cifar10_mobile/).
You can easily write your own components to replace any of the pre-configured ones. 

First, copy the jobs to the admin console transfer folder:

```commandline
cp -r ./jobs/* ./edge_example/prod_00/admin@nvidia.com/transfer
```

Start the admin console to interact with the NVFlare system:

```commandline
./edge_example/prod_00/admin@nvidia.com/startup/fl_admin.sh
```

Submit a job:

```
submit_job cifar10_mobile
```

You will then see the device start receiving the model from the server and complete local training.
The server will perform aggregation and proceed to the next round.
After the configured rounds have finished, the training is complete!

## [Optional] End-to-end Cifar10 Example

Let's run an end-to-end example with Cifar10 dataset with baseline comparisons.
### Baselines
1. Run the centralized baseline
```commandline
cd baselines
python cifar_train_central.py
```
2. Run the federated baseline under regular single-layer setting
```commandline
python cifar_fl_base_job.py
```

## [Optional] Run Device Emulator

If you don't have a device at hand or want to develop a new algorithm,
you can utilize our NVFlare device emulator.

The emulator can be used to test all the features of the edge system.

To start the emulator with an emulator config:

```
python ../emulator/run_emulator.py [emulator config file]
```

The emulator polls the NVFlare system for job assignments. It runs one job and then quits.
Please refer to [emulator docs](../emulator/README.md) for more details.

