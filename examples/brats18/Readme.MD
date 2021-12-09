# Federated Learning with Differential Privacy for Brats18 segmentation

## Introduction to MONAI, BraTS and Differential Privacy

### MONAI
This example shows how to use NVFlare on medical image applications.
It uses a python package [MONAI](https://github.com/Project-MONAI/MONAI),
which is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of PyTorch Ecosystem.

### BraTS
The application shown in this example is volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data.
It uses a deep network model published by [Myronenko 2018](https://arxiv.org/abs/1810.11654).

The model is trained to segment 3 nested subregions of primary brain tumors (gliomas): the "enhancing tumor" (ET), the "tumor core" (TC), the "whole tumor" (WT) based on 4 aligned input MRI scans (T1c, T1, T2, FLAIR). 

![](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_mri_segmentation_workflow.png)

- The ET is described by areas that show hyper intensity in T1c when compared to T1, but also when compared to "healthy" white matter in T1c. 
- The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. 
-  The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.

To run this example, please make sure you have downloaded BraTS 2018 data, which can be obtained from [MSD](http://medicaldecathlon.com/). The Task01 is BraTS18 data. After unzipping it, the brats image data are in `$dataset_base_dir`, which should have a subfolder `${dataset_base_dir}/training`.

### Differential Privacy (DP)
[Differential Privacy (DP)](https://arxiv.org/pdf/1910.00962.pdf) is an encryption method for Federated Learning (FL).
This instance can be used as an example of monai trainer for medical image applications, as well as an example of the usage of filters as DP is added as a filter in `config_fed_client.json`.
The original [DP paper](https://arxiv.org/pdf/1910.00962.pdf) split BraTS18 dataset into 13 datasets for 13 clients.
Each client requires a 12G GPU to run. Thus we need 13 GPUs to reproduce the results in [DP paper](https://arxiv.org/pdf/1910.00962.pdf)

In the readme, we will first go through the process of running 2 clients for users to try it. 
Then we will show the training results for 13 clients, which follows the same settings in the original [DP paper](https://arxiv.org/pdf/1910.00962.pdf).


## (Optional) 1. Set up a virtual environment
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
(If needed) make all shell scripts executable using
```
find . -name ".sh" -exec chmod +x {} \;
```
initialize virtual environment.
```
source ./virtualenv/set_env.sh
```
install required packages for training
```
pip install --upgrade pip
pip install -r ./virtualenv/min-requirements.txt
```
(optional) if you would like to plot the TensorBoard event files as shown below, please also install
```
pip install -r ./virtualenv/plot-requirements.txt
```

## 2. Create your FL workspace 

### 2.1 POC ("proof of concept") workspace
To run FL experiments in POC mode, create your local FL workspace the below command. 
In the following experiments, we will be using 2 clients. Press y and enter when prompted. 
```
./create_poc_workpace.sh [n_clients]
./create_poc_workpace.sh 2
```

### 2.2 (Optional) Secure FL workspace
We only cover POC mode in this example. To run it with Secure mode, please refer to the [`cifar10`](../cifar10) example.

<!-- The project file for creating the secure workspace used in this example is shown at 
[./workspaces/secure_project.yml](./workspaces/secure_project.yml).

To create the secure workspace, please use the following to build a package and copy it 
to `secure_workspace` for later experimentation.
```
cd ./workspaces
provision -p ./secure_project.yml
cp -r ./workspace/secure_project/prod_00 ./secure_workspace
cd ..
```
For more information about secure provisioning see the [documentation](https://nvidia.github.io/NVFlare/user_guide/provisioning_tool.html).
 -->
> **_NOTE:_** **POC** stands for "proof of concept" and is used for quick experimentation 
> with different amounts of clients.
> It doesn't need any advanced configurations while provisioning the startup kits for the server and clients. 
>
> The **secure** workspace on the other hand is needed to run experiments that require encryption keys such as the 
> homomorphic encryption (HE) one shown below. These startup kits allow secure deployment of FL in real-world scenarios 
> using SSL certificated communication channels.


## 3. Run automated experiments

### 3.1 FedAvg with and without differential privacy
Next, FL training will start automatically. 

The [run_poc.sh](./run_poc.sh) and [run_secure.sh](./run_secure.sh) scripts follows this pattern:
```
./run_poc.sh [n_clients] [config] [run] [dataset_base_dir] [datalist_json_path]
```

`[run]` is the run number for nvflare experiment, you need to assign a unique `[run]` for each experiment.
`[datalist_json_path]` is the path for datalist json files, chosen from the subfolders in ./datalists/brats/

For example, 
to run brats FL without differential privacy:
```
./run_poc.sh 2 brats18_fedavg 2 /dataset/brats18/ ${PWD}/datalists/brats/brats_13clients
```
to run brats FL with differential privacy:
```
./run_poc.sh 2 brats18_fedavg_dp 3 /dataset/brats18/ ${PWD}/datalists/brats/brats_13clients
```

These scripts will start the FL server and `[n_clients]` clients automatically to run FL experiments on localhost. 
Each client can be assigned a GPU using `export CUDA_VISIBLE_DEVICES=${gpu_idx}` in the [run_poc.sh](./run_poc.sh). 
In this example, we run 2 clients each on one GPU with 12 GB memory.

The commands above will use the data split in `./datalists/brats/brats_13clients` to simulate each client having different data distributions.
The `[config]` argument controls which experiment to run. The respective folder under `[configs]` will be selected and 
uploaded to the server for distribution to each client using the admin API with [run_fl.py](./run_fl.py). 
The run will time out if not completed in 36 hours. You can adjust this within the `run()` call of the customized admin API script in [run_fl.py](./run_fl.py). 

### 3.2 Centralized training

To simulate a centralized training baseline, we run FL with 1 client using all the training data. 
```
./run_poc.sh 1 brats18_central 1 /dataset/brats18/ ${PWD}/datalists/brats/brats_1clients
```
You can visualize the training progress by running `tensorboard --logdir=[workspace]/.`


### 3.3 Reproduce FedAvg with and without differential privacy on 13 clients and GPUs
The original [DP paper](https://arxiv.org/pdf/1910.00962.pdf) split brats18 dataset into 13 datasets for 13 clients.
So we will run the following cmd:
```
./create_poc_workpace.sh 13
./run_poc.sh 13 brats18_fedavg 2 /dataset/brats18/ ${PWD}/datalists/brats/brats_13clients
./run_poc.sh 13 brats18_fedavg_dp 3 /dataset/brats18/ ${PWD}/datalists/brats/brats_13clients
```
In [run_poc.sh](./run_poc.sh), the default server's IP address is `servername="localhost"`. It works when your server and clients run on the same machine. However, if you need to run server and clients on different machines, you would start the server first, then change `servername` to connect clients to the server's IP address. The IP address can be found by `echo "The server starts on IP address: $(hostname --all-ip-addresses)"`.


> **_NOTE:_** You can always use the admin console to manually abort the automatically started runs 
  using `abort all`. If the admin API script is running, the FL system will automatically shut down using
  the current setting defined in [run_fl.py](./run_fl.py). An automatic shutdown is useful here for development as code changes 
> in your FL components will only be picked up on a restart of the FL system. 
> For real-world deployments, the system should be kept running but the admin restart command can be used, 
> see [here](https://nvidia.github.io/NVFlare/user_guide/admin_commands.html).

> To log into the POC workspace admin console, use username "admin" and password "admin". 
> For the secure workspace admin console, use username "admin@nvidia.com"

After training, each client's best model will be used for cross-site validation. The results can be shown with
for example
```
  cat ./workspaces/poc_workspace/server/run_2/cross_val_results/cross_val.json
```

### 3.4 Running all examples

You can use `./run_experiments.sh` to execute all above-mentioned experiments sequentially if preferred. 
This script uses the secure workspace to also support the HE experiment.

## 4. Results on 13 clients for Central vs. FedAvg vs. FedAvg with DP

### 4.1 Validation curve 
Let's summarize the result of the experiments run above. Next, we will compare the final validation scores of 
the global models for different settings. In this example, all clients compute their validation scores using the
same Brats validation set. 

The tensorboard curve for validation dice of the global model for the 600 epochs during training is shown below
![All training curve](./figs/all_training.jpg)

### 4.2 Best model
One can see that FedAvg with DP can achieve a similar performance compared with FedAvg while adding security to the aggregation step.

| Config	|  	Val score	| 
| ----------- | ----------- |  
| brats18_central 	| 	0.82758	| 
| brats18_fedavg  	| 	0.81983	| 
| brats18_fedavg_dp | 	0.81036	|