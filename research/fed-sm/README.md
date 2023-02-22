# Personalized Federated Learning with FedSM Algorithm

## Introduction to MONAI, and FedSM

### MONAI
This example shows how to use [NVIDIA FLARE](https://nvidia.github.io/NVFlare) on medical image applications.
It uses [MONAI](https://github.com/Project-MONAI/MONAI),
which is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of the PyTorch Ecosystem.

### FedSM
This example illustrates the personalized federated learning algorithm [FedSM](https://arxiv.org/abs/2203.10144) accpeted to [CVPR2022](https://cvpr2022.thecvf.com/). It bridges the different data distributions across clients via a SoftPull mechanism and a Super Model setting. 

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
pip3 install --upgrade pip
pip3 install -r ./virtualenv/min-requirements.txt
```
(optional) if you would like to plot the TensorBoard event files as shown below, please also install
```
pip install -r ./virtualenv/plot-requirements.txt
```
## 2. Multi-source Prostate Segmentation
This example uses 2D (axial slices) segmentation of the prostate in T2-weighted MRIs based on multiple datasets.

Please refer to [Prostate Example](https://github.com/NVIDIA/NVFlare/tree/dev/examples/advanced/prostate) for details of data preparation and task specs. In the following, we assume the data has been prepared in the same way to `./data_preparation`.

## 3. Run automated experiments
We use the NVFlare simulator to run FL training automatically, the 6 clients are named `client_I2CVB, client_MSD, client_NCI_ISBI_3T, client_NCI_ISBI_Dx, client_Promise12, client_PROSTATEx`
### 3.1 Prepare local configs
First, we add the image directory root to `config_train.json` files for generating the absolute path to dataset and datalist. In the current folder structure, it will be `${PWD}/..`, it can be any arbitary path where the data locates.  
```
for job in fedsm_prostate
do
  sed -i "s|DATASET_ROOT|${PWD}/data_preparation|g" job_configs/${job}/app/config/config_train.json
done
```
### Use NVFlare simulator to run the experiments
NWe use NVFlare simulator to run the FL training experiments, following the pattern:
```
nvflare simulator job_configs/[job] -w ${PWD}/workspaces/[job] -c [clients] -gpu [gpu] -t [thread]
```
`[job]` is the experiment job that will be submitted for the FL training, in this example, this is `fedsm_prostate`.  
The combination of `-c` and `-gpu`/`-t` controls the resource allocation. 

## 4. Results on three clients for FedSM
In this example, we run three clients on 1 GPU with three threads `-t 3`. The minimum GPU memory requirement is 12 GB. 

### Validation curve on each site
In this example, each client computes their validation scores using their own
validation set. 

We provide a script for plotting the tensorboard records, running
```
python3 ./result_stat/plot_tensorboard_events.py
```
The TensorBoard curves (smoothed with weight 0.8) for validation Dice for the 100 epochs (100 rounds, 1 local epochs per round) during training are shown below:
![All training curve](./figs/all_training.png)

### Testing score
The testing score is computed based on the Super Model for FedSM.
We provide a script for performing validation on testing data split, please add the correct paths and job_ids, and run

```
bash ./result_stat/testing_models_2d.sh
```

The Dice results for the above run are:

| Config	          | 	Val Dice	 | 
|------------------|------------|
| fedsm_prostate | |
| prostate_central | 	0.8590	 | 
| prostate_fedavg  |   0.8324   | 
| prostate_fedprox |   0.8131   | 
| prostate_ditto   | 	0.8474	 |
