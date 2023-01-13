# Federated SVM with Scikit-learn

## Introduction to Scikit-learn, tabular data, and federated SVM
### Scikit-learn
This example shows how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tubular data.
It uses [Scikit-learn](https://scikit-learn.org/),
which is a widely used open source machine learning library that supports supervised and unsupervised learning.
### Tabular data
The data used in this example is tabular in a format that can be handled by [pandas](https://pandas.pydata.org/), such that:
- rows correspond to data samples
- first column represents the label 
- the other columns cover the features.    

Each client is expected to have 1 local data file containing both training and validation samples. To load the data for each client, the following parameters are expected by local learner:
- data_file_path: string, full path to the client's data file 
- train_start: int, start row index for training set
- train_end: int, end row index for training set
- valid_start: int, start row index for validation set
- valid_end: int, end row index for validation set

### Federated SVM
The machine learning algorithm shown in this example is [SVM with RBF kernel](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
Under this setting, federated learning can be formulated in two steps:
- local training: each client trains a local SVM model with their own data
- global training: server collects the support vectors from all clients, and train a global SVM model based on them

Note that unlike other iterative federated algorithms, federated SVM only involves these two training steps.

## Data preparation 
This example uses the breast cancer dataset available from Scikit-learn's dataset API.  
```commandline
bash data_prepare.sh
```
This will load the data, format it properly by removing the header, order the label and feature columns, and save it to a CSV file with comma separation, the default path is `/tmp/nvflare/dataset/sklearn_breast_cancer.csv`. 

## Prepare clients' configs with proper data information 
For realworld FL applications, the config json files are expected to be specified by each client individually, according to their own local data path and splits for training and validation.

In this simulated study, in order to efficiently generate the config files for a study under a particular setting, we provide a script to automate the process, note that manual copying and content modification can achieve the same.

For an experiment with `K` clients, we split one dataset into `K+1` parts in a non-overlapping fashion: `K` clients' training data, and `1` common validation data. To simulate data imbalance among clients, we provided several options for client data splits by specifying how a client's data amount correlates with its ID number (from `1` to `K`):
- Uniform
- Linear
- Square
- Exponential

These options can be used to simulate no data imbalance (uniform), moderate data imbalance (linear), and high data imbalance (square for larger client number e.g. `K=20`, exponential for smaller client number e.g. `K=5` as it will be too aggressive for larger client numbers)

This step is performed by 
```commandline
bash prepare_job_config.sh
```
In this example, we perform experiment with 3 clients under uniform data split. 

Below is a sample config for site-1, saved to `/job_configs/sklearn_svm_3_uniform/app_site-1/config/config_fed_client.json`:
```json

```

## Run experiments with FL simulator
### 3.1 Training with FL simulator
FL simulator is used to simulate FL experiments or debug codes, not for real FL deployment.
In this example, we assume four local GPUs with at least 12GB of memory are available.

Let's create an empty folder to serve as the workspace.
```
mkdir ./workspace_brats
```

Then, we can run the FL simulator with 1 client for centralized training
```
nvflare simulator './configs/brats_central' -w './workspace_brats/brats_central' -n 1 -t 1 -gpu 0
```
or
```
python3 -u -m nvflare.private.fed.app.simulator.simulator './configs/brats_central' -w './workspace_brats/brats_central' -n 1 -t 1 -gpu 0
```
Similarly, run the FL simulator with 4 clients for federated learning by running
```
nvflare simulator './configs/brats_fedavg' -w './workspace_brats/brats_fedavg' -n 4 -t 4 -gpu 0,1,2,3
```
Run the FL simulator with 4 clients for federated learning with differential privacy by running
```
nvflare simulator './configs/brats_fedavg_dp' -w './workspace_brats/brats_fedavg_dp' -n 4 -t 4 -gpu 0,1,2,3
```

### 3.2 Testing with FL simulator
The best global models are stored at
```
workspace_brats/[job]/simulated_job/app_server/best_FL_global_model.pt
```

Please then add the correct paths to the testing script, and run
```
cd ./result_stat
bash testing_models_3d.sh
```

## 4. Run experiments with POC ("proof of concept") FL setting
After verifying the codes with FL simulator, we have more confidence to perform FL experiments in POC setting.
### 4.1 Create your POC ("proof of concept") workspace
In this example, we run FL experiments in POC mode, starting with creating local FL workspace.
The [create_poc_workspace.sh](./create_poc_workspace.sh) script follows this pattern:
```
./create_poc_workspace.sh [n_clients]
```
In the following experiments, we will be using 4 clients. 
```
./create_poc_workspace.sh 4
```
Press y and enter when prompted.

### 4.2 GPU resource and Multi-tasking
In this example, we assume four local GPUs with at least 12GB of memory are available. 

As we use the POC workspace without `meta.json`, we control the client GPU directly when starting the clients by specifying `CUDA_VISIBLE_DEVICES`. 

To enable multitasking (if there are more computation resources - e.g. 4 x 32 GB GPUs), we can adjust the default value in `workspace_server/server/startup/fed_server.json` by setting `max_jobs: 2` (default value 1). Please adjust this properly according to resource available and task demand. 

For details, please refer to the [documentation](https://nvflare.readthedocs.io/en/main/user_guide/job.html).

### 4.3 Training with POC FL setting
The next scripts will start the FL server and clients automatically to run FL experiments on localhost.
#### 4.3.1 Start the FL system and submit jobs
Next, we will start the FL system and submit jobs to start FL training automatically.

Start the FL system with either 1 client for centralized training, or 4 clients for federated learning by running
```
bash start_fl_poc.sh "All"
```
or
```
bash start_fl_poc.sh "1 2 3 4"
```
This script will start the FL server and clients automatically to run FL experiments on localhost. 
Each client will be alternately assigned a GPU using `export CUDA_VISIBLE_DEVICES=${gpu_idx}` in the [start_fl_poc.sh](./start_fl_poc.sh). 
In this example, we run each client on a single GPU: 4 clients on 4 GPUs with 12 GB memory.

Then FL training will be run with an automatic script utilizing the FLAdminAPI functionality.    
The [submit_job.sh](./submit_job.sh) script follows the pattern:
```
bash ./submit_job.sh [config]
```
`[config]` is the experiment job that will be submitted for the FL training, in this example, this includes `brats_central`, `brats_fedavg`, and `brats_fedavg_dp`.  

Note that in order to make it working under most system resource conditions, the current config set `"cache_dataset": 0.0`, which could be slow. If resource permits, it will make the training much faster by caching the dataset. More information available [here](https://docs.monai.io/en/stable/data.html#cachedataset).  
For reference, with `"cache_dataset": 0.5` setting (cache half the data), the centralized training for 100 round, 1 epoch per round takes around 24.5 hours on a 12GB NVIDIA TITAN Xp GPU. 
#### 4.3.2 Centralized training
To simulate a centralized training baseline, we run FL with 1 client using all the training data. 
```
bash start_fl_poc.sh "All"
bash submit_job.sh brats_central
```
#### 4.3.3 Federated learning
Start 4 FL clients
```
bash start_fl_poc.sh "1 2 3 4"
```
To run FL with [FedAvg](https://arxiv.org/abs/1602.05629), we use
```
bash submit_job.sh brats_fedavg
``` 
To run FL with differential privacy, we use
```
bash submit_job.sh brats_fedavg_dp 
```

### 4.4 Control the process with admin console
You can always use the admin console to manually abort a running job.
To access the admin console, run:
```
bash ./workspace_brats/admin/startup/fl_admin.sh
``` 

Then using `abort_job [JOB_ID]` to abort a job, where `[JOB_ID]` is the ID assigned by the system when submitting the job. 
For a complete list of admin commands, see [here](https://nvflare.readthedocs.io/en/main/user_guide/operation.html).
The `[JOB_ID]` can be found from site folder like `./workspace_brats/site-1`.

To log into the POC workspace admin console no username is required 
(use "admin" for commands requiring conformation with username). 

### 4.5 Testing with POC FL setting
After training, each client's best model will be used for cross-site validation.
The results can be downloaded and shown with the admin console using
```
  download_job [JOB_ID]
```
where `[JOB_ID]` is the ID assigned by the system when submitting the job.

The result will be downloaded to your admin workspace (the exact download path will be displayed when running the command).
The best global models are stored at
```
[DOWNLOAD_DIR]/[JOB_ID]/workspace/app_server/best_FL_global_model.pt
```

Then for each job, please add the correct paths and `[JOB_ID]` to the testing script `./result_stat/testing_models_3d_poc.sh`, and run the following code to get the validation score of the best FL global model.
```
cd ./result_stat
bash testing_models_3d_poc.sh
```

## 5. Results on 4 clients for Central vs. FedAvg vs. FedAvg with DP 
In this example, only the global model gets evaluated at each round, and saved as the final model. 
### 5.1 Validation curve
We can use tensorboard tool to view the training and validation curves for each setting and each site, e.g.,
```
tensorboard --logdir='./workspace_brats'
```

We compare the validation curves of the global models for different settings during FL. In this example, all clients compute their validation scores using the same BraTS validation set. 

We provide a script for plotting the tensorboard records, running the following code.

For FL simulator, run:
```
python3 ./result_stat/plot_tensorboard_events.py
```

For FL with POC mode, run:
```
python3 ./result_stat/plot_tensorboard_events_poc.py
```

The TensorBoard curves (smoothed with weight 0.8) for validation Dice for 600 epochs (600 rounds, 1 local epoch per round) during training are shown below:
![All training curve](./figs/nvflare_brats18.png)

As shown, FedAvg achieves similar accuracy as centralized training, while DP will lead to some performance degradation based on the specific [parameter settings](./configs/brats_fedavg_dp/config/config_fed_client.json). Different DP settings will have different impacts over the performance. 

### 5.2 Validation score
The accuracy metrics under each settings are:

| Config	| Val Overall Dice | 	Val TC Dice	 | 	Val WT Dice	 | 	Val ET Dice	 | 
| ----------- |------------------|---------------|---------------|---------------|  
| brats18_central 	| 	0.8558	         | 	0.8648	      | 0.9070	       | 0.7894	       | 
| brats18_fedavg  	| 	0.8573	         | 0.8687	       | 0.9088	       | 0.7879	       | 
| brats18_fedavg_dp | 	0.8209	    | 0.8282	       | 0.8818	       | 0.7454	       |



## References
[1] Myronenko A. 3D MRI brain tumor segmentation using autoencoder regularization. InInternational MICCAI Brainlesion Workshop 2018 Sep 16 (pp. 311-320). Springer, Cham.

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[4] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[6] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF

[7] Li, W., Milletarì, F., Xu, D., Rieke, N., Hancox, J., Zhu, W., Baust, M., Cheng, Y., Ourselin, S., Cardoso, M.J. and Feng, A., 2019, October. Privacy-preserving federated brain tumour segmentation. In International workshop on machine learning in medical imaging (pp. 133-141). Springer, Cham.

[8] Lyu, M., Su, D., & Li, N. (2016). Understanding the sparse vector technique for differential privacy. arXiv preprint arXiv:1603.01699.
