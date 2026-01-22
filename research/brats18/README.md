# Federated Learning with Differential Privacy for BraTS18 Segmentation

Please make sure you set up virtual environment and follows [example root readme](../../README.md).
This example uses the NVIDIA FLARE Job Recipe API (similar to `examples/hello-world/hello-dp`).

## Introduction to MONAI, BraTS and Differential Privacy
### MONAI
This example shows how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/) on medical image applications.
It uses [MONAI](https://github.com/Project-MONAI/MONAI),
which is a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of the PyTorch Ecosystem.
### BraTS
The application shown in this example is volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data.
It uses a deep network model published by [Myronenko 2018](https://arxiv.org/abs/1810.11654) [1].

The model is trained to segment 3 nested subregions of primary brain tumors (gliomas): the "enhancing tumor" (ET), the "tumor core" (TC), the "whole tumor" (WT) based on 4 aligned input MRI scans (T1c, T1, T2, FLAIR). 

![](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_mri_segmentation_workflow.png)

- The ET is described by areas that show hyper intensity in T1c when compared to T1, but also when compared to "healthy" white matter in T1c. 
- The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. 
- The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.

To run this example, please make sure you have downloaded BraTS 2018 data, which can be obtained from [Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018](https://www.med.upenn.edu/cbica/brats2018.html) [2-6]. Please download the data to [./dataset_brats18/dataset](./dataset_brats18/dataset). It should result in a sub-folder `./dataset_brats18/dataset/training`.

In this example, we split BraTS18 dataset into [4 subsets](./dataset_brats18/datalist) for 4 clients:
- `site-1.json`, `site-2.json`, `site-3.json`, `site-4.json`: Individual client data splits (60-61 training samples each)
- `site-All.json`: Combined dataset for centralized training (242 training samples total)
- All clients share the same validation set (43 samples) for fair comparison

Each client requires at least a 12 GB GPU to run. 

Note that for achieving FL and centralized training curves with a validation score that can be directly compared, we use an identical validation set across all clients and experiments without withholding a standalone testing set. In this case all scores will be computed against the same dataset, and when combining all clients' data we will have the same dataset as the centralized training. In reality though, each site will usually have its own validation set (in which case the validation curves are not directly comparable), and a testing set is usually withheld from the training process.

### Differential Privacy (DP)
[Differential Privacy (DP)](https://arxiv.org/abs/1910.00962) [7] is method for ensuring that Federated Learning (FL) preserves privacy by obfuscating the model updates sent from clients to the central server.
This example shows the usage of a MONAI-based trainer for medical image applications with NVFlare, as well as the usage of DP filters in your FL training. DP is added as a Recipe output filter in `job.py`. Here, we use the "Sparse Vector Technique", i.e. the [SVTPrivacy](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.filters.svt_privacy.html) protocol, as utilized in [Li et al. 2019](https://arxiv.org/abs/1910.00962) [7] (see [Lyu et al. 2016](https://arxiv.org/abs/1603.01699) [8] for more information).

## NVIDIA FLARE Installation
For complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html).
```
pip install nvflare
```
Install the dependencies:
```
pip install -r requirements.txt
```

## Code Structure
```
brats18/
|-- job.py                         # Job Recipe entrypoint
|-- client.py                      # Client API script (MONAI training)
|-- model.py                       # Model definition (BratsSegResNet wrapper)
|-- dataset_brats18/               # Dataset and datalist splits
|-- result_stat/                   # Evaluation and plotting scripts
```

## Prepare dataset paths
Pass dataset locations to the client script through `job.py` arguments:
```
DATASET_ROOT="${PWD}/dataset_brats18/dataset"
DATALIST_ROOT="${PWD}/dataset_brats18/datalist"
```

## Run with Recipe API
This example uses the [Job Recipe API](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/job_recipe.html)
to run in simulation mode.

Run centralized training (1 client, uses all data from `site-All.json`):
```
python job.py --n_clients 1 --num_rounds 600 --gpu 0 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

Run FedAvg (4 clients on 4 GPUs):
```
python job.py --n_clients 4 --num_rounds 600 --gpu 0,1,2,3 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

Run FedAvg (4 clients on single GPU):
```
python job.py --n_clients 4 --num_rounds 600 --gpu 0 --threads 4 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

Run FedAvg with DP (4 clients):
```
python job.py --n_clients 4 --num_rounds 600 --gpu 0 --threads 4 --enable_dp \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

By default, results are stored under `/tmp/nvflare/simulation/<job_name>`.
The job name follows the format `brats18_{n_clients}` (e.g., `brats18_4`) or `brats18_{n_clients}_dp` with DP enabled (e.g., `brats18_4_dp`).
Use `--workspace` to change the workspace root and `--threads` to control simulator threads.

### Testing
The best global models are stored at:
```
<workspace_root>/<job_name>/server/simulate_job/app_server/best_FL_global_model.pt
```
For example: `/tmp/nvflare/simulation/brats18_1/server/simulate_job/app_server/best_FL_global_model.pt`

Update the testing script paths and run:
```
cd ./result_stat
bash testing_models_3d.sh
```

## Results for Central vs. FedAvg vs. FedAvg with DP 
In this example, only the global model gets evaluated at each round, and saved as the final model.
The results below are from experiments with 4 clients. 
### Validation curve
We can use tensorboard tool to view the training and validation curves for each setting and each site, e.g.,
```
tensorboard --logdir='/tmp/nvflare/simulation'
```

We compare the validation curves of the global models for different settings during FL. In this example, all clients compute their validation scores using the same BraTS validation set. 

We provide a script for plotting the tensorboard records:
```
python3 ./result_stat/plot_tensorboard_events.py
```

The TensorBoard curves (smoothed with weight 0.8) for validation Dice for 600 epochs (600 rounds, 1 local epoch per round) during training are shown below:
![All training curve](./figs/nvflare_brats18.png)

As shown, FedAvg achieves similar accuracy as centralized training, while DP will lead to some performance degradation based on the specific SVTPrivacy parameter settings in `job.py`.
Different DP settings will have different impacts over the performance.

### Validation score
The accuracy metrics under each setting are:

| Config	| Val Overall Dice | 	Val TC Dice	 | 	Val WT Dice	 | 	Val ET Dice	 | 
| ----------- |------------------|---------------|---------------|---------------|  
| brats18_1 (central) 	| 	0.8558	         | 	0.8648	      | 0.9070	       | 0.7894	       | 
| brats18_4 (fedavg)  	| 	0.8573	         | 0.8687	       | 0.9088	       | 0.7879	       | 
| brats18_4_dp (fedavg+dp) | 	0.8209	    | 0.8282	       | 0.8818	       | 0.7454	       |



## References
[1] Myronenko A. 3D MRI brain tumor segmentation using autoencoder regularization. InInternational MICCAI Brainlesion Workshop 2018 Sep 16 (pp. 311-320). Springer, Cham.

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[4] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[6] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF

[7] Li, W., Milletarì, F., Xu, D., Rieke, N., Hancox, J., Zhu, W., Baust, M., Cheng, Y., Ourselin, S., Cardoso, M.J. and Feng, A., 2019, October. Privacy-preserving federated brain tumour segmentation. In International workshop on machine learning in medical imaging (pp. 133-141). Springer, Cham.

[8] Lyu, M., Su, D., & Li, N. (2016). Understanding the sparse vector technique for differential privacy. arXiv preprint arXiv:1603.01699.
