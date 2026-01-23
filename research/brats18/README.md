# Privacy-preserving Federated Brain Tumour Segmentation

This repository contains the reference implementation of the paper [Privacy-preserving Federated Brain Tumour Segmentation](https://arxiv.org/abs/1910.00962) published at MLMI 2019 (International Workshop on Machine Learning in Medical Imaging, held in conjunction with MICCAI 2019).

**Authors:** Wenqi Li, Fausto Milletarì, Daguang Xu, Nicola Rieke, Jonny Hancox, Wentao Zhu, Maximilian Baust, Yan Cheng, Sébastien Ourselin, M. Jorge Cardoso, Andrew Feng  

## Abstract

Due to medical data privacy regulations, it is often infeasible to collect and share patient data in a centralised data lake. This poses challenges for training machine learning algorithms, such as deep convolutional networks, which often require large numbers of diverse training examples. Federated learning sidesteps this difficulty by bringing code to the patient data owners and only sharing intermediate model raining updates among them. Although a high-accuracy model could be achieved by appropriately aggregating these model updates, the model shared could indirectly leak the local training examples. In this paper, we investigate the feasibility of applying differential-privacy techniques to protect the patient data in a federated learning setup. We implement and evaluate practical federated learning systems for brain tumour segmentation on the BraTS dataset. The experimental results show that there is a tradeoff between model performance and privacy protection costs.

![Brain Tumor Segmentation](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_mri_segmentation_workflow.png)

This implementation uses the NVIDIA FLARE Job Recipe API to reproduce the experiments from the paper.

## Installation

### Prerequisites
Set up a virtual environment and follow the [example root readme](../../README.md).

### Install NVIDIA FLARE
```bash
pip install nvflare
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

This will install MONAI and other required packages for medical image processing and model training.

For complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html).

## Project Structure

```
brats18/
├── job.py                         # Job Recipe entrypoint
├── client.py                      # Client training script (MONAI + Client API)
├── model.py                       # Model definition (BratsSegResNet wrapper)
├── dataset_brats18/               # Dataset and datalist splits
│   ├── dataset/                   # BraTS18 training data (download required)
│   └── datalist/                  # Data split JSONs (site-1.json, site-All.json, etc.)
├── result_stat/                   # Evaluation and plotting scripts
└── figs/                          # Result figures
```

## Background

### Medical Image Segmentation with MONAI

This example demonstrates how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/) for medical image applications with [MONAI](https://github.com/Project-MONAI/MONAI), a PyTorch-based, open-source framework for deep learning in healthcare imaging.

### Brain Tumor Segmentation Task

The application focuses on volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs using the BraTS 2018 dataset. The segmentation model is based on [Myronenko 2018](https://arxiv.org/abs/1810.11654) [1].

**Segmentation Targets**: 3 nested subregions of primary brain tumors (gliomas):
- **Enhancing Tumor (ET)**: Areas with hyper-intensity in T1c
- **Tumor Core (TC)**: The bulk of the tumor (ET + necrotic + non-enhancing parts)  
- **Whole Tumor (WT)**: Complete extent (TC + peritumoral edema)

**Input Modalities**: 4 aligned MRI scans per patient (T1c, T1, T2, FLAIR)

### Differential Privacy in Federated Learning

The key contribution of this work is applying differential privacy to federated learning for medical imaging. We use the **Sparse Vector Technique (SVT)** [8] to add calibrated noise to model updates, providing formal privacy guarantees while maintaining competitive segmentation accuracy. The implementation uses NVFlare's [SVTPrivacy](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.filters.svt_privacy.html) filter.

## Data

### Download BraTS18 Dataset

Please request access and download the BraTS 2018 training data from [Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018](https://www.med.upenn.edu/cbica/brats2018.html) [2-6].

Place the downloaded data in `./dataset_brats18/dataset`. This should create the folder structure `./dataset_brats18/dataset/training` containing 285 patient scans (242 for training, 43 for validation in our splits).

### Data Splits

The dataset is split into 4 subsets for federated learning:

| File | Purpose | Training Samples | Validation Samples |
|------|---------|-----------------|-------------------|
| `site-1.json` | Client 1 data | 60 | 43 |
| `site-2.json` | Client 2 data | 61 | 43 |
| `site-3.json` | Client 3 data | 61 | 43 |
| `site-4.json` | Client 4 data | 60 | 43 |
| `site-All.json` | Centralized training | 242 | 43 |

**Note**: All clients use the same validation set (43 samples) for fair comparison. For centralized training with 1 client, `site-All.json` is automatically used to include all training data.

### Prepare Dataset Paths

Set environment variables for dataset locations:
```bash
export DATASET_ROOT="${PWD}/dataset_brats18/dataset"
export DATALIST_ROOT="${PWD}/dataset_brats18/datalist"
```

## Client

### Model

The model uses MONAI's SegResNet architecture wrapped in a custom `BratsSegResNet` class. The wrapper explicitly stores constructor arguments as attributes, which is required for proper serialization by NVFlare's Job API.

Model architecture (defined in `model.py`):
- **Input**: 4 MRI channels (T1c, T1, T2, FLAIR)
- **Architecture**: SegResNet with residual blocks
- **Output**: 3 segmentation channels (ET, TC, WT)
- **Parameters**: 16 initial filters, 0.2 dropout

### Client Training Script

The client script (`client.py`) implements:

1. **Data Loading**: Uses MONAI's data loaders with client-specific data splits
2. **Model Training**: Standard PyTorch training loop with optional FedProx regularization
3. **Validation**: Computes Dice metrics for each tumor subregion
4. **Communication**: Uses NVFlare Client API (`flare.receive()` / `flare.send()`)
5. **Weight Diff**: Automatically handled by the API when `params_transfer_type=TransferType.DIFF`

Training hyperparameters:
- Learning rate: 1e-4 
- Optimizer: Adam with weight decay 1e-5
- Loss: Dice Loss
- Local epochs per round: 1 
- ROI size: 224×224×144

## Server

### Aggregation

The server uses the `FedAvg` workflow, which implements federated averaging, the aggregator uses **weighted averaging** based on the number of training steps (`NUM_STEPS_CURRENT_ROUND`).

### Model Selection

The `IntimeModelSelector` tracks validation metrics (`val_dice`) across rounds and saves the best performing global model as `best_FL_global_model.pt`.

### Differential Privacy

When `--enable_dp` is specified, the **SVTPrivacy** filter implements the Sparse Vector Technique for differential privacy:

**Mechanism**:
1. **Sparse Selection**: Select the top k weights with largest absolute values (controlled by `fraction` parameter)
2. **Noise Addition**: Add Laplace noise calibrated to sensitivity and privacy budget (`epsilon`)
3. **Clipping**: Apply threshold `gamma` to limit sensitivity of weight updates
4. **Privacy Guarantee**: The mechanism satisfies (ε, δ)-differential privacy, formally bounding information leakage

**Parameters**:
- `--dp_fraction=0.9`: Share top 90% of weights by magnitude (reduces communication and limits information exposure)
- `--dp_epsilon=0.001`: Privacy budget (smaller = stronger privacy, lower accuracy)
- `--dp_noise_var=1.0`: Laplace noise scale
- `--dp_gamma=1e-4`: Gradient clipping threshold

*Defaults are example values. Adjust these parameters to explore different privacy-utility trade-offs.See [Li et al. 2019](https://arxiv.org/abs/1910.00962) [7] for the full privacy analysis and theoretical guarantees.*

## Job Recipe

The `job.py` file uses the `FedAvgRecipe` to configure the federated learning job:

```python
recipe = FedAvgRecipe(
    name=f"brats18_{n_clients}",
    min_clients=n_clients,
    num_rounds=num_rounds,
    initial_model=BratsSegResNet(),
    train_script="client.py",
    train_args="...",
    key_metric="val_dice",
    params_transfer_type=TransferType.DIFF,
)
```

## Run Job

### Centralized and Federated

**Centralized training** (baseline, 1 client with all data):
```bash
python job.py --n_clients 1 --num_rounds 600 --gpu 0 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

**Federated learning without privacy** (4 clients on 4 GPUs):
```bash
python job.py --n_clients 4 --num_rounds 600 --gpu 0,1,2,3 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

**Federated learning without privacy** (4 clients on single 48GB GPU):
```bash
python job.py --n_clients 4 --num_rounds 600 --gpu 0 --threads 4 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

**Privacy-preserving federated learning** (4 clients with SVT differential privacy):
```bash
python job.py --n_clients 4 --num_rounds 600 --gpu 0 --threads 4 --enable_dp \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

### Workspace and Output

By default, results are stored under `/tmp/nvflare/simulation/<job_name>`:
- Job names follow the format `brats18_{n_clients}` (e.g., `brats18_4`)
- With DP enabled: `brats18_{n_clients}_dp` (e.g., `brats18_4_dp`)

Use `--workspace` to specify a custom workspace root.

### Additional Options

**Customize training hyperparameters:**
```bash
python job.py --n_clients 4 --num_rounds 100 \
  --learning_rate 5e-5 --aggregation_epochs 2 --cache_dataset 0.5 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

**Customize differential privacy parameters:**
```bash
python job.py --n_clients 4 --num_rounds 600 --enable_dp \
  --dp_epsilon 0.01 --dp_fraction 0.8 --dp_noise_var 0.5 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

**Use custom job name:**
```bash
python job.py --job_name my_experiment --n_clients 4 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

## Results

### Model Evaluation

The best global models are stored at:
```
<workspace_root>/<job_name>/server/simulate_job/app_server/best_FL_global_model.pt
```

Example: `/tmp/nvflare/simulation/brats18_4/server/simulate_job/app_server/best_FL_global_model.pt`

To evaluate the models:
```bash
cd ./result_stat
bash testing_models_3d.sh
```

### Training and Validation Curves

View training progress with TensorBoard:
```bash
tensorboard --logdir='/tmp/nvflare/simulation'
```

Generate comparison plots:
```bash
cd ./result_stat
python3 plot_tensorboard_events.py
```

The TensorBoard curves (smoothed with weight 0.8) for validation Dice over 600 rounds (1 local epoch per round):

![All training curve](./figs/nvflare_brats18.png)

**Key Findings:**
- **Federated learning matches centralized training**: FedAvg achieves comparable performance (0.8573 vs 0.8558 Dice), demonstrating that FL can effectively train medical imaging models without centralizing data
- **Privacy with acceptable trade-off**: SVT differential privacy provides formal privacy guarantees with ~4% Dice reduction (0.8573 → 0.8209), making privacy-preserving FL practical for medical applications
- **Convergence behavior**: All methods converge within 600 rounds, with DP showing slower convergence due to noisy updates

### Validation Metrics

Quantitative results after 600 rounds of training:

| Configuration | Val Overall Dice | Val TC Dice | Val WT Dice | Val ET Dice |
|---------------|------------------|-------------|-------------|-------------|
| brats18_1 (central) | 0.8558 | 0.8648 | 0.9070 | 0.7894 |
| brats18_4 (fedavg) | 0.8573 | 0.8687 | 0.9088 | 0.7879 |
| brats18_4_dp (fedavg+dp) | 0.8209 | 0.8282 | 0.8818 | 0.7454 |

**Analysis:**
1. **Federated Learning Effectiveness**: FedAvg achieves 0.8573 Dice compared to 0.8558 for centralized training (0.15% difference), demonstrating that collaborative learning across distributed institutions can match centralized performance without sharing raw medical data.

2. **Privacy-Utility Trade-off**: The SVT differential privacy mechanism provides formal privacy guarantees with a 4.2% accuracy reduction (0.8573 → 0.8209). This demonstrates the feasibility of privacy-preserving federated learning for sensitive medical applications.

**Note**: The results above use the default DP parameters from the paper. You can explore different privacy-utility trade-offs by adjusting `--dp_fraction`, `--dp_epsilon`, `--dp_noise_var`, and `--dp_gamma`. Lower epsilon values provide stronger privacy but typically reduce accuracy.

## Technical Notes

### Framework Details

- **Framework**: MONAI + PyTorch + NVIDIA FLARE
- **FL Algorithm**: FedAvg (Federated Averaging)
- **Privacy**: SVT (Sparse Vector Technique) Differential Privacy
- **Communication**: Weight differences (`TransferType.DIFF`) for applying DP
- **Aggregation**: Weighted averaging based on local training steps

### Hardware Requirements

- Minimum 12 GB GPU per client
- For 4 clients on single GPU: Recommend 48 GB GPU

## Citing This Work

If you use this implementation or build upon this work, please cite the original paper:

```bibtex
@inproceedings{li2019privacy,
  title={Privacy-preserving federated brain tumour segmentation},
  author={Li, Wenqi and Milletar{\`\i}, Fausto and Xu, Daguang and Rieke, Nicola and Hancox, Jonny and Zhu, Wentao and Baust, Maximilian and Cheng, Yan and Ourselin, S{\'e}bastien and Cardoso, M Jorge and others},
  booktitle={International workshop on machine learning in medical imaging},
  pages={133--141},
  year={2019},
  organization={Springer}
}
```

## References

[1] Myronenko A. 3D MRI brain tumor segmentation using autoencoder regularization. InInternational MICCAI Brainlesion Workshop 2018 Sep 16 (pp. 311-320). Springer, Cham.

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[4] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[6] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF

[7] Li, W., Milletarì, F., Xu, D., Rieke, N., Hancox, J., Zhu, W., Baust, M., Cheng, Y., Ourselin, S., Cardoso, M.J. and Feng, A., 2019, October. Privacy-preserving federated brain tumour segmentation. In International workshop on machine learning in medical imaging (pp. 133-141). Springer, Cham.

[8] Lyu, M., Su, D., & Li, N. (2016). Understanding the sparse vector technique for differential privacy. arXiv preprint arXiv:1603.01699.
