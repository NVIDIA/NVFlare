# Federated Learning with Differential Privacy for BraTS18 Segmentation

This example demonstrates federated learning for 3D medical image segmentation using the NVIDIA FLARE Job Recipe API.

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

## Data

### BraTS18 Dataset

This example uses the BraTS 2018 dataset for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs. The model is based on [Myronenko 2018](https://arxiv.org/abs/1810.11654) [1].

**Task**: Segment 3 nested subregions of primary brain tumors (gliomas):
- **Enhancing Tumor (ET)**: Areas with hyper-intensity in T1c
- **Tumor Core (TC)**: The bulk of the tumor (ET + necrotic + non-enhancing parts)
- **Whole Tumor (WT)**: Complete extent (TC + peritumoral edema)

**Input**: 4 aligned MRI scans per patient (T1c, T1, T2, FLAIR)

![Brain Tumor Segmentation](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_mri_segmentation_workflow.png)

### Download Dataset

Download BraTS 2018 data from [Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018](https://www.med.upenn.edu/cbica/brats2018.html) [2-6].

Place the data in `./dataset_brats18/dataset`. It should result in a sub-folder `./dataset_brats18/dataset/training`.

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

### Differential Privacy (Optional)

When `--enable_dp` is specified, the **SVTPrivacy** filter is applied to client outputs:

- **Method**: Sparse Vector Technique (SVT) [7, 8]
- **Effect**: Adds Laplace noise and selectively shares only a fraction of weight updates
- **Parameters** (configurable in `job.py`):
  - `fraction=0.9`: Share top 90% of weights
  - `epsilon=0.001`: Privacy budget
  - `noise_var=1.0`: Noise variance
  - `gamma=1e-4`: Clipping threshold

**Privacy-Utility Trade-off**: DP provides privacy guarantees but reduces model accuracy and convergence (see Results section).

## Job Recipe

The `job.py` file uses the `FedAvgRecipe` to configure the federated learning job:

```python
recipe = FedAvgRecipe(
    name=f"brats18_{n_clients}",
    min_clients=n_clients,
    num_rounds=num_rounds,
    initial_model=create_brats_model(),
    train_script="client.py",
    train_args="...",
    key_metric="val_dice",
    params_transfer_type=TransferType.DIFF,
)
```

## Run Job

### Basic Commands

**Centralized training** (1 client, all data):
```bash
python job.py --n_clients 1 --num_rounds 600 --gpu 0 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

**FedAvg** (4 clients on 4 GPUs):
```bash
python job.py --n_clients 4 --num_rounds 600 --gpu 0,1,2,3 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

**FedAvg** (4 clients on single 48GB GPU):
```bash
python job.py --n_clients 4 --num_rounds 600 --gpu 0 --threads 4 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"
```

**FedAvg with Differential Privacy** (4 clients on single 48GB GPU):
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

**Key Observations:**
- FedAvg achieves similar accuracy to centralized training
- Differential Privacy reduces accuracy and convergence but provides privacy guarantees
- All methods converge within 600 rounds

### Validation Metrics

Accuracy metrics after 600 rounds:

| Configuration | Val Overall Dice | Val TC Dice | Val WT Dice | Val ET Dice |
|---------------|------------------|-------------|-------------|-------------|
| brats18_1 (central) | 0.8558 | 0.8648 | 0.9070 | 0.7894 |
| brats18_4 (fedavg) | 0.8573 | 0.8687 | 0.9088 | 0.7879 |
| brats18_4_dp (fedavg+dp) | 0.8209 | 0.8282 | 0.8818 | 0.7454 |

**Key Findings:**
- **FedAvg vs Centralized**: Minimal difference (0.8573 vs 0.8558) - demonstrates effectiveness of federated learning
- **DP Impact**: ~4% Dice reduction (0.8573 → 0.8209) - privacy-utility trade-off with the chosen SVTPrivacy parameters

Different DP settings will have different impacts on performance. Adjust `fraction`, `epsilon`, `noise_var`, and `gamma` in `job.py` to tune the privacy-utility trade-off.

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

## References

[1] Myronenko A. 3D MRI brain tumor segmentation using autoencoder regularization. InInternational MICCAI Brainlesion Workshop 2018 Sep 16 (pp. 311-320). Springer, Cham.

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[4] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[6] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF

[7] Li, W., Milletarì, F., Xu, D., Rieke, N., Hancox, J., Zhu, W., Baust, M., Cheng, Y., Ourselin, S., Cardoso, M.J. and Feng, A., 2019, October. Privacy-preserving federated brain tumour segmentation. In International workshop on machine learning in medical imaging (pp. 133-141). Springer, Cham.

[8] Lyu, M., Su, D., & Li, N. (2016). Understanding the sparse vector technique for differential privacy. arXiv preprint arXiv:1603.01699.
