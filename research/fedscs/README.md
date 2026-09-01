# FedSCS: Robust Federated Learning via Stable Cosine Similarity

**Paper Title:**
FedSCS: Robust Federated Learning via Stable Cosine Similarity

**Authors:**
Rakib Ul Haque and Panagiotis (Panos P.) Markopoulos

**Affiliation:**
The University of Texas at San Antonio

🏆 This paper received the **Distinguished Conference Paper Award** at **IEEE ICCST 2025**.

## Overview

FedSCS (Federated Learning with Stable Cosine Similarity) is a robust federated learning aggregation method that evaluates client model updates based on their cosine similarity with the aggregated updates of peer clients.

The method maintains a historical stability score for each client and uses the resulting scores to determine aggregation weights across communication rounds.

This directory contains an NVIDIA FLARE implementation of FedSCS using PyTorch and CIFAR-10.

## Directory Structure

```text
research/fedscs/
├── client.py
├── job.py
├── prepare_data.sh
├── requirements.txt
├── train.py
├── README.md
└── src/
    ├── fedscs.py
    ├── fedscs_aggregator.py
    ├── fedscs_controller.py
    └── model.py
```

The CIFAR-10 dataset is **not included in this repository**. It is downloaded and prepared locally using `prepare_data.sh`.

## Requirements

The example requires:

* Python 3.9+
* PyTorch
* torchvision
* NVIDIA FLARE
* CUDA-enabled GPU (optional)

Install the Python dependencies with:

```bash
pip install -r research/fedscs/requirements.txt
```

If you are running the example from the NVFlare repository root:

```bash
cd /path/to/NVFlare
```

## 1. Prepare the CIFAR-10 Dataset

Before running the training example, download and extract the CIFAR-10 dataset:

```bash
./research/fedscs/prepare_data.sh
```

The script downloads the official CIFAR-10 Python dataset and extracts it to:

```text
research/fedscs/data/cifar-10-batches-py/
```

The script can be run again safely. If the dataset directory already exists, it will not download the dataset again.

You can verify that the dataset was prepared successfully with:

```bash
ls research/fedscs/data/cifar-10-batches-py
```

The directory should contain the CIFAR-10 batch files, including:

```text
batches.meta
data_batch_1
data_batch_2
data_batch_3
data_batch_4
data_batch_5
test_batch
```

## 2. Run Standalone Training

A standalone PyTorch training run can be used to verify that the CIFAR-10 data, model, and training environment are configured correctly.

From the NVFlare repository root, run:

```bash
python research/fedscs/train.py
```

The script:

1. Loads the CIFAR-10 training and test datasets.
2. Creates the PyTorch data loaders.
3. Initializes the CIFAR-10 CNN model.
4. Performs one local training epoch.
5. Evaluates the trained model on the CIFAR-10 test set.

The expected output includes the training loss and test accuracy, for example:

```text
Site: site-1
Device: cuda
Epoch 1/1 - loss: ...
Test accuracy: ...%
```

## 3. Run FedSCS with NVIDIA FLARE

After verifying standalone training, run the complete NVIDIA FLARE simulation:

```bash
python research/fedscs/job.py
```

The example uses the NVIDIA FLARE simulator with:

* **2 simulated clients**
* **2 federated communication rounds**
* **CIFAR-10**
* **PyTorch**
* **FedSCS aggregation**

The simulation creates two independent NVFlare client processes. Each client creates its own training and test `DataLoader` and performs local training before sending its model parameters to the server.

The server uses the `FedSCSAggregator` to calculate client stability scores and aggregation weights before producing the next global model.

## 4. Expected Workflow

The complete workflow is:

```bash
cd /path/to/NVFlare

pip install -r research/fedscs/requirements.txt

./research/fedscs/prepare_data.sh

python research/fedscs/train.py

python research/fedscs/job.py
```

The standalone training step is optional but recommended as a sanity check before launching the federated simulation.

## 5. Output

The NVFlare simulation stores its workspace and results under:

```text
/tmp/nvflare/fedscs/
```

The example job is configured with the workspace:

```text
/tmp/nvflare/fedscs
```

and the job name:

```text
fedscs_cifar10
```

## 6. Implementation

The main FedSCS aggregation logic is implemented in:

```text
research/fedscs/src/fedscs_aggregator.py
```

The implementation:

1. Receives client model parameters.
2. Computes each client's model update relative to the current global model.
3. Computes the cosine similarity between each client update and the sum of peer-client updates.
4. Clips negative similarity scores to zero.
5. Updates the historical stability score.
6. Converts stability scores into normalized aggregation weights.
7. Produces the aggregated global model.

The NVFlare controller integration is implemented in:

```text
research/fedscs/src/fedscs_controller.py
```

The PyTorch model is defined in:

```text
research/fedscs/src/model.py
```

The NVFlare client logic is implemented in:

```text
research/fedscs/client.py
```

and the simulation job is configured in:

```text
research/fedscs/job.py
```

## 7. Dataset Configuration

The current example uses the standard CIFAR-10 dataset for the simulated clients.

The data-loading implementation is located in:

```text
research/fedscs/train.py
```

The `site_id` is passed to the data-loading function so that client-specific data partitioning can be added in future extensions.

## 8. Citation

If you use FedSCS in your research, please cite the corresponding paper:

```bibtex
@inproceedings{haque2025fedscs,
  title={FedSCS: Robust Federated Learning via Stable Cosine Similarity},
  author={Haque, Rakib Ul and Markopoulos, Panagiotis},
  booktitle={IEEE International Conference on Communications, Control, and Technology (ICCST)},
  year={2025}
}
```

## Acknowledgment

This implementation is provided as a research example for NVIDIA FLARE and is intended to facilitate experimentation with robust federated learning aggregation.
