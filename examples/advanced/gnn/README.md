# Federated GNN on Graph Dataset using Inductive Learning

In this example, we demonstrate how to train a Graph Neural Network (GNN) for node classification tasks using federated learning with NVIDIA FLARE's **recipe-based approach**. 

Graph Neural Networks (GNNs) show a promising future in research and industry, with potential applications in various domains, including social networks, e-commerce, recommendation systems, and more. GNNs excel in learning, modeling, and leveraging complex relationships within graph-structured data.

We provide two federated learning tasks:
1. **Protein Classification**: Classify protein roles based on cellular functions
2. **Financial Transaction Classification**: Detect illicit transactions in Bitcoin networks

## Data

This example supports two datasets for different node classification tasks:

### 1. Protein-Protein Interaction (PPI) Dataset
- **Task**: Classify protein roles based on their cellular functions from gene ontology
- **Source**: [GraphSAGE PPI Dataset](http://snap.stanford.edu/graphsage/#code)
- **Structure**: Multiple graphs, where each graph represents a specific human tissue
- **Nodes**: Proteins
- **Edges**: Interactions between proteins
- **Features**: Gene sets, positional gene sets, immunological signatures, and gene ontology sets
- **Usage**: Commonly used in graph-based machine learning tasks in bioinformatics
- **Download**: Dataset will be automatically downloaded when running the example via `torch_geometric.datasets.PPI`

### 2. Elliptic++ Dataset
- **Task**: Classify whether a given Bitcoin transaction is licit or illicit
- **Source**: [Elliptic++ GitHub](https://github.com/git-disl/EllipticPlusPlus)
- **Structure**: Single large graph representing the Bitcoin transaction network
- **Scale**: 203k Bitcoin transactions and 822k wallet addresses
- **Files Required**:
  - `txs_classes.csv`: Transaction IDs and their class labels (licit/illicit)
  - `txs_edgelist.csv`: Connections between transaction IDs
  - `txs_features.csv`: Transaction IDs and their features
- **Reference**: [Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network](https://arxiv.org/pdf/2306.06108.pdf)
- **Download**: Manual download required to `/tmp/nvflare/datasets/elliptic_pp` from source

## Model

Both tasks use **GraphSAGE** (Graph SAmple and aggreGatE), an inductive representation learning method for graph-structured data.

- **Framework**: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- **Paper**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)
- **Key Feature**: Inductive learning - the model learns to generate embeddings for unseen nodes, making it suitable for federated learning scenarios

### Model Architecture
- **Protein Classification**: 
  - Unsupervised learning approach
  - Based on [PyG's unsupervised PPI example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup_ppi.py)
  - Multi-layer GraphSAGE with mean aggregation
  
- **Financial Transaction Classification**:
  - Supervised learning approach
  - Custom SAGE model defined in `finance/model.py`
  - Uses node labels with supervised classification loss

Since inductive learning is used, the locally learned model is independent of the specific graph structure, enabling the use of standard FedAvg aggregation.

## Client Side Code

Each task contains a `client.py` file that implements the federated learning client logic:

### Key Components:
1. **Data Loading**: Loads the task-specific graph dataset for the client
2. **Model Initialization**: Creates the GraphSAGE model
3. **Training Loop**: Trains the model locally for specified epochs
4. **Model Evaluation**: Validates the model on local validation data
5. **Model Export**: Prepares model weights for aggregation

### Client Responsibilities:
- Load client-specific subgraph or data split
- Perform local training with the global model
- Compute training loss and validation metrics
- Send updated model weights to the server
- Receive and apply global model updates

## Server Side Code

The server side uses NVIDIA FLARE's built-in components via the FedAvg recipe:

### Key Components:
1. **Aggregation**: FedAverage algorithm aggregates client model weights
2. **Workflow**: Scatter and Gather (SAG) workflow
   - **Scatter**: Distribute global model to clients
   - **Gather**: Collect updated models from clients and aggregate
3. **Model Persistence**: Saves global model checkpoints

### Server Responsibilities:
- Initialize the global model
- Coordinate federated learning rounds
- Aggregate client model updates using weighted averaging
- Distribute updated global model to clients
- Track training progress and metrics

## Job Recipe

This example uses **NVFlare's FedAvgRecipe** to simplify job creation and configuration.

### Recipe Structure:
Each task directory contains:
- `job.py`: Job creation and execution script using FedAvgRecipe
- `client.py`: Federated learning client implementation
- `local_train.py`: Standalone local training script for baseline comparison

### FedAvgRecipe Configuration:
The recipe is configured in `job.py` with parameters such as:
- Number of clients
- Number of federated learning rounds
- Epochs per round
- Learning rate and optimizer settings
- Data paths

The recipe uses **per-site configuration** (`per_site_config`) to provide site-specific training arguments:
```python
per_site_config = {}
for i in range(1, num_clients + 1):
    site_name = f"site-{i}"
    per_site_config[site_name] = {
        "train_args": f"--data_path {data_path} --epochs {epochs_per_round} ..."
    }
```

This pattern allows each site to receive customized arguments, making it easy to:
- Specify different data paths per site
- Configure site-specific hyperparameters
- Support heterogeneous client configurations

The recipe automatically handles:
- Job structure creation
- Workflow configuration (ScatterAndGather)
- Model aggregation setup (FedAvgAggregator)
- Client deployment configuration with per-site customization

### Folder Structure

```
gnn/
├── protein/                        # Protein classification task
│   ├── job.py                      # Job creation and execution
│   ├── client.py                   # FL client script
│   └── local_train.py              # Local training baseline
└── finance/                        # Financial transaction classification task
    ├── job.py                      # Job creation and execution
    ├── client.py                   # FL client script
    ├── local_train.py              # Local training baseline
    ├── model.py                    # Custom SAGE model
    └── prepare_data.py             # Data preprocessing
```

## Run Job

### Installation

Follow the [Installation](../../getting_started/README.md) instructions to install NVIDIA FLARE.

Install additional requirements:

```bash
python3 -m pip install -r requirements.txt
```

Install PyTorch Geometric dependencies (refer to [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)):

```bash
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

Note: If you already have a specific version of nvflare installed in your environment, you may want to remove nvflare from the requirements to avoid reinstalling.

### Task 1: Protein Classification

Navigate to the protein directory:

```bash
cd protein
```

#### Run Local Training (Baseline)

Establish baselines by running local training on individual clients:

```bash
python local_train.py --client_id 0
python local_train.py --client_id 1
python local_train.py --client_id 2
```

#### Run Federated Training

Run federated learning with 2 clients:

```bash
python job.py \
  --num_clients 2 \
  --num_rounds 7 \
  --epochs_per_round 10 \
  --data_path /tmp/nvflare/datasets/ppi \
  --threads 2
```

**Parameters:**
- `--num_clients`: Number of federated learning clients (default: 2)
- `--num_rounds`: Number of federated learning rounds (default: 7)
- `--epochs_per_round`: Local training epochs per round (default: 10)
- `--data_path`: Path to PPI dataset (will auto-download if not present)
- `--threads`: Number of parallel threads for simulation

### Task 2: Financial Transaction Classification

Navigate to the finance directory:

```bash
cd finance
```

#### Prepare Dataset

Download the Elliptic++ dataset to `/tmp/nvflare/datasets/elliptic_pp` folder. You will need:
- `txs_classes.csv`: Transaction IDs and their class labels (licit/illicit)
- `txs_edgelist.csv`: Connections between transaction IDs
- `txs_features.csv`: Transaction IDs and their features

#### Run Local Training (Baseline)

Establish baselines by running local training on individual clients:

```bash
python local_train.py --client_id 0
python local_train.py --client_id 1
python local_train.py --client_id 2
```

#### Run Federated Training

Run federated learning with 2 clients:

```bash
python job.py \
  --num_clients 2 \
  --num_rounds 7 \
  --epochs_per_round 10 \
  --data_path /tmp/nvflare/datasets/elliptic_pp \
  --threads 2
```

**Parameters:**
- `--num_clients`: Number of federated learning clients (default: 2)
- `--num_rounds`: Number of federated learning rounds (default: 7)
- `--epochs_per_round`: Local training epochs per round (default: 10)
- `--data_path`: Path to Elliptic++ dataset
- `--threads`: Number of parallel threads for simulation

## Results

Results are saved in the local and federated learning workspaces under `/tmp/nvflare/gnn`.

### Color Scheme

**Local Training:** 
- Black curve: Centralized training (whole dataset)
- Green curve: Client 1 (local data only)
- Purple curve: Client 2 (local data only)

**Federated Learning:** 
- Blue curve: Client 1
- Red curve: Client 2

### Protein Classification Results

The training losses are shown below: 
![All training curves](./figs/protein_train_loss.png)

Note the "bumps" in federated learning curves due to global model aggregation and synchronization.

The validation F1 scores are shown below:

![All validation curves](./figs/protein_val_f1.png)

**Observations:**
- Since we validate the global model after each round, both clients' validation scores are identical (blue and red curves overlap)
- Federated learning achieves better performance than local training using individual site data only
- Performance is below centralized training (black curve) using the whole dataset, which is expected

### Financial Transaction Classification Results

The training losses are shown below: 
![All training curves](./figs/finance_train_loss.png)

The validation AUC scores are shown below:

![All validation curves](./figs/finance_val_auc.png)

**Observations:**
- Since we validate the global model after each round, both clients' validation scores are identical (blue and red curves overlap)
- Federated learning achieves significantly better performance than local training using individual site data only
- Performance is comparable to centralized training (black curve) using the whole dataset


## Citation for Elliptic++ Dataset

> Youssef Elmougy and Ling Liu. 2023. Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA, USA. ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3580305.3599803

BibTeX
```
@article{elmougy2023demystifying,
  title={Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics},
  author={Elmougy, Youssef and Liu, Ling},
  journal={arXiv preprint arXiv:2306.06108},
  year={2023}
}
```
