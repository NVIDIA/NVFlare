# FedHCA2: Clean NVFLARE Implementation

A clean, faithful NVFLARE implementation of FedHCA2 (Federated Heterogeneous Client Adaptive learning) that preserves the original algorithm's core logic and achieves the same results as the standalone version.

## Key Features

- **Faithful Implementation**: Preserves original FedHCA2 algorithms without modification
- **Clean Architecture**: Minimal NVFLARE wrappers around original code
- **Real Datasets**: Uses actual PASCAL-Context and NYUD-v2 datasets
- **Exact Model Architecture**: Maintains Swin Transformer + multi-decoder structure
- **Heterogeneous Clients**: Supports both single-task and multi-task clients
- **Personalized Aggregation**: Implements conflict-averse encoder and cross-attention decoder aggregation

## Architecture

```
research/fedhca2_nvflare/
├── jobs/fedhca2_pascal_nyud/
│   └── app/custom/
│       ├── fedhca2_learner.py          # Thin NVFLARE wrapper
│       ├── fedhca2_aggregator.py       # Thin NVFLARE wrapper  
│       ├── fedhca2_core/               # Original FedHCA2 algorithms
│       │   ├── aggregate.py            # Exact copy from original
│       │   ├── models/                 # Exact copy from original
│       │   ├── datasets/               # Exact copy from original
│       │   ├── losses.py               # Exact copy from original
│       │   └── utils.py                # Exact copy from original
│       └── data_utils/
│           ├── data_loader.py          # NVFLARE data integration
│           └── partitioner.py          # Client data partitioning
└── data/                               # Real datasets
    ├── PASCALContext/
    └── NYUDv2/
```

## Quick Start

1. **Setup datasets**:
   ```bash
   # Download and extract datasets to data/ directory
   ./setup_datasets.sh
   ```

2. **Run experiment**:
   ```bash
   python run_experiment.py --rounds 100 --workspace /tmp/fedhca2_nvflare
   ```

## Design Principles

1. **Preserve Original Logic**: Core FedHCA2 algorithms remain unchanged
2. **Minimal NVFLARE Integration**: Only adapt interfaces, not internals  
3. **Real Data**: Use actual PASCAL-Context and NYUD-v2 datasets
4. **Clean Separation**: NVFLARE-specific code separate from algorithm code
5. **Faithful Results**: Achieve same performance as original implementation

## Experiment Configuration

- **5 Pascal Context Single-Task Clients**: semseg, human_parts, normals, edge, sal
- **1 NYU Depth Multi-Task Client**: semseg, normals, edge, depth
- **Model**: Swin Transformer-Tiny backbone with task-specific decoders
- **Aggregation**: Conflict-averse encoder + cross-attention decoder
- **Hyperweights**: Learnable personalization parameters

## Citation

```bibtex
@InProceedings{Lu_2024_CVPR,
    author    = {Lu, Yuxiang and Huang, Suizhi and Yang, Yuwen and Sirejiding, Shalayiding and Ding, Yue and Lu, Hongtao},
    title     = {FedHCA2: Towards Hetero-Client Federated Multi-Task Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
    pages     = {5599-5609}
}
```


