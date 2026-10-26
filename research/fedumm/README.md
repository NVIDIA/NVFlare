# FedUMM: Federated Learning for Unified Multimodal Models

This example shows how to run [FedUMM](https://arxiv.org/abs/2601.15390) with the [FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/nvflare_cli/fl_simulator.html).

###### Abstract:
> Unified multimodal models (UMMs) are emerging as strong foundation models that can do both generation and understanding tasks in a single architecture. However, they are typically trained in centralized settings where all training and downstream datasets are gathered in a central server, limiting the deployment in privacy-sensitive and geographically distributed scenarios. In this paper, we present FedUMM, a general federated learning framework for UMMs under non-IID multimodal data with low communication cost. Built on NVIDIA FLARE, FedUMM instantiates federation for a BLIP3o backbone via parameter-efficient fine-tuning: clients train lightweight LoRA adapters while freezing the foundation models, and the server aggregates only adapter updates. We evaluate on VQA v2 and the GenEval compositional generation benchmarks under Dirichlet-controlled heterogeneity with up to 16 clients. Results show slight degradation as client count and heterogeneity increase, while remaining competitive with centralized training. We further analyze computation–communication trade-offs and demonstrate that adapter-only federation reduces per-round communication by over an order of magnitude compared to full fine-tuning, enabling practical federated UMM training. This work provides empirical experience for future research on privacy-preserving federated unified multimodal models.

## License
The code in this directory is released under Apache v2 License.
Model weights are subject to the [Salesforce Research license](https://huggingface.co/Salesforce/blip-vqa-base).

> **Note:** For illustration purposes, this example uses [`Salesforce/blip-vqa-base`](https://huggingface.co/Salesforce/blip-vqa-base) as a lightweight stand-in. The paper uses BLIP3o as the backbone.

## 1. Setup

We recommend creating a [conda environment](https://www.anaconda.com):
```commandline
conda create -n fedumm python=3.10 -y
conda activate fedumm
pip install -r requirements.txt
```

## 2. Run a federated learning experiment

### Verify setup with a centralized baseline

```commandline
python centralized_baseline.py \
    --batch_size 8 \
    --num_epochs 1 \
    --max_train_samples 500 \
    --max_eval_samples 100
```

### Run federated learning with the NVFlare simulator

```commandline
python job.py \
    --num_clients 4 \
    --num_rounds 5 \
    --local_epochs 1 \
    --dirichlet_alpha 0.5 \
    --batch_size 32 \
    --lr 2e-5
```

The `--dirichlet_alpha` argument controls data heterogeneity across clients: lower values (e.g., `0.1`) produce more extreme non-IID splits, while higher values (e.g., `1.0`) approach near-IID. Set `--dirichlet_alpha 0` for IID round-robin partitioning.

The simulator runs all clients on a single machine.

### Key arguments

| Argument | Description | Default |
|---|---|---|
| `--num_clients` | Number of FL clients | `2` |
| `--num_rounds` | Communication rounds | `3` |
| `--local_epochs` | Local training epochs per round | `1` |
| `--dirichlet_alpha` | Non-IID level (`0`=IID, lower=more skewed) | `0` |
| `--lora_r` | LoRA rank | `16` |
| `--lora_alpha` | LoRA scaling factor | `32` |
| `--lr` | Learning rate | `5e-5` |
| `--batch_size` | Per-client batch size | `8` |
| `--data_path` | HuggingFace cache directory | `""` |

## 3. GPU Requirements

The simulator runs all clients in the same process on a single GPU. Each client loads its own model copy, so memory scales with the number of clients. Measured on an NVIDIA RTX 6000 Ada (48 GB):

| Clients | GPU memory |
|---|---|
| 4 | ~30 GB (~7.5 GB per client) |

Reduce `--batch_size` (e.g. to 4) if running on a smaller GPU.

## 4. Results

The paper evaluates FedUMM on VQA v2 with K = {2, 4, 6, 8, 10, 12, 14, 16} clients and
Dirichlet alpha = {0.1, 0.5, 1.0}. FedUMM remains competitive with centralized training
(centralized: 82.4%; K=8, alpha=0.5: ~80.2%) while exchanging only LoRA adapter weights,
achieving over 10x communication reduction compared to full fine-tuning.

## Acknowledgments

This work was partially supported by the NVIDIA Academic Grant Program.

## Citation

> Su, Zhaolong, et al. "FedUMM: A General Framework for Federated Learning with Unified Multimodal Models." arXiv preprint arXiv:2601.15390 (2026).

BibTeX
```
@article{su2026fedumm,
    title={FedUMM: A General Framework for Federated Learning with Unified Multimodal Models},
    author={Su, Zhaolong and Zhao, Leheng and Wu, Xiaoying and Xu, Ziyue and Wang, Jindong},
    journal={arXiv preprint arXiv:2601.15390},
    year={2026}
}
```
