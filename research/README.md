# Research with NVIDIA FLARE

<img src="../docs/resources/researcher.svg" alt="Researcher Icon" width="100">

NVIDIA FLARE has been used in several research studies. This directory contains
reference implementations, reusable research support material, and a template for
new community contributions.

Research projects in this directory do not all use the same layout. Some provide
NVFlare jobs under `jobs/`, some use a `job.py` recipe, and others include
scripts, notebooks, or links to external reference code. Start with the local
README in each subfolder for setup, data, and run instructions.

## Published and Reference Implementations

| Project | Description |
| --- | --- |
| [Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE](./fsi-fraud-detection/README.md) | Payment fraud detection across heterogeneous financial institutions ([arXiv 2026](https://arxiv.org/abs/2603.13617)). |
| [FedUMM: Federated Learning for Unified Multimodal Models](./fedumm/README.md) | Federated unified multimodal model training with parameter-efficient updates ([arXiv 2026](https://arxiv.org/abs/2601.15390)). |
| [FedNCA - Equitable Federated Learning with NCA](./FedNCA/README.md) | Equitable federated learning with NCA ([MICCAI 2025](https://arxiv.org/abs/2506.21735)). |
| [FedHCA2: Towards Hetero-Client Federated Multi-Task Learning](./fedhca2/README.md) | Heterogeneous-client federated multi-task learning ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FedHCA2_Towards_Hetero-Client_Federated_Multi-Task_Learning_CVPR_2024_paper.html), [arXiv](https://arxiv.org/abs/2311.13250)). |
| [FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models](./fed-bpt/README.md) | Black-box prompt tuning for federated language model adaptation ([arXiv 2023](https://arxiv.org/abs/2310.01467)). |
| [ConDistFL: Conditional Distillation for Federated Learning from Partially Annotated Data](./condist-fl/README.md) | Conditional distillation for partially annotated medical image segmentation ([arXiv 2023](https://arxiv.org/abs/2308.04070)). |
| [FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning](./fedobd/README.md) | Communication-efficient block dropout for large-scale federated models ([IJCAI 2023](https://www.ijcai.org/proceedings/2023/0394.pdf), [arXiv](https://arxiv.org/abs/2208.05174)). |
| [FedCE: Fair Federated Learning via Client Contribution Estimation](./fed-ce/README.md) | Fair federated medical image segmentation via contribution estimation ([CVPR 2023](https://arxiv.org/abs/2303.16520)). |
| [One-shot Vertical Federated Learning with CIFAR-10](./one-shot-vfl/README.md) | Communication-efficient vertical federated learning with limited overlapping samples ([ICCV 2023](https://arxiv.org/abs/2303.16270)). |
| [Personalized Federated Learning with FedSM Algorithm](./fed-sm/README.md) | Personalized federated medical image segmentation ([CVPR 2022](https://arxiv.org/abs/2203.10144)). |
| [Quantifying Data Leakage in Federated Learning](./quantifying-data-leakage/README.md) | Gradient inversion attacks and data leakage analysis ([IEEE Transactions on Medical Imaging](https://arxiv.org/abs/2202.06924)). |
| [Auto-FedRL](./auto-fed-rl/README.md) | Federated hyperparameter optimization for medical image segmentation ([ECCV 2022](https://arxiv.org/abs/2203.06338)). |
| [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](./fed-bn/README.md) | Local batch normalization for feature-shifted federated learning ([ICLR 2021](https://arxiv.org/abs/2102.07623)). |
| [Privacy-preserving Federated Brain Tumour Segmentation](./brats18/README.md) | Federated brain tumor segmentation with privacy-preserving training ([MLMI 2019](https://arxiv.org/abs/1910.00962)). |

## Agent-Assisted Research Workflow

- [Auto-FL Research with NVFlare](./auto-fl-research/README.md) provides
  an autoresearch-style control plane, CIFAR-10 simulation harness, bounded
  mutation workflow, and reporting tools for agent-assisted FL experiments.

## Shared Data and Task Support

- [Prostate multi-source data preparation](./prostate/README.md) contains
  shared data preparation instructions used by prostate segmentation research
  projects such as FedCE and FedSM.

## Contributor Template

- [Sample Research Project](./sample-research/README.md) is a flexible README
  template for new research contributions. It shows the common information
  contributors should provide while allowing each project to use the code,
  job, notebook, or external-reference layout that best fits the work.

## Contributing

To provide your own research implementation, start with the
[sample research template](./sample-research/README.md) and follow the
[research contribution guide](./CONTRIBUTING.md).
