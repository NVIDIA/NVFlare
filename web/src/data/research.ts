// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import type { ImageMetadata } from "astro";
import fedcore from "../images/research/highlights/fedcore-modality-gap.png";
import autoFl from "../images/research/highlights/auto-fl-research-loop.png";
import fraud from "../images/research/highlights/fraud-detection-results.svg";
import fedumm from "../images/research/highlights/fedumm-overview.png";
import fedrevive from "../images/research/highlights/fedrevive-framework.png";
import fednca from "../images/research/highlights/fednca-results.svg";

// Keep this catalog aligned with research/README.md AND the project READMEs.
// Figure versions, credits, and source URLs are recorded alongside the assets.
export interface ResearchProject {
  directory: string;
  name: string;
  topic: string;
  publication: string;
  paperTitle: string;
  paper: string;
  summary: string;
  highlight?: {
    title: string;
    description: string;
    example: string;
    image: ImageMetadata;
    alt: string;
    caption: string;
    credit: string;
    figureUrl: string;
  };
}

export const researchProjects: ResearchProject[] = [
  {
    directory: "fedcore",
    name: "FedCoRe",
    topic: "Multimodal learning",
    publication: "DeCaF 2026",
    paperTitle:
      "FedCoRe: Target-Adaptive Completion for Missing Modalities in Healthcare Federated Learning",
    paper: "https://arxiv.org/abs/2608.18311",
    summary:
      "Learn corrections from paired observations to support clients with missing modalities.",
    highlight: {
      title: "Learning when a modality is missing",
      description:
        "FedCoRe learns representation or logit corrections from clients with paired observations. Validation determines whether to apply completion or keep the original prediction.",
      example:
        "The public starter uses frozen Qwen3-VL with MNIST images and simulated OCR reports to demonstrate missing-image completion. It is a tutorial, not a reproduction of the clinical study.",
      image: fedcore,
      alt: "Complete EHR, chest X-ray, and ECG inputs assumed at evaluation, compared with federated clients that have different subsets of those modalities.",
      caption:
        "Figure 1. The gap between complete-modality evaluation and uneven modality availability across clients.",
      credit: "Roth, Xu, and Cnudde, 2026",
      figureUrl: "https://arxiv.org/html/2608.18311v1#S1.F1",
    },
  },
  {
    directory: "auto-fl-research",
    name: "Auto-FL-Research",
    topic: "Automation",
    publication: "arXiv 2026",
    paperTitle:
      "Auto-FL-Research: Agentic Search for Federated Learning Algorithms",
    paper: "https://arxiv.org/abs/2607.01366",
    summary:
      "Explore FL algorithms with coding agents, bounded edits, fixed evaluation, and an experiment ledger.",
    highlight: {
      title: "An agent-assisted research loop",
      description:
        "Auto-FL-Research studies coding agents that propose and evaluate FL training recipes within a fixed task profile. The paper examines repeated gains, tuning effects, and failure cases across FLamby and LEAF tasks.",
      example:
        "The repository provides task profiles, a simulation harness, candidate review and reporting tools, and a literature-review loop for stalled searches.",
      image: autoFl,
      alt: "Auto-FL loop from research intent through a task profile, bounded code changes, NVFlare experiments, and a results ledger, with a literature-review path when progress stalls.",
      caption:
        "Figure 2, left panel. Bounded algorithm search connects each experiment to a recorded review decision.",
      credit: "Roth et al., 2026",
      figureUrl: "https://arxiv.org/html/2607.01366v1#S3.F2",
    },
  },
  {
    directory: "fsi-fraud-detection",
    name: "Federated fraud detection",
    topic: "Financial services",
    publication: "arXiv 2026",
    paperTitle:
      "Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE",
    paper: "https://arxiv.org/abs/2603.13617",
    summary:
      "Study payment fraud across heterogeneous institutions, including interpretability and DP-SGD.",
    highlight: {
      title: "Collaborative payment fraud detection",
      description:
        "This proof-of-concept study compares local, federated, and centralized training across heterogeneous financial institutions. It also investigates feature attribution and privacy–utility trade-offs with DP-SGD.",
      example:
        "Start with configurable synthetic payment data and anomaly injection, then follow the example's federated training and evaluation workflow.",
      image: fraud,
      alt: "F1-score curves compare local, federated, and centralized payment fraud detection over training rounds, with shaded variability bands.",
      caption:
        "Figure 3(a). Fraud-detection performance over training rounds in the paper's experimental setting.",
      credit: "Roth et al., 2026",
      figureUrl: "https://arxiv.org/html/2603.13617v1#S2.F3",
    },
  },
  {
    directory: "fedumm",
    name: "FedUMM",
    topic: "Multimodal learning",
    publication: "arXiv 2026",
    paperTitle:
      "FedUMM: A General Framework for Federated Learning with Unified Multimodal Models",
    paper: "https://arxiv.org/abs/2601.15390",
    summary:
      "Federate lightweight LoRA adapters for multimodal foundation models under non-IID data.",
    highlight: {
      title: "Federating multimodal foundation models",
      description:
        "FedUMM trains LoRA adapters while keeping the foundation model frozen. The paper studies BLIP3o on visual question answering and image generation with heterogeneous client data.",
      example:
        "The runnable example uses Salesforce/blip-vqa-base as a lightweight stand-in for the paper's BLIP3o backbone and demonstrates federated visual question answering.",
      image: fedumm,
      alt: "Clients with different vision and text inputs train LoRA modules and send adapter updates to a central aggregator while backbone modules stay frozen.",
      caption:
        "Figure 3. Adapter aggregation across clients in the FedUMM framework.",
      credit: "Su et al., 2026",
      figureUrl: "https://arxiv.org/html/2601.15390v1#S3.F3",
    },
  },
  {
    directory: "fedrevive",
    name: "FedRevive",
    topic: "Efficient training",
    publication: "arXiv 2025 · revised 2026",
    paperTitle:
      "Reviving Stale Updates: Data-Free Knowledge Distillation for Asynchronous Federated Learning",
    paper: "https://arxiv.org/abs/2511.00655",
    summary:
      "Combine asynchronous model updates with server-side data-free knowledge distillation.",
    highlight: {
      title: "Making stale client updates useful",
      description:
        "FedRevive combines parameter aggregation with data-free knowledge distillation at the server. A synthetic-data generator and recent client models help transfer knowledge from delayed updates to the current model.",
      example:
        "The Collab API implementation compares FedAvg, FedBuff, and FedRevive on CIFAR-10 using reproducible simulated arrival schedules.",
      image: fedrevive,
      alt: "Fast and slow clients send updates with different staleness; the server blends parameter updates with knowledge distilled from buffered teachers and generated samples.",
      caption:
        "Figure 1. Asynchronous clients, hybrid server aggregation, and the knowledge-distillation module.",
      credit: "Askin et al., 2026 revision",
      figureUrl: "https://arxiv.org/html/2511.00655v2#S2.F1",
    },
  },
  {
    directory: "FedNCA",
    name: "FedNCA",
    topic: "Medical imaging",
    publication: "MICCAI 2025",
    paperTitle: "Equitable Federated Learning with NCA",
    paper: "https://arxiv.org/abs/2506.21735",
    summary:
      "Use neural cellular automata for medical segmentation with low communication and compute demands.",
    highlight: {
      title: "Medical imaging on limited resources",
      description:
        "FedNCA uses compact neural cellular automata to study federated segmentation where bandwidth and compute are limited, including heterogeneous devices and encrypted model aggregation.",
      example:
        "The research folder introduces the method and links to the authors' external implementation and setup instructions.",
      image: fednca,
      alt: "Ultrasound and X-ray segmentation Dice scores versus transmission cost on a logarithmic scale, comparing FedNCA with UNet and TransUNet variants.",
      caption:
        "Figure 3. Segmentation quality versus model transmission cost on the paper's ultrasound and X-ray tasks.",
      credit: "Lemke et al., 2025 · CC BY 4.0",
      figureUrl: "https://arxiv.org/html/2506.21735v1#S5.F3",
    },
  },
  {
    directory: "fedhca2",
    name: "FedHCA²",
    topic: "Multitask learning",
    publication: "CVPR 2024",
    paperTitle: "FedHCA2: Towards Hetero-Client Federated Multi-Task Learning",
    paper:
      "https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FedHCA2_Towards_Hetero-Client_Federated_Multi-Task_Learning_CVPR_2024_paper.html",
    summary:
      "Collaborate across clients with different dense prediction tasks and model architectures.",
  },
  {
    directory: "fed-bpt",
    name: "FedBPT",
    topic: "Language models",
    publication: "ICML 2024",
    paperTitle:
      "FedBPT: Efficient Federated Black-box Prompt Tuning for Large Language Models",
    paper: "https://arxiv.org/abs/2310.01467",
    summary:
      "Adapt language models with federated black-box prompt tuning; includes an SST-2 example.",
  },
  {
    directory: "condist-fl",
    name: "ConDistFL",
    topic: "Medical imaging",
    publication: "DeCaF 2023",
    paperTitle:
      "ConDistFL: Conditional Distillation for Federated Learning from Partially Annotated Data",
    paper: "https://arxiv.org/abs/2308.04070",
    summary:
      "Learn medical image segmentation from sites with different partial organ annotations.",
  },
  {
    directory: "fedobd",
    name: "FedOBD",
    topic: "Efficient training",
    publication: "IJCAI 2023",
    paperTitle:
      "FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning",
    paper: "https://arxiv.org/abs/2208.05174",
    summary:
      "Explore communication-efficient learning through the contributed ADAQUANT quantization scheme.",
  },
  {
    directory: "fed-ce",
    name: "FedCE",
    topic: "Medical imaging",
    publication: "CVPR 2023",
    paperTitle:
      "Fair Federated Medical Image Segmentation via Client Contribution Estimation",
    paper: "https://arxiv.org/abs/2303.16520",
    summary:
      "Estimate client contributions to improve fairness in federated medical image segmentation.",
  },
  {
    directory: "one-shot-vfl",
    name: "One-shot vertical FL",
    topic: "Efficient training",
    publication: "ICCV 2023",
    paperTitle:
      "Communication-Efficient Vertical Federated Learning with Limited Overlapping Samples",
    paper: "https://arxiv.org/abs/2303.16270",
    summary:
      "Explore vertical learning with limited overlapping samples and reduced communication on CIFAR-10.",
  },
  {
    directory: "quantifying-data-leakage",
    name: "Quantifying data leakage",
    topic: "Privacy",
    publication: "IEEE TMI 2023",
    paperTitle: "Do Gradient Inversion Attacks Make Federated Learning Unsafe?",
    paper: "https://arxiv.org/abs/2202.06924",
    summary:
      "Analyze gradient inversion and quantify reconstruction leakage during federated training.",
  },
  {
    directory: "fed-sm",
    name: "FedSM",
    topic: "Medical imaging",
    publication: "CVPR 2022",
    paperTitle:
      "Closing the Generalization Gap of Cross-silo Federated Medical Image Segmentation",
    paper: "https://arxiv.org/abs/2203.10144",
    summary:
      "Study personalized models and model selection for cross-silo medical image segmentation.",
  },
  {
    directory: "auto-fed-rl",
    name: "Auto-FedRL",
    topic: "Automation",
    publication: "ECCV 2022",
    paperTitle:
      "Auto-FedRL: Federated Hyperparameter Optimization for Multi-institutional Medical Image Segmentation",
    paper: "https://arxiv.org/abs/2203.06338",
    summary:
      "Optimize federated hyperparameters with reinforcement learning; includes CIFAR-10 simulations.",
  },
  {
    directory: "fed-bn",
    name: "FedBN",
    topic: "Personalization",
    publication: "ICLR 2021",
    paperTitle:
      "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization",
    paper: "https://arxiv.org/abs/2102.07623",
    summary:
      "Keep batch normalization local to handle feature shifts across client datasets.",
  },
  {
    directory: "brats18",
    name: "Federated brain tumor segmentation",
    topic: "Medical imaging",
    publication: "MLMI 2019",
    paperTitle: "Privacy-preserving Federated Brain Tumour Segmentation",
    paper: "https://arxiv.org/abs/1910.00962",
    summary:
      "Compare centralized and federated BraTS18 segmentation and explore privacy-preserving training.",
  },
];
