.. _roadmap:

####################
NVIDIA FLARE Roadmap
####################

This page outlines planned features and target release milestones for upcoming NVIDIA FLARE versions.
Dates and features are subject to change.

.. note::
   This roadmap reflects current planning and is provided for informational purposes.
   Feature scope and release timing may shift as development progresses.

********************************
FLARE 2.9.1 — Target: Q4 2026
********************************

**Confidential Federated AI Support**

Building on existing support for AMD SEV-SNP CPU CVMs with NVIDIA GPUs:

- Intel TDX CPU support for CPU-based confidential computing workloads
- CoCo (Confidential Containers) support for container-level confidential execution

***********************************
FLARE 2.10.0 — Target: Early 2027
***********************************

**Possible Features**

- Documentation and Tutorial Transformation
- Better GPU resource utilization
- Job Statistics
- Operation and Security
- Audit log enhancements
- AI Agents in Federated Data Networks

**************************
Research Directions
**************************

These are earlier-stage research threads that may feed a future FLARE
release. They are not yet scheduled or committed to a specific version.

- **Memorization in LLMs** — Understanding what a federated LLM training run
  memorizes from participant data, a prerequisite for defensible privacy
  claims about federated training itself.
- **FedRevive (Async FL Algorithm)** — An asynchronous federated learning
  algorithm where participants contribute without waiting on a synchronous
  round barrier.
- **Missing Modality** — Robust federated learning when different sites hold
  different, incomplete subsets of a multimodal dataset.
- **Contribution Estimation** — Attributing a trained model's
  quality/behavior back to each participating site's actual contribution.
