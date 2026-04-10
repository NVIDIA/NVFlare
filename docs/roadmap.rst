.. _roadmap:

####################
NVIDIA FLARE Roadmap
####################

This page outlines planned features and target release milestones for upcoming NVIDIA FLARE versions.
Dates and features are subject to change.

.. note::
   This roadmap reflects current planning and is provided for informational purposes.
   Feature scope and release timing may shift as development progresses.

*******************************
FLARE 2.8.0 — Target: Q2 2026
*******************************

**Native Kubernetes Support**

- Separate the parent control pod from the job execution pod, enabling independent lifecycle management and better resource isolation
- Simplified deployment across major cloud Kubernetes environments (GKE, EKS, AKS, and on-prem)

**Improved Docker Deployment**

- Separate parent container from job execution container, mirroring the Kubernetes pod separation model
- Ready-to-use Dockerfiles provided for common deployment scenarios, reducing setup friction

**Multi-Study Support**

- Enable multiple concurrent studies within a single FLARE deployment
- Enforce data isolation between studies via Docker and Kubernetes pod-level data separation

**Distributed Provisioning**

- Enable the distributed provisioning workflow so site administrators can generate their own key pairs locally and receive signed certificates from the project administrator
- Eliminates the need for centralized private key generation and distribution

**Expanded CLI Commands**

- Extend the ``nvflare`` CLI to cover all FLARE Admin Console commands
- Enables full administrative control from the command line without requiring the interactive console

**Server-Side Memory Optimization**

- Reduce server-side memory usage during federated learning jobs
- Improved memory management for large model and large dataset workloads

*******************************
FLARE 2.9.0 — Target: Q3 2026
*******************************

**New Collab API**

- Introduce a new Collaboration (Collab) API designed to improve research productivity
- Enables more flexible and composable FL workflow definitions with reduced boilerplate

**FLARE Agent Readiness**

- Platform features enabling FLARE to be used as a backend for AI agent workflows

**Better Kubernetes User Experience**

- Simplified Kubernetes deployment and operational experience building on the 2.8.0 foundation
- Usability improvements for data scientists and operators managing multi-site Kubernetes deployments

**Slurm Support**

- Better integration with Slurm workload managers for HPC cluster environments
- Enable FL jobs to run natively within Slurm-managed compute environments

********************************
FLARE 2.10.0 — Target: Q4 2026
********************************

**Advanced Kubernetes Enhancements**

- Advanced Kubernetes feature set building on prior releases
- Deeper platform integration and operational controls for large-scale multi-site Kubernetes deployments

**Confidential Federated AI Support**

Building on existing support for AMD SEV-SNP CPU CVMs with NVIDIA GPUs and Azure Confidential Computing:

- Intel TDX CPU support for CPU-based confidential computing workloads
- CoCo (Confidential Containers) support for container-level confidential execution
- Expanded Cloud Service Provider (CSP) integration beyond Azure to additional major cloud platforms
