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

- **Documentation and Tutorial Transformation** — Redesign the onboarding
  path so a first-time user runs a real federation before seeing a feature
  list, and bring the 100+ notebook self-paced course and research-project
  guides up to a verified, reproducible standard.
- **Better GPU resource utilization** — Release GPUs during aggregation,
  barriers, and downloads instead of holding them idle. A durable, GPU-free
  site supervisor keeps a job's identity alive across time-bounded Slurm
  allocations via checkpoint-and-resume, with no change to the
  FedAvg/FedOpt/DiLoCo math.
- **Job Statistics** — A usage proxy for cost: FLARE has no access to
  billing rates, so the goal is to report compute hours, GPU hours, storage
  volume/time, and ingress/egress size per job, so real cost can be mapped
  in later once rates are supplied.
- **Operation and Security** — Federated identity via OpenID Connect (sites
  federate through their existing enterprise IdP) plus short-lived,
  job-scoped ephemeral keys, so a leaked credential's blast radius is
  limited to one job and one window.
- **Audit log enhancements** — Still a brainstorm, not yet scoped. Today's
  per-site audit log (unchanged since FLARE 2.2.1) captures event ID, user
  identity, action name, and job ID as one flat text line; possible gaps
  include artifact-level detail, site identity, and a structured,
  queryable format.
- **AI Agents in Federated Data Networks** — Cover sites whose data isn't
  mapped to a shared schema yet: reach sites outside existing
  standardization programs, adaptively discover unmapped data, turn
  ambiguous questions into queries within FLARE's allowlisted sandbox, and
  reconcile same-schema/different-labeling mismatches across sites.

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
