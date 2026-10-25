# NVFLARE Feature Tutorials

Focused Jupyter notebooks for common NVFLARE tasks. This directory is the
authoritative index for the maintained feature tutorials: each notebook has a
single outcome, stated prerequisites, and an expected completion signal.

Looking for something else?

- **First run:** the [Quick Start](https://nvflare.readthedocs.io/en/main/quickstart.html)
  runs a meaningful federated job in the simulator and explains the result.
- **Introductory applications:** the [hello-world examples](../hello-world)
  run a complete federated job per framework with a single `python job.py`.
- **Example catalog:** browse all runnable examples on the
  [NVFLARE website](https://nvidia.github.io/NVFlare/catalog/).
- **Structured training:** the [self-paced course](./self-paced-training)
  (NVFLARE 2.6 edition) remains available as versioned training.

## Setup

1. Create a virtual environment and install NVFLARE following the
   [example root README](../README.md). Supported Python: 3.10 or later.
2. Install and start JupyterLab from this directory, then open a notebook and
   run the cells in order.

Notebooks that exercise the simulator run jobs under `/tmp/nvflare` by
default; notebooks that exercise POC mode require a provisioned POC system
(see each notebook for its terminal prerequisites).

## Notebook index

### Build and test applications

| Notebook | Outcome | Prerequisites | Expected completion signal | Owning area |
|---|---|---|---|---|
| [Intro to the FL Simulator](./flare_simulator.ipynb) | Run NVFLARE jobs locally with the simulator-managed server and simulated clients; know when to use Recipe-style `job.py` runs vs. `nvflare simulator` | `nvflare`, `tensorboard`; this repository checkout (uses `../hello-world/hello-numpy`) | hello-numpy finishes `FINISHED:COMPLETED`; results under `/tmp/nvflare/simulation/hello-numpy` | Simulator |
| [Job Recipe](./job_recipe.ipynb) | Build and run federated jobs with the high-level Recipe API (FedAvg with PyTorch) | `torch`, `tensorboard`; this repository checkout (uses `../hello-world/hello-pt`) | A multi-round FedAvg simulation completes with the run summary (models, metrics, evaluation paths) | Recipe API |

### Operate and deploy systems

| Notebook | Outcome | Prerequisites | Expected completion signal | Owning area |
|---|---|---|---|---|
| [NVFLARE in POC Mode](./setup_poc.ipynb) | Configure and operate POC mode: workspace, prepare, start/stop, and inspect a full local FL system | `nvflare` CLI; `tree` command optional (Linux) | A two-client POC system is prepared, started, and inspected from the notebook | POC mode |
| [NVFlare CLI](./nvflare_cli.ipynb) | End-to-end CLI workflow against a running POC system: recipes, job export and submit, studies, monitoring, result download | POC system started from a terminal first (commands in the notebook); `torch` for the hello-pt job | Submitted job completes and results download to the notebook's `RESULT_DIR` | NVFlare CLI |
| [Hello FLARE API](./flare_api.ipynb) | Connect an admin session with the FLARE API and drive a running FL system programmatically | A running POC system with the admin startup kit | `new_secure_session()` connects and `get_system_info()` reports server, clients, and jobs | FLARE API |

### Observe and troubleshoot runs

| Notebook | Outcome | Prerequisites | Expected completion signal | Owning area |
|---|---|---|---|---|
| [FLARE Logging](./logging.ipynb) | Configure logging modes (`concise`, `progress`, `msg_only`, `full`, `verbose`) and custom JSON log configs | `nvflare`, `tensorboard`; `tree` command optional | hello-numpy runs with the selected log config; log files shown under `/tmp/nvflare/simulation/hello-numpy` | Logging configuration |

## Verification

Verification records a maintainer executing the notebook's flow end to end in
the stated environment.

- **2026-10-24 — simulator-based notebooks** (FL Simulator, Job Recipe,
  Logging): core flows re-verified on `main` (2.10.0.dev0, commit
  `b329c8113`), Python 3.14 on Linux. The hello-numpy and hello-pt simulation
  runs used by these notebooks completed successfully.
- **2026-10-24 — POC-based notebooks** (POC Mode, NVFlare CLI, FLARE API):
  core flows re-verified on `main` (2.10.0.dev0, commit `b329c8113`),
  Python 3.14 on Linux: POC prepare and start of a two-client system, FLARE
  API admin session (`new_secure_session` + `get_system_info`), and job
  export, submit, completion (`FINISHED:COMPLETED`), and result download
  for a hello-pt job.

Report problems as [GitHub issues](https://github.com/NVIDIA/NVFlare/issues).

## Self-paced course

- [Federated Learning with NVIDIA FLARE](./self-paced-training/README.md)
  - A five-part course covering fundamentals, system architecture, privacy
    and security, advanced topics, and industry applications.
  - Developed with NVFLARE 2.6: not all content reflects the latest APIs.

## NVIDIA DLI courses

Hosted courses for guided, hands-on FLARE training. They are versioned
independently of the current NVFLARE release and may use earlier NVFLARE
versions.

- [Introduction to Federated Learning with NVIDIA FLARE](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-28+V1)
- [Decentralized AI at Scale with NVIDIA FLARE](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-29+V1)

Visit the [NVIDIA FLARE developer portal](https://developer.nvidia.com/flare)
for product information, tutorials, documentation, and developer resources.
