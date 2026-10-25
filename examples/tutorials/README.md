# NVFLARE Feature Tutorials

Focused Jupyter notebooks for common NVFLARE tasks. This directory is the
authoritative index for the maintained feature tutorials: each notebook has a
single outcome, stated prerequisites, and an expected completion signal.

Looking for something else?

- **First run:** the [Quick Start](https://nvflare.readthedocs.io/en/main/quickstart.html)
  runs a meaningful federated job in the simulator and explains the result.
- **Introductory applications:** the [hello-world examples](../hello-world)
  demonstrate complete federated applications across frameworks. Follow each
  example's README for dependencies, data or model preparation, and the exact
  run command; some examples require arguments to `job.py`.
- **Example catalog:** browse all runnable examples on the
  [NVFLARE website](https://nvidia.github.io/NVFlare/catalog/).
- **Structured training:** the [self-paced course](./self-paced-training)
  (NVFLARE 2.6 edition) remains available as versioned training.

## Setup

1. Create a virtual environment and install NVFLARE following the
   [example root README](../README.md). Supported Python: 3.10 or later.
2. Install and start JupyterLab from this directory, then open a notebook in
   a fresh kernel. Follow each section's prerequisites before running it;
   optional Docker, HE, and production sections need additional setup.

Notebooks that exercise the simulator run jobs under `/tmp/nvflare` by
default; notebooks that exercise POC mode require a provisioned POC system
(see each notebook for its terminal prerequisites).

## Notebook index

### Build and test applications

| Notebook | Outcome | Prerequisites | Expected completion signal | Owning area |
|---|---|---|---|---|
| [Intro to the FL Simulator](./flare_simulator.ipynb) | Run NVFLARE jobs locally with the simulator-managed server and simulated clients; know when to use Recipe-style `job.py` runs vs. `nvflare simulator` | `nvflare`, `tensorboard`; this repository checkout (uses `../hello-world/hello-numpy`) | hello-numpy finishes `FINISHED:COMPLETED`; results under `/tmp/nvflare/simulation/hello-numpy` | Simulator |
| [Job Recipe](./job_recipe.ipynb) | Build and run federated jobs with the high-level Recipe API (FedAvg with PyTorch) | `torch`, `tensorboard`; this repository checkout (uses `../hello-world/hello-pt`) | A multi-round FedAvg simulation completes with the run summary (models and metrics) | Recipe API |

### Operate and deploy systems

| Notebook | Outcome | Prerequisites | Expected completion signal | Owning area |
|---|---|---|---|---|
| [NVFLARE in POC Mode](./setup_poc.ipynb) | Configure and operate POC mode: workspace, prepare, start/stop, and inspect a full local FL system | `nvflare` CLI; `tree` command optional | A two-client POC system is prepared, started, and inspected using the documented commands | POC mode |
| [NVFlare CLI](./nvflare_cli.ipynb) | End-to-end CLI workflow against a running POC system: recipes, job export and submit, studies, monitoring, result download | POC system started from a terminal first (commands in the notebook); `torch` for the hello-pt job | Submitted job completes and results download to the notebook's `RESULT_DIR` | NVFlare CLI |
| [Hello FLARE API](./flare_api.ipynb) | Connect an admin session with the FLARE API and drive a running FL system programmatically | A running POC system with the admin startup kit; `tensorboard` for the hello-numpy job | `new_secure_session()` connects and `get_system_info()` reports server, clients, and jobs | FLARE API |

### Observe and troubleshoot runs

| Notebook | Outcome | Prerequisites | Expected completion signal | Owning area |
|---|---|---|---|---|
| [FLARE Logging](./logging.ipynb) | Configure logging modes (`concise`, `progress`, `msg_only`, `full`, `verbose`) and custom JSON log configs | `nvflare`, `tensorboard`; `tree` command optional | hello-numpy runs with the selected log config; log files shown under `/tmp/nvflare/simulation/hello-numpy` | Logging configuration |

## Verification

Verification is scoped to the workflows explicitly listed below, not every
cell or optional deployment section. Development-checkout results do not
establish compatibility with a published release; use notebooks and examples
from the revision matching your installed NVFLARE version.

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
- **2026-10-24 — notebook refresh during review:** Python 3.13.15 on macOS,
  using this PR's development checkout based on `b329c8113`. The revised
  Job Recipe cells completed three rounds with two clients in both simulation
  and native POC mode. hello-numpy simulations completed with default and
  custom JSON logging. All FLARE API code cells ran against a temporary
  two-client POC deployment, substituting only its admin-kit path: connect,
  export, submit, monitor, download, resubmit, abort, delete, and close.
  The CLI helper exported a job; all six notebooks passed schema validation,
  77 code cells compiled, and 46 runnable CLI invocations parsed. The
  notebook regression tests and related POC/export/hello-pt tests passed
  (36 tests).

The review-time checks did not execute every POC/CLI cell, optional Docker,
HE, checkpoint-loading, or production-deployment sections, and did not run
against released wheels. Those sections still require their stated
environment and assets.

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
