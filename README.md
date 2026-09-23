<img src="https://raw.githubusercontent.com/NVIDIA/NVFlare/main/docs/resources/nvidia_eye.wwPt122j.png" alt="NVIDIA Logo" width="200">

# NVIDIA FLARE

Build federated computing applications that collaborate across organizational and geographic boundaries while each
site keeps control of its data and local execution.

[Website](https://nvidia.github.io/NVFlare) |
[NVIDIA FLARE Developer Site](https://developer.nvidia.com/flare) |
[Documentation](https://nvflare.readthedocs.io/en/main/) |
[Quick Start](https://nvflare.readthedocs.io/en/stable/quickstart.html) |
[Examples](https://nvidia.github.io/NVFlare/catalog/) |
[Discussions](https://github.com/NVIDIA/NVFlare/discussions)

[Paper](https://arxiv.org/abs/2210.13291) |
[Blogs](https://developer.nvidia.com/blog/tag/federated-learning) |
[Talks & Papers](https://nvflare.readthedocs.io/en/main/publications_and_talks.html) |
[Webinars](https://nvidia.github.io/NVFlare/webinars) |
[Research](./research/README.md)

[![Blossom-CI](https://github.com/NVIDIA/nvflare/workflows/Blossom-CI/badge.svg?branch=main)](https://github.com/NVIDIA/nvflare/actions)
[![documentation](https://readthedocs.org/projects/nvflare/badge/?version=main)](https://nvflare.readthedocs.io/en/main/?badge=main)
[![license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](./LICENSE)
[![pypi](https://badge.fury.io/py/nvflare.svg)](https://badge.fury.io/py/nvflare)
[![pyversion](https://img.shields.io/pypi/pyversions/nvflare.svg)](https://badge.fury.io/py/nvflare)
[![downloads](https://static.pepy.tech/badge/nvflare)](https://pepy.tech/project/nvflare)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NVFlare)

## What is NVIDIA FLARE?

**[NVIDIA FLARE™](https://developer.nvidia.com/flare)** (**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment) is a
domain-agnostic, open-source, extensible Python SDK for federated learning and other federated-computing applications.
It coordinates approved local computation and exchanges application-defined results across independently controlled
sites, while each site retains its source datasets and controls what results may leave.

Start with a small federated-learning example on your computer. Then adapt your training code or explore statistics,
custom workflows, and deployment options.

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA/NVFlare/f0aac04afdd7f0fab7675e8f5259a5a40cc9592c/docs/resources/federated_learning_overview.png"
       alt="Two hospitals train models on private local data and send model updates to a federated server, which aggregates them into a global model."
       width="650">
</p>

In this federated-learning workflow, the server sends a model to participating clients. Each client trains the model
on its local data and returns model updates, keeping its source dataset local. The server aggregates the updates into
a global model and repeats the process across training rounds.

The example below runs this workflow on your computer, with two simulated clients and a server.

## Try it locally

Create and activate a Python virtual environment as described in the
[installation guide](https://nvflare.readthedocs.io/en/stable/installation.html), then install the latest stable
NVFLARE release with its PyTorch integration, retrieve the matching Hello PyTorch example, and run it:

```bash
python -m pip install "nvflare[PT]"
nvflare examples get hello-pt
cd hello-pt
python job.py --log_config progress
```

This runs a two-client simulation on CPU using synthetic data; no dataset download is required.

Continue with the [Quick Start for the stable release](https://nvflare.readthedocs.io/en/stable/quickstart.html) to
understand the run and inspect its results. If you use an older release, select its version in the documentation.
Follow the example README for that release for dependencies, commands, and troubleshooting.

## Choose your path

Start with the guide for the concept and supported workflow, then use a maintained example as the runnable reference.

| Goal | Read first | Then run or adapt |
|---|---|---|
| Adapt existing training code | [Quick Start](https://nvflare.readthedocs.io/en/stable/quickstart.html), then [Agent Skills](https://nvflare.readthedocs.io/en/main/user_guide/agent_skills/index.html) for supported projects or [API selection](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/api_selection.html) for manual and custom integration | Try the [Agent Skills conversion examples](./examples/hello-world/agent-skills/README.md), or use [Hello PyTorch with the Client API and a Recipe](./examples/hello-world/hello-pt/README.md) |
| Fine-tune large language models | [Federated LLM guide](https://nvflare.readthedocs.io/en/main/programming_guide/llm_fine_tuning.html) | [Hugging Face LoRA fine-tuning](./examples/hello-world/hello-huggingface/README.md) or [NeMo supervised fine-tuning](./integration/nemo/examples/supervised_fine_tuning/README.md) |
| Federated analytics and statistics | [Agent Skills](https://nvflare.readthedocs.io/en/main/user_guide/agent_skills/index.html) for supported tabular and image datasets, or the [Federated Statistics guide](https://nvflare.readthedocs.io/en/main/examples/federated_statistics_overview.html) for manual and custom workflows | Agent-assisted [tabular](./examples/hello-world/agent-skills/fedstats-tabular/README.md) or [image statistics](./examples/hello-world/agent-skills/fedstats-image/README.md), or manual [DataFrame](./examples/advanced/federated-statistics/df_stats/README.md) or [image statistics](./examples/advanced/federated-statistics/image_stats/README.md) |
| Research and design custom FL algorithms or workflows | [Collaboration API guide (Technical Preview)](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/collab_api.html) | [Hello FedAvg with the Collab API](./examples/hello-world/hello-collab/README.md), then [split learning, asynchronous aggregation, and swarm examples](./examples/advanced/collab/README.md) |
| Learn a specific NVIDIA FLARE feature interactively | [Feature Tutorials](https://nvflare.readthedocs.io/en/main/tutorials.html#feature-tutorials) | [Simulator](./examples/tutorials/flare_simulator.ipynb), [Recipe](./examples/tutorials/job_recipe.ipynb), or [logging](./examples/tutorials/logging.ipynb) notebook |
| Follow structured NVIDIA FLARE training | [Tutorials and training](https://nvflare.readthedocs.io/en/main/tutorials.html) | [Self-paced curriculum](./examples/tutorials/self-paced-training/README.md) or [NVIDIA DLI introductory course](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-28+V1) |
| Validate POC and production environments | [Deployment guide](https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/deployment/overview.html) | [Hello PyTorch environment example](./examples/advanced/hello-pt-environments/README.md) |
| Add privacy and governance controls | [Security overview](https://nvflare.readthedocs.io/en/main/system_architecture/security_overview.html) | [DP](./examples/hello-world/hello-dp/README.md), [HE](./examples/advanced/cifar10/pt/cifar10-real-world/cifar10_fedavg_he/README.md), [PSI](./examples/advanced/psi/user_email_match/README.md), [confidential computing](./examples/advanced/cc_provision/README.md), or [site policies](./examples/advanced/federated-policies/README.rst) |

Use the [NVIDIA FLARE website](https://nvidia.github.io/NVFlare) to discover tutorials, examples, research, webinars,
and events. The [documentation](https://nvflare.readthedocs.io/en/main/) provides the authoritative guides. After
installing NVFLARE 2.10.0 or later, `nvflare examples list` shows the curated catalog and `nvflare examples get <name>`
retrieves an example matched to that installation's source revision. Each example README then owns its exact setup
and run steps.

## Federated learning and computing

**Federated learning** lets sites collaboratively train or evaluate models on their local data. Training and evaluation
happen where the data resides; the underlying training examples are not exchanged between sites or sent to a central
server. This can enable collaboration where privacy, regulation, data sovereignty, intellectual property, data
ownership, or the cost of moving data makes central collection impractical. Federated learning is one application of
the broader federated computing paradigm.

**Federated computing** brings approved computation to data distributed across multiple parties or locations. In
addition to model training and evaluation, it supports analytics, statistics, site-local data processing, and custom
multi-site computation. Participating sites run tasks locally and share only outputs permitted by the collaboration.

NVIDIA FLARE builds its federated-learning capabilities on top of a general federated-computing core. That core
provides workflow orchestration, task execution, communication, security, and lifecycle services shared by training,
evaluation, analytics, statistics, and custom distributed applications.

A **job** packages the application logic and configuration. A **workflow** coordinates tasks across participating
sites. Each site executes its task against local resources, and policies determine what can run and what may leave the
site. Depending on the application, the shared result can be an aggregate model, evaluation, statistic, intersection,
or another collaboration artifact.

NVIDIA FLARE also supports decentralized and client-controlled workflows, as well as federated computing applications
whose shared result is a statistic, evaluation, intersection, or another permitted application result rather than a
model.

> **Keep data at its source.** NVIDIA FLARE moves approved computation to participating sites. In a properly designed
> and governed federation, raw datasets remain at their source and only outputs permitted by the collaboration and
> each site's policies leave the site. Application owners and site operators define and enforce those policies for
> their data and threat model.

## Capabilities

| Area | Capabilities |
|---|---|
| Federated applications | Training and fine-tuning, evaluation and cross-site validation, analytics and statistics, site-local data processing, and custom multi-site computation |
| Models and frameworks | Any model type or ML/AI framework through extensible Python APIs, with maintained integrations and examples for PyTorch, TensorFlow, JAX, scikit-learn, XGBoost, Hugging Face, NeMo, and Flower |
| Algorithms and workflows | FedAvg, FedProx, FedOpt, SCAFFOLD, Ditto, cyclic and swarm learning, horizontal and vertical FL, and custom server- or client-controlled workflows |
| Privacy, security, and governance | Site-controlled authorization, audit logging, secure provisioning, differential privacy, homomorphic encryption, private set intersection, and confidential computing options |
| Runtime and operations | Local simulation, proof-of-concept environments, production provisioning, cloud and on-premises deployment, resiliency, monitoring, and experiment tracking |

FLARE's component and event architecture lets applications replace or extend controllers, aggregators, executors,
filters, persistence, communication, and deployment behavior without rewriting the entire system.

## From local development to production

The application stays consistent while its execution environment changes:

| Stage | Purpose |
|---|---|
| [Simulator](https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html) (`SimEnv`) | Run a local job directly to test application code and algorithms, without first starting and administering a FLARE deployment. |
| [Proof of Concept (POC)](https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/poc_command.html) (`PocEnv`) | Start a local FLARE deployment to practice job submission, administration, and deployment behavior before moving to provisioned sites. |
| [Production](https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/deployment/overview.html) (`ProdEnv`) | Run across provisioned sites with production identities, authorization, networking, policies, and operations. |

See the [run modes](https://nvflare.readthedocs.io/en/main/run_mode.html) and
[deployment guide](https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/deployment/overview.html) for the
differences and operational requirements.

## Industry use cases

Federated computing is useful when organizations need a shared result but cannot freely pool the source data. NVIDIA
FLARE has been applied to:

| Industry | Representative problems and evidence |
|---|---|
| Healthcare and oncology | Multi-institution medical imaging, tumor segmentation, survival analysis, and collaborative cancer research. See the [industry use cases](https://nvflare.readthedocs.io/en/main/industry_use_cases.html) and [healthcare research implementations](./research/README.md). |
| Life sciences and drug discovery | Collaborative protein-property prediction, molecular modeling, and model fine-tuning while organizations retain proprietary research data. See the [life-sciences case studies](https://nvflare.readthedocs.io/en/main/industry_use_cases.html#healthcare-life-sciences). |
| Financial services | Cross-institution fraud detection, risk analysis, and federated statistics over transaction data. See the [federated fraud-detection research workflow](./research/fsi-fraud-detection/README.md). |
| Automotive, edge, and industrial systems | Learning from distributed vehicles, devices, sensors, and facilities without centralizing every raw stream. See the [FLARE Day deployments and talks](https://developer.nvidia.com/flare-day-2025). |
| Government and scientific computing | Collaboration across facilities where sovereignty, classification, scale, or data movement constrains centralization. See the [industry use cases](https://nvflare.readthedocs.io/en/main/industry_use_cases.html#government-national-security). |

These references describe particular deployments and studies. The privacy, security, governance, and model-quality
properties of a new application still depend on its design, policies, data, and validation.

### Solution ecosystem

Independent software vendors (ISVs) and global systems integrators (GSIs) use NVIDIA FLARE to build federated
platforms, governed data-collaboration solutions, and customer implementations. Public technical sessions include
Rhino Health's federated computing platform, an Apheris and BioNeMo data-collaboration implementation, and Deloitte's
FedRAG implementation. Explore these and other ecosystem work in the [NVIDIA FLARE webinar
series](https://nvidia.github.io/NVFlare/webinars/) and [FLARE Day](https://nvidia.github.io/NVFlare/flareDay/).

## Project and community

- Read [What's New](https://nvflare.readthedocs.io/en/main/whats_new.html) and the
  [talks and publications](https://nvflare.readthedocs.io/en/main/publications_and_talks.html).
- Ask questions and share ideas in [GitHub Discussions](https://github.com/NVIDIA/NVFlare/discussions).
- Review the [contributing guide](./CONTRIBUTING.md) and open
  [good first issues](https://github.com/NVIDIA/NVFlare/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- Cite the [NVIDIA FLARE paper](https://arxiv.org/abs/2210.13291) when the project supports your work.

NVIDIA FLARE is released under the [Apache 2.0 license](./LICENSE).
