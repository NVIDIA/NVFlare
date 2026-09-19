<img src="https://raw.githubusercontent.com/NVIDIA/NVFlare/main/docs/resources/nvidia_eye.wwPt122j.png" alt="NVIDIA Logo" width="200">

# NVIDIA FLARE

Build federated computing applications that collaborate across organizational and geographic boundaries while each
site keeps control of its data and local execution.

[Website](https://nvidia.github.io/NVFlare) |
[Documentation](https://nvflare.readthedocs.io/en/main/) |
[Quick Start](https://nvflare.readthedocs.io/en/main/quickstart.html) |
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

## Federated computing with NVIDIA FLARE

**Federated computing** brings computation to data distributed across multiple parties or locations. Raw records stay
at the site where they are held instead of being collected in a central location. Participating sites run approved
tasks locally and share only the outputs permitted by the collaboration, such as aggregate statistics, model updates,
or evaluation metrics.

**Federated learning** is a form of federated computing in which sites collaboratively train or evaluate models on
their local data. Training and evaluation happen where the data resides; the underlying training examples are not
exchanged between sites or sent to a central server. This can enable collaboration where privacy, regulation, data
sovereignty, intellectual property, data ownership, or the cost of moving data makes central collection impractical.

**[NVIDIA FLARE](https://nvidia.github.io/NVFlare)** (**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment) is an open-source,
extensible Python SDK for building these applications. NVIDIA FLARE coordinates approved code, tasks, and model state
across participating sites while each site retains its raw dataset and controls what results may leave. Data scientists
can adapt existing Python workflows, algorithm developers can implement new collaboration patterns, and platform teams
can operate the same applications from local simulation through provisioned multi-site deployment.

> **Data stays at its source.** NVIDIA FLARE moves approved computation to participating sites. Raw datasets do not
> move to the server or to other sites; only outputs allowed by the collaboration and each site's policies leave the
> site.

## What you can do with NVIDIA FLARE

| Area | Capabilities |
|---|---|
| Federated applications | Training and fine-tuning, evaluation and cross-site validation, analytics and statistics, site-local data processing, and custom multi-site computation |
| Framework integrations | PyTorch, TensorFlow, JAX, scikit-learn, XGBoost, Hugging Face, NeMo, Flower, and other Python workloads |
| Algorithms and workflows | FedAvg, FedProx, FedOpt, SCAFFOLD, Ditto, cyclic and swarm learning, horizontal and vertical FL, and custom server- or client-controlled workflows |
| Privacy, security, and governance | Site-controlled authorization, audit logging, secure provisioning, differential privacy, homomorphic encryption, private set intersection, and confidential computing options |
| Runtime and operations | Local simulation, proof-of-concept environments, production provisioning, cloud and on-premises deployment, resiliency, monitoring, and experiment tracking |

FLARE's component and event architecture lets applications replace or extend controllers, aggregators, executors,
filters, persistence, communication, and deployment behavior without rewriting the entire system.

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

## How NVIDIA FLARE works

```text
                         Federated job
                workflow · tasks · policies
                              │
                    NVIDIA FLARE runtime
             orchestration · transport · aggregation
                  ┌───────────┼───────────┐
                  │           │           │
               Site A      Site B      Site C
             local data   local data   local data
             local task   local task   local task
                  │           │           │
                  └── permitted results ──┘
```

A **job** packages the application logic and configuration. A **workflow** coordinates tasks across participating
sites. Each site executes its task against local resources, and policies determine what can run and what may leave the
site. Depending on the application, the shared result can be an aggregate model, evaluation, statistic, intersection,
or another collaboration artifact.

## From local development to production

The application stays consistent while its execution environment changes:

| Stage | Purpose |
|---|---|
| Simulator (`SimEnv`) | Run server and client logic on one system for fast development and validation. |
| Proof of Concept (`PocEnv`) | Simulate a production deployment on one local host using separate server and client processes and locally generated startup kits. |
| Production (`ProdEnv`) | Run across provisioned sites with production identities, authorization, networking, policies, and operations. |

See the [run modes](https://nvflare.readthedocs.io/en/main/run_mode.html) and
[deployment guide](https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/deployment/overview.html) for the
differences and operational requirements.

## Try NVIDIA FLARE locally

This four-command path installs the PyTorch integration, retrieves the Hello PyTorch example that matches the installed
NVFLARE revision, and runs a two-client federation with focused progress output:

```bash
python -m pip install "nvflare[PT]"
nvflare examples get hello-pt
cd hello-pt
python job.py --log_config progress
```

The default example runs on CPU, downloads no dataset, and uses deterministic site-local synthetic data. Continue with
the [Quick Start](https://nvflare.readthedocs.io/en/main/quickstart.html) to understand the run, inspect its artifacts,
adapt training code, and choose a deployment mode. The
[Hello PyTorch README](./examples/hello-world/hello-pt/README.md) remains the authoritative reference for dependencies,
options, artifacts, and troubleshooting.

## Choose your path

Start with the guide for the concept and supported workflow, then use a maintained example as the runnable reference.

| Goal | Read first | Then run or adapt |
|---|---|---|
| Adapt existing training code | [Quick Start](https://nvflare.readthedocs.io/en/main/quickstart.html), then [Agent Skills](https://nvflare.readthedocs.io/en/main/user_guide/agent_skills/index.html) for supported projects or [API selection](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/api_selection.html) for manual and custom integration | Try the [Agent Skills conversion examples](./examples/hello-world/agent-skills/README.md), or use the [Client API and Recipe examples](https://nvidia.github.io/NVFlare/catalog/) |
| Federated analytics and statistics | [Agent Skills](https://nvflare.readthedocs.io/en/main/user_guide/agent_skills/index.html) for supported tabular and image datasets, or the [Federated Statistics guide](https://nvflare.readthedocs.io/en/main/examples/federated_statistics_overview.html) for manual and custom workflows | Try the [tabular and image Agent Skills examples](./examples/hello-world/agent-skills/README.md), or use the [runnable statistics examples](https://nvidia.github.io/NVFlare/catalog/) |
| Research and design custom FL algorithms or workflows | [Researcher Guide](https://nvflare.readthedocs.io/en/main/user_guide/researcher_guide/index.html), then the [Collaboration API guide](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/collab_api.html) | [Algorithm and workflow examples](https://nvidia.github.io/NVFlare/catalog/) and [research implementations](./research/README.md) |
| Learn a specific NVIDIA FLARE feature interactively | [Feature Tutorials](https://nvflare.readthedocs.io/en/main/tutorials.html#feature-tutorials) | Open the focused Simulator, POC, FLARE API, CLI, Recipe, or logging notebook |
| Fine-tune large language models | [Federated LLM guide](https://nvflare.readthedocs.io/en/main/programming_guide/llm_fine_tuning.html) | [Hugging Face and NeMo examples](https://nvidia.github.io/NVFlare/catalog/) |
| Validate POC and production environments | [Deployment guide](https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/deployment/overview.html) | [Hello PyTorch environment example](./examples/advanced/hello-pt-environments/README.md) |
| Add privacy and governance controls | [Security overview](https://nvflare.readthedocs.io/en/main/system_architecture/security_overview.html) | [DP, HE, PSI, and confidential-computing examples](https://nvidia.github.io/NVFlare/catalog/) |

Use the [NVIDIA FLARE website](https://nvidia.github.io/NVFlare) to discover tutorials, examples, research, webinars,
and events. The [documentation](https://nvflare.readthedocs.io/en/main/) provides the authoritative guides. After
installing NVFLARE, `nvflare examples list` shows the curated catalog and `nvflare examples get <name>` retrieves an
example matched to that installation's source revision. Each example README then owns its exact setup and run steps.

## Project and community

- Read [What's New](https://nvflare.readthedocs.io/en/main/whats_new.html) and the
  [talks and publications](https://nvflare.readthedocs.io/en/main/publications_and_talks.html).
- Ask questions and share ideas in [GitHub Discussions](https://github.com/NVIDIA/NVFlare/discussions).
- Review the [contributing guide](./CONTRIBUTING.md) and open
  [good first issues](https://github.com/NVIDIA/NVFlare/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
- Cite the [NVIDIA FLARE paper](https://arxiv.org/abs/2210.13291) when the project supports your work.

NVIDIA FLARE is released under the [Apache 2.0 license](./LICENSE).
