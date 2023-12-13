[![Blossom-CI](https://github.com/NVIDIA/nvflare/workflows/Blossom-CI/badge.svg?branch=main)](https://github.com/NVIDIA/nvflare/actions)
[![documentation](https://readthedocs.org/projects/nvflare/badge/?version=main)](https://nvflare.readthedocs.io/en/main/?badge=main)
[![license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](./LICENSE)
[![pypi](https://badge.fury.io/py/nvflare.svg)](https://badge.fury.io/py/nvflare)
[![pyversion](https://img.shields.io/pypi/pyversions/nvflare.svg)](https://badge.fury.io/py/nvflare)
[![downloads](https://static.pepy.tech/badge/nvflare)](https://pepy.tech/project/nvflare)

# NVIDIA FLARE

[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) (**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment)
is a domain-agnostic, open-source, extensible SDK that allows researchers and data scientists to adapt existing ML/DL workflows to a federated paradigm.
It enables platform developers to build a secure, privacy-preserving offering for a distributed multi-party collaboration.

## Features
FLARE is built on a componentized architecture that allows you to take federated learning workloads
from research and simulation to real-world production deployment.

Application Features
* Support both deep learning and traditional machine learning algorithms (eg. PyTorch, TensorFlow, Scikit-learn, XGBoost etc.)
* Support horizontal and vertical federated learning
* Built-in Federated Learning algorithms (e.g., FedAvg, FedProx, FedOpt, Scaffold, Ditto, etc.)
* Support multiple server and client-controlled training workflows (e.g., scatter & gather, cyclic) and validation workflows (global model evaluation, cross-site validation)
* Support both data analytics (federated statistics) and machine learning lifecycle management
* Privacy preservation with differential privacy, homomorphic encryption, private set intersection (PSI)

From Simulation to Real-World
* FLARE Client API to transition seamlessly from ML/DL to FL with minimal code changes
* Simulator and POC mode for rapid development and prototyping
* Fully customizable and extensible components with modular design
* Deployment on cloud and on-premise
* Dashboard for project management and deployment
* Security enforcement through federated authorization and privacy policy
* Built-in support for system resiliency and fault tolerance

> _Take a look at [NVIDIA FLARE Overview](https://nvflare.readthedocs.io/en/main/flare_overview.html) for a complete overview, and [What's New](https://nvflare.readthedocs.io/en/main/whats_new.html) for the lastest changes._

## Installation
To install the [current release](https://pypi.org/project/nvflare/):
```
$ python3 -m pip install nvflare
```
## Getting Started

You can quickly get started using the [FL simulator](https://nvflare.readthedocs.io/en/main/getting_started.html#the-fl-simulator).
A detailed [getting started](https://nvflare.readthedocs.io/en/main/getting_started.html) guide is available in the [documentation](https://nvflare.readthedocs.io/en/main/index.html).
 
Examples and notebook tutorials are located at [NVFlare/examples](./examples).

## Community

We welcome community contributions! Please refer to the [contributing guidelines](https://github.com/NVIDIA/NVFlare/blob/main/CONTRIBUTING.md) for more details.

Ask and answer questions, share ideas, and engage with other community members at [NVFlare Discussions](https://github.com/NVIDIA/NVFlare/discussions).

## Related Talks and Publications

Take a look at our growing list of [talks, blogs, and publications](https://nvflare.readthedocs.io/en/main/publications_and_talks.html) related to NVIDIA FLARE.

## License

NVIDIA FLARE is released under an [Apache 2.0 license](https://github.com/NVIDIA/NVFlare/blob/main/LICENSE).
