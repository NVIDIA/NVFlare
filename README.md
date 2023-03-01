**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment

[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) is a domain-agnostic, open-source, extensible SDK that 
allows researchers and data scientists to adapt existing ML/DL workflows (PyTorch, TensorFlow, Scikit-learn, XGBoost) to a federated paradigm; and enables platform developers to build a 
secure, privacy preserving offering for a distributed multi-party collaboration. 

NVIDIA FLARE is built on a componentized architecture that gives you the flexibility to take federated learning workloads 
from research and simulation to real-world production deployment. Some of the key components include:

* Simulator for rapid development and prototyping
* NVFLARE Dashboard UI for simplified project management and deployment  
* Reference FL algorithms (e.g., FedAvg, FedProx) and workflows (e.g., Scatter and Gather, Cyclic)
* Privacy preservation with differential privacy, homomorphic encryption, and more
* Management tools for secure provisioning and deployment, orchestration, and management
* Specification-based API for extensibility

## Installation
To install the [current release](https://pypi.org/project/nvflare/), you can simply run:
```
$ python3 -m pip install nvflare
```
Please refer to the installation guide (TODO) for other installation options.
> TODO: do we need an installation guide similar to Monai

## Quick Start

Clone NVFLARE repo to get examples, switch main branch (latest stable branch)

```
$ git clone https://github.com/NVIDIA/NVFlare.git
$ cd NVFlare
$ git switch main
```

#### **Quick Start with Simulator**

```
nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 examples/hello-world/hello-numpy-sag
```
Now you can watch the simulator run two clients (n=2) with two threads (t=2) and logs are saved in the /tmp/nvflare/hello-numpy-sag workspace.

To learn more about NVFLARE and understand the concepts, details of above commands, examples, 
you can look into the following topics 

* [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html) and
 
* [Examples](https://github.com/NVIDIA/NVFlare/tree/main/examples/)

overall documentation can be found at [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html)

## Related talks and publications

For a list of talks, blogs, and publications related to NVIDIA FLARE, see [here](docs/publications_and_talks.md).

## License

NVIDIA FLARE has Apache 2.0 license, as found in [LICENSE](https://github.com/NVIDIA/NVFlare/blob/dev/LICENSE) file 
 

