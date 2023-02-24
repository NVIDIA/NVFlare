**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment

[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) (NVIDIA Federated Learning Application Runtime Environment)
is a domain-agnostic, open-source, extensible SDK that allows researchers and data scientists to adaptexisting 
ML/DL workflows (PyTorch, RAPIDS, Nemo, TensorFlow) to a federated paradigm; and enables platform developers to build a 
secure, privacy preserving offering for a distributed multi-party collaboration. 
Our mission: _Bring privacy preserved compute and machine learning to data in a federated setting, keep it simple and production ready_

NVIDIA FLARE is built on a componentized architecture that gives you the flexibility to take federated learning workloads 
from research and simulation to real-world production deployment. Some of the key components of this architecture include:

* [FL Simulator](https://nvflare.readthedocs.io/en/main/user_guide/fl_simulator.html) for rapid development and prototyping

Different from other Federated learning framework, the FL simulator not only can support single thread debugging and multi-thread simulation,
the code used in simulator can be directly deployed to real-world production without change. 

* [FLARE Dashboard UI](https://nvflare.readthedocs.io/en/main/user_guide/dashboard_ui.html) for simplified project management and deployment  

Enable user to easily manager Federated Learning project and distributed start up package to collaborating organizations. 

* Reference FL algorithms (e.g., FedAvg, FedProx) and workflows (e.g., Scatter and Gather, Cyclic)

NVFLARE has various built-in workflows that supports Federated Learning (horizontal, vertical, traditional machine learning) 
and Federated Statistics. You can find all of them in [examples](https://github.com/NVIDIA/NVFlare/tree/dev/examples)

* Privacy preservation with differential privacy, homomorphic encryption, and more

 NVFLARE control data privacy in various ways: privacy filter, [privacy policy management](https://nvflare.readthedocs.io/en/main/user_guide/site_policy_management.html), 
 privacy algorithms and tools 

* Management tools for secure provisioning and deployment, orchestration, and management

NVFLARE has provided a set of tools to help manage the provision and deployment (both on cloud and on premise) system to production.  

* Specification-based API for extensibility

## Quick Start

#### Install NVFLARE
```
$ python3 -m pip install nvflare
```
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

To learn more about NVFLARE and understand the concepts, details of above commands, examples, you can look into the following topics [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html) and  [Examples](https://github.com/NVIDIA/NVFlare/tree/main/examples/)

## Related talks and publications

For a list of talks, blogs, and publications related to NVIDIA FLARE, see [here](docs/publications_and_talks.md).

## License

NVIDIA FLARE has Apache 2.0 license, as found in [LICENSE](https://github.com/NVIDIA/NVFlare/blob/dev/LICENSE) file 
 

