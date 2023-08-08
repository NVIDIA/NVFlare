**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment

[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) is a domain-agnostic, open-source, extensible SDK that 
allows researchers and data scientists to adapt existing ML/DL workflows(PyTorch, TensorFlow, Scikit-learn, XGBoost etc.) 
to a federated paradigm. It enables platform developers to build a secure, privacy-preserving offering 
for a distributed multi-party collaboration. 

**NVIDIA FLARE** is built on a componentized architecture that allows you to take federated learning workloads 
from research and simulation to real-world production deployment. Key components include:

* Support both deep learning and traditional machine algorithms
* Support horizontal and vertical federated learning
* Built-in FL algorithms (e.g., FedAvg, FedProx, FedOpt, Scaffold, Ditto )
* Support multiple training workflows (e.g., scatter & gather, cyclic) and validation workflows (global model evaluation, cross-site validation)
* Support both data analytics (federated statistics) and machine learning lifecycle management
* Privacy preservation with differential privacy, homomorphic encryption
* Security enforcement through federated authorization and privacy policy 
* Easily customizable and extensible
* Deployment on cloud and on premise 
* Simulator for rapid development and prototyping
* Dashboard UI for simplified project management and deployment
* Built-in support for system resiliency and fault tolerance

## Installation
To install the [current release](https://pypi.org/project/nvflare/), you can simply run:
```
$ python3 -m pip install nvflare
```
## Getting started

You can quickly get started using the [FL simulator](https://nvflare.readthedocs.io/en/main/getting_started.html#the-fl-simulator).

A detailed [getting started](https://nvflare.readthedocs.io/en/main/getting_started.html) guide is available in the [documentation](https://nvflare.readthedocs.io/en/main/index.html).
 
Examples and notebook tutorials are located [here](./examples).

## Related talks and publications

For a list of talks, blogs, and publications related to NVIDIA FLARE, see [here](https://nvflare.readthedocs.io/en/main/publications_and_talks.html).

## License

NVIDIA FLARE has Apache 2.0 license, as found in [LICENSE](https://github.com/NVIDIA/NVFlare/blob/main/LICENSE) file. 
