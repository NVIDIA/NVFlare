# Getting Started with NVFlare
NVFlare is an open-source framework that allows researchers and data scientists to seamlessly move their 
machine learning and deep learning workflows into a federated paradigm.

### Basic Concepts
At the heart of NVFlare lies the concept of collaboration through "tasks." An FL controller assigns tasks 
(e.g., training on local data) to one or more FL clients, processes returned results (e.g., model weight updates), 
and may assign additional tasks based on these results and other factors (e.g., a pre-configured number of training rounds). 
The clients run executors which can listen for tasks and perform the necessary computations locally, such as model training. 
This task-based interaction repeats until the experiment’s objectives are met.

We can also add data filters (for example, for [homomorphic encryption](https://www.usenix.org/conference/atc20/presentation/zhang-chengliang)
or [differential privacy filters](https://arxiv.org/abs/1910.00962)) to the task data
or results received or produced by the server or clients.

![NVIDIA FLARE Overview](../../docs/resources/controller_executor_no_filter.png)


### Examples
We provide several examples to quickly get you started using NVFlare's Job API. 
Each example folder includes basic job configurations for running different FL algorithms. 
Starting from [FedAvg](https://arxiv.org/abs/1602.05629), to more advanced ones, 
such as [FedOpt](https://arxiv.org/abs/2003.00295), or [SCAFFOLD](https://arxiv.org/abs/1910.06378).

### 1. [PyTorch Examples](./pt/README.md)
### 2. [Tensorflow Examples](./tf/README.md)
### 3. [Scikit-Learn Examples](./sklearn/README.md)

Once you finished above examples, you also read about [getting started documentation](https://nvflare.readthedocs.io/en/main/getting_started.html), 
look at the ["hello-world"](../hello-world) examples or checkout more examples at tutorial catelog https://nvidia.github.io/NVFlare/.


