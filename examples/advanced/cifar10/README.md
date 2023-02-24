# Federated Learning with CIFAR-10

## Setup
### Install requirements

Install required packages for training
```
pip install --upgrade pip
pip install -r ./requirements.txt
```

> **_NOTE:_**  We recommend either using a containerized deployment or virtual environment, 
> please refer to [getting started](https://nvflare.readthedocs.io/en/latest/getting_started.html).

Set `PYTHONPATH` to include custom files of this example:
```
export PYTHONPATH=${PWD}
```

### Download the CIFAR-10 dataset 
To speed up the following experiments, first download the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:
```
python3 ./pt/utils/cifar10_download_data.py
```

> **_NOTE:_** This is important for running multitask experiments or running multiple clients on the same machine.
> Otherwise, each job will try to download the dataset to the same location which might cause a file corruption.

## Examples

### [Simulated Federated Learning with CIFAR-10](./cifar10-sim/README.md)
This example includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629), 
[FedProx](https://arxiv.org/abs/1812.06127), [FedOpt](https://arxiv.org/abs/2003.00295), 
and [SCAFFOLD](https://arxiv.org/abs/1910.06378) algorithms using NVFlare's 
[FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/fl_simulator.html).

### [Real-world Federated Learning with CIFAR-10](./cifar10-real-world/README.md)
Real-world FL deployment requires secure provisioning and the admin API to submit jobs. 
This example runs you through the process and includes instructions on running 
[FedAvg](https://arxiv.org/abs/1602.05629) with streaming of TensorBoard metrics to the server during training 
and [homomorphic encryption](https://developer.nvidia.com/blog/federated-learning-with-homomorphic-encryption/)
for secure server-side aggregation.
