# One-shot Vertical Federated Learning with CIFAR-10

This example includes instructions on how to run single-client [one-shot vertical federated learning](https://arxiv.org/abs/2303.16270) using the 
CIFAR-10 dataset and the [FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/fl_simulator.html).

We assume one client holds the images, and the other client holds the labels to compute losses and accuracy metrics. 
Activations and corresponding gradients are being exchanged between the clients using NVFlare.

<img src="./figs/oneshotVFL.png" alt="One-shot VFL setup" width="500"/>

For instructions of how to run CIFAR-10 in real-world deployment settings, 
see the example on ["Real-world Federated Learning with CIFAR-10"](../../cifar10/cifar10-real-world/README.md).

## 1. Setup
This examples uses [JupyterLab](https://jupyter.org).

We recommend creating a [virtual environment](../../../README.md#set-up-a-virtual-environment).

## 2. Start JupyterLab
Set `PYTHONPATH` to include custom files of this example and some reused files from the [CIFAR-10](../../examples/advanced/cifar10) examples:
```
export PYTHONPATH=${PWD}/src:${PWD}/../../examples/advanced/cifar10
```
Start Jupyter Lab
```
jupyter lab .
```
and open [cifar10_oneshot_vfl.ipynb](./cifar10_oneshot_vfl.ipynb).

## 3. Example results
An example local training curve with an overlap of 10,000 samples is shown below.
One-shot VFL only requires the client to conduct two uploading and one downloading, which reduces the communication cost significantly. This CIFAR10 example can achieve test accuracy of 79.0%, which is nearly the same with the results of vanilla [single-client VFL (split learning)](https://github.com/jeremy313/NVFlare/tree/dev/examples/advanced/vertical_federated_learning/cifar10-splitnn).

![Local training curves](./figs/oneshotVFL_results.png)
