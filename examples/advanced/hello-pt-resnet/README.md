# Hello PyTorch ResNet

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629))
and [PyTorch](https://pytorch.org/) as the deep learning training framework. Comparing with the Hello PyTorch example, it uses the torchvision ResNet, 
instead of the SimpleNetwork.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the client train code.

The Job API only supports the object instance created directly out of the Python Class. It does not support 
the object instance created through using the Python function. Comparing with the hello-pt example, 
if we replace the SimpleNetwork() object with the resnet18(num_classes=10), 
the "resnet18(num_classes=10)" creates an torchvision "ResNet" object instance out of the "resnet18" function. 
As shown in the [torchvision reset](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L684-L705), 
the resnet18 is a Python function, which creates and returns a ResNet object. The job API can 
only use the "ResNet" object instance for generating the job config. It can not detect the object creating function logic in the "resnet18".

This example demonstrates how to wrap up the resnet18 Python function into a Resnet18 Python class. Then uses the Resnet18(num_classes=10)
object instance in the job API. After replacing the SimpleNetwork() with the Resnet18(num_classes=10),
you can follow the exact same steps in the hello-pt example to run the fedavg_script_runner_pt.py.
