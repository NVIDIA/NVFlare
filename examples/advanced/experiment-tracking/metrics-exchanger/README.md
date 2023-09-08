# Hello PyTorch Examples using MetricsExchanger

Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to train an image classifier
using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/)
as the deep learning training framework.

These examples are the same as the respective examples for experiment tracking with each system, but swap out
the `LogWriter` component with the `LogWriterForMetricsExchanger` version to work with `MetricsRetriever`.

> **_NOTE:_** This example uses the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and will load its data within the trainer code.
