# Experiment Tracking Overview

Please make sure you set up a virtual environment and follow the installation steps on the [example root readme](../../README.md).

This folder contains examples for [experiment tracking](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to
train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and
[PyTorch](https://pytorch.org/) as the deep learning training framework.

## Tensorboard

The `tensorboard` folder contains the [TensorBoard Streaming](./tensorboard/README.md) example
showcasing the [TensorBoard streaming](https://nvflare.readthedocs.io/en/main/examples/tensorboard_streaming.html)
capability of streaming from the clients to the server.

## MLflow

The `mlflow` folder contains [examples](./mlflow/README.md) showcasing the usage of the capability to stream from
the clients to the server to write to an MLflow tracking server. The `hello-pt-mlflow` job demonstrates how a
`PTLearner` with logging using MLflow syntax can be used with `MLflowWriter`, and the `hello-pt-tb-mlflow` job
shows how `PTLearner` for Tensorboard can be used with `TBWriter` and still stream to `MLflowReceiver`.

## Weights and Biases

The `wandb` folder contains the [Hello PyTorch with Weights and Biases](./wandb/README.md) example
showing how Weights and Biases can be used as a back end for the `WandBReceiver` to write to. This
example uses a `PTLearner` with Weights and Biases syntax.
