# Experiment Tracking Overview


This section demonstrates how NVIDIA FLARE supports flexible experiment tracking through various backends such as MLflow, TensorBoard, and Weights & Biases. It also enables both centralized and decentralized metric collection.

## Key Highlights

1. **Centralized Streaming to Server Receiver**  
   FLARE can stream all client training metrics to a server-side receiver, allowing a consolidated view of all clients' training progress.

2. **Pluggable Metrics Receivers**  
   FLARE allows plugging in different metrics receivers, independent of whether they are used on the server side or client side. This enables streaming metrics to various observability frameworks such as:
   - MLflow
   - TensorBoard
   - Weights & Biases

   **Benefit**: This makes it easy to switch between different experiment tracking frameworks without modifying the training code that logs metrics—only the receiver configuration needs to change.


3. **Site-Specific Metric Streaming**  
   FLARE also supports streaming metrics from each client to site-specific receivers. This enables local tracking at each site. Again with configuration changes only


## Configuration Flexibility

FLARE allows seamless switching between centralized and decentralized experiment tracking by modifying only the configuration:

- The training code remains unchanged.
- You can control:
  - Where the metrics are sent (server or site-local).
  - Which experiment tracking framework is used.

This flexible design enables easy integration with different observability platforms, tailored to your deployment needs.

--- 

## Examples

Depends your preference of the experiment tracking framework, we offers different API for each framework, but you only need to write once even you need to switch tracking framework. 

---

Please make sure you set up a virtual environment and follow the installation steps on the [example root readme](../../README.md).

This folder contains examples for [experiment tracking](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html)
using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) to
train an image classifier using federated averaging ([FedAvg](https://arxiv.org/abs/1602.05629)) and
[PyTorch](https://pytorch.org/) as the deep learning training framework.

### Tensorboard

The `tensorboard` folder contains the [TensorBoard Streaming](./tensorboard/README.md) example
showcasing the [TensorBoard streaming](https://nvflare.readthedocs.io/en/main/examples/tensorboard_streaming.html)
capability of streaming from the clients to the server.

### MLflow

The `mlflow` folder contains [examples](./mlflow/README.md) showcasing the usage of the capability to stream from
the clients to the server to write to an MLflow tracking server, as well as concurrent streaming to Tensorboard.
It also illustrate how to stream the metrics to site-specific mlflow server.

### Weights and Biases

The `wandb` folder contains the [Hello PyTorch with Weights and Biases](./wandb/README.md) example
showing how Weights and Biases can be used as a back end for the `WandBReceiver` to write to.  

