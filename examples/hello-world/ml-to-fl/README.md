# Deep Learning to Federated Learning transition with NVFlare

Converting Deep Learning (DL) to Federated Learning (FL) is not easy, as it involves:

1. Algorithms formulation, how to formulate a DL to FL algorithm and what information needs to be passed between Client and Server

2. Convert existing standalone, centralized DL code to FL code.

3. Configure the workflow to use the newly changed code.

In these examples, we assume algorithm formulation is fixed (FedAvg).
We are showing how to quickly convert the centralized DL to FL.
We will demonstrate different techniques depending on the existing code structure and preferences.

To configure the workflow, one can reference the config we have here and the documentation.


We cover the following use cases:

  1. PyTorch and PyTorch Lightning: [pt](./pt/README.md)
  2. TensorFlow: [tf](./tf/README.md)


Note: Avoid install TensorFlow and PyTorch on the same virtual environment due to library conflicts.
