# Deep Learning to Federated Learning transition with NVFlare

Converting Deep Learning (DL) to Federated Learning (FL) is not easy, as it involves:

1. Algorithms formulation, how to formulate a DL to FL algorithm and what information needs to be passed between Client and Server

2. Convert existing standalone, centralized DL code to FL code.

3. Configure the workflow to use the newly changed code.

In this example, we assume algorithm formulation is fixed (FedAvg).
We are showing how to quickly convert the centralized DL to FL.
We will demonstrate different techniques depending on the existing code structure and preferences.

To configure the workflow, one can reference the config we have here and the documentation.

In this directory, we are providing job configurations to showcase how to utilize 
`LauncherExecutor`, `Launcher`, and several NVFlare interfaces to simplify the transition from your DL code to FL with NVFlare.


We will demonstrate how to transform an existing DL code into an FL application:

  1. From PyTorch / PyTorch + Lightning: [pt](./pt/README.md)
  2. From TensorFlow: [tf](./tf/README.md)


Note: Avoid install tensorflow and pytorch on the same virtual environment due to library conflicts.
