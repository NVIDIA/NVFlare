# ML to FL transition with NVFlare

Converting Machine Learning or Deep Learning to FL is not easy, as it involves:

1. Algorithms formulation, how to formulate an ML/DL to FL algorithm and what information needs to be pass between Client and Server

2. Convert existing standalone, centralized ML/DL code to FL code.

3. Configure the workflow to use the newly changed code.

In this example, we assume #1 algorithm formulation is fixed (FedAvg).
We are showing #2, that is how to quickly convert the centralized DL to FL.
We will demonstrate different techniques depending on the existing code structure and preferences.

For #3 one can reference to the config we have here and the documentation.

In this directory, we are providing job configurations to showcase how to utilize 
`LauncherExecutor`, `Launcher` and several NVFlare interfaces to simplify the
transition from your ML code to FL with NVFlare.


## Examples

- [client_api1](./jobs/client_api1/): Re-write CIFAR10 PyTorch example to federated learning job using NVFlare client API
- [client_api2](./jobs/client_api2/): Re-write CIFAR10 PyTorch example to federated learning job using NVFlare client API with model selection
- [decorator](./jobs/decorator/): Re-write CIFAR10 PyTorch example to federated learning job using NVFlare client API decorator
- [lightning_client_api](./jobs/lightning_client_api/): Re-write PyTorch Lightning auto-encoder example to federated learning job using NVFlare lightning client API
