# Federated Learning with CIFAR-10 using PyTorch

Please make sure you set up virtual environment and follows [example root readme](../../README.md)

### [Simulated Federated Learning with CIFAR-10](./cifar10-sim/README.md)
This example includes instructions on running job recipes for [FedAvg](https://arxiv.org/abs/1602.05629), 
[FedProx](https://arxiv.org/abs/1812.06127), [FedOpt](https://arxiv.org/abs/2003.00295), 
and [SCAFFOLD](https://arxiv.org/abs/1910.06378) algorithms with NVFlare's 
[FL simulator](https://nvflare.readthedocs.io/en/main/user_guide/nvflare_cli/fl_simulator.html) using the **SimEnv**.

### [Real-world Federated Learning with CIFAR-10](./cifar10-real-world/README.md)
Real-world FL deployment requires secure [provisioning](https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_system.html) and the [admin API](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/flare_api.html#flare-api) to manage jobs via **ProdEnv**. 
This example runs you through the process and includes instructions on running job recipes for
[FedAvg](https://arxiv.org/abs/1602.05629) with [experiment tracking](https://nvflare.readthedocs.io/en/main/programming_guide/experiment_tracking.html) (supporting MLFlow, TensorBoard, or Weights & Biases) 
and [homomorphic encryption](https://developer.nvidia.com/blog/federated-learning-with-homomorphic-encryption/) for secure server-side aggregation.