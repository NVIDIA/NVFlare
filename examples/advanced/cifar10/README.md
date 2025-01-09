# Federated Learning with CIFAR-10

Please make sure you set up virtual environment and follows [example root readme](../../README.md)

### [Simulated Federated Learning with CIFAR-10](./cifar10-sim/README.md)
This example includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629), 
[FedProx](https://arxiv.org/abs/1812.06127), [FedOpt](https://arxiv.org/abs/2003.00295), 
and [SCAFFOLD](https://arxiv.org/abs/1910.06378) algorithms using NVFlare's 
[FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/nvflare_cli/fl_simulator.html).

### [Real-world Federated Learning with CIFAR-10](./cifar10-real-world/README.md)
Real-world FL deployment requires secure provisioning and the admin API to submit jobs. 
This example runs you through the process and includes instructions on running 
[FedAvg](https://arxiv.org/abs/1602.05629) with streaming of TensorBoard metrics to the server during training 
and [homomorphic encryption](https://developer.nvidia.com/blog/federated-learning-with-homomorphic-encryption/)
for secure server-side aggregation.
