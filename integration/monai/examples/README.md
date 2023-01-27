# Examples of MONAI-NVFlare Integration

### [Simulated Federated Learning for 3D spleen CT segmentation](./spleen_ct_segmentation_sim/README.md)
An example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) 
to train a medical image analysis model using federated averaging ([FedAvg]([FedAvg](https://arxiv.org/abs/1602.05629))) 
and a [MONAI Bundle](https://docs.monai.io/en/latest/mb_specification.html).

This example will also guide you on using MONAI FL with FLARE to 
collect client data statistics and visualize both global and local 
intensity histograms using FLARE's [FL simulator](https://nvflare.readthedocs.io/en/main/user_guide/fl_simulator.html).

The example can be extended to use additional FL [algorithms](https://nvflare.readthedocs.io/en/main/example_applications_algorithms.html) 
available in NVIDIA FLARE.


### [Real-world Federated Learning for 3D spleen CT segmentation](./spleen_ct_segmentation_real-world/README.md)
Real-world FL deployment requires secure provisioning and an admin API to submit jobs. 
This example runs you through the process and includes instructions on running [FedAvg](https://arxiv.org/abs/1602.05629)
with streaming of TensorBoard metrics to the server during training and 
[homomorphic encryption](https://developer.nvidia.com/blog/federated-learning-with-homomorphic-encryption/) for secure server-side aggregation.

In this example, we use an already prepared [provisioning](https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_system.html)
file (*project.yml*) to run experiments on a single machine. 
For [real-world deployment](https://nvflare.readthedocs.io/en/main/real_world_fl.html), 
additional considerations must be taken into account.
Please see the [real-world FL docs](https://nvflare.readthedocs.io/en/main/real_world_fl.html) 
for further details on using FL in a real-world deployment.
