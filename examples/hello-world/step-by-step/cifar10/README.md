
# Training an image classifier with CIFAR10 dataset
 
We will use the original [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
example from pytorch as the code base.  

The CIFAR10 dataset has the following classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

![image](data/cifar10.png)

In the following examples, we will show various Federated Learning workflows and features:

* [stats](stats) - Federated statistics image intensity histogram calculation.
* [sag](sag) - FedAvg with Client API
* [sag_deploy_map](sag_deploy_map) - FedAvg with site-specific configs.
* [sag_executor](sag_executor) - FedAvg with Executor API
* [sag_mlflow](sag_mlflow) - FedAvg with MLflow experiment tracking logs.
* [sag_he](sag_he) - FedAvg with homomorphic encryption using POC -he mode.
* [cse](cse) - Cross-site evaluation with server-side controller.
* [cyclic](cyclic) - Cyclic Weight Transfer (cyclic) workflow with server-side controller.
* [cyclic_ccwf](cyclic_ccwf) - Client-controlled cyclic workflow with client-side controller.
* [swarm](swarm) - Swarm learning and client-controlled cross-site evaluation.
