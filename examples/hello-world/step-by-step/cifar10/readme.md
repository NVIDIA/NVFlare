
# Training a image classifier with CIFAR10 data
 
We will use the original [Training a Classifer](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) example
in pytorch as the code base.  

The CIFAR10 dataset has the following classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

![image](data/cifar10.png)

In the follow examples, we will show various Federated Learning workflows.

* [Image intensity histogram caculation](stats)
* Scatter and Gather (SAG) workflow with NVFLARE Client APIs
* Cyclic Weight Transfer workflow with NVFLARE Client APIs
* Swarm Learning Workflow with NVFLARE Client APIs
* SAG with NVFLARE model learner APIs
* SAG with NVFLARE Executor APIs
