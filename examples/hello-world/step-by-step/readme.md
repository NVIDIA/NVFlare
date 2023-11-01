#  Step-by-Step Examples

When given a machine learning problem, we probably wonder, where do we start to formulate the federated learning problem. 

* What does the data look like?
* How do we compare global statistics with the site's local data statistics? 
* How to formulate the federated algorithms
  * https://developer.download.nvidia.com/healthcare/clara/docs/federated_traditional_machine_learning_algorithms.pdf
* Given the formulation, how to convert the existing machine learning or deep learning code to Federated learning code.
  * [ML to FL examples](https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-world/ml-to-fl/README.md)
* For different types of federated learning workflows: Scatter and Gather, Cyclic Weight Transfer, Swarming learning, 
Vertical learning, ..., what do we need to change ?
* Further how can apply the experiment log, so all sites' metrics and global metrics can be viewed 
* in experiment tracking tools such as Weights & Biases, MLFLow, or simply Tensorboard

In this "step-by-step" examples, we will dive these questions in two series of examples: 

## Imaging classification problem using CIFAR10 data

The CIFAR10 dataset has the following classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.
 
![image](cifar10/data/cifar10.png)

We will use the [pytorch](https://pytorch.org/) deep learning framework to illustrate how to formulate and convert the deep learning training
program to a federated learning training program. The example will include:

* Federated Histogram analysis with Federated Statistics
* Scatter and Gather (SAG) workflow with NVFLARE Client APIs 
* Cyclic Weight Transfer workflow with NVFLARE Client APIs
* Swarm Learning Workflow with NVFLARE Client APIs
* SAG with NVFLARE model learner APIs
* SAG with NVFLARE Executor APIs


## Tabular HIGGs dataset  

With the HIGGs Dataset, we like to demonstrate traditional machine learning techniques in federated learning.
These include:

* Federated Statistics for tabular data
* Federated Linear and Logistics Regression
* Federated Kmeans
* Federated SVM with non-learner kernel
* Federated (Horizontal) XGBoost
