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

## Multi-class classification with image data using CIFAR10 dataset

The CIFAR10 dataset has the following 10 classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
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
* SAG with NVFLARE Client APIs + MLflow


## Binary classification with tabular data using HIGGS dataset  

### HIGGS Dataset

[HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs) contains 11 million instances, each with 28 attributes, for binary classification to predict whether an event corresponds to the decayment of a Higgs boson or not.

The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. 
The data has been produced using Monte Carlo simulations. The first 21 features are kinematic properties measured by the particle detectors in the accelerator. The last 7 features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes.

Please note that the [UCI's website](https://archive.ics.uci.edu/dataset/280/higgs) may experience occasional downtime.

With the HIGGs Dataset, we like to demonstrate traditional machine learning techniques in federated learning.
These include:

* Federated Statistics for tabular data
* Federated Logistic Regression
* Federated Kmeans
* Federated SVM
* Federated Horizontal XGBoost

These examples demostrate:
* How to use the NVFlare Client APIs to convert the traditional machine learning code to federated learning code. Most of them contains local training scripts as baselines for comparison.
* How different machine learning methods can be applied to the same problem. Different behaviors and accuracies can be observed, as a reference for choosing the right method for the problem.
* How federated learning impacts different machine learning methods. Some methods are more sensitive to the federated learning process, and some are less.
