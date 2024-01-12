#  Step-by-Step Examples

These step-by-step example series are aimed to help users quickly get started and learn about FLARE.
For consistency, each example in the series uses the same dataset- CIFAR10 for image data and the HIGGS dataset for tabular data.
The examples will build upon previous ones to showcase different features, workflows, or APIs, allowing users to gain a comprehensive understanding of FLARE functionalities.

Given a machine learning problem, here are some common questions we aim to cover when formulating a federated learning problem:

* What does the data look like?
* How do we compare global statistics with the site's local data statistics? 
* How to formulate the federated algorithms
  * https://developer.download.nvidia.com/healthcare/clara/docs/federated_traditional_machine_learning_algorithms.pdf
* Given the formulation, how to convert the existing machine learning or deep learning code to Federated learning code.
  * [ML to FL examples](https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-world/ml-to-fl/README.md)
* For different types of federated learning workflows: Scatter and Gather, Cyclic Weight Transfer, Swarming learning, 
Vertical learning, ... what do we need to change ?
* How can we capture the experiment log, so all sites' metrics and global metrics can be viewed in experiment tracking tools such as Weights & Biases, MLfLow, or Tensorboard

In these "step-by-step" examples, we will dive into these questions in two series of examples (See the README in each directory for more details about each series):

* [cifar10](cifar10) - Multi-class classification with image data using CIFAR10 dataset
* [higgs](higgs) - Binary classification with tabular data using HIGGS dataset


