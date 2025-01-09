#  Step-by-Step Examples

To run the notebooks in each example, please make sure you first set up a virtual environment and install "./requirements.txt" and JupyterLab following the [example root readme](../README.md).

* [cifar10](cifar10) - Multi-class classification with image data using CIFAR10 dataset
* [higgs](higgs) - Binary classification with tabular data using HIGGS dataset

These step-by-step example series are aimed to help users quickly get started and learn about FLARE.
For consistency, each example in the series uses the same dataset- CIFAR10 for image data and the HIGGS dataset for tabular data.
The examples will build upon previous ones to showcase different features, workflows, or APIs, allowing users to gain a comprehensive understanding of FLARE functionalities (Note: each example is self-contained, so going through them in order is not required, but recommended). See the README in each directory for more details about each series.

## Common Questions

Here are some common questions we aim to cover in these examples series when formulating a federated learning problem:

* What does the data look like?
* How do we compare global statistics with the site's local data statistics? 
* How to formulate the [federated algorithms](https://developer.download.nvidia.com/healthcare/clara/docs/federated_traditional_machine_learning_algorithms.pdf)?
* How do we convert the existing machine learning or deep learning code to federated learning code? [ML to FL examples](https://github.com/NVIDIA/NVFlare/blob/main/examples/hello-world/ml-to-fl/README.md)
* How do we use different types of federated learning workflows (e.g. Scatter and Gather, Cyclic Weight Transfer, Swarming learning,
Vertical learning) and what do we need to change?
* How can we capture the experiment log, so all sites' metrics and global metrics can be viewed in experiment tracking tools such as Weights & Biases MLfLow, or Tensorboard
