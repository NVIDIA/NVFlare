
# Training traditional ML classifiers with HIGGS dataset

The [HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs) contains 11 million instances, each with 28 attributes, for binary classification to predict whether an event corresponds to the decayment of a Higgs boson or not. Follow the [prepare_data.ipynb](prepare_data.ipynb) notebook to download the HIGGS dataset and prepare the data splits.
(Please note that the [UCI's website](https://archive.ics.uci.edu/dataset/280/higgs) may experience occasional downtime)

The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. 
The data has been produced using Monte Carlo simulations. The first 21 features are kinematic properties measured by the particle detectors in the accelerator. The last 7 features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes.

Key Concepts:
* How to use the NVFlare Client APIs to convert the traditional machine learning code to federated learning code. Most of them contains local training scripts as baselines for comparison.
* How different machine learning methods can be applied to the same problem. Different behaviors and accuracies can be observed, as a reference for choosing the right method for the problem.
* How federated learning impacts different machine learning methods. Some methods are more sensitive to the federated learning process, and some are less.

In the following examples, we will demonstrate traditional machine learning techniques with tabular data for federated learning:

* [stats](stats) - Federated statistics for tabular histogram calculation.
* [sklearn-linear](sklearn-linear) - Federated linear model (logistic regression on binary classification) learning.
* [sklearn-svm](sklearn-svm) - Federated SVM model learning.
* [sklearn-kmeans](sklearn-kmeans) - Federated k-Means clustering.
* [xgboost](xgboost) - Federated horizontal xgboost learning with bagging collaboration.


