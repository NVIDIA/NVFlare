
# Training traditional ML classifiers with HIGGS data

[HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs) contains 11 million instances, each with 28 attributes, for binary classification to predict whether an event corresponds to the decayment of a Higgs boson or not.

The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. 
The data has been produced using Monte Carlo simulations. The first 21 features are kinematic properties measured by the particle detectors in the accelerator. The last 7 features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes.

Please note that the [UCI's website](https://archive.ics.uci.edu/dataset/280/higgs) may experience occasional downtime.

With the HIGGs Dataset, in the following examples, we like to demonstrate traditional machine learning techniques in federated learning.
These include:

* Federated Statistics for tabular data
* Federated Logistic Regression
* Federated Kmeans
* Federated SVM
* Federated Horizontal XGBoost

