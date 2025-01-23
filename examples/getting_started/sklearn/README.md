# Getting Started with NVFlare (scikit-learn)
[![Scikit-Learn Logo](https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg)](https://scikit-learn.org/)

We provide examples to quickly get you started using NVFlare's Job API. 
All examples in this folder are based on using [scikit-learn](https://scikit-learn.org/), a popular library for general machine learning with Python.

## Setup environment
First, install NVFlare and its dependencies:
```commandline
pip install -r requirements.txt
```

## Examples
You can also run any of the below scripts directly using
```commandline
python "script_name.py"
```
### 1. [Federated K-Means Clustering](./kmeans_script_runner_higgs.py)
Implementation of [K-Means](https://arxiv.org/abs/1602.05629). For more details see this [example](../../advanced/sklearn-kmeans/README.md).
```commandline
python kmeans_script_runner_higgs.py
```

> [!NOTE]
> More examples can be found at https://nvidia.github.io/NVFlare.
