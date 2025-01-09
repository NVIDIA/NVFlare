# Federated SVM with Scikit-learn and cuML

Please make sure you set up virtual environment and Jupyterlab follows [example root readme](../../README.md)

## Introduction to Scikit-learn, tabular data, and federated SVM
### Scikit-learn
This example shows how to use [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) on tabular data.
It uses [Scikit-learn](https://scikit-learn.org/), a widely used open-source machine learning library that supports supervised and unsupervised learning.
Follow along in this [notebook](./sklearn_svm_cancer.ipynb) for an interactive experience.
### cuML
You can also use [cuML](https://docs.rapids.ai/api/cuml/stable/) as backend instead of Scikit-learn.
Please install cuml following instructions: https://rapids.ai/start.html

### Tabular data
The data used in this example is tabular in a format that can be handled by [pandas](https://pandas.pydata.org/), such that:
- rows correspond to data samples.
- the first column represents the label. 
- the other columns cover the features.    

Each client is expected to have one local data file containing both training and validation samples. 
To load the data for each client, the following parameters are expected by local learner:
- data_file_path: (`string`) the full path to the client's data file. 
- train_start: (`int`) start row index for the training set.
- train_end: (`int`) end row index for the training set.
- valid_start: (`int`) start row index for the validation set.
- valid_end: (`int`) end row index for the validation set.

### Federated SVM
The machine learning algorithm shown in this example is [SVM for Classification (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
Under this setting, federated learning can be formulated in two steps:
- local training: each client trains a local SVM model with their own data
- global training: server collects the support vectors from all clients and 
  trains a global SVM model based on them

Unlike other iterative federated algorithms, federated SVM only involves 
these two training steps. Hence, in the server config, we have
```
"num_rounds": 2
```
The first round is the training round, performing local training and global aggregation. 
Next, the global model will be sent back to clients for the second round, 
performing model validation and local model update. 
If this number is set to a number greater than 2, the system will report an error and exit.

## Data preparation 
This example uses the breast cancer dataset available from Scikit-learn's dataset API.

First, we will load the data, format it properly by removing the header, order 
the label and feature columns, and save it to a CSV file with comma separation. 
The default path is `/tmp/nvflare/dataset/sklearn_breast_cancer.
csv`.
```commandline
bash prepare_data.sh
``` 

## Prepare clients' configs with proper data information 
For real-world FL applications, the config JSON files are expected to be 
specified by each client, according to their own local data path and splits for training and validation.

In this simulated study, to efficiently generate the config files for a 
study under a particular setting, we provide a script to automate the process. 
Note that manual copying and content modification can achieve the same.

For an experiment with `K` clients, we split one dataset into `K+1` parts in a non-overlapping fashion: `K` clients' training data and `1` common validation data. 
To simulate data imbalance among clients, we provided several options for client data splits by specifying how a client's data amount correlates with its ID number (from `1` to `K`):
- Uniform
- Linear
- Square
- Exponential

These options can be used to simulate no data imbalance (`uniform`), 
moderate data imbalance (`linear`), and high data imbalance (`square` for 
larger client number e.g., `K=20`, exponential for smaller client number e.g., 
`K=5` as it will be too aggressive for larger client numbers)

This step is performed by 
```commandline
bash prepare_job_config.sh
```
In this example, we chose the Radial Basis Function (RBF) kernel to experiment with three clients under the uniform data split. 

Below is a sample config for site-1, saved to `./jobs/sklearn_svm_3_uniform/app_site-1/config/config_fed_client.json`:
```json
{
    "format_version": 2,
    "executors": [
        {
            "tasks": [
                "train"
            ],
            "executor": {
                "id": "Executor",
                "path": "nvflare.app_opt.sklearn.sklearn_executor.SKLearnExecutor",
                "args": {
                    "learner_id": "svm_learner"
                }
            }
        }
    ],
    "task_result_filters": [],
    "task_data_filters": [],
    "components": [
        {
            "id": "svm_learner",
            "path": "svm_learner.SVMLearner",
            "args": {
                "data_path": "/tmp/nvflare/dataset/sklearn_breast_cancer.csv",
                "train_start": 114,
                "train_end": 265,
                "valid_start": 0,
                "valid_end": 114
            }
        }
    ]
}
```

## Run experiment with FL simulator
[FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/nvflare_cli/fl_simulator.html) is used to simulate FL experiments or debug codes, not for real FL deployment.
We can run the FL simulator with three clients under the uniform data split with
```commandline
bash run_experiment_simulator.sh
```
Running with default [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) classifier, the 
resulting global model's AUC is 0.8088 which can be seen in the clients' logs.
