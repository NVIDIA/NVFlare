# Swarm Learning with Cross-Site Evaluation
This example shows how to use swarm learning with [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) using PyTorch with the CIFAR-10 dataset.

Before starting please make sure you set up a [virtual environment](../../../README.md#set-up-a-virtual-environment) and install the additional requirements:
```
python3 -m pip install -r requirements.txt
```

## Introduction to Swarm Learning

As introduced in the [Nature article](https://www.nature.com/articles/s41586-021-03583-3), swarm learning is essentially a decentralized form of federated learning, wherein responsibilities are distributed to all peers rather than consolidated in a central server. The main goal is to support data sovereignty, security, and confidentiality. One key difference to note is that the authors in the article use blockchain for secure onboarding, elections and communication, while instead FLARE uses its own secure messaging.

Our implementation of swarm learning leverages the newly added Client Controlled Workflow (CCWF) framework in FLARE, which allows for secure peer-to-peer communication between clients. The server is simply responsible for the job lifecycle management (health of client sites and monitoring of job status), while the clients are now responsible for training logic and aggregration management (where tasks are assigned via peer-to-peer communication).

Algorithmically, swarm learning is identical to federated averaging with the main differences being that the server will no longer control the aggregation process and will not have access to any sensitive information, such as trained model weights.

## Swarm Learning with FLARE

In swarm learning, training and aggregation is done accoss multiple rounds. The starting client is responsible for the initial model which is loaded from the configured persistor. In each round:
1. An aggregator client is randomly chosen from all clients.
2. The training task is sent to all training clients on the current global model.
3. The clients then send their training results to the designated client for aggregration.
4. The aggregration client applies the results to the current global model, which then becomes the base for the next round of training.

<img src="./figs/swarm_learning.png" alt="swarm learning diagram" width="500"/>

In order to convert regular PyTorch code into a swarm learning workflow, there are a few new required configurations:

- We first start with `CIFAR10ModelLearner`, which is a simple PyTorch trainer subclassing ModelLearner that implements the required `train()`, `validate()`, and `get_model()` methods.
- The server side `SwarmServerController` is set to manage the job lifecycle, and thus the server is not involved with any sensitive information or training logic.
- The client side `SwarmClientController` sends `learn_task` to all training clients for each round, which is mapped to the `train` task implemented by the `CIFAR10ModelLearner`. The aggregrator and persistor components must also be configured.
- Filter direction can now be set for both `task_result_filters` and `task_data_filters`, with options of `in`, `out`, or `inout`. Since clients send tasks to each other, a task has both a sending (`out`) and a receiving (`in`) direction.
- A model selection widget can be configured to determine the best global model.

## Client-Controlled Cross-Site Evaluation

In client-controlled CSE, rather than sending client models to the server for distribution, clients instead communicate directly with each other to share their models for validation:
1. The initial config task contains information about who the evaluators and evaluatees are, and which client contains the global model.
2. Next, the global models are evaluated if available and then the clients' local models are evaluated next. 
When evaluating an evaluatees' model, an `eval` task is sent to all evaluators.
3. When an evalutor receives an `eval` task, they send a `get_model` task to the evaluatee client that owns the model, who in return obtains the model with `submit_model` and sends it back to the evaluator.
4. The evaluator then performs `validate` on the received model.
4. All validation results are sent back to the server for easy access.

<img src="./figs/cse.png" alt="client-controlled cse diagram" width="500"/>

In order to convert regular PyTorch code into a client-controlled CSE workflow, there are a few new required configurations:

- The `CrossSiteEvalServerController` is set to manage the configuration and evaluation workflow.
- The `CrossSiteEvalClientController` `submit_model_task_name` and `validation_task_name` are mapped to the `submit_model` and `validate` tasks of the `CIFAR10ModelLearner`.
- The global model client must have a model persistor that implements `get_model_inventory()` in order to return the names of available global models, as well as `get_model()` to get the model for other clients to evaluate.


## Preparing CIFAR-10 Dataset
Run the following command to prepare the data splits (note: change `num_sites` in `cifar10_data_utils.py` if using more than 2 clients):
```
./prepare_data.sh
```

## Running Swarm Learning with Cross-Site Evaluation Job
First we create the swarm learning with cross-site evaluation job using the predefined swarm_cse_pt template:
```
nvflare job create -j ./jobs/swarm_cse_cifar10 -w swarm_cse_pt
```

Feel free to change job configuration parameters, and ensure that the components are set correctly as described above. Then run the job:

```
nvflare simulator ./jobs/swarm_cse_cifar10 -w /tmp/nvflare/swarm_cse_cifar10 -n 2 -t 2
```

## Results
To view the cross validation results:
```
python -m json.tool /tmp/nvflare/swarm_cse_cifar10/simulate_job/cross_site_val/cross_val_results.json
```

Models and results can found in `/tmp/nvflare/swarm_cse_cifar10/simulate_job/`, and we can confirm that the server in `/tmp/nvflare/swarm_cse_cifar10/simulate_job/app_server` does not contain any sensitive training data.

Lastly, since swarm learning and federated averaging are algorithmically the same, this can be proven experimentally by creating a CIFAR-10 SAG FedAvg job with the same data splits and training parameters, which will yield similar results.
