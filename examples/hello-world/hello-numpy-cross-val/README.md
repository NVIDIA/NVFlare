# Hello Numpy Cross-Site Validation

The cross-site model evaluation workflow uses the data from clients to run evaluation with the models of other clients. Data is not shared. Rather the collection of models is distributed to each client site to run local validation. The server collects the results of local validation to construct an all-to-all matrix of model performance vs. client dataset. It uses the [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.

> **_NOTE:_** This example uses a Numpy-based trainer and will generate its data within the code.

You can follow the [hello_world notebook](../hello_world.ipynb) or the following:

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.

### 2. Run the experiment

Use nvflare simulator to run the hello-examples:

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 hello-numpy-cross-val/jobs/hello-numpy-cross-val
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```

# Run cross site validation using the previous trained results

## Introduction

The "hello-numpy-cross-val-only" and "hello-numpy-cross-val-only-list-models" jobs show how to run the NVFlare cross-site validation without the training workflow, making use of the previous run results. The first one uses the default single server model. The second enables a list of server models. You can provide / use your own previous trained models for the cross-validation.

### Generate the previous run best global model and local best model

Run the following command to generate the pre-trained models:

```
python pre_train_models.py 
```

### How to run the Job

Define two OS system variable "SERVER_MODEL_DIR" and "CLIENT_MODEL_DIR" to point to the absolute path of the server best model and local best model location respectively. Then use the NVFlare admin command "submit_job" to submit and run the cross-validation job.

For example, define the system variable "SERVER_MODEL_DIR" like this:

```
export SERVER_MODEL_DIR="/path/to/model/location/at/server-side"
```

