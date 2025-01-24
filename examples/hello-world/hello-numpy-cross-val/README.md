# Hello Numpy Cross-Site Validation

The cross-site model evaluation workflow uses the data from clients to run evaluation with the models of other clients. Data is not shared. Rather the collection of models is distributed to each client site to run local validation. The server collects the results of local validation to construct an all-to-all matrix of model performance vs. client dataset. It uses the [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.


## Installation

Follow the [Installation](../../getting_started/README.md) instructions.

# Run training and cross site validation right after training

This example uses a NumPy-based trainer to simulate the training steps.

We first perform FedAvg training and then conduct cross-site validation.

So you will see two workflows (ScatterAndGather and CrossSiteModelEval) are configured.

## 1. Prepare the job and run the experiment using simulator

We use Job API to generate the job and run the job using simulator:

```bash
python3 job_train_and_cse.py
```

## 2. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir/
server/  site-1/  site-2/  startup/
```

The cross-site validation results:

```bash
$ cat /tmp/nvflare/jobs/workdir/server/simulate_job/cross_site_val/cross_val_results.json
```

# Run cross site evaluation using the previous trained results

We can also run cross-site evaluation without the training workflow, making use of the previous results or just want to evaluate on the pretrained models.

You can provide / use your own pretrained models for the cross-site evaluation.

## 1. Generate the pretrained model

In reality, users would use any training workflows to obtain these pretrained models

To mimic that, run the following command to generate the pre-trained models:

```bash
python3 generate_pretrain_models.py
```

## 2. Prepare the job and run the experiment using simulator

Note that our pre-trained models are generated under:

```python
SERVER_MODEL_DIR = "/tmp/nvflare/server_pretrain_models"
CLIENT_MODEL_DIR = "/tmp/nvflare/client_pretrain_models"
```

In our job_cse.py we also specify that.

Then we can use Job API to generate the job and run it using simulator:

```bash
python3 job_cse.py
```

## 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace:

```bash
$ ls /tmp/nvflare/jobs/workdir/
server/  site-1/  site-2/  startup/
```

The cross-site validation results:

```bash
$ cat /tmp/nvflare/jobs/workdir/server/simulate_job/cross_site_val/cross_val_results.json
```