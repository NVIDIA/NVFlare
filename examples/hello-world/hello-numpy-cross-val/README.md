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
