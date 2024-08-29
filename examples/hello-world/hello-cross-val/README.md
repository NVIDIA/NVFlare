# Hello Cross-Site Validation

The cross-site model evaluation workflow uses the data from clients to run evaluation with the models of other clients. Data is not shared. Rather the collection of models is distributed to each client site to run local validation. The server collects the results of local validation to construct an all-to-all matrix of model performance vs. client dataset. It uses the [CrossSiteEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_eval.html) controller workflow.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/getting_started.html) instructions.

### 2. Run the experiment

Use nvflare simulator to run the example:

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 hello-cross-val/jobs/hello-cross-val
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```
