# Hello Client Controlled Workflow (CCWF)

[Client Controlled Workflows](https://nvflare.readthedocs.io/en/main/programming_guide/controllers/client_controlled_workflows.html) are managed
by logic from clients. This example shows the components used in a job for a client controlled workflow.

### 1. Install NVIDIA FLARE

Follow the [Installation](https://nvflare.readthedocs.io/en/main/quickstart.html) instructions.

### 2. Run the experiment

Use nvflare simulator to run the hello-examples:

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 hello-ccwf/jobs/numpy-swcse
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```
