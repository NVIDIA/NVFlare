# Hello Client Controlled Workflow (CCWF)

[Client Controlled Workflows](https://nvflare.readthedocs.io/en/main/programming_guide/controllers/client_controlled_workflows.html) are managed
by logic from clients. This example shows the components used in a job for a client controlled workflow.

### 1. Install NVIDIA FLARE

Follow the [Installation](../../getting_started/README.md) instructions.

### 2. Run the experiment

Run the script using the job API to create the job and run it with the simulator:

```
python3 swarm_script_runner_np.py
```

### 3. Access the logs and results

You can find the running logs and results inside the simulator's workspace/simulate_job

```bash
$ ls /tmp/nvflare/simulate_job/
app_server  app_site-1  app_site-2  log.txt

```
