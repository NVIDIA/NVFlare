# Code Pre-Installation Example

This example demonstrates how to use NVFLARE's code pre-installer in a real-world scenario.


## Overview

In production environments, application code often needs to be pre-installed on target machines due to:
- Security policies restricting dynamic code loading
- Network isolation requirements
- Need for consistent code deployment

The pre-installer tool provides two commands:
- `prepare`: Package application code for distribution
- `install`: Install code on target machines

## Prerequisites

- NVFLARE installed
- Access to example jobs (fedavg)
- Python environment with required packages

## Steps

### 1. Prepare Application Package

First, we'll package the fedavg example job:
[FedAvg](./jobs/fedavg)

# Create application package from fedavg job

```bash
nvflare pre-install prepare -j jobs/fedavg -o /tmp/prepare -r jobs/requirements.txt
```

# Verify the package
ls -l /tmp/prepare/application.zip
```

### 2. Pre-install the application package on each site, simulating the real-world deployment

since we are simulating the deployment on a single machine, we need to pre-install the application package on different locations, one for each site.

```bash
# Install on server
nvflare pre-install install -a /tmp/prepare/application.zip -s server -p /tmp/opt/nvflare/server/fedavg/

# Install on client-1
nvflare pre-install install -a /tmp/prepare/application.zip -s site-1 -p /tmp/opt/nvflare/site-1/fedavg/

# Install on client-2
nvflare pre-install install -a /tmp/prepare/application.zip -s site-2 -p /tmp/opt/nvflare/site-2/fedavg/
```/local/custom
We also need to setup PYTHONPATH to include the installed prefix path and shared directory path.

```bash
export PYTHONPATH=/local/custom:$PYTHONPATH
export PYTHONPATH=/tmp/opt/nvflare/server/fedavg/:$PYTHONPATH
export PYTHONPATH=/tmp/opt/nvflare/site-1/fedavg/:$PYTHONPATH
export PYTHONPATH=/tmp/opt/nvflare/site-2/fedavg/:$PYTHONPATH
```

### 2. Test in Simulator

Now, we have pre-installed the application code on different locations, we can test the pre-installed code using NVFLARE's simulator:

First, we need to delete the custom code directories from the fedavg job folder.

Next, we need to update the job configuration to use the pre-installed code prefix. This step involves updating the client job configuration to point to the pre-installed code prefix.

These changes have already been applied in [fedavg_config](./jobs/fedavg_config).

For site-1: 
the code install prefix: "/tmp/opt/nvflare/site-1/fedavg/"
```json

       "task_script_path": "/tmp/opt/nvflare/site-1/fedavg/src/client.py",
```
For site-2: 
the code install prefix: "/tmp/opt/nvflare/site-2/fedavg/"
```json

       "task_script_path": "/tmp/opt/nvflare/site-2/fedavg/src/client.py",
```

The rest of the configuration should be the same. 


Test the pre-installed code using NVFLARE's simulator:

```bash
nvflare simulator jobs/fedavg_config -w /tmp/workspace -n 2
```

### 3. Test in POC Mode


```bash
# Create POC workspace
nvflare poc prepare -n 2
nvflare poc start -ex admin@nvidia.com

```

We use CLI to submit the job

```bash
nvflare job submit -j jobs/fedavg_config
```

once you finish, you can do 

```bash

nvflare poc stop
nvflare poc clean
```
