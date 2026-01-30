# Example for Job-level authorization

# Overview

The purpose of this example is to demonstrate following features of NVFlare,

1. Run NVFlare in secure mode
2. Demonstrate job-level authorization policy

## System Requirements

1. Install Python and set up a Virtual Environment,
```
python3 -m venv nvflare-env
source nvflare-env/bin/activate
```
2. Install NVFlare
```
pip install -r requirements.txt
```
3. The example is part of the NVFlare source code. The source code can be obtained like this,
```
git clone https://github.com/NVIDIA/NVFlare.git
```
4. TLS requires domain names. Please add the following line in the `/etc/hosts` file,
```
127.0.0.1	server1
```

### Setup

```
cd NVFlare/examples/advanced/job-level-authorization
./setup.sh
```

All the startup kits will be generated in this folder,
```
/tmp/nvflare/poc/job-level-authorization/prod_00
```

**Important**: The `setup.sh` script performs the following operations:
1. Removes the workspace folder (if it exists) and regenerates the POC environment
2. Prepares the POC deployment with the specified configuration
3. **Overwrites site_a's security settings** by copying the custom security handler from `security/site_a/*` to `/tmp/nvflare/poc/job-level-authorization/prod_00/site_a/local`

This custom security configuration installs the `CustomSecurityHandler` that enforces job-level authorization on site_a, blocking jobs named "FL-Demo-Job2" while allowing all other jobs.

Note that the "workspace" folder is removed every time `setup.sh` is run. Please do not save customized files in this folder.

### Starting NVFlare

This script will start up the server and 2 clients,
```
nvflare poc start
```

### Submitting Jobs to ProdEnv

Here, we treat the created POC environment as a production environemnt running in the background.
You can submit jobs programmatically using the Job API with `ProdEnv`. Two example scripts are provided:

**job1.py** - Submits a job named "hello-numpy" (**ALLOWED by site_a**):
```
python job1.py
```

**job2.py** - Submits a job named "FL-Demo-Job2" (**BLOCKED by site_a**):
```
python job2.py
```

Both scripts use `ProdEnv` to connect to the production deployment and submit jobs via the Flare API. The jobs demonstrate how site_a's `CustomSecurityHandler` enforces authorization based on job name:
- Job 1 with name "hello-numpy" will be accepted by both site_a and site_b
- Job 2 with name "FL-Demo-Job2" will be rejected by site_a but accepted by site_b

You can customize the startup kit location and username using command-line arguments:
```
python job1.py --startup_kit_location /path/to/startup_kit --username user@example.com
```

## Participants

### Site
* `server1`: NVFlare server
* `site_a`: Site_a has a CustomSecurityHandler set up which does not allow the job "FL-Demo-Job2" to run. Any other named jobs will be able to deploy and run on site_a.
* `site_b`: Site_b does not have the extra security handling codes. It allows any job to be deployed and run.

### Jobs

* job1: The job is called `hello-numpy`. site_a will allow this job to run.
* job2: The job is called `FL-Demo-Job2`. site_a will block this job to run.

### Output

For job1, you will see successful completion with training on both clients (site_a & site_b).

For job2, you will see an output like this in the POC log messages:

```
2026-01-30 12:41:51,006 - site_security - ERROR - Authorization failed. Reason: Job 'FL-Demo-Job2' BLOCKED by site_a's CustomSecurityHandler - not authorized to execute: check_resources
2026-01-30 12:41:51,008 - ServerEngine - ERROR - Client reply error: Job 'FL-Demo-Job2' BLOCKED by site_a's CustomSecurityHandler - not authorized to execute: check_resources
```
Only site_b will execute the training run.

