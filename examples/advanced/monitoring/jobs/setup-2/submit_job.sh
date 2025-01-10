#!/bin/bash


# generate job config folder
python3 fedavg_script_runner_pt.py -j /tmp/nvflare/jobs/job_config

# Submit the NVFlare job
nvflare job submit -j /tmp/nvflare/jobs/job_config/hello-pt
