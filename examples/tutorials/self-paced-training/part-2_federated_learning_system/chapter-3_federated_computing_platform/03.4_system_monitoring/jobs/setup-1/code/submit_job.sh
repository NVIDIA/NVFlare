#!/bin/bash


# generate job config folder
python3 fl_job.py -j /tmp/nvflare/jobs/job_config

# Submit the NVFlare job
nvflare job submit -j /tmp/nvflare/jobs/job_config/fedavg
