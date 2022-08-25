#!/usr/bin/env bash
bash start_fl_poc.sh 5
bash submit_job.sh 5 bagging uniform uniform
bash submit_job.sh 5 cyclic uniform uniform
bash submit_job.sh 5 bagging exponential uniform
bash submit_job.sh 5 cyclic exponential uniform
bash submit_job.sh 5 bagging exponential scaled


