#!/usr/bin/env bash
bash start_fl_poc.sh 20
bash submit_job.sh 20 bagging uniform uniform
bash submit_job.sh 20 cyclic uniform uniform
bash submit_job.sh 20 bagging square uniform
bash submit_job.sh 20 cyclic square uniform
bash submit_job.sh 20 bagging square scaled


