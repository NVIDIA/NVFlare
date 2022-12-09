#!/usr/bin/env bash

n=3
for study in uniform #linear
do
    nvflare simulator job_configs/sklearn_kmeans_${n}_${study} -w ${PWD}/workspaces/workspace_${n}_${study} -n ${n} -t ${n}
done
