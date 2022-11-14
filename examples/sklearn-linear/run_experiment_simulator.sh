#!/usr/bin/env bash

n=20 #5
for study in uniform #exponential square 
do
    nvflare simulator job_configs/higgs_${n}_${study} -w /tmp/nvflare/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}
done
