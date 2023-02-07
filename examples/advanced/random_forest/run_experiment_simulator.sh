#!/usr/bin/env bash

n=5
for subsample in 0.5 0.05 0.005
do
    for study in uniform_split_uniform_lr exponential_split_uniform_lr exponential_split_scaled_lr 
    do
        nvflare simulator job_configs/higgs_${n}_${subsample}_${study} -w ${PWD}/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}
    done
done


n=20
for subsample in 0.8 0.2 0.02
do
    for study in uniform_split_uniform_lr square_split_uniform_lr square_split_scaled_lr 
    do
        nvflare simulator job_configs/higgs_${n}_${subsample}_${study} -w ${PWD}/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}
    done
done
