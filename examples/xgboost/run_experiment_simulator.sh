#!/usr/bin/env bash

for n in 5 
do
  for study in bagging_uniform_split_uniform_lr bagging_exponential_split_uniform_lr bagging_exponential_split_scaled_lr cyclic_uniform_split_uniform_lr cyclic_exponential_split_uniform_lr
  do
    python3 -u -m nvflare.private.fed.app.simulator.simulator ${PWD}/job_configs/higgs_${n}_${study} -w ${PWD}/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}
  done
done

for n in 20
do
  for study in bagging_uniform_split_uniform_lr bagging_square_split_uniform_lr bagging_square_split_scaled_lr cyclic_uniform_split_uniform_lr cyclic_square_split_uniform_lr
  do
    python3 -u -m nvflare.private.fed.app.simulator.simulator ${PWD}/job_configs/higgs_${n}_${study} -w ${PWD}/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}
  done
done
