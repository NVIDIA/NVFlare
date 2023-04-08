#!/usr/bin/env bash

n=5
for study in bagging_uniform_split_uniform_lr \
             bagging_exponential_split_uniform_lr \
             bagging_exponential_split_scaled_lr \
             cyclic_uniform_split_uniform_lr \
             cyclic_exponential_split_uniform_lr
do
  nvflare simulator jobs/higgs_${n}_${study} -w ${PWD}/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}
done


n=20
for study in bagging_uniform_split_uniform_lr \
            bagging_square_split_uniform_lr \
            bagging_square_split_scaled_lr \
            cyclic_uniform_split_uniform_lr \
            cyclic_square_split_uniform_lr
do
  nvflare simulator jobs/higgs_${n}_${study} -w ${PWD}/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}
done
