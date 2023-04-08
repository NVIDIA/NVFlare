#!/usr/bin/env bash

n=2
study=histogram_uniform_split_uniform_lr
nvflare simulator jobs/higgs_${n}_${study} -w ${PWD}/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}

n=5
study=histogram_uniform_split_uniform_lr
nvflare simulator jobs/higgs_${n}_${study} -w ${PWD}/workspaces/xgboost_workspace_${n}_${study} -n ${n} -t ${n}
