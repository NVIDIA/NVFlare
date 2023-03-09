#!/usr/bin/env bash

WORKSPACE_ROOT="/tmp/nvflare/random_forest/HIGGS/workspaces"

n=5
for subsample in 0.5 0.05 0.005
do
    for study in uniform_split_uniform_lr exponential_split_uniform_lr exponential_split_scaled_lr 
    do
        nvflare simulator jobs/higgs_${n}_${subsample}_${study} -w $WORKSPACE_ROOT/workspace_${n}_${subsample}_${study} -n ${n} -t ${n}
    done
done


n=20
for subsample in 0.8 0.2 0.02
do
    for study in uniform_split_uniform_lr square_split_uniform_lr square_split_scaled_lr 
    do
        nvflare simulator jobs/higgs_${n}_${subsample}_${study} -w $WORKSPACE_ROOT/workspace_${n}_${subsample}_${study} -n ${n} -t ${n}
    done
done
