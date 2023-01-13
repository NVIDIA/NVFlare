#!/usr/bin/env bash

n=3
for study in uniform
do
    nvflare simulator job_configs/sklearn_svm_3_uniform -w ${PWD}/workspaces/sklearn_svm_${n}_${study} -n ${n} -t ${n}
done
