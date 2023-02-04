#!/usr/bin/env bash
task_name="sklearn_linear"
n=5
for study in uniform
do
    nvflare simulator job_configs/${task_name}_${n}_${study} -w ${PWD}/workspaces/${task_name}_${n}_${study} -n ${n} -t ${n}
done
