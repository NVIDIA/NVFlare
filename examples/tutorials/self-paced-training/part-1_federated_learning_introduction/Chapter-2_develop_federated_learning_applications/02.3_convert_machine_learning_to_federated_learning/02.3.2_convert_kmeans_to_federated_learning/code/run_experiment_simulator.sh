#!/usr/bin/env bash

task_name="sklearn_kmeans"

for site_num in 3;
do
    for split_mode in uniform;
    do
        nvflare simulator jobs/${task_name}_${site_num}_${split_mode} \
            -w "${PWD}"/workspaces/${task_name}_${site_num}_${split_mode} \
            -n ${site_num} \
            -t ${site_num}
    done
done
