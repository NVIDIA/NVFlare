#!/usr/bin/env bash

task_name="sklearn_svm"

for site_num in 3;
do
    for split_mode in uniform;
    do
        for backend in sklearn;
        do
            nvflare simulator jobs/${task_name}_${site_num}_${split_mode}_${backend} \
                -w "${PWD}"/workspaces/${task_name}_${site_num}_${split_mode}_${backend} \
                -n ${site_num} \
                -t ${site_num}
        done
    done
done
