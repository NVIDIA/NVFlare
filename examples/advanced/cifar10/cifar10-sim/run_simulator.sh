#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

job=$1
alpha=$2
threads=$3
n_clients=$4

# specify output workdir
RESULT_ROOT=/tmp/nvflare/sim_cifar10
out_workspace=${RESULT_ROOT}/${job}_alpha${alpha}

# run FL simulator
./set_alpha.sh "${job}" "${alpha}"
echo "Running ${job} using FL simulator with ${threads} threads and ${n_clients} clients. Save results to ${out_workspace}"
nvflare simulator "jobs/${job}" --workspace "${out_workspace}" --threads "${threads}" --n_clients "${n_clients}"
