#!/usr/bin/env bash
echo "PYTHONPATH is ${PYTHONPATH}"

config=$1
alpha=$2
threads=$3
n_clients=$4

# specify output workdir
RESULT_ROOT=/tmp/nvflare/sim_cifar10
if [ 1 -eq "$(echo "${alpha} > 0" | bc)" ]
then
  out_workspace=${RESULT_ROOT}/${config}_alpha${alpha}
else
  out_workspace=${RESULT_ROOT}/${config}
fi

# run FL simulator
./set_alpha.sh "${config}" "${alpha}"
echo "Running ${config} using FL simulator with ${threads} threads and ${n_clients} clients. Save results to ${out_workspace}"
nvflare simulator "jobs/${config}" --workspace "${out_workspace}" --threads "${threads}" --n_clients "${n_clients}"
