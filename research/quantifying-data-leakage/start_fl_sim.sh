#!/bin/bash

n_clients=${1}
mmar=${2}
exp_name=${3}

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:./src"
echo "Job ID ${NGC_JOB_ID}"
echo "Starting FL simulation with ${n_clients} clients."
echo "Results will be placed in ${result_dir}"

workspace="/tmp/nvflare/${n_clients}clients_${exp_name}"

# get num of gpus and assign to clients in a balanced manner
n_gpus=$(nvidia-smi --list-gpus | wc -l)
echo "There are ${n_gpus} GPUs."

gpus="0"
for i in $(eval echo "{1..$n_clients}")
do
  gpu_idx=$((${i} % ${n_gpus}))
  if (( ${i} < ${n_clients} ))
  then
    gpus+=",${gpu_idx}"
  fi
done

# Start FL run
# enable determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ ${n_clients} -gt 1 ]]
then
  COMMAND="nvflare simulator ${mmar} -w ${workspace} -n ${n_clients} -t ${n_clients} --gpu ${gpus}"
else
  COMMAND="nvflare simulator ${mmar} -w ${workspace} -n ${n_clients} -t ${n_clients} --clients site-9"
fi

echo "================================================================================================================="
echo "Executing: ${COMMAND}"
echo "================================================================================================================="
${COMMAND}
echo "================================================================================================================="

# Show cross-site validation results
echo "Cross-site validation results:"
cat "${workspace}/server/simulate_job/cross_site_val/cross_val_results.json"
echo
echo "================================================================================================================="
