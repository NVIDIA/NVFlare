#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
TASK_DIR=${TASK_DIR:-tasks/cifar10}
CLIENT_CONTRACT_PATH=${CLIENT_CONTRACT_PATH:-${TASK_DIR}/client.py}
N_CLIENTS=${N_CLIENTS:-8}
NUM_ROUNDS=${NUM_ROUNDS:-20}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-4}
LOCAL_TRAIN_STEPS=${LOCAL_TRAIN_STEPS:-0}
BATCH_SIZE=${BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1024}
ALPHA=${ALPHA:-0.5}
SEED=${SEED:-0}
AGGREGATOR=${AGGREGATOR:-weighted}
FINAL_EVAL_CLIENTS=${FINAL_EVAL_CLIENTS:-site-1}
NAME=${NAME:-autofl_full_eval}
DESCRIPTION=${DESCRIPTION:-full_eval}
TARGET=${TARGET:-${CLIENT_CONTRACT_PATH}}

if [[ "${TASK_DIR}" != "tasks/cifar10" ]]; then
  echo "ERROR: scripts/run_full_eval.sh has a built-in full-eval budget only for TASK_DIR=tasks/cifar10." >&2
  echo "Use scripts/run_iteration.sh with the active task profile's budget, or add a task-specific full-eval wrapper." >&2
  exit 2
fi

"${PYTHON}" scripts/validate_contract.py "${CLIENT_CONTRACT_PATH}"

bash scripts/run_iteration.sh \
  --description "${DESCRIPTION}" \
  --target "${TARGET}" \
  -- --n_clients "${N_CLIENTS}" \
     --num_rounds "${NUM_ROUNDS}" \
     --aggregation_epochs "${LOCAL_EPOCHS}" \
     --local_train_steps "${LOCAL_TRAIN_STEPS}" \
     --batch_size "${BATCH_SIZE}" \
     --eval_batch_size "${EVAL_BATCH_SIZE}" \
     --alpha "${ALPHA}" \
     --seed "${SEED}" \
     --aggregator "${AGGREGATOR}" \
     --final_eval_clients "${FINAL_EVAL_CLIENTS}" \
     --name "${NAME}" \
     "$@"
