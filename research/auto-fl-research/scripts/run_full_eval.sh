#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
N_CLIENTS=${N_CLIENTS:-8}
NUM_ROUNDS=${NUM_ROUNDS:-10}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1024}
ALPHA=${ALPHA:-0.5}
SEED=${SEED:-0}
AGGREGATOR=${AGGREGATOR:-weighted}
FINAL_EVAL_CLIENTS=${FINAL_EVAL_CLIENTS:-site-1}
NAME=${NAME:-autofl_full_eval}
DESCRIPTION=${DESCRIPTION:-full_eval}
TARGET=${TARGET:-client.py}

"${PYTHON}" scripts/validate_contract.py client.py

bash scripts/run_iteration.sh \
  --description "${DESCRIPTION}" \
  --target "${TARGET}" \
  -- --n_clients "${N_CLIENTS}" \
     --num_rounds "${NUM_ROUNDS}" \
     --aggregation_epochs "${LOCAL_EPOCHS}" \
     --batch_size "${BATCH_SIZE}" \
     --eval_batch_size "${EVAL_BATCH_SIZE}" \
     --alpha "${ALPHA}" \
     --seed "${SEED}" \
     --aggregator "${AGGREGATOR}" \
     --final_eval_clients "${FINAL_EVAL_CLIENTS}" \
     --name "${NAME}" \
     "$@"
