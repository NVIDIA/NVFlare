#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=lib/gpu-operator-readiness.sh
source "${SCRIPT_DIR}/lib/gpu-operator-readiness.sh"

for command in kubectl python3; do need_cmd "${command}"; done
gpu_count="$(python3 "${SCRIPT_DIR}/lib/kata-runtime-profile.py" target "${RUNTIME_CLASS}" --field gpu_count)"
if ((gpu_count == 0)); then
  printf 'RuntimeClass %s is CPU-only; GPU Operator is not required.\n' "${RUNTIME_CLASS}"
  exit 0
fi
load_gpu_operator_expectations
assert_gpu_operator_readiness
printf 'GPU Operator operands and confidential-computing readiness are intact.\n'
