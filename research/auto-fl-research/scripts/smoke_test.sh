#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
TASK_DIR=${TASK_DIR:-tasks/cifar10}
CLIENT_CONTRACT_PATH=${CLIENT_CONTRACT_PATH:-${TASK_DIR}/client.py}
PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/auto-fl-pycache}
export PYTHONPYCACHEPREFIX

"${PYTHON}" scripts/validate_contract.py "${CLIENT_CONTRACT_PATH}"
"${PYTHON}" scripts/pycompile_sources.py .

SMOKE_ARGS_TEXT=${SMOKE_ARGS:-}
if [[ -z "${SMOKE_ARGS_TEXT}" && "${TASK_DIR}" == "tasks/cifar10" ]]; then
  SMOKE_ARGS_TEXT="--n_clients 2 --num_rounds 1 --aggregation_epochs 1 --batch_size 32 --alpha 0.5 --seed 0 --aggregator weighted --name smoke_autofl"
fi

if [[ -z "${SMOKE_ARGS_TEXT}" ]]; then
  echo "Skipping runtime smoke test because no default smoke budget is defined for TASK_DIR=${TASK_DIR}."
  echo "Set SMOKE_ARGS to run a task-specific no-ledger smoke through scripts/run_iteration.sh."
  exit 0
fi

if "${PYTHON}" - <<'PY'
import importlib.util, sys

mods = [
    'numpy',
    'torch',
    'torchvision',
    'nvflare',
    'nvflare.app_common.abstract.fl_model',
    'nvflare.app_common.aggregators.model_aggregator',
    'nvflare.app_opt.pt.recipes.fedavg',
    'nvflare.client',
    'nvflare.recipe',
    'nvflare.recipe.utils',
]
missing = []
for mod in mods:
    try:
        spec = importlib.util.find_spec(mod)
    except ModuleNotFoundError:
        spec = None
    except Exception as exc:
        missing.append(f'{mod} ({exc})')
        continue
    if spec is None:
        missing.append(mod)
if missing:
    print('missing or incompatible:', ', '.join(missing))
    sys.exit(1)
print('all required runtime modules present')
PY
then
  read -r -a SMOKE_ARGS_ARRAY <<< "${SMOKE_ARGS_TEXT}"
  RUN_ITERATION_REQUIRE_SCORE=0 bash scripts/run_iteration.sh \
    --no-log-results \
    --description "smoke" \
    --target "${CLIENT_CONTRACT_PATH}" \
    -- "${SMOKE_ARGS_ARRAY[@]}"
else
  echo "Skipping runtime smoke test because the active environment does not have the required NVFlare API paths or other runtime modules installed."
  echo "Run this later in a proper NVFlare environment:"
  echo "bash scripts/run_iteration.sh --no-log-results --description \"smoke\" --target \"${CLIENT_CONTRACT_PATH}\" -- ${SMOKE_ARGS_TEXT}"
fi
