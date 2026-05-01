#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/auto-fl-pycache}
export PYTHONPYCACHEPREFIX

"${PYTHON}" scripts/validate_contract.py client.py
"${PYTHON}" scripts/pycompile_sources.py .

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
  RUN_ITERATION_REQUIRE_SCORE=0 bash scripts/run_iteration.sh \
    --no-log-results \
    --description "smoke" \
    --target client.py \
    -- --n_clients 2 --num_rounds 1 --aggregation_epochs 1 --batch_size 32 --alpha 0.5 --seed 0 --aggregator weighted --name smoke_autofl
else
  echo "Skipping runtime smoke test because the active environment does not have the required NVFlare API paths or other runtime modules installed."
  echo "Run this later in a proper NVFlare environment:"
  echo "bash scripts/run_iteration.sh --no-log-results --description \"smoke\" --target client.py -- --n_clients 2 --num_rounds 1 --aggregation_epochs 1 --batch_size 32 --alpha 0.5 --seed 0 --aggregator weighted --name smoke_autofl"
fi
