PYTHON ?= python3
PYTHONPYCACHEPREFIX ?= /tmp/auto-fl-pycache
TASK_DIR ?= tasks/cifar10
CLIENT_CONTRACT_PATH ?= $(TASK_DIR)/client.py
TAG ?=

.PHONY: validate pycompile smoke full init-run

validate:
	PYTHONPYCACHEPREFIX="$(PYTHONPYCACHEPREFIX)" $(PYTHON) scripts/validate_contract.py "$(CLIENT_CONTRACT_PATH)"
	PYTHONPYCACHEPREFIX="$(PYTHONPYCACHEPREFIX)" $(PYTHON) scripts/pycompile_sources.py .

pycompile:
	PYTHONPYCACHEPREFIX="$(PYTHONPYCACHEPREFIX)" $(PYTHON) scripts/pycompile_sources.py .

smoke:
	PYTHON="$(PYTHON)" bash scripts/smoke_test.sh

full:
	PYTHON="$(PYTHON)" bash scripts/run_full_eval.sh

init-run:
	bash scripts/init_run.sh $(TAG)
