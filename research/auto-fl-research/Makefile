PYTHON ?= python3
PYTHONPYCACHEPREFIX ?= /tmp/auto-fl-pycache
TAG ?=

.PHONY: validate pycompile smoke full init-run

validate:
	PYTHONPYCACHEPREFIX="$(PYTHONPYCACHEPREFIX)" $(PYTHON) scripts/validate_contract.py client.py
	PYTHONPYCACHEPREFIX="$(PYTHONPYCACHEPREFIX)" $(PYTHON) scripts/pycompile_sources.py .

pycompile:
	PYTHONPYCACHEPREFIX="$(PYTHONPYCACHEPREFIX)" $(PYTHON) scripts/pycompile_sources.py .

smoke:
	PYTHON="$(PYTHON)" bash scripts/smoke_test.sh

full:
	PYTHON="$(PYTHON)" bash scripts/run_full_eval.sh

init-run:
	bash scripts/init_run.sh $(TAG)
