# AGENTS.md

## Entry point

Start with `program.md`. It is the single research-org entry point for the agent.

Use `mutation_schema.yaml` only when `program.md` directs you to the hard mutation bounds or when choosing a mutation axis. If anything here conflicts with `program.md`, follow `program.md` unless a human explicitly overrides it.

## Mission

Improve this Auto-FL NVFlare harness without breaking the federated contract.

## Files you may edit

Preferred mutation files:
- `client.py`
- `custom_aggregators.py`
- `job.py`
- `model.py` for registered architecture variants under the active parameter cap

Do not change unless explicitly requested:
- `data/*`

## Hard invariants

You must preserve all of the following unless a human explicitly asks for a protocol upgrade:
- `flare.init()`
- `while flare.is_running():`
- `input_model = flare.receive()`
- `flare.send(output_model)`
- `model.load_state_dict(input_model.params, strict=True)`
- `compute_model_diff(...)`
- `output_model.params_type == ParamsType.DIFF`
- `output_model.meta["NUM_STEPS_CURRENT_ROUND"]`
- the optional `if flare.is_evaluate():` branch
- the same selected `model_arch` must be instantiated on server and clients for a run
- `model_arch` and `max_model_params` are fixed budget fields unless the campaign is explicitly labeled as an architecture subcampaign

## Required workflow after every edit

1. Run `make validate`
2. Run either `make smoke` or `bash scripts/run_iteration.sh ...`
3. Record the mutation in `results.tsv`
4. Summarize the mutation in `templates/mutation_report.md`
