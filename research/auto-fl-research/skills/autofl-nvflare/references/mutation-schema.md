# Mutation schema

## Target
The active task's `client.py`, plus task-local files allowed by that task profile.

## Goal
Mutate local client behavior and bounded registered model architectures without changing the active task's server contract.

## Fixed invariants
- `flare.init()`
- `while flare.is_running()`
- `input_model = flare.receive()`
- `flare.send(output_model)`
- `model.load_state_dict(input_model.params, strict=True)`
- `compute_model_diff(model, global_model)`
- `output_model.params_type == ParamsType.DIFF`
- `output_model.meta["NUM_STEPS_CURRENT_ROUND"]`
- same selected `model_arch` on server and clients for a run
- active `max_model_params` cap for architecture campaigns
- dataset, site mapping, and evaluation contract defined by the active task profile

## Safe arguments
Use the active task's `mutation_schema.yaml` as the source of truth. Common
examples include:

- `aggregation_epochs`
- `local_train_steps`
- `lr`
- `batch_size`
- `num_workers`
- `momentum`
- `weight_decay`
- `no_lr_scheduler`
- `cosine_lr_eta_min_factor`
- `evaluate_local`
- `fedproxloss_mu`
- `model_arch`
- `max_model_params`

`local_train_steps=0` uses epoch-based training with `aggregation_epochs`. Positive `local_train_steps` values use exact optimizer steps per client per round and should not be varied in the same narrow sweep as `aggregation_epochs`.

## Forbidden mutations
- switch DIFF uploads to FULL uploads
- remove `NUM_STEPS_CURRENT_ROUND`
- rewrite the flare task loop
- change the model architecture outside registered `model_arch` variants or above `max_model_params`
- add server-coupled metadata fields
