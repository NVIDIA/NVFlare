# Mutation schema

## Target
`client.py`

## Goal
Mutate local client behavior and bounded registered model architectures without changing the server contract.

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
- dataset partition contract under `train_idx_root`

## Safe arguments
- `aggregation_epochs`
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

## Forbidden mutations in v0
- switch DIFF uploads to FULL uploads
- remove `NUM_STEPS_CURRENT_ROUND`
- rewrite the flare task loop
- change the model architecture outside registered `model_arch` variants or above `max_model_params`
- add server-coupled metadata fields
