# HuggingFace Client API Sharded Training Proposal

Status: Proposal for next-release design

## Summary

The HuggingFace Client API currently supports one process and distributed data
parallel (DDP) `transformers.Trainer` workloads. It explicitly rejects
`TrainingArguments.fsdp` and `TrainingArguments.deepspeed`. This is a real
runtime limitation, not a statement that sharded training is unsupported by
NVFlare generally.

This proposal adds a sharded-exchange adapter for HuggingFace Trainer workloads.
It preserves the existing federated server contract: each site contributes one
logical model/update to FedAvg or another existing workflow. It does not send
FSDP or ZeRO shard objects over the federation. Instead, ranks cooperate at the
site boundary to import a federated model and export one CPU-resident,
canonical state dict from rank 0.

FSDP and DeepSpeed should be delivered as separate capabilities. Their state
dict and checkpoint semantics differ enough that enabling them behind one
generic "sharded" switch would be unsafe.

## Current Limitation

The current path assumes an ordinary replicated `state_dict`:

```text
server -> rank 0 receives full FLModel parameters
       -> rank 0 broadcasts the full parameter map to every local rank
       -> every rank calls model.load_state_dict(...)

every rank trains with DDP

rank 0 calls accelerator.get_state_dict()/state_dict()
       -> rank 0 sends one full FLModel to the server
```

That works for DDP because every rank holds a complete model. It is not valid
for FSDP or DeepSpeed ZeRO:

- FSDP full-state export is collective: all ranks must participate even when
  only rank 0 obtains the populated full state dict.
- Broadcasting a complete incoming model to every FSDP/ZeRO rank defeats the
  memory purpose of sharding and can cause host-memory or GPU-memory OOM.
- Model and optimizer checkpoint save/load are coordinated sharded operations,
  not rank-0 file writes.
- Current Client API checkpoint injection and `submit_model` assumptions use
  ordinary HF checkpoint files and cannot safely interpret every distributed
  checkpoint format.

Therefore removing the validation in `nvflare.app_opt.hf.patch()` is not a
valid implementation.

## Goals

- Support HuggingFace Trainer with FSDP full-parameter sharding for federated
  full-model and PEFT-adapter training.
- Keep the federated wire format a canonical CPU tensor mapping produced only
  by rank 0.
- Avoid materializing a full incoming model on every rank.
- Preserve model, optimizer, scheduler, RNG, and trainer resume correctness
  across local FL rounds.
- Bound rank-0 host memory and make the aggregation-size tradeoff explicit.
- Add DeepSpeed ZeRO support only after FSDP is validated through the same
  public capability contract.

## Non-Goals

- Do not aggregate rank-local FSDP or ZeRO shards across sites.
- Do not make arbitrary `accelerate` training loops part of the Trainer API.
- Do not silently fall back to DDP or replicated weights when sharded import or
  export is unavailable.
- Do not claim that sharding makes federated full-model aggregation practical
  for a model too large for rank-0 CPU memory or server aggregation memory.

## Capability Contract

Add an internal capability resolver, selected during `flare.patch(trainer)`:

```text
replicated-ddp       current implementation
fsdp-full-state      new FSDP adapter
deepspeed-zero       future DeepSpeed adapter
unsupported          explicit descriptive error
```

The public `patch()` API remains unchanged. The selected capability and its
state-dict mode must be recorded in logs and the outgoing FL model metadata so
an operator can identify the site-side execution mode.

Only configurations validated by the adapter are accepted. Unsupported FSDP
versions, hybrid-shard modes, incompatible `use_orig_params` configurations,
or checkpoint formats fail before a federated task begins, with a remediation
message. FSDP support must not rely on private `accelerate` internals.

## FSDP Design

### Export: site shards to one federated update

All local ranks enter an FSDP full-state-dict collective after the local train
round. The adapter requests a full state dict with CPU offload and rank-0-only
materialization. Non-zero ranks participate in the collective but do not build
or send an `FLModel`; rank 0 validates, converts, and sends the canonical
parameter mapping.

```text
all ranks: FSDP full-state collective
rank 0:    full CPU state dict -> NVFlare FLModel -> server
other:     empty state dict -> wait for task completion broadcast
```

The adapter must use an explicit FSDP state-dict context and restore the
previous context after the operation. It must not assume that
`accelerator.get_state_dict()` is safe to invoke only on rank 0.

### Import: federated model to site shards

Rank 0 receives the canonical server model once. Import is collective and
must not use the current `broadcast_object_list` full-map path. The adapter
uses the FSDP-supported load path to distribute/shard the incoming model across
ranks. The exact API is version-gated, but the invariant is:

```text
rank 0 holds the federated full CPU state dict
all ranks enter the FSDP import collective
each rank receives only its required local shard
```

The adapter checks parameter keys, shapes, and dtypes against the canonical
unsharded state dict before the collective. It reports a coordinated failure to
all ranks if validation or import fails, preventing distributed hangs.

### Adapter/PEFT scope

For `params_scope="adapter"`, the first release should support PEFT only when
the selected FSDP wrapping and PEFT integration produce a canonical adapter
state dict and a collective adapter load path. This is a separate compatibility
matrix entry, not an assumption that every PEFT+FSDP combination works.

## Checkpoint and Resume

The existing rank-0 checkpoint-injection fallback must not be used for sharded
training. The adapter owns a coordinated checkpoint protocol:

1. Every rank saves its local model/optimizer/scheduler/RNG state using the
   supported FSDP/HuggingFace checkpoint mechanism.
2. A durable checkpoint manifest records the framework versions, world size,
   state-dict mode, model identity, adapter scope, and completion generation.
3. The manifest is published atomically only after every required shard is
   durable.
4. Resume validates the manifest and restores collectively before the next
   train call.

Changing world size across local FL rounds is out of scope for the first FSDP
release unless the selected checkpoint format explicitly supports resharding
and it is covered by integration tests.

## Memory and Transport Policy

FSDP reduces device memory for local training; it does not eliminate the need
for one full federated model at the site and server aggregation boundaries.
Before enabling a full-model FSDP FL job, the site must provision rank-0 CPU
memory for the full state dict, conversion buffer, and configured safety margin.

For models where full-model aggregation is not practical, users should use a
supported PEFT adapter scope, quantized/filtered exchange once supported by the
workflow, or a different algorithmic/runtime design. The adapter must expose
export/import timing and peak host-memory metrics for admission and capacity
planning.

Large rank-local coordination should use the existing file-exchange mechanism
only when all ranks share the configured path. The file is an internal
site-local optimization, never a new federated transport contract. Cleanup
occurs after all ranks acknowledge completion or a coordinated failure is
recorded.

## DeepSpeed Follow-On

DeepSpeed ZeRO has equivalent high-level needs but different APIs, checkpoint
layouts, optimizer ownership, and configuration semantics. It is a separate
follow-on after FSDP:

- define a ZeRO stage support matrix, starting with the configuration that can
  export/import a canonical consolidated state dict safely;
- implement coordinated rank participation and rank-0 federated exchange;
- add a ZeRO-native checkpoint manifest and resume contract;
- validate CPU/NVMe offload configurations independently.

The FSDP adapter should share only common orchestration interfaces with
DeepSpeed, not state-dict implementation details.

## Validation Plan

FSDP support requires real multi-rank integration tests, not fake Trainer-only
unit tests:

1. Two-rank FSDP full-model round-trip: server params import, local train,
   rank-0 export, and parameter equality checks.
2. Two consecutive FL rounds with optimizer/scheduler/RNG checkpoint resume.
3. Failure on one rank during import/export/checkpoint, with all ranks released
   and no distributed hang.
4. Strict and non-strict key/shape validation before collective import.
5. CPU-offloaded, rank-0-only export memory assertions and metrics.
6. Supported PEFT+FSDP configuration round trip.
7. Version-matrix jobs for supported PyTorch, Transformers, and Accelerate
   combinations.

DeepSpeed gets an equivalent, independent matrix before its rejection is
removed.

## Delivery Sequence

1. Introduce the internal capability resolver and retain explicit rejection.
2. Implement and test FSDP full-state import/export collectives.
3. Add FSDP coordinated checkpoint/resume and a documented compatibility
   matrix.
4. Publish an FSDP example and operator memory-sizing guidance.
5. Implement DeepSpeed ZeRO in a separate feature sequence.
6. Remove the respective rejection only after its integration suite passes.

## Open Decisions

- Which PyTorch/Transformers/Accelerate version combinations form the initial
  supported FSDP matrix?
- Which FSDP state-dict mode and PEFT wrapping combinations are in the first
  release?
- What rank-0 CPU-memory admission policy should reject an unsafe full-model
  exchange before training starts?
- Is world-size-changing resume required for the first release?
