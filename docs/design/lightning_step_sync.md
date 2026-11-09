# Step-Level Sync for the Lightning Client API

Status: Implemented (PR #5700)
Author: Ruda Porto Filgueiras
Date: 2026-11-05

## Objective

Let one long `trainer.fit()` exchange the model with the server every N optimizer steps:

```python
import nvflare.client.lightning as flare

trainer = Trainer(strategy="ddp", devices=4, val_check_interval=10)
flare.patch(trainer, sync_every_n_steps=10)
trainer.fit(model, datamodule=dm)
```

| | Fit-per-round | N-steps-per-round |
| --- | --- | --- |
| One round is | One `trainer.fit()` call | N optimizer steps |
| Round starts | When `fit()` starts | After the new global model is loaded |
| Result sent | When `fit()` ends | After the Nth step |
| Next round | The script calls `fit()` again | Training keeps running; the callback loads the next global model |
| Round length | As many steps as the epochs need | N steps at every site |
| Training ends | When the script stops calling `fit()` | When the server stops sending training tasks |

At each sync, training pauses, sends the local model, loads the next global model in place and resumes with the
same optimizer, LR scheduler, AMP scaler and data position. Three hooks at the sync point let user code extend the
payload or react to a new global model without forking `FLCallback`.

Compared with today's only way to get short rounds, one `fit()` per round, this uses less host memory and less time
per round, and the same GPU memory (see [Memory](#memory)).

## Background

`FLCallback` (`nvflare/app_opt/lightning/api.py`) receives in `on_train_start` and sends in `on_train_end`, so one FL
round is one `fit()` call. The shortest natural round is one epoch.

For large models, one epoch is far too long a round. One workload that motivates this design trains a large model with
DDP on multi-GPU nodes: one epoch takes about 5 hours, and the clients sync every 10 optimizer steps, about every
11 minutes, which is some 25 rounds per epoch.

The only way to get short rounds today is `Trainer(max_steps=N)` with one `fit()` per round. Each `fit()` then:

| Effect | Consequence |
| --- | --- |
| Starts a new epoch from the first batch | Without shuffling, the same first N batches train every round and the rest of the data is never seen; with shuffling, epochs stop meaning a pass over the data |
| Rebuilds the optimizer and LR scheduler | State survives only through `RestoreState`, which keeps a host-memory `deepcopy` of all optimizer state for the whole job (for Adam, about twice the size of the weights) |
| Re-runs the per-fit setup | Dataloader workers, `setup`/`teardown`, and on GPU a round trip to CPU of the module (`pl_module.cpu()` before each send) and of the optimizer state (Lightning's teardown) |

Step-bounded rounds also hit a round-length bug (#5453), fixed in #5454.

## Goals

- `sync_every_n_steps` on `flare.patch()`; `None` keeps today's behaviour.
- Rounds of N optimizer steps inside one `fit()`, including across epoch boundaries and with gradient accumulation.
- No change to the optimizer, LR scheduler, AMP scaler, dataloader iterator or epoch counter at a sync.
- Validation configured on the trainer (`val_check_interval`, epoch end) keeps working and runs while the server
  aggregates.
- DDP: all ranks pause at the same step and hold identical weights after each sync.
- SCAFFOLD and FedProx work per sync as they work per round today.
- Three hooks at the sync point, in both modes.

## Non-Goals

- `train_with_evaluation` in step mode (see [`train_with_evaluation`](#train_with_evaluation)).
- Step sync across several `fit()` calls.
- Sharded strategies (FSDP, DeepSpeed); they are not supported by the patched path today either.
- Workload-specific behaviour: receive heartbeats, restart sentinels, payload dtype normalisation. The hooks let user
  code add these.

## Proposed API

```python
def patch(
    trainer: pl.Trainer,
    restore_state: bool = True,
    load_state_dict_strict: bool = True,
    update_fit_loop: bool = True,
    sync_every_n_steps: Optional[int] = None,
): ...
```

`sync_every_n_steps`:

- `None` (default): one round per `fit()`, as today.
- A positive `int`: a round ends every N optimizer steps (`trainer.global_step`), so gradient accumulation is handled
  and the count matches `NUM_STEPS_CURRENT_ROUND`. Anything else raises `ValueError` (`bool` included).

In step mode:

- The client script calls `fit()` once.
- The server's `num_rounds` decides how long training runs, as in fit-per-round mode. The trainer's `max_epochs`
  and `max_steps` do not bound it: step mode lifts both at `on_train_start`, epochs keep cycling the data, and `fit()`
  ends when a receive returns no training task. `update_fit_loop` has no effect.
- `restore_state` has no effect and `RestoreState` is not installed: the optimizer lives through the single `fit()`.
  With task lifetime each task runs the script again, so every round starts with a fresh optimizer; use job lifetime
  to keep optimizer state across rounds.
- `train_with_evaluation=True` in the job config raises `ValueError` when `patch()` runs.
- A site that stops on its own (`EarlyStopping`, `max_time`, `trainer.should_stop`) sends its open round, which may be
  shorter than N steps, and leaves. A round with no optimizer step yet is not sent: it would carry the received model
  unchanged, which FedAvg counts with weight 1 (it treats `NUM_STEPS_CURRENT_ROUND <= 0` as 1), and SCAFFOLD cannot
  compute a control update from zero steps. The callback logs a warning instead. The server then handles the missing site as it handles any client that stops
  responding.
- Every site reports `NUM_STEPS_CURRENT_ROUND = N`, so FedAvg weights sites equally per round, whatever their dataset
  size. Use `aggregation_weight` on the server for unequal weights.

## Sync Lifecycle

```text
fit() ─ on_train_start: rank 0 receives round 1 → broadcast (no training task → trainer.should_stop = True)
        ├─ on_train_batch_start: every rank loads the received model in place → start round
        ├─ batch … ─ on_train_batch_end: N steps since round start → finish round, rank 0 sends,
        │      then rank 0 receives the next task → broadcast
        │      (no model → trainer.should_stop = True; a non-train task → stop, leave it pending)
        │      (validation from val_check_interval / epoch end runs here, on the local model)
        ├─ on_train_batch_start: load the received model → start round
        ├─ …
        └─ on_train_end: send the open round, if the trainer stopped itself
```

Rules:

1. **Sync condition.** At `on_train_batch_end`, the round finishes when
   `trainer.global_step - round_start_step == sync_every_n_steps`. With gradient accumulation this first holds on the
   batch that ran the optimizer step. A batch that takes the round past N steps (manual optimization with several
   optimizer steps per batch) raises `RuntimeError`, because the round can no longer end at N; N must then be a
   multiple of the steps per batch.
2. **Receive at batch end, load at batch start.** The send and the next receive both happen at batch end, so the
   callback can stop with `trainer.should_stop` before another batch trains. The received model is loaded at the next
   batch start, so in-fit validation between the two still evaluates the local model. The cost: a validation that
   falls right after a sync waits for the server's aggregation instead of overlapping it, which adds up to the
   shorter of the two per such sync. Lightning gives a callback no hook between batch end and its stop check that
   says whether validation runs next, so the receive cannot be moved after it without copying Lightning's private
   scheduling logic.
3. **The server's horizon.** `on_train_start` sets `fit_loop.max_epochs` and `epoch_loop.max_steps` to `-1`, and
   clears `min_epochs` and `min_steps`, which would otherwise keep Lightning from honouring the stop, so only the
   server ends a normal run. `on_train_end` and `on_exception` put the four limits back, so a later `fit()` on the
   same trainer starts from the user's values. Optimizers and schedulers are configured before `on_train_start`, so a schedule
   sized from the trainer's original limits keeps that shape. Step-counting schedulers (warmup and decay,
   `StepLR`) are unaffected; a scheduler sized from the run length (`OneCycleLR`, cosine with `T_max` from
   `estimated_stepping_batches`) must be sized for `num_rounds * sync_every_n_steps`. The callback cannot fix
   this, since the scheduler exists before the first receive, so it records the trainer's step budget at
   `on_train_start` and logs one warning when training passes it. A parameter that keeps the trainer's limits
   was considered and rejected: it brings back sites that leave before the server's last round. When the
   trainer stops for its own reason (`EarlyStopping`, `max_time`), `on_train_end` sends the open round.
4. **No send without a receive.** NVFlare rejects a send that no receive preceded. A round is open only between a
   receive that returned a train task and the next send; `on_train_end` sends only an open round.
5. **Receive results.**
   - A train task: load in place, start the next round.
   - No model (the server finished, or the job stopped): set `trainer.should_stop`. The receive happens at batch end
     (or at `on_train_start` for the first round), so Lightning stops before the next batch; no round is open and no
     step is lost. As for any `should_stop` (`EarlyStopping`), Lightning then runs one validation pass if a
     validation dataloader is set and that batch was not already a validation batch. It evaluates the last local
     model and does not reach the server. Skipping it would mean changing `trainer.limit_val_batches` during
     `fit()`, which would also drop a scheduled validation on the same batch. Raising `StopIteration` from
     `on_train_batch_start`, used before, avoided that pass but relied on `TrainingEpochLoop.run` catching it,
     which is not a documented contract, and left the batch counters inconsistent.
   - A non-train task (evaluate, submit model): stop the same way, without loading it or clearing the Client API
     cache. The task stays pending, and the script can handle it after `fit()` with the usual loop:

     ```python
     trainer.fit(model, datamodule=dm)
     while flare.is_running():
         flare.receive()
         trainer.validate(model, datamodule=dm)
     ```
6. **In-place load.** The received weights go through the existing `_update_model` (key and shape checks, strict and
   non-strict loading) into the existing parameters with `load_state_dict`. No optimizer or loop state is touched.
7. **DDP.** All ranks evaluate the sync condition on the same `global_step`, so they reach the send and the receive
   at the same batch. Rank 0 sends and receives; the received model is broadcast with `trainer.strategy.broadcast`,
   as `FLCallback._receive_model` does today.
8. **Payload.** Rank 0 sends a CPU copy of `pl_module.state_dict()` (`v.detach().to("cpu", copy=True)`), because
   training continues on the device after the send. The module stays on its device. Fit-per-round mode keeps `pl_module.cpu()`, so existing users see no change in where the
   module lives after `fit()`.
9. **Algorithms.** Each sync runs the existing `_AlgorithmHandlerManager.finish_round` (SCAFFOLD control diff, step
   count) before the send, and each receive runs `start_round`.
10. **Metadata.** `__fl_meta__` is read at every send, as at the end of `fit()` today, with the same conflict check
    against algorithm metadata.

## Hooks

Custom training code often has to act at the sync point:

| Need | When |
| --- | --- |
| Add exponential moving average (EMA) weights or other state to the outgoing payload | before send |
| Log drift between local and global weights | after receive, before load |
| Take entries that are not module weights (EMA) out of the payload and load them | after receive, before load |
| Refresh the EMA from the new global weights | after load |

Three hooks, defined as methods on any `pl.Callback` in `trainer.callbacks`, called when present:

| Hook | Arguments | Called | May |
| --- | --- | --- | --- |
| `on_fl_before_send(trainer, pl_module, fl_model)` | `fl_model` is `None` on ranks other than 0 | before every send | edit `fl_model.params` / `meta` |
| `on_fl_receive(trainer, pl_module, fl_model)` | the broadcast model | after receive, before load, once per received model | edit `fl_model` before NVFlare loads it |
| `on_fl_model_loaded(trainer, pl_module, fl_model)` | the loaded model | after load | read the loaded module |

- Called on every rank in the same order, so a hook may use collectives.
- Called in both modes. In fit-per-round mode, `on_fl_before_send` also runs before the metrics-only send of an
  evaluate task, where `fl_model.params` is `None`.
- Not called for an empty receive or for a non-train task left pending in step mode.
- Called in `trainer.callbacks` order.
- Hooks get a copy of the received model, so edits change what is loaded, not the model the Client API keeps as
  the baseline for DIFF transfers. When a callback defines `on_fl_receive` or `on_fl_model_loaded`, `meta` is
  deep-copied, so in-place edits of nested values (such as SCAFFOLD's control tensors) are safe. `params` is copied
  one level deep, except that with DIFF transfers rank 0 clones the tensors, so in-place edits are safe too. That clone costs
  M host memory per sync and is only paid when a hook could change the DIFF baseline.
- A model received by `trainer.validate()` before `fit()` runs the receive hooks once; `fit()` reuses it.

```python
class EMASync(pl.Callback):
    def on_fl_before_send(self, trainer, pl_module, fl_model):
        if fl_model is not None:
            fl_model.meta["ema"] = {k: v.cpu() for k, v in pl_module.ema.state_dict().items()}

    def on_fl_receive(self, trainer, pl_module, fl_model):
        self._global_ema = fl_model.meta.pop("ema", None)

    def on_fl_model_loaded(self, trainer, pl_module, fl_model):
        if self._global_ema is not None:
            pl_module.ema.load_state_dict(self._global_ema)


trainer = Trainer(strategy="ddp", devices=4, val_check_interval=10, callbacks=[EMASync()])
flare.patch(trainer, sync_every_n_steps=10)
trainer.fit(model, datamodule=dm)
```

The hooks expose the same points that `_AlgorithmHandlerManager` already uses internally for SCAFFOLD and FedProx.

### Alternatives considered

| Option | Pros | Cons |
| --- | --- | --- |
| **A. Methods on trainer callbacks (chosen)** | The Lightning way to add behaviour; several callbacks compose; no new `patch()` arguments; same duck typing as `__fl_meta__` | A misspelled hook name is silently ignored |
| B. Callables passed to `patch(...)` | Explicit in the signature | One function per hook; several consumers must be combined by hand |
| C. A separate callback class per mode | Each mode keeps only its own hooks; no mode checks; `patch()` picks the class, so a conflicting existing callback can be rejected | `isinstance(cb, FLCallback)` is False for the step callback; no known external use |
| Two hooks (merge `on_fl_receive` into `on_fl_model_loaded`) | Smaller surface | No way to log drift or take entries out of the payload before the load |

## `train_with_evaluation`

With `train_with_evaluation`, the server expects each update to carry the metrics of the received global model on
local data and uses them for model selection. In fit-per-round mode they come from `trainer.validate()` before
`fit()`. Step mode has no such call: in-fit validation evaluates the local model, and `FLCallback` already keeps in-fit
metrics out of these metrics. Validating the global model after each receive would mean running Lightning's
validation loop from inside `fit()`, which `trainer.validate()` does not support, and would delay training by one
validation pass per sync. Step mode therefore raises `ValueError` for `train_with_evaluation=True`; a global-model
validation per sync can follow if there is demand.

## Code Structure

One class per mode, on a shared base:

```text
_FLCallbackBase      receive, load and send; algorithm handlers; metadata; step counting;
                     validation hooks; on_fl_* hooks
FLCallback           fit-per-round: on_train_start, on_train_end, _set_next_round_budget;
                     pre-existing constructor and behaviour
_StepSyncFLCallback  N steps: on_train_start lifts the trainer's limits, on_train_batch_start,
                     on_train_batch_end, on_train_end; rank-0 CPU copy
```

1. `FLCallback` keeps its name, export, constructor (`rank`, `load_state_dict_strict`, `update_fit_loop`) and
   fit-per-round behaviour; the shared code moved to the base unchanged. A test pins its constructor and methods.
2. `_StepSyncFLCallback` takes `sync_every_n_steps` and rejects `train_with_evaluation` in its constructor.
3. `patch()` picks the class from `sync_every_n_steps`, finds an existing callback through the base class, and raises
   when `sync_every_n_steps` is given and the existing callback is not a step callback with the same N. It installs
   `RestoreState` only for `FLCallback`.
4. `_params_to_send` picks the payload per class: `pl_module.cpu().state_dict()` for `FLCallback`, a rank-0 CPU copy
   for step mode, `None` on other ranks.
5. `_send` calls `on_fl_before_send` on every rank and sends from rank 0; the training send and the metrics-only send
   of an evaluate task both go through it. `_load_received_model` wraps `_update_model` with `on_fl_receive` and
   `on_fl_model_loaded`.
6. `_call_fl_hooks` calls a hook on each callback in `trainer.callbacks` that defines it.

## Memory

Compared with one `fit()` per round, step mode uses less host memory and less time per round, and the same GPU
memory:

- **Host memory:** no `RestoreState` copy of the optimizer state, which today is held per rank for the whole job.
  Adam keeps two state tensors per weight, so for Adam that copy is about twice the size of the weights, on every
  rank.
- **Time:** no move of the module and the optimizer state to CPU and back, and no per-`fit()` setup, at each round.
- **GPU memory:** the optimizer state is on the GPU during training in both modes. The temporary broadcast buffer
  described below exists in both modes, but step mode allocates it at every sync instead of once per `fit()`.

Per sync, with M the size of the state dict:

| Phase | Step mode | Fit-per-round mode, today |
| --- | --- | --- |
| Send | Rank 0: a CPU copy, +M host memory until the send returns; GPU unchanged; other ranks: nothing | `pl_module.cpu()` on every rank moves the module to host memory; the next `fit()` moves it back |
| Receive (rank 0) | The received model, +M host memory, held in the Client API cache until the round is sent | Same |
| Broadcast | `broadcast_object_list` pickles the model: about +M host memory on each rank; with NCCL, the pickled bytes are staged on each rank's GPU, about +M GPU memory for the duration of the broadcast | Same, once per round at `fit()` start |
| Load | `load_state_dict` copies into the existing parameters; no new GPU memory | Same |
| Optimizer | Lives through the single `fit()` on the training device; no copy | Lightning's teardown moves the optimizer state to CPU at every `fit()` end, where `RestoreState` keeps a `deepcopy` for the whole job (about 2× the weights for Adam), restored to the device at the next `fit()` |
| FedProx | Keeps the global weights on the training device during each round (+M GPU), as today | Same |

Payload extras added through `on_fl_before_send` (EMA weights) add their own size to the send.

The main risk is the temporary +M GPU memory of the broadcast: in step mode it happens every N steps, at a batch
boundary, with the optimizer state and the allocator's cached activation blocks resident. A callback with the same
broadcast every 10 steps runs in production on the workload above, so the existing broadcast is kept.

### Alternative: per-tensor broadcast

Rank 0 loads the received weights, then every parameter and buffer is broadcast in place with
`torch.distributed.broadcast`, as DDP syncs a module at start-up. This needs no pickling and no extra GPU memory, but:

1. It covers the module weights only. `meta`, `current_round` and the task flags still need the object broadcast, and
   for SCAFFOLD `meta` holds the global controls, about M again.
2. Ranks other than 0 would not see the payload, which breaks `on_fl_receive` on every rank.
3. The key and shape checks would run on rank 0 only. An error there leaves the other ranks blocked in the broadcast
   until the NCCL timeout, unless a status is broadcast first.
4. With `load_state_dict_strict=False`, the other ranks need the list of loaded keys.
5. It is only correct for replicated weights; `trainer.strategy.broadcast` works for any strategy.
6. One collective per tensor is slow for models with thousands of tensors; bucketing needs a private PyTorch helper
   (`_broadcast_coalesced`) or custom code.

It stays the documented fallback if the broadcast's GPU memory causes out-of-memory errors.

## Example

```python
trainer = Trainer(strategy="ddp", devices=2, val_check_interval=50)
flare.patch(trainer, sync_every_n_steps=50)
trainer.fit(model, datamodule=dm)
```

The Lightning section of `docs/user_guide/data_scientist_guide/client_api_usage.rst` documents the parameter, the
`num_rounds` rule and the hooks.

## Testing Plan

Unit tests in `tests/unit_test/app_opt/lightning/api_test.py`, with a real `pl.Trainer` on CPU and the Client API
mocked at the `receive`/`send` seam, as the existing `test_real_lightning_*` tests do. The `receive`/`send` mock
enforces NVFlare's exchange order: a send without a preceding receive fails.

| Expected outcome | Test |
| --- | --- |
| One `fit()` sends every N optimizer steps with `NUM_STEPS_CURRENT_ROUND == N`, including with gradient accumulation | count sends and step metadata |
| Each received model is what trains next | distinct weights per round, checked at the next batch |
| Optimizer, LR scheduler and data order carry over across syncs | same optimizer object and momentum buffers, `lr_scheduler.last_epoch`, uninterrupted sample sequence |
| The server decides how many rounds run, past the trainer's `max_epochs` and `max_steps` | rounds crossing epochs, epochs shorter than N, `max_steps` below the run |
| A trainer that stops itself sends its open round | `should_stop` mid-round |
| The server's end stops `fit()` even when the trainer sets `min_steps` or `min_epochs` | server finishing below both minimums |
| Passing the trainer's own step budget logs one scheduler warning | warning count past `max_epochs` |
| In-fit validation runs between send and receive | call order of send, validation and receive |
| The server finishing first stops training cleanly | `receive()` returns `None` |
| A non-train task mid-fit stops training and is handled by a later `trainer.validate()` | evaluate task after a sync |
| SCAFFOLD and FedProx produce per-sync metadata | existing algorithm fixtures in step mode |
| Hooks are called in order, once per received model, in both modes; edits take effect | recording callback; a hook that adds to `meta` and one that edits the received params |
| Invalid `sync_every_n_steps`, `train_with_evaluation` in step mode, and a conflicting existing callback are rejected | `ValueError` from `patch()` |
| Fit-per-round mode is unchanged | the existing tests; a test pins `FLCallback`'s constructor and methods |
| Under DDP, all ranks pause at the same step, hold identical weights after each sync, only rank 0 sends, no hang | 2-rank CPU gloo test in `tests/integration_test/fast/lightning_step_sync_ddp_test.py`, run serially by `ci/run_process_contracts.py` |

## Follow-ups

- A `--sync_every_n_steps` flag in `examples/advanced/multi-gpu/lightning/client.py`, and a simulator run that records
  peak GPU memory per sync.
- Global-model validation per sync, to support `train_with_evaluation` in step mode, if there is demand.
