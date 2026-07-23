# HuggingFace Trainer Client API

Status: Implemented (PR #4948)
Author: NVFlare Team
Date: 2026-07-23

## Objective

Provide a first-class Client API integration for the HuggingFace `transformers.Trainer`
(and subclasses such as `trl.SFTTrainer`), analogous to the existing PyTorch Lightning
integration (`nvflare.app_opt.lightning.patch()`), so that an existing HF training
script becomes federated with a one-line patch plus a round loop:

```python
import nvflare.client.hf as flare

trainer = SFTTrainer(model=model, args=train_args, ...)
flare.patch(trainer)

while flare.is_running():
    # no need to pass input_model to the trainer — patch() loads the global
    # model internally; this is for round info/logging, as in the Lightning examples
    input_model = flare.receive()
    if input_model:   # None on non-rank-0 ranks
        print(f"round={input_model.current_round}")

    trainer.evaluate()   # optional: global-model metrics for server-side model selection
    trainer.train()
```

The public surface is the standard Client API plus `patch()` — no new user-facing
concepts. Everything `patch()` installs (wrappers, callbacks, task state) is
private implementation.

## Background

Today, HF users write the raw Client API loop by hand. The reference is
`examples/advanced/llm_hf/client.py`, which contains ~150 lines of FL plumbing that
every HF user must reinvent:

1. `flare.init(rank=...)`, rank-0-only `receive()`/`send()`, manual
   `dist.broadcast_object_list` of round number and params to other ranks.
2. Key-name remapping between the server model definition (`model.` prefix) and the
   local model's `state_dict()` keys.
3. PEFT vs full-model branching (`get/set_peft_model_state_dict` vs
   `load_state_dict`/`state_dict`).
4. Per-round training on a Trainer designed for one-shot runs: a `StopCallback`
   that sets `control.should_training_stop` at epoch end, `num_train_epochs`
   pre-multiplied by `num_rounds`, and `resume_from_checkpoint=True` with global
   weights manually written into the last checkpoint directory to preserve
   optimizer/LR-scheduler state across rounds.
5. Global-model evaluation before local training, metrics packaging.
6. dtype handling (`bf16 -> fp32` cast when exchanging via numpy).
7. `NUM_STEPS_CURRENT_ROUND` meta for weighted aggregation.

The Lightning integration (`nvflare/app_opt/lightning/api.py`) already solved the
same problems for `pl.Trainer` via an `FLCallback` + `patch()`. This proposal ports
that design to HF, with HF-specific handling where the two trainers differ.

## Goals

- `flare.patch(trainer)` for `transformers.Trainer` and subclasses (`Seq2SeqTrainer`,
  `trl.SFTTrainer`).
- Support task types `train`, `evaluate`, and `submit_model` (same as Lightning).
- Full fine-tuning (SFT) and PEFT/LoRA (send/receive adapter weights only).
- Optimizer and LR-scheduler state preserved across rounds by default.
- Multi-GPU (DDP via `torchrun`) and multi-node: rank-0-only FLARE interaction,
  transparent broadcast to other ranks.
- Incoming-params validation with the same strictness contract as Lightning
  (`inspect_model_params`: shape-mismatch and zero-match fail fast; unexpected keys
  fail in strict mode, filter-with-warning in non-strict mode).
- Works in both in-process and subprocess (`ScriptRunner` / launcher executor)
  execution modes — it is built purely on the existing Client API.

## Non-Goals

- No new server-side workflow or controller. Existing FedAvg/recipes work unchanged.
- No change to FL algorithms.
- DeepSpeed ZeRO-3 and FSDP full support is phased (see Phasing); Phase 1 targets
  single-GPU and DDP.
- `Trainer`-free HF training loops (raw `accelerate` loops) — those users keep the
  raw Client API.

## Package Layout

```
nvflare/app_opt/hf/
    __init__.py          # exports patch, FLCallback
    api.py               # patch()
    callbacks.py         # FLCallback, FLMetricsCallback (transformers.TrainerCallback)
    utils.py             # state-dict extraction/load helpers (PEFT, accelerator)
nvflare/client/hf/
    __init__.py          # re-exports patch + client API, plus rank-aware is_running()
                         # (not a pure re-export — see Task Dispatch section)
tests/unit_test/app_opt/hf/
```

Naming: `hf` (matches `examples/advanced/llm_hf`, avoids any ambiguity with the
`transformers` import name). `transformers`, `peft`, and `safetensors` are optional
dependencies, imported lazily inside `nvflare/app_opt/hf` (same policy as
`app_opt/lightning` w.r.t. `pytorch_lightning`); `peft` is only required when a
`PeftModel` is detected. `packaging` is an optional app-opt dependency used only
for the `transformers` version gate. If it is absent, the adapter conservatively
uses checkpoint injection instead of the in-memory restore strategy. `trl` is a
**test-only** dependency (min pin `trl >= 0.18`, in-constructor PEFT wrapping):
the API depends solely on `transformers` callback/trainer interfaces, and
`SFTTrainer` support is verified by the integration test, not by code paths.

## Proposed API

```python
def patch(
    trainer: transformers.Trainer,
    restore_state: bool = True,         # continue optimizer/LR-scheduler/RNG state across rounds;
                                        # model weights ALWAYS come from the received global model
    load_state_dict_strict: bool = True,
    params_scope: str = "auto",         # "auto" | "model" | "adapter" — which weights to exchange:
                                        # full model state dict vs PEFT adapter only
    server_key_prefix: str | None = None,  # e.g. "model." — strip on receive, add on send
    local_epochs: float | None = None,  # per-round budget; default captured at first train():
                                        # args.max_steps if set, else args.num_train_epochs
    local_steps: int | None = None,     # alternative per-round budget in optimizer steps
    stream_metrics: bool = False,       # forward HF on_log scalars to server-side experiment tracking
):
```

Budget semantics: when neither `local_epochs` nor `local_steps` is given, the
per-round budget is captured from the trainer at the first `train()` call —
`args.max_steps` if set (taking precedence, matching HF's own semantics), else
`args.num_train_epochs`. This preserves the user's mental model with zero config:
"my script trains 1 epoch" becomes "1 epoch per round", same as Lightning and the
example's `local_epoch`. Passing **both** `local_epochs` and `local_steps`
explicitly is an error, not a warning — silent precedence between two explicit
budgets trains the wrong amount quietly.

Two naming clarifications:

- `params_scope` is deliberately **not** named `params_type`: `FLModel.params_type`
  already means `FULL` vs `DIFF` (full weights vs delta update), and that axis is
  orthogonal — it is already handled by the existing Client API config
  (`params_transfer_type` on `ScriptRunner` / client config; the send path computes
  the diff automatically). This API adds nothing there. `params_scope` selects
  *which* weights are exchanged: the full model state dict (`"model"`) or the PEFT
  adapter weights only (`"adapter"`), i.e. the SFT-vs-PEFT branch of the `llm_hf`
  example. `"auto"` picks `"adapter"` when `trainer.model` is a `PeftModel`.
  Edge cases are explicit: `"adapter"` on a non-`PeftModel` is a `patch()`-time
  error; `"model"` on a `PeftModel` is allowed but logs a WARNING — the exchanged
  keyspace is then the PEFT-wrapped `state_dict()` (`base_model.model.*` + adapter
  keys), which the server model definition must match; most PEFT users want
  `"auto"`.
- `restore_state` restores **training progress state only**: optimizer moments,
  LR-scheduler position, RNG state, `global_step`. It never determines model
  weights — those are always overridden by the received global model each round.
  The name matches the Lightning API's `restore_state` (documented there as
  "restore optimizer and learning rate scheduler states"). It defaults to `True`
  deliberately — parity with Lightning and with the `llm_hf` flow — and the
  default's costs are bounded and enforced rather than implicit: storage is
  capped (`save_total_limit=2`), static incompatible settings
  (`save_only_model`, `load_best_model_at_end`) are rejected at `patch()` time,
  and dataloader-dependent budget errors are rejected at the first `train()`
  call when the real train dataloader is available.

`patch()` does three things:

1. Calls `flare.init(rank=<resolved rank>)` if not already initialized
   (idempotent), exactly as Lightning's `FLCallback.__init__` does with
   `trainer.global_rank`. The FL control rank is resolved defensively, in order:
   `torch.distributed.get_rank()` if the process group is initialized, else the
   `RANK` env var, else `trainer.args.process_index`, else `0`. **Never
   `LOCAL_RANK`** — it is node-local, and using it would make one process per
   *node* believe it owns the FLARE pipe. `patch()` must be called after
   `Trainer` construction — already the natural pattern — which under `torchrun`
   also guarantees `torch.distributed` is initialized before the broadcast
   machinery is set up. "Idempotent" is precise: if the user already called
   `flare.init()` with the **same** rank, `patch()` reuses that context; if the
   ranks differ, `patch()` raises rather than silently re-initializing (the
   Client API reuses a context only for the same rank/config pair — a silent
   re-init would split control ownership).
2. Registers an `FLCallback(transformers.TrainerCallback)` on the trainer
   (idempotent, like Lightning: skipped if already present).
3. Wraps **both** `trainer.train` and `trainer.evaluate`. This intentionally goes
   beyond Lightning's callbacks-only patch, for two HF-specific reasons:
   - `Trainer.train()` rebuilds the optimizer and scheduler on every call unless
     resuming from a checkpoint, so round-to-round state continuity cannot be
     achieved from callbacks alone (see Round-Loop Semantics).
   - HF has **no pre-evaluation callback hook**: `on_evaluate` fires *after* the
     eval loop, with the metrics (there is no `on_validation_start` equivalent).
     Receiving and loading the global model before evaluation therefore must
     happen in the `evaluate()` wrapper, not in a callback. Callbacks are used
     only where HF's hook timing is correct: `on_train_begin` (re-apply global
     weights after checkpoint restore), budget stop, `on_train_end` (send),
     `on_evaluate` (capture metrics).

   The wrappers share a small per-round task-state object (working name
   `_HFTaskState`) that owns task receipt and dispatch (see Task Dispatch below).
   This is **private implementation, not public API** — the doc refers to it as
   "the session" for brevity, but no `Session`/`Dispatcher` class is exported and
   the Client API spec is unchanged.

`FLCallback` hooks:

| Hook | Action |
|---|---|
| `on_train_begin` | (re)apply the round's received global params to the model — fires after HF's checkpoint restore, so global weights win (see Round-Loop Semantics); the params themselves were already received by the session in the wrapper |
| `on_epoch_end` / `on_step_end` | stop training when the per-round budget (`local_epochs`/`local_steps`) is reached (`control.should_training_stop = True`) |
| `on_train_end` | extract params, package `FLModel` (params, metrics, meta), send (rank 0) |
| `on_evaluate` | capture `metrics` **only when fired by the wrapped pre-train `evaluate()`**; firings triggered mid-training by `eval_strategy="steps"/"epoch"` are excluded from `FLModel.metrics` (they reflect a partially-trained model) and feed metric streaming only. This hook fires *after* eval and cannot load weights |

User-visible round loop is identical to Lightning, including the optional
`receive()` for round info:

```python
while flare.is_running():
    input_model = flare.receive()   # round info only — patch() loads the model internally
    trainer.evaluate()   # optional — evaluates the freshly received global model
    trainer.train()
```

**`flare.receive()` ownership:** the wrappers never depend on the user calling
`receive()` — they consume the task the client API already cached when rank-0
`is_running()` performed its internal blocking receive. The explicit `receive()`
in the loop reads that same cache (idempotent — never double-consumes) and exists
for round info/logging, exactly as in the Lightning examples. Caveat carried over
from the raw Client API: it returns the cached task on rank 0 and `None` on all
other ranks — multi-rank scripts guard with `if input_model:` as the multi-gpu
Lightning example does.

Note for implementers: "consume the cache" means the wrappers call the **public
`flare.receive()`** on rank 0 — not a private cache read — because `send()` gates
on the API's `receive_called` state (both in-process and ex-process
implementations enforce it).

**When is `trainer.evaluate()` required?** "Optional" in the loop means optional
for plain training; the full contract is:

- train task, `train_with_evaluation=False`: `trainer.train()` alone is enough.
- train task, `train_with_evaluation=True`: the user must call
  `trainer.evaluate()` before `trainer.train()` — otherwise `on_train_end` fails
  with the same actionable error as Lightning ("missing training metrics, please
  remember to call evaluate").
- `evaluate` task: the loop must call `trainer.evaluate()`; a loop that only
  calls `train()` hits the fail-fast pending-task error (see Failure semantics).

## Round-Loop Semantics (the crux)

HF `Trainer.train()` is one-shot: each call builds a fresh optimizer/scheduler and
counts epochs from zero, unless `resume_from_checkpoint` is given.

What `resume_from_checkpoint` buys is **not** model weights — it is the only HF
mechanism to carry optimizer state (Adam moments), LR-scheduler position, RNG
state, and `global_step` into the next `train()` call. The checkpoint's model
weights are loaded by HF as part of resume and then immediately overridden by the
received global model; that redundant weight load is expected and is exactly what
the hand-written `llm_hf` example does today (it writes the global weights *into*
the checkpoint directory so that resume loads global weights together with local
optimizer state).

**Checkpoint provenance:** the session resumes only from checkpoints **recorded
for this FL job** — at each round end it stores the path of the checkpoint just
written, and round r > 0 resumes from that record. It never scans the directory
(no blind `get_last_checkpoint()`): a stale `checkpoint-*` left in `output_dir` by
a previous run can therefore never be resumed. Round 0 never resumes, regardless
of directory contents. This replaces the example's `rmtree`-on-startup hygiene;
the session logs a WARNING at startup if stale checkpoint dirs are present.

**The provenance record is persisted, not in-memory:** the session writes a small
state file (`<output_dir>/_fl_exchange/fl_state.json`, atomic
temp-write + `os.replace`) at each round end containing the FL job ID, world
size, last completed round, the recorded checkpoint path, the computed cumulative
`max_steps` target, and the per-round budget. **Single-writer rule:** only rank 0
ever writes it; at startup all ranks pass a barrier after rank 0 has read (or
created) it, so a relaunch cannot race a partially-written record and no two
processes ever write concurrently. On startup, a session that finds a state file for the *same*
job ID restores it — so a mid-job script relaunch (client crash + rejoin, or
`launch_once=False` relaunch-per-task) resumes correctly instead of silently
degrading to stateless rounds with an undefined target. A state file from a
*different* job ID is treated as stale (WARNING, ignored). This also removes any
hard dependency on `launch_once=True`, though it remains the recommended default.

Two modes:

### `restore_state=True` (default) — continuous training

- Round 0: patched `train()` runs normally with the per-round budget; the callback
  stops it at the budget boundary; HF's `save_strategy` produces a checkpoint (the
  patch forces at least one checkpoint save at round end if the user's
  `save_strategy="no"`, into `output_dir`). When `restore_state=True` and the user
  set no `save_total_limit`, it defaults to `2` (logged at INFO) — resume needs ≥1
  checkpoint, 2 protects against a corrupted latest write, and unbounded
  checkpoints of a multi-GB model over many rounds fill disks. A user-set value is
  never overridden.
- Round r > 0: patched `train()` injects
  `resume_from_checkpoint=<session-recorded checkpoint from round r-1>` so HF
  restores optimizer, LR scheduler, RNG, and `global_step`.
- Cumulative training target: at round 0 the session sets `args.max_steps` **once**
  to `per_round_budget_steps × total_rounds` (available from `FLModel.total_rounds`
  meta), and each round ends via the budget-stop callback. Ordering constraint:
  the per-round budget may itself be captured *from* `args.max_steps`, which this
  write then overwrites — the capture must happen strictly before the cumulative
  target is written, or the budget silently becomes the whole-run total. When the
  budget is in epochs, it is converted at round 0 as
  `steps_per_epoch = ceil(len(train_dataloader) / gradient_accumulation_steps)`
  from the real dataloader (so world size and `drop_last` are already accounted
  for). A length-less `IterableDataset` cannot be converted: the fixed-total mode
  then requires an explicit `local_steps`, otherwise the first `train()` call
  errors with that instruction. An empty train dataloader is also rejected with
  a data-preparation hint, because a zero-step local budget would silently send
  unchanged weights. If the server schedules more train rounds than `total_rounds`
  (resume would start with `global_step ≥ max_steps`, and HF would train zero
  steps yet still send unchanged weights), the session extends the target by one
  budget and logs a WARNING that the schedule tail deviates from the planned
  curve — never silently contributes a zero-step round. This is preferred over
  bumping `num_train_epochs`/`max_steps` every round (Lightning's
  `update_fit_loop` analog) because HF *recreates* the LR scheduler from the
  current totals on each `train()` call and `LambdaLR`-family schedulers do not
  serialize their lambda — only `last_epoch` — so per-round bumping silently
  re-derives decay/warmup schedules from a different total each round. The
  per-round bump remains as fallback when `total_rounds` is not provided by the
  workflow, documented as safe only for `constant` schedules.
- Global weights override checkpoint weights via `on_train_begin`, which HF fires
  *after* checkpoint restore and before the epoch loop, so the received global
  params win over the checkpointed local ones. This in-memory override is the
  **primary** strategy: it is checkpoint-format-agnostic (no coupling to
  safetensors/bin/sharded layouts) and avoids a full-model disk write per round.
  The proven alternative (used by `llm_hf/client.py` today) — writing the global
  weights into the last checkpoint directory before resume — is kept as an
  **automatic fallback**: Phase 0 produces a *verified version range*, and when
  `patch()` creates task state, any `transformers` version outside that range
  conservatively selects the fallback (a version check cannot detect a regression
  — it can only trust what was verified; a config/env override exists for users
  who have validated a newer version themselves). If the optional `packaging`
  dependency needed for version parsing is unavailable, the same conservative
  fallback is used. The fallback is also expected to become the preferred path
  under ZeRO-3/FSDP in Phase 2, since it rides HF's own sharded-checkpoint load
  machinery. The Phase 0 spike pins the supported version range; current
  `transformers` restores optimizer/scheduler in `_inner_training_loop` before
  firing `on_train_begin`. When the injection fallback is active, the implementation
  loads the received params into the in-memory model, saves that model through
  HF/PEFT-compatible checkpoint writers, and then resumes from the checkpoint.
  The `on_train_begin` re-apply is **skipped**: the resumed checkpoint already
  contains the global weights, and a redundant full-model `load_state_dict`
  costs real time on large models.

### Forced checkpoint: contents and conflicting `TrainingArguments`

The round-end save forced by `restore_state=True` is a **standard, full HF
checkpoint** (the same `_save_checkpoint` path as `save_strategy`): model weights
— or adapter weights only, for PEFT models, via HF's PEFT save integration —
optimizer state, LR-scheduler state, RNG states, and `TrainerState`
(`global_step`), in HF's default formats (safetensors for weights). Nothing
custom: resume must be able to consume it with stock `transformers`.

Mechanism: the budget-stop callback sets `control.should_save = True` together
with `control.should_training_stop` — the public `TrainerControl` path, the same
one `save_strategy` uses — rather than calling the private
`trainer._save_checkpoint()`. Phase 0 must verify this produces **each required
component at the stop boundary**: optimizer state, scheduler state, RNG states,
`TrainerState`, and model weights — including the PEFT adapter-save integration
for `PeftModel`s. If any component is missing on any supported version, the
design needs an agreed fallback (accepting the private `_save_checkpoint` call
and its drift risk) **before** implementation starts — not a workaround invented
mid-PR.

Conflicting settings are rejected at `patch()` time with actionable errors rather
than silently misbehaving:

- `save_only_model=True` + `restore_state=True`: incompatible — the checkpoint
  would carry no optimizer/scheduler state, making resume a silent
  `restore_state=False`. Error tells the user to pick one.
- `load_best_model_at_end=True`: incompatible with FL train tasks in either mode —
  HF would swap in the *best* checkpoint at the end of `train()`, so
  `on_train_end` would send weights that are not this round's training result.
  Error directs the user to server-side model selection (which the pre-train
  `evaluate()` metrics already drive).

### `restore_state=False` — stateless rounds

Fresh optimizer/scheduler each round; each `train()` call simply runs exactly the
per-round budget (epochs or steps) from the received global weights, with no
resume and no cumulative target. Simpler, matches FedAvg-with-restart semantics
some papers assume; loses LR continuity.

### LR schedule across rounds

With the fixed cumulative target (`max_steps` set once at round 0, above),
decay/cosine/warmup schedules keep a consistent shape across the whole federated
run: the scheduler is recreated from the *same* total every round, and the
restored `last_epoch` places it correctly on that curve. Users who need to size
warmup or a custom schedule themselves get a documented helper
(`nvflare.app_opt.hf.utils.total_train_steps(dataset_len, args, total_rounds)`)
rather than magic. `constant` works out of the box in every mode, including the
per-round-bump fallback.

## Receive Path

1. Rank 0 calls `receive()`; task type flags (`is_train/is_evaluate/is_submit_model`)
   and the `FLModel` are broadcast to all ranks via `torch.distributed`
   (`broadcast_object_list`), matching the hand-written example. Non-rank-0 processes
   never touch the FLARE pipe.
2. Optional `server_key_prefix` stripped from incoming keys.
3. Validation via `nvflare.app_opt.pt.utils.inspect_model_params` — identical
   contract and error messages as Lightning (`FLCallback._receive_and_update_model`),
   but against the **scope-appropriate reference keyspace**: for
   `params_scope="adapter"` the reference is `get_peft_model_state_dict(unwrapped)`
   keys, *not* `model.state_dict()` — the two differ (PEFT strips the adapter name,
   e.g. `...lora_A.weight` vs `...lora_A.default.weight`), so a literal Lightning
   port would fail every PEFT round with zero matched keys.
4. Load:
   - `PeftModel` (or `params_scope="adapter"`): `peft.set_peft_model_state_dict`.
   - otherwise: `model.load_state_dict(..., strict=load_state_dict_strict)`.
   - All detection and load/extract operate on the **unwrapped** model: a
     `utils.unwrap_model(trainer)` helper based on
     `trainer.accelerator.unwrap_model()`. HF Trainer nominally keeps the raw
     model at `trainer.model` (the DDP/DeepSpeed-wrapped one at
     `trainer.model_wrapped`), but unwrapping defensively costs nothing and
     survives backend relocation. PEFT detection runs against the unwrapped model
     *after* the trainer's own PEFT wrapping (TRL ≥ 0.18 wraps inside
     `SFTTrainer.__init__`), so `patch()` after trainer construction sees the
     final model type. Wrapped-model handling is part of the Phase 1 DDP tests.

## Send Path

1. Extraction on rank 0 at `on_train_end`, always from the **unwrapped** model via
   the same `utils.unwrap_model(trainer)` helper as the receive path (never raw
   `trainer.model` — send and receive must agree on the key space):
   - PEFT: `peft.get_peft_model_state_dict(unwrapped)` (adapter-only — tiny
     payload, the main reason LLM users want this API).
   - Full: `trainer.accelerator.get_state_dict(unwrapped)` rather than raw
     `state_dict()` — this is the seam that later makes ZeRO-3/FSDP gathering work
     without API changes; for single-GPU/DDP it degenerates to a plain state dict.
2. Tensors moved to CPU. If the configured exchange format is numpy, cast
   bf16/fp16 → fp32 automatically (numpy has no bf16); in tensor mode, send native
   dtype (halves payload for bf16 models). `TensorDecomposer` registered via
   `fobs.register` in `patch()`, as Lightning does.
3. Optional `server_key_prefix` re-applied.
4. Meta: `NUM_STEPS_CURRENT_ROUND` is the **effective aggregation weight**. The
   default is samples processed this round, **estimated** as
   `global_step delta × per_device_train_batch_size × gradient_accumulation_steps × world_size`
   (an estimate: the final partial batch and dataloader `drop_last` make it
   slightly high). Optimizer-step count alone would under-weight sites with larger
   world size or batch size; raw dataset-row counts (the example's approach)
   mis-weight partial epochs and step-budgeted rounds. When an exact count is
   available it is preferred: with `include_num_input_tokens_seen=True`, the delta
   of `state.num_input_tokens_seen` gives exact **token-based** weighting — often
   the right unit for LLM fine-tuning with variable-length sequences. Users with
   different weighting needs override via the same `__fl_meta__` attribute
   convention as Lightning, which also carries arbitrary extra meta. Lookup
   location is exactly one place: the **unwrapped model**
   (`utils.unwrap_model(trainer)`) — set it before training starts; attributes
   on the wrapped/DDP model or on the trainer itself are not consulted (PEFT/DDP
   wrapping makes "on the model" ambiguous unless pinned).
5. Metrics: only what the **wrapped pre-train `evaluate()`** captured this round
   (e.g. `eval_loss` of the global model) rides on the outgoing `FLModel` when
   `train_with_evaluation` is configured — same contract and same "call evaluate()
   or fail with an actionable error" rule as Lightning. Mid-training evaluations
   triggered by `eval_strategy` never populate `FLModel.metrics` (see the
   `on_evaluate` hook rule).

## Task Types and Task Dispatch

The user loop stays `while flare.is_running(): trainer.evaluate(); trainer.train()`,
but dispatch is an explicit state machine in the session shared by the two wrappers —
not callback-side suppression:

1. Within a loop iteration, the **first wrapped call** (`evaluate()` or `train()`)
   makes the session receive the task: rank 0 calls `receive()` and the task type +
   params are distributed to all ranks (see Distributed section). The session holds
   the received params for the rest of the round.
2. Dispatch by task type:
   - **`train`**: `evaluate()` loads the global params, runs eval, captures metrics.
     `train()` runs local training (with `on_train_begin` re-applying the global
     params after checkpoint restore) and `on_train_end` sends params + metrics,
     completing the task.
   - **`evaluate`**: `evaluate()` loads the global params, runs eval, sends a
     metrics-only `FLModel`, and marks the task complete. A subsequent `train()`
     in the same iteration is a **no-op** (returns immediately, logged at INFO).
   - **`submit_model`**: the first wrapped call completes it without running eval
     or training; the other wrapped call in the iteration is a no-op. **What is
     sent is defined by provenance, not by which wrapper fired**: the session
     loads and sends the params (scope-extracted) from its recorded round-end
     checkpoint — the authoritative "latest local result", correct even on a
     fresh relaunch. Only when no recorded checkpoint exists (no train round has
     completed on this site) does it fall back to the current in-memory model,
     with a WARNING stating the payload is the untrained/initial weights.
3. Task completion resets the session for the next loop iteration. No-op calls
   leave all session accounting untouched — in particular, a skipped `train()`
   (for an `evaluate`/`submit_model` task) does not advance the cumulative
   training target or round counter.
4. **Train-internal evaluation passes through.** With `eval_strategy="steps"/
   "epoch"`, HF calls `self.evaluate()` from *inside* `train()` — through the
   wrapper. The train wrapper sets an in-train flag **in `try/finally`** (an
   exception mid-train must clear it, or the poisoned flag makes every later
   user `evaluate()` a silent passthrough); while it is set, the
   evaluate wrapper is a pure passthrough: no task receive, **no global-param
   load** (it would clobber the weights being trained mid-round), no FL send.
   Its metrics feed metric streaming only, consistent with the `on_evaluate`
   rule. Only a user-called `evaluate()` outside `train()` participates in task
   dispatch.

This makes the same user script serve all three task types deterministically:
each wrapped call either executes for the current task, or is a logged no-op.

### Failure semantics

The state machine includes failure transitions, not just the happy path:

- **Exception inside a wrapped call** (training crash, load failure, validation
  error): the session marks the round aborted and **re-raises** — it never
  swallows. The script exits nonzero; the existing executor lifecycle reports the
  task failed (subprocess mode: launcher detects process exit; in-process mode:
  the executor surfaces the exception), and under `torchrun` the elastic agent
  tears down the remaining ranks, so no rank is left hanging in a collective.
- **`receive()` returns `None` / job ended:** `is_running()` returns `False` on
  all ranks (rank-0 result is broadcast) and the loop exits cleanly.
- **Task received but never completed** (user script never makes the wrapped call
  the task needs — e.g. an `evaluate` task but the loop only calls `train()`,
  which no-ops): fail fast, not spin. Because the cached task makes `is_running()`
  return instantly, an ERROR-and-continue policy would busy-loop at full CPU until
  the server-side task timeout, leaving an undefined stale cache behind. Instead,
  the rule is stated in terms `is_running()` can actually observe (it sees calls,
  not the user's loop boundary): **if `is_running()` is called while a previously
  received task is still pending — received but not completed — it raises**
  `RuntimeError` naming the task type and the missing wrapped call. The script exits; the exception path above reports the
  task failure through the normal executor lifecycle.
- **Wrapped call with no task available** (job ended; e.g. a trailing
  `trainer.evaluate()` after the loop, or a loop not guarded by `is_running()`):
  rank 0's receive yields `None`; the session broadcasts a stop sentinel in the
  dispatch slot (so no rank hangs in a collective) and the wrapped call becomes a
  logged no-op returning HF's normal "nothing to do" result.

### `is_running()` under multiple ranks

`nvflare.client.hf` is **not** a pure re-export: it provides a rank-aware
`is_running()` in which only rank 0 queries the FLARE pipe and the boolean is
broadcast to all ranks via `torch.distributed` when initialized. Non-zero ranks
never touch the pipe (today's hand-written loops must broadcast rank-0 running
state themselves — e.g. `examples/advanced/multi-gpu/lightning/client.py`; the HF
API absorbs that). Single-process runs degenerate to the plain client API call.
This broadcast behavior is an **HF-specific convenience** documented as such; the
core Client API `is_running()` spec is unchanged. Two scoping rules make it
well-defined: **one patched trainer per process** (a second `patch()` on a
different trainer in the same process raises — the session, wrappers, and rank
context are process-global); and before any `patch()` call,
`nvflare.client.hf.is_running()` simply delegates to the standard Client API
(no broadcast — there is no distributed context to consult yet).

Implementation fact this design leans on: the client API's `is_running()`
internally performs the blocking `receive()` and caches the result — the wait for
the next task happens *inside* rank 0's `is_running()`, and the session's "first
wrapped call receives" then consumes the cache instantly.

### Per-iteration collective sequence

Every rank must execute the same collectives in the same order, or the process
group hangs (the classic NCCL/Gloo failure, undebuggable for users). The session
owns this sequence; per loop iteration it is exactly:

1. `is_running()` boolean broadcast (rank 0 has already done the blocking receive).
2. Task dispatch broadcast: task type + params payload — either the object itself
   or, above the size threshold, the exchange-file path.
3. If file exchange was used: post-write barrier (readers released), then
   post-load cleanup barrier.
4. Training/eval collectives — owned entirely by HF/torch, outside the session.
5. No session collective after `send()` — rank 0 sends to the pipe alone.
   (Phase 2 forward note: under ZeRO-3/FSDP, `accelerator.get_state_dict()` is
   itself a collective gather that *all* ranks must enter, so the sequence gains a
   gather step between training and send — this section will be amended then;
   items 1–4 are unchanged.)

User code must not introduce rank-conditional wrapped calls (e.g. calling
`trainer.evaluate()` only on rank 0): the wrappers run their collectives on all
ranks. This constraint is documented in the API docs and asserted where cheap
(the session detects a dispatch-broadcast mismatch on rank 0 vs others by
including the call name in the broadcast payload and raising on divergence).

## Metrics and Experiment Tracking

Two distinct metric flows; the design supports both:

1. **Per-round metrics on `FLModel.metrics`** (already covered above): one value
   set per round (e.g. `eval_loss` of the global model), consumed server-side for
   model selection and reporting. This is FL-protocol data, not experiment
   tracking.
2. **Fine-grained training curves** (loss/accuracy/LR per logging step). Two
   coexisting paths:
   - **Site-local (unchanged):** HF's own `report_to="tensorboard" | "mlflow" |
     "wandb"` integrations keep working untouched — the FL callbacks do not
     interfere. Logs stay at the site, as in the `llm_hf` example today.
   - **Federated streaming (opt-in):** `patch(trainer, stream_metrics=True)`
     registers an additional `FLMetricsCallback` hooking HF's `on_log` — the
     single funnel through which the Trainer emits every scalar log (train loss,
     learning rate, `eval_*`). Scalars are forwarded via the existing
     `nvflare.client.tracking` writers (`SummaryWriter` by default;
     `MLflowWriter`/`WandBWriter` selectable) through the FL pipe to the
     server-side receivers configured in the job (`TBAnalyticsReceiver`,
     `MLflowReceiver`, `WandBReceiver`). `FLMetricsCallback` gates on the
     **NVFlare rank explicitly** (the rank `flare.init()` was given) before
     touching any tracking writer — it does not rely on HF's own world-zero
     `on_log` gating, which is a logging behavior, not a contract. It also
     forwards **finite scalars only**: non-numeric entries in the `logs` dict
     (e.g. `epoch` strings from some integrations) are dropped, and NaN/inf
     values are skipped with a debug log rather than streamed. No new
     infrastructure: this is the same metric-streaming layer the raw Client API
     and Lightning use.

Properties worth stating:

- **Client writer and server backend are decoupled** — the writers convert to
  NVFlare analytics events; the job-side receiver decides whether they land in
  TensorBoard, MLflow, or W&B on the server. The training script does not change
  when the server backend changes.
- **Sites need no direct network path to a tracking server** — metrics ride the
  existing FL channel (firewall/privacy friendly); the aggregated cross-site view
  lives at the FL server.
- **X-axis continuity:** the step axis is `state.global_step`. With
  `restore_state=True` it is monotonic across rounds for free, so each site's
  curve is continuous over the whole federated run. With `restore_state=False`,
  `FLMetricsCallback` offsets by the cumulative step count to keep the axis
  monotonic.

## Distributed / Parallelism Matrix

| Backend | Phase | Notes |
|---|---|---|
| Single GPU / CPU | 1 | trivial |
| DDP (`torchrun`, multi-node) | 1 | rank via the resolution order in `patch()` step 1 (never `LOCAL_RANK`); rank-0 pipe + object broadcast, as in `llm_hf` today |
| DeepSpeed ZeRO-1/2 | 2 | extraction already correct via `accelerator.get_state_dict`; needs testing |
| DeepSpeed ZeRO-3 | 2 | `get_state_dict` gathers full params on rank 0 — peak-memory risk for large models; document, consider shared-FS exchange (below) |
| FSDP | 2 | full-state-dict gathering via accelerator; same memory caveat |

Phase 1 **rejects** unsupported backends explicitly rather than leaving them
"untested but maybe works": `patch()` raises at patch time when the trainer is
configured with DeepSpeed (`args.deepspeed`) or FSDP (`args.fsdp`), with a
message naming Phase 2. A backend that would fail subtly mid-round (collective
mismatch, sharded-load corruption) must fail loudly at setup instead.

For multi-node SFT of large models, `broadcast_object_list` of a full state dict
duplicates the model in CPU RAM per node. The established pattern avoids this:
rank 0 saves the params to a file and broadcasts only the file path; other ranks
load from the path. This is one local rank-distribution strategy, not a
server-to-client model-transfer redesign; the FL transfer path remains the
existing Client API / Cellnet / download-service mechanism. The session applies
file exchange automatically above a configurable payload-size threshold, while
small payloads — PEFT adapters, metrics, task metadata — keep using object
broadcast. Operators can also force object broadcast or force file exchange.
The Phase 1 contract for the file exchange:

- Location: `<args.output_dir>/_fl_exchange/`, written as a temp file + atomic
  `os.replace` rename; readers are released by a `dist.barrier()` after the write.
- Format: `safetensors` where possible, with
  `torch.save`/`torch.load(weights_only=True)` fallback for payloads that are not
  safetensors-compatible. This file handoff avoids pushing large parameter
  objects through `broadcast_object_list` and reduces serialization pressure, but
  each reader rank still materializes the tensors it must load into its model.
- Cleanup: each round's payload file is deleted after the post-load barrier; the
  directory is bounded to one round's payload plus the tiny persistent
  `fl_state.json` (see Checkpoint provenance).
- Reach: single-node multi-GPU works on local disk. Multi-node requires
  `output_dir` on a shared filesystem — **not a new requirement**: multi-node
  checkpoint resume from `output_dir` already demands it, as the `llm_hf`
  multi-node setup does today. If a deployment cannot share `output_dir`,
  a config flag forces object broadcast with its documented per-rank memory cost.

## Job-Side Integration

No new components required:

- Subprocess mode: existing `ScriptRunner` / `PTClientAPILauncherExecutor` +
  `CellPipe`, unchanged.
- In-process mode: existing in-process client API executor, unchanged.
- Server: `FedAvgRecipe` (+ `PTModel` or an initial-model persistor). For large
  models enable `enable_tensor_disk_offload=True` in the recipe.
- Reference configuration for large models (the misconfiguration to prevent:
  leaving `server_expected_format` at its numpy default forces the fp32 cast and
  disables tensor disk offload entirely):

  ```python
  runner = ScriptRunner(
      script="client.py",
      launch_external_process=True,                     # subprocess for training
      framework=FrameworkType.PYTORCH,
      server_expected_format=ExchangeFormat.PYTORCH,    # tensor mode: native dtype (bf16), enables offload
      params_transfer_type=TransferType.FULL,
  )
  recipe = FedAvgRecipe(
      ...,
      initial_model=model,
      enable_tensor_disk_offload=True,                  # server-side peak-memory control
  )
  ```
- PEFT jobs (`params_scope="adapter"`): the server's initial model must be
  **adapter-shaped** — the persistor/model definition contains only the adapter
  keys, since that is what round 0 sends and what the aggregator averages. The
  clean example must show the PEFT-mode server model alongside the SFT one.
- The `server_key_prefix` option removes the example's hand-rolled `"model."`
  renames; sites that prefer executor-side conversion can keep using a
  `ParamsConverter` instead — the two mechanisms are alternatives, and the doc for
  the API must say "pick one".

## Example

A clean `examples/advanced/hf_client_api` example demonstrates the API without
carrying the legacy `llm_hf` comparison scripts and figures. Its `client.py`
contains ordinary dataset/model/`SFTConfig`/`SFTTrainer` setup, one default
`flare.patch(trainer)` call, and the standard `while flare.is_running():
evaluate(); train()` loop. The example server model wrappers expose state-dict
keys that match the trainer model, so first-time users do not need
`params_scope`, `server_key_prefix`, `local_epochs`, or metrics-streaming options
on the main path. Those knobs stay documented and are shown as comments in the
example for users who need them. Synthetic data generation is kept in the
standard example helper `prepare_data.py`; `job.py` only consumes prepared
per-site data paths.

## Testing Plan

- Unit tests (`tests/unit_test/app_opt/hf/`), no network: tiny model built from
  config (e.g. 2-layer `GPT2LMHeadModel(GPT2Config(...))`), synthetic dataset,
  client API mocked at the `receive`/`send` seam (same approach as Lightning tests).
  Cover: patch idempotency; round budget stop (epochs and steps); state restore
  across two rounds (optimizer moments survive, global weights override checkpoint
  weights — this pins the `on_train_begin` ordering assumption); strict/non-strict
  key validation paths; PEFT extract/load round-trip; numpy fp32 cast; meta
  step-count under gradient accumulation; evaluate-task and submit_model flows
  (including that a no-op `train()` leaves the cumulative target and round
  accounting untouched); checkpoint provenance (stale `checkpoint-*` in
  `output_dir` is never resumed; round 0 never resumes); global-model metrics
  exclude mid-training `eval_strategy` evaluations; `stream_metrics` forwarding of
  `on_log` scalars (writer mocked) with monotonic step axis in both
  `restore_state` modes; the **checkpoint-injection fallback strategy** (forced via
  its config override — the fallback must not ship as dead code); patch-time
  rejections (`save_only_model`, `load_best_model_at_end`, both explicit budgets,
  `"adapter"` scope on non-PEFT); first-train rejection for epoch budget on
  length-less `IterableDataset`; empty train dataloader rejection;
  session state-file persistence and restore (relaunch mid-job resumes the
  recorded checkpoint and target; different-job state file ignored).
- Multi-process unit tests: 2-process CPU/`gloo` `torch.distributed` tests for the
  session's collective sequence — `is_running()` broadcast, dispatch broadcast,
  non-rank-0 no-ops, file-exchange write/barrier/cleanup, and the
  divergent-wrapped-call assertion. The highest-risk code in this design is the
  multi-rank session logic; it cannot be left to manual HPC runs alone.
- Version matrix in CI: `transformers` min supported pin + latest (the Trainer
  callback/resume internals are the main drift risk), and `trl` 0.18 + latest
  (integration test only).
- Integration: `hf_client_api` example under simulator (single GPU) per release;
  multi-node remains a manual/HPC run.

## Phasing

- **Phase 0 (spike — hard prerequisite, gates the `restore_state=True` default):**
  verify `on_train_begin`-after-checkpoint-restore ordering on the supported
  `transformers` range and pick the min version pin; also verify that
  `control.should_save` set at the budget-stop boundary produces a full standard
  checkpoint (the forced round-end save mechanism). Phase 1 does not ship the
  in-memory override as default until this is verified; if verification fails for
  part of the range, those versions get the checkpoint-injection fallback via the
  runtime version check. (The strategy decision itself is made: in-memory override
  primary, checkpoint-injection as automatic fallback — see Round-Loop Semantics.)
- **Phase 1:** `patch()` + `FLCallback`; SFT + PEFT; single-GPU + DDP; train /
  evaluate / submit_model; unit tests; add the clean `hf_client_api` example.
- **Phase 2 (HF-specific; primarily validation):** exercise the Phase 1 code paths
  under DeepSpeed (ZeRO-1/2/3) and FSDP — `accelerator.get_state_dict()` extraction
  and the resume/override path are designed to be backend-agnostic, so the bulk of
  this phase is verifying that claim, not new code. Known concrete deltas it will
  trigger: (a) backend-based selection of the weight-override strategy
  (checkpoint-injection under ZeRO-3/FSDP — in-memory `load_state_dict` into a
  sharded model does not work naively); (b) FSDP full-state-dict-type context
  handling if the accelerator seam proves insufficient; (c) `Seq2SeqTrainer`
  predict/generate-based eval metrics capture (small real implementation — those
  metrics flow through different hooks than `eval_loss`).
