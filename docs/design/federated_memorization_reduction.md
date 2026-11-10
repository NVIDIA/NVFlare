# Experimental memorization mitigation and auditing

Status: draft for API and design review, updated November 5, 2026.

Issue: [NVIDIA/NVFlare#5680](https://github.com/NVIDIA/NVFlare/issues/5680).

Target: the next NVFlare release; all new components are experimental.
Constructing the TRAP and auditing components emits NVFlare's `@experimental` warning;
their APIs may change. The decorator is applied to constructors to preserve class
identity and signatures for job configuration export and loading.

## Problem and scope

Federated fine-tuning can retain information about individual training records.
Keeping raw training data at a site does not establish that a released model, a
client update, or a reference model is safe against membership or extraction attacks.
Applications need reusable training and measurement components with explicit
observation and failure semantics.

This proposal adds a PyTorch Target Reference Advantage Penalty (TRAP), frozen
reference scoring and scheduled report-only model audits built on the separately
reviewed `ModelFilter` foundation. Applications continue to use their ordinary Client API
training loops and FedAvg/FedOpt aggregation. The components do not choose a
reference, tune a defense, impose an audit acceptance threshold, or claim a privacy
guarantee.

TRAP is introduced by Muhammed Ustaomeroglu et al. (2026) in
[TRAP: Understanding and Mitigating Privacy Memorization in Language Models](https://arxiv.org/abs/2609.32293),
arXiv:2609.32293. This design covers the NVFlare integration of the preprint's
reference-based penalty and the supporting audit interfaces.

The [benchmark report](https://github.com/NVIDIA/NVFlare/blob/main/research/memorization-reduction/TRAP_DP_RESULTS.md)
includes the completed GB300 research study. Representative SDK validation on H100
and focused unit tests check this integration's behavior; the separate benchmark
evidence and its limits are summarized below. TRAP is not differential privacy,
and this proposal adds neither a DP mechanism nor a privacy accountant.

## Components and ownership

| Component | Location | Responsibility |
|---|---|---|
| `PTTRAPLoss`, `causal_token_log_probs` | `app_opt/pt/trap_loss.py` | Masked penalty, causal alignment and diagnostics |
| `FrozenReferenceScorer` | `app_opt/pt/reference_scorer.py` | Frozen inference and optional local score cache |
| `MemorizationAuditor`, `UnsupportedAudit` | `app_common/abstract/memorization_auditor.py` | Framework-neutral application audit contract |
| `AuditSnapshotFilter`, `MemorizationAuditFilter` | `app_common/filters/memorization_audit_filter.py` | Task-context baseline, scheduling and failure classification |
| `PTMemorizationAuditor` | `app_opt/pt/memorization_auditor.py` | Model reconstruction, RNG isolation and membership aggregates |
| `MemorizationAuditReporter` | `app_common/widgets/memorization_audit_reporter.py` | Local JSON reports and explicitly selected analytics |

The runnable research entry point is
[`research/memorization-reduction/`](https://github.com/NVIDIA/NVFlare/tree/main/research/memorization-reduction).
Its `job.py`, client loop and synthetic model/cohorts stay with the research project.
The reusable components in the table remain in `nvflare/`; the programming guide
covers their API contracts. Experiment-specific orchestration and future benchmark
material belong under the research project.

PyTorch remains an optional framework dependency. There is no runtime dependency
on Hugging Face or an external research repository. Dataset preparation, reference
training, attack selection, extraction probes and experiment orchestration belong
to the application, not to the SDK.

## Local objective

For valid observed tokens M, target p, fixed reference q, weight lambda and
nonnegative advantage threshold epsilon, train with ordinary cross-entropy plus:

```text
TRAP = lambda / |M| * sum(max(0, log p(token | prefix)
                              - log q(token | prefix) - epsilon))
```

`PTTRAPLoss(target_log_probs, reference_log_probs, mask, normalizer=None)` returns
only this additive penalty. Target gradients are retained; reference values are
detached. Shapes and devices must match, and the mask must be boolean. Weight and
threshold must be finite and nonnegative. Values at selected tokens are expected
to be valid log probabilities; the primitive does not repair nonfinite inputs.

`causal_token_log_probs` shifts logits and labels once. Applications mask padding
and excluded prompt tokens in the unshifted labels with `-100`. CE and TRAP must use
the same tokens and normalization. An empty mask or zero weight yields a
differentiable zero penalty. A zero-weight caller may omit reference scoring.
Weights above one can reverse the active token's CE gradient and require deliberate
evaluation.

For gradient accumulation, normalize each microbatch sum by the valid-token count
of the entire accumulation window, then sum the contributions. Distributed data
parallel users must also account for the framework's gradient averaging; this API
does not infer global token counts or rescale DDP gradients.

Diagnostics report valid-token count, mean reference advantage, active fraction
and penalty. They describe the objective, not leakage or a privacy budget.

## Reference provenance and cache

The intended reference is a separate, same-task model without training exposure to
the protected records. Its data provenance is the application's responsibility.
A global model that has already incorporated those records does not satisfy that
assumption. A peer reference can itself reveal another site's data; exchanging one
requires an explicit threat model. Unknown pretraining exposure remains a limitation.

`FrozenReferenceScorer` freezes parameters, enforces evaluation mode, and returns
observed-token log probabilities and their mask. It must remain separate from the
target optimizer and transmitted target state. Construct a new scorer when the
reference changes. Ordinary in-place parameter/buffer changes, tensor/storage
replacement, and dtype/device conversions are detected. Finalize precision and
placement before constructing the scorer. In-place writes through `.data` or
external storage aliases bypass version tracking and remain unsupported, as do
arbitrary forward-code changes.

Optional cache keys bind checkpoint tensor contents, tokenizer and scoring
identities, tokenized inputs, labels and supplied tensor model inputs. The caller's
scoring identity must cover model configuration, context construction, precision
and other settings affecting inference. Entries are written atomically and read
with `weights_only=True`. The cache is trusted local storage, not an authenticated
artifact exchange. It is batch-addressed: regrouping records can produce misses.
It has no built-in size limit or eviction policy; use a site-managed quota and
retention policy, fixed-order precomputation, or online scoring without a cache.
Online scoring does not hash inputs for cache lookup. Cached scores and tensor
fingerprints may be sensitive and need site-controlled access, retention and
deletion policies.

## Dependency on the ModelFilter foundation

The general `ModelFilter` interface, its focused tests, and generic documentation
were merged into `main` in [PR #5683](https://github.com/NVIDIA/NVFlare/pull/5683)
([issue #5682](https://github.com/NVIDIA/NVFlare/issues/5682)). This proposal builds
on that merged implementation. Callback metrics handling is local to `ModelFilter`;
`FLModelUtils` and `ModelLearnerExecutor` retain their upstream behavior.

Auditing subclasses use the callback contract documented in
[Model Filters](../programming_guide/filters.rst#model-filters).
That contract owns supported encodings, borrowed parameters and nested values,
shallow copies of metadata and metrics dictionaries, `None` pass-through semantics,
and immutable round identity. A callback returning `None` must leave shared data
unchanged. This proposal adds application-specific baseline ownership, audit
scheduling, measurement and reporting. It uses existing Recipe filter registration
and nested constructor-argument export; no Recipe or shared runtime changes are needed.

On current main, `ClientRunner` materializes lazy input references for active input
filters, and `ClientAPIExecutor` materializes external/attached-process results for
active output filters or declared local consumers. This draft uses those existing
paths and does not restore the removed launcher/pass-through configuration API.
The audit runs in the client runtime, not in the trainer's live module.

## Audit lifecycle and observation

```text
incoming full model -> input transformations -> baseline snapshot -> trainer
                                                                  |
                                      CE + TRAP with local reference
                                                                  |
server <- encryption <- report-only audit <- outgoing transformations/update
```

Pass the auditor instance directly to the output filter:

```python
recipe.add_client_input_filter(AuditSnapshotFilter(), tasks=["train"])
recipe.add_client_output_filter(
    MemorizationAuditFilter(auditor=my_auditor), tasks=["train"]
)
```

The filter stores `self.auditor`; job export serializes it as a nested `path`/`args`
configuration and the runtime component builder reconstructs it. Constructor
arguments are exported and must not contain secrets. Nested auditors are not
registered event handlers, so they must not depend on component lifecycle events.

`ClientRunner` passes the same task `FLContext` through input filters, the executor,
and output filters. `AuditSnapshotFilter` deep-copies the incoming full model into
a private, non-sticky context property. The property is excluded from public peer
metadata and is not inherited by later task contexts. The filters retain no model
state themselves; there is no snapshot component, ID lookup, job/task/round storage
key or run-event cleanup. One paired filter chain per task context is supported.

The output filter consumes and removes the property on every output path, including
skipped rounds, empty or unsupported results, failed task replies, and conversion/
reporting failures. A missing baseline is unsupported, never silently an unpaired
audit. If execution exits before output filtering, the baseline lives only as long
as that task context is retained. Direct auditor calls still accept `baseline=None`
for unpaired full-model/global evaluation.

Both observer callbacks return `None`, preserving model contents, metadata and
transport history without model re-encoding; observation status is recorded in the
separate audit report. Initial support is sequential training tasks with one full
model or a complete parameter difference. Collections, partial updates, incompatible
names/shapes and missing round identities are unsupported.
The input filter captures every selected training task, including rounds skipped
by the output audit schedule. Budget that full-state copy even with sparse audits;
the filters currently have independent placement and no shared capture schedule.

Capture after incoming transformations. Audit after outgoing clipping/noise when
the intent is to measure the transmitted update, and before encryption. Executor
conversions must preserve parameter values and names; applications that alter the
training baseline need a matching adapter. Register filters for `tasks=["train"]`.

`PTMemorizationAuditor` builds an independent module and strictly loads its state.
Complete state means **all `state_dict()` entries, including persistent buffers**,
not only trainable parameters. Adapter-only/LoRA updates are unsupported by this
adapter; an application must reconstruct the exact base/adapters/buffers in an
independent task-specific auditor. Missing entries are never inferred.
For differences it reconstructs the candidate from the matching full baseline. It
preserves Python, NumPy and PyTorch CPU/selected-CUDA-device RNG state, including on
failure. Arbitrary application RNGs, other CUDA devices and concurrent training
threads are outside that guarantee. Audit model construction and scoring must not
mutate borrowed inputs.

Subclasses implement bounded local `evaluate(module, deadline)` over fixed member
and nonmember records. Both cohorts must be nonempty and scores finite. Higher
scores indicate membership; AUROC counts ties as half, and `nAUC = 2*abs(AUROC-0.5)`
also exposes inverted separation. With a baseline, the adapter additionally scores
per-record candidate-minus-baseline changes. Equal cohort lengths are checked;
identical record ordering is the subclass's responsibility. The adapter checks its
budget between reconstruction, construction/loading, candidate evaluation, and
baseline evaluation. Subclasses must also check between batches. A deadline is
cooperative and cannot interrupt an operation already in progress or a blocked
framework/CUDA call.

Periodic client audits concern that client model at that filter stage. They do not
establish leakage from the selected global checkpoint, unseen attacks, reference
artifacts or unobserved rounds. The same auditor/reporter can be called by a
separate global-checkpoint evaluation job. Applications without a valid local
nonmember cohort must report unsupported coverage, never a zero-leakage result.

## Reporting and failure behavior

| Status | Meaning | Valid privacy measurement? |
|---|---|---|
| `completed` | Auditor returned serializable aggregate results | Only within that auditor's stated observation scope |
| `skipped` | Round is outside the configured schedule or the task result is not OK | No |
| `unsupported` | Representation, baseline or cohort cannot be audited | No |
| `failed` | Runtime, deadline or serialization failure | No |

The audit is report-only. Failures leave the update on its normal path; they do not
enforce a privacy policy. Operators requiring fail-closed release control need a
separate reviewed mechanism. Configuration errors are distinct from failed runtime
measurements. A DXO filter-application event is not proof of a completed audit.

Schema version 1 reports contain job/task/site/stage/round, status/reason, elapsed
time, scorer/model identities, counts and aggregate metrics. Local files live under
the job run directory, use atomic replacement and owner-only permissions for newly
created files/directories. Retry identity includes job, task, stage, round and model
fingerprint. A `global_model` stage separates final evaluation from client audits.
Reporting failures are logged and return no report path.

Raw text, prompts and per-record scores must not enter the auditor's result dict.
This is an application contract, not automatic content redaction. No analytics are
emitted by default. Only names explicitly listed in `summary_metrics` are emitted
on completed reports; configured event handling determines whether they are
relayed. Aggregate metrics and model hashes are not inherently DP releases.

## Resource costs and alternatives

Online reference scoring adds a forward pass and reference model memory; caching
trades that compute for storage and input hashing. Baseline snapshots require a
full parameter copy. An audit adds an evaluation module, reconstruction buffers,
model hashing and potentially before/after inference. Lazy-reference
materialization increases client-runtime memory/transfer use. Sample counts,
frequency, device and cooperative deadlines must be set for the deployment.

The [Hugging Face adoption example](https://github.com/NVIDIA/NVFlare/tree/main/research/memorization-reduction/huggingface)
provides reference preparation, completion cohorts, likelihood/TRA attacks, final
global-checkpoint evaluation, and an expected site/round coverage checker. Its
[SDK validation report](https://github.com/NVIDIA/NVFlare/blob/main/research/memorization-reduction/huggingface/SDK_VALIDATION.md)
records numerical parity and memory/runtime scope for the actual SDK. These bounded
measurements do not establish capacity for arbitrary models or client concurrency.
Operators must budget live trainer/reference/evaluation models, snapshots,
materialized updates, activations/logits, optimizer state and temporary copies.

Expected audit identities must come from the task schedule, rather than from the
reports that happened to arrive. Missing/duplicate reports and incomplete cohorts
need alerts alongside explicit failed/unsupported/skipped statuses. Completed-only
analytics cannot demonstrate full coverage. Cooperative deadlines need an external
supervisor for hard bounds; release approval needs a separate fail-closed gate.
The example's post-run coverage checker supplies neither of those mechanisms.

Keeping everything in research scripts would avoid a new SDK surface but duplicate
alignment, state reconstruction and failure handling. A TRAP-specific aggregation
controller would couple a local objective to an unrelated transport/aggregation
choice. Using the incoming global model as the automatic reference would conceal
the protected-record exposure assumption. Threshold-based update rejection is
deferred because attack calibration and training consequences need separate design.

## Validation and review gates

The draft tests loss values/gradients/masking/accumulation, cache identity and
mutation rejection, audit scheduling, stale baselines, RNG isolation, unsupported
inputs and local report behavior. The separate foundation owns model-filter
representation and round-trip tests.
SDK revision `778d632e` passed 42 focused tests on H100, including both CUDA tests,
and a two-client/two-round Pythia workflow with four completed client audits and separate
final global-checkpoint evaluation. Reference scores, loss and all 76 parameter
gradients matched independent calculations with zero measured error. The
[validation report](https://github.com/NVIDIA/NVFlare/blob/main/research/memorization-reduction/huggingface/SDK_VALIDATION.md)
records the measured memory/runtime scope and identifies subsequent fixes validated
locally; those later revisions have not rerun the GPU workflow. A tiny model example is
also available as a wiring smoke check. Current-main Client API materialization
tests cover the reused runtime paths.

Before removing experimental status, review the API and security contracts and
publish paired utility/leakage measurements, all defense strengths and individual
seeds, reference provenance/cost, memory/timing/compute and negative/unattainable
outcomes. Freeze development choices before final testing and identify tuning
seeds. Evaluate global models, intermediate observations and reference artifacts
separately; compare against relevant undefended and DP controls without implying
that empirical mitigation has a formal DP guarantee.

Open review questions include API maturity/naming, audit coverage reporting,
deployment-specific reference provisioning and storage retention, and whether a
future isolated audit worker is needed for hard time/memory bounds.

## Benchmark evidence

The [Pythia-70M benchmark report](https://github.com/NVIDIA/NVFlare/blob/main/research/memorization-reduction/TRAP_DP_RESULTS.md)
provides all 525 fits and 210 final evaluations, complete strength/seed tables,
negative outcomes, privacy and compute accounting, and eight PNG/PDF figure pairs.
It uses the frozen research implementation preceding this SDK refactor; the
separate H100 workflow validates its recorded SDK revision; later fixes have
focused local regression coverage.

TRAP passed 60/84 paired accuracy-and-leakage screens, falling to 24/84 when the
separate CE criterion was also required. The tested DP settings passed 0/84 in this
bounded six-configuration search. Seed 13 is tuning; seeds 29/47 reuse its frozen
configurations. These descriptive results motivate the APIs without establishing
statistical non-inferiority, optimal DP performance, or a formal privacy guarantee
for TRAP. They do not by themselves remove the experimental status or the API and
security review gates above.
