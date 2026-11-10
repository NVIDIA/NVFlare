# SDK GPU validation and operating limits

Validated November 6, 2026 at
[`778d632e35bd46c8b40a928837d43a55d0931373`](https://github.com/holgerroth/NVFlare/commit/778d632e35bd46c8b40a928837d43a55d0931373).
[Machine-readable results, source hashes and artifact hashes](validation-h100.json)
accompany this report and bind the Python sources at that commit. A subsequent
coverage-checker fix handles SDK failure reports with unknown site/round identities;
an in-process regression generates an actual filter report and checks JSON diagnostics
and exit status 1. That fix was validated locally, without rerunning the H100 workflow.
A later review tightened reference mutation detection, contained non-ValueError
DXO decoding failures, made observer callbacks avoid model re-encoding, skipped
input hashing when caching is off, and
added deadline checks between audit phases. These changes have focused local
regression coverage; the H100 workflow has not been
rerun after them. The measurements below belong to the pinned revision, not the
updated implementation. The current API also replaces the historical
`AuditModelSnapshot`/component-ID wiring with a private, non-sticky task-context
baseline and a nested auditor constructor argument. Its local export/reload and
lifetime tests do not establish GPU runtime or memory parity. Current `validate.py`
checks ownership through the public input/output filters and includes output-filter
reporting in the full-audit measurement; the historical measurements below used
the removed snapshot store and direct auditor calls. The TRAP equation and audit
metric formulas are unchanged. Later local follow-up also adds matching input
snapshot schedules, checks paired round identity in the generic filter, and loads
one reference per HF audit for both evaluation phases, releasing it at exit. The
H100 workflow has not been rerun for these changes; the timings and memory figures
below do not measure their effects.
The historical [525-fit benchmark](../TRAP_DP_RESULTS.md) used the earlier research
implementation; this run separately exercises the SDK classes at the pinned revision. Neither establishes
general production readiness.

## Workload and runtime

One NVIDIA H100 NVL (96 GB nominal), with two simulator clients sharing the device.
The isolated container had eight CPUs, a 24-GiB host-memory limit and 4-GiB shared
memory. Its 40-minute external watchdog did not fire; cleanup released the GPU.

- Image: `nvcr.io/nvidia/nemo-automodel:26.08`, pinned digest
  `sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee`.
- Python 3.12.3, PyTorch `2.13.0a0+8145d630e8.nv26.06`, Transformers 5.12.1,
  NumPy 1.26.4, safetensors 0.8.0; editable NVFlare from the validated checkout.
- Pythia-70M model/tokenizer revision
  `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; complete fp32 state, eager attention,
  KV cache disabled and TF32 disabled (including the container's override).
- 112 distinct fictional records; shared prompt format without split labels.
  Actual sequence lengths **16–22 tokens**. The code rejects lengths above 128,
  but this run **does not measure capacity at 128 tokens**.
- Batch size 2; public reference: 32 records/16 AdamW steps. Two clients each use
  16 training records/eight AdamW steps per round, for two FedAvg rounds. Learning
  rate 5e-5, gradient clipping norm 1.0, TRAP weight 0.5 and threshold 0. Seed 17.
  Pre-training validation uses 16 separate records in one batch.

The exact execution sequence is the [README workflow](README.md#run): preparation,
parity validation, federation, coverage check, final checkpoint evaluation and job
export. The container first installed the checkout with `pip install -e /workspace`
and ran the six focused test modules for the loss, scorer, auditor, filter, reporter
and HF example. Because its shallow checkout lacked usable Versioneer metadata,
the runner set `NVFL_BASE_VERSION=2.10.0`, producing distribution version
`2.10.0.dev261106`; this changed packaging metadata only.

## Numerical and workflow checks

**42 focused tests passed, including both CUDA tests; none skipped.** Four new
in-process tests cover cohort separation/template consistency, completion masking
and truncation rejection, incomplete coverage, and offline report persistence.
GPU/model downloads and the simulator run are explicit integration validation,
not hidden unit-test subprocesses or an automatic CI guarantee.

`validate.py` exercised the actual `FrozenReferenceScorer`, `PTTRAPLoss`,
`AuditModelSnapshot` and `PTMemorizationAuditor`:

| Check | Observed result |
|---|---|
| Reference token log probabilities vs independently shifted cross-entropy | Maximum absolute error **0** |
| CE + TRAP loss vs literal masked penalty equation | Absolute error **0** |
| All **76** parameter-gradient tensors | Maximum absolute error **0** |
| Nontrivial penalty and masking | **6/12** valid completion tokens active; prompts/padding excluded |
| Likelihood and TRA AUROC/nAUC and paired score changes | Matched independent pairwise AUROC calculation within **1e-6** |
| Full-state vs complete-difference reconstruction | Same aggregate metrics and reconstructed-state fingerprint |
| Baseline ownership | Independent snapshot storage, with the matching round identity |

Reference tolerances were atol 1e-5/rtol 1e-6; loss 1e-6/1e-6; parameter gradients
1e-6/1e-4. They were set before the first GPU attempt and were not relaxed. Audit
aggregate tolerance was 1e-6. These are fp32 implementation checks, separate from
the historical study's retained-evidence verification tolerances. A single SGD
perturbation forms the parity harness's nontrivial reconstruction candidate; the
actual federated workflow uses AdamW as specified above.

The real Recipe run exercised incoming snapshots and outgoing
`MemorizationAuditFilter` reports through the merged `ModelFilter`:

- **Two clients × two rounds**, all four reports `completed`, with correct site/round
  identities, matching per-round baseline fingerprints and **8 members/8 nonmembers**
  for both likelihood and TRA. The expected-coverage checker passed.
- The saved last `FL_global_model.pt` was separately evaluated with the actual
  `DossierAuditor` and `MemorizationAuditReporter`, `stage="global_model"`, on
  **16 previously unaudited members/16 new nonmembers**. Its file SHA-256 matched
  the report; it differs from the initial checkpoint. Final cohorts did not select
  a checkpoint or training configuration.
- Prepared asset hashes, exported auditor/filter configuration and terminal exit
  code were checked. Source hashes in the receipt identify the validated commit.
  No SDK numerical/runtime behavior was changed to obtain the pass.

These tiny synthetic cohorts validate execution and arithmetic. Their attack
metrics, retained in the receipt, are not evidence that this short run protects
privacy or reproduces the historical benchmark.

## Measured resource envelope

GiB means 2^30 bytes. CUDA values below are **process allocator peaks**, including
other live tensors in that process, not incremental costs of just the named phase.
Host RSS is a **process-lifetime high-water mark**, not a phase allocation delta.
See [PyTorch CUDA memory accounting](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-memory-management)
for the distinction between allocated tensors, allocator reservations and device-level usage.

| Phase | Wall time | CUDA allocated peak | CUDA reserved peak |
|---|---:|---:|---:|
| Public reference training, 16 steps including checkpoint save | 2.165 s | 1.383 GiB | 1.426 GiB |
| Frozen-reference construction/fingerprinting in parity harness | 1.613 s | 0.263 GiB | 0.281 GiB |
| First reference scoring batch | 0.875 s | 0.580 GiB | 0.611 GiB |
| First SDK CE+TRAP forward/backward | 0.089 s | 0.854 GiB | 0.885 GiB |
| Actual client training, eight steps, four site/round observations | 0.430–0.526 s | 1.645–1.677 GiB | 1.693–2.043 GiB |
| Actual paired client audit, four observations | 4.143–4.194 s | 1.965–1.972 GiB | 2.023–2.043 GiB |
| Separate final global audit | 6.495 s | 0.590 GiB | 0.631 GiB |

The full state contains **281,706,496 bytes (268.656 MiB)**. The standalone CPU
baseline snapshot took **0.052 s** and owns an additional full-state copy. Largest
client process RSS was **3.777 GiB**; the diagnostic reconstruction harness reached
**3.839 GiB**. Whole-device memory sampled every second peaked at **6,422 MiB**
(**6.271 GiB**) while both clients shared the GPU. Sampling can miss short peaks;
CUDA contexts/runtime allocations also explain why this differs from tensor peaks.
The container's host limit is not a measurement of total host usage.

The simulator reported **58.4 s** for the federated run, including its orchestration
and communication. Download/setup, reference preparation, parity checks, final
audit and export are outside that number. First-call initialization, warmed kernels,
and concurrent client work differ across phases: these times do not support a
causal percentage-slowdown claim against CE-only training.

For this exact starter, an initial admission budget to **re-test** is a dedicated
GPU slot with at least 12 GiB available and 24 GiB host memory for the container.
That provides headroom over the observed workload; a smaller GPU was not validated.
Do not extrapolate it to longer contexts, larger vocabularies/models, mixed
precision, different optimizers, extra concurrent clients, PEFT, quantization or
DDP/FSDP. Measure those workloads first. Include baseline/update copies, reference
and evaluation modules, activations/logits, optimizer states, loading/hashing
copies, allocator reservations and runtime contexts in the budget.

A requested **one-microsecond** audit deadline took **2.019 s** to raise
`TimeoutError`, because construction/reference setup is cooperative and not
preemptible. The normal 120-second deadline is not a hard wall-time or memory bound.
Hard limits require a supervised worker/process and cleanup policy. Audit failures
still permit normal model transmission; release approval needs separate enforcement.

## Coverage and deployment acceptance

Use the [coverage and complete-state contract](README.md#complete-state-and-adapter-only-updates)
before adapting this example. In particular:

1. Provision the exact reference, tokenizer and task cohorts at each site, and bind
   the evaluation model's complete state, including buffers. Adapter-only state is
   unsupported by the supplied adapter; reconstruct the exact base/adapters in an
   application-specific auditor.
2. Derive expected job/task/site/round coverage from the execution schedule, with
   required attack/cohort counts. Alert on missing, failed, unsupported, duplicate
   or incomplete reports and reporter errors; distinguish scheduled skips. The
   sample checker checks the every-round site/round schedule in a fresh run root.
3. Monitor memory headroom and elapsed time alongside coverage. Completed-only
   scalar analytics cannot show absent reports. Check final global-checkpoint
   coverage separately and apply access/retention policy to aggregate reports too.
4. For approval-sensitive deployments, a separate fail-closed policy must authorize
   transmission/release. The SDK filters and this post-run checker are not that gate.

The API remains experimental. This representative run closes the missing SDK GPU
exercise and supplies a runnable adoption/resource example; it does not replace
application-specific security review, task/attack validation or production qualification.

## Development findings retained

Setup attempts exposed shallow-checkout version metadata handling; the first
independent gradient calculation used a transposed softmax layout that produced
fp32 gradient differences. A diagnostic showed identical repeated SDK gradients
and exact agreement with flattened cross-entropy. The harness adopted that usual
layout and explicitly disabled the image's TF32 override; tolerances and SDK code
were unchanged. A standalone workspace omission was fixed with a focused regression.
The preliminary SGD starter deteriorated in CE and was replaced with fixed AdamW
and clipping. Finally, split labels were removed from scored prompts and the entire
workflow was rerun at the validated commit above. Earlier attempts and diagnostics
remain retained; only the final cohort-corrected run supplies this report's results.
