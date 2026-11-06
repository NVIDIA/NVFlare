# Hugging Face TRAP and audit workflow

This bounded research example uses the **SDK classes in this checkout** with
Pythia-70M, complete model exchange, two clients, and two FedAvg rounds on one CUDA
GPU. It prepares a public reference, trains with TRAP, audits each client result,
checks coverage, and audits the saved final global checkpoint. It is an adoption
example and implementation validation, not a new privacy/utility benchmark or a
production certification. TRAP has no differential privacy guarantee.

## Run

Use Python 3.12 and a CUDA PyTorch environment. From the repository root:

```bash
python -m pip install -e .
python -m pip install -r research/memorization-reduction/huggingface/requirements.txt
cd research/memorization-reduction/huggingface
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
# Choose a new absolute directory for every run. Commands reject existing outputs.
RUN=/tmp/trap-hf-run-1
python prepare.py --assets "$RUN/assets"
python validate.py --assets "$RUN/assets" --output "$RUN/parity"
python job.py --assets "$RUN/assets" --output "$RUN/federation"
python check_coverage.py "$RUN/federation/simulation"
# Locate the unique last global checkpoint; do not select it using final audit scores.
find "$RUN/federation" -name FL_global_model.pt
python evaluate.py --assets "$RUN/assets" \
  --checkpoint <absolute-path-to-FL_global_model.pt> --output "$RUN/final"
```

The model and tokenizer are pinned to
`EleutherAI/pythia-70m-deduped@e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`.
Only preparation downloads from Hugging Face. Subsequent steps load local assets.
`manifest.json` records file hashes and reference training resources. The example
uses fp32, eager attention, no KV cache, no TF32, batch size 2, a 128-token limit,
AdamW at 5e-5 with gradient-norm clipping at 1.0, and TRAP weight 0.5/threshold 0. Reference training uses 16 steps;
each client uses eight steps per round. These fixed choices demonstrate the API;
they are not tuned recommendations. Validation uses predeclared numerical tolerances.

`job.py --assets ... --output <fresh-path> --export --export-dir <job-directory>`
exports the same recipe. For a real deployment, provision the reference/tokenizer
assets and cohort files locally at each site, adjust their configured paths, and
provide a site-local writable measurements directory. The simulation's common
filesystem and fictional data are conveniences, not a data-sharing architecture.
The server needs architecture/initial weights; it need not receive private cohorts
or the local frozen reference. Do not deploy the shared synthetic bundle as a
pattern for collecting private patient/user data centrally.

## Task, cohorts, and attack adapter

The task completes the access code in a fictional dossier. All records are generated
locally from fixed identifiers; no real personal data are used. All cohorts use
the same prompt format and hashed display identifiers: split/site labels appear
only in cohort metadata, never in the text scored by the attacks. `make_cohorts()` in
`hf_model.py` makes 32 public-reference records and 16 validation records disjoint
from both sites' training records. Each site trains on 16 records. Local audits
use eight training members and eight disjoint nonmembers. Final evaluation uses
the other eight members per site and 16 new nonmembers, never used for local audits
or checkpoint selection. Membership in a model's training cohort is intentional;
reference/nonmember overlap is not. Unknown pretraining exposure cannot be excluded.

Prompt and completion are tokenized separately and concatenated. Labels mask the
prompt and padding with `-100`; the SDK shifts them once. Overlong/empty examples
fail rather than silently changing the cohort. For your task, replace this explicit
construction and keep the exact tokenizer, prompt formatting, truncation policy,
cohort membership, and record order consistent across target and reference scoring.

`DossierAuditor` implements likelihood and target-minus-reference advantage (TRA)
attacks: mean completion log probability, and mean target-minus-reference token
log probability. Higher means more member-like. The SDK computes directional
AUROC and nAUC, plus paired before/after score-change AUROC/nAUC. The report also
includes completion CE, token accuracy and counts. Individual scores exist only
inside the adapter; only aggregates are reported. This does **not** implement
extraction, exposure or every attack from the larger benchmark. Applications must
choose attacks and cohort sizes for their threat model. Tiny-sample nAUC is noisy
and is not an approval threshold.

`evaluate.py` uses the same adapter/reporter on the explicit final global checkpoint,
with `stage="global_model"`, its file SHA-256 and the independent final cohorts.
It fails if evaluation/report persistence fails. Local client audits alone cannot
establish final global-model coverage.

## Complete state and adapter-only updates

`PTMemorizationAuditor` requires **all keys and shapes in the evaluation module's
`state_dict()`**, including persistent buffers, not just trainable parameters.
Full updates load strictly. Differences require every key and a matching full
baseline for the same task round. This example exchanges full Pythia state.

LoRA/PEFT adapter-only updates do not satisfy that contract and report `unsupported`.
They are not automatically merged. An application supporting adapter-only exchange
must reconstruct a separate evaluation model from the exact immutable base,
architecture, adapters and buffers, with matching baseline/candidate identities,
and provide its own `MemorizationAuditor` implementation (or explicitly produce
complete state for the supplied adapter). Never treat an adapter dictionary as a
complete model or silently fill missing values from an unrelated checkpoint.

## Resource budgets and visibility

See [SDK GPU validation and operating limits](SDK_VALIDATION.md) for measured scope.
Plan for the trainer, frozen reference, incoming baseline copy, materialized outgoing
state, reconstructed evaluation model, temporary loading/hashing copies, activations,
logits, gradients, optimizer state and allocator reservations. State size alone is
not a peak-memory estimate. Increasing sequence length/vocabulary/batch size can
dominate the logits budget; multiple concurrent clients multiply process costs.

The scripts emit phase wall time and CUDA allocator allocated/reserved peaks.
Host RSS is explicitly a **process-lifetime** high-water mark. Measurements include
synchronization at phase boundaries; they do not measure whole-node power, total
GPU use, or isolate all shared runtime overhead. Budget headroom and measure your
actual model, optimizer, precision, client concurrency and audit cadence before use.

The auditor's 120-second deadline is cooperative. It is checked between batches and
after evaluation; model construction, fingerprinting or a blocked CUDA operation
can exceed it. `validate.py` deliberately requests a tiny budget and records observed
latency to demonstrate that limitation. A deployment needing a hard limit needs an
external worker/process supervisor and resource controls, with explicit cleanup.

`check_coverage.py` requires exactly one completed report for every expected
(site, round), both attack cohorts of the expected size, and model/baseline identities.
Missing, duplicate, failed, skipped and unsupported reports make it exit nonzero.
It checks the every-round schedule here; deployments with sparse schedules must
generate their expected identities from their task schedule, and separately track
intentionally skipped tasks. Run it after tasks finish (or on a declared observation
window); an in-progress report is not a terminal missing-audit diagnosis.

Monitor report statuses, missing expected reports, cohort counts, reference/model
identities, audit elapsed time, memory headroom, and reporter errors in client logs.
A report file can be absent when writing fails; scalar analytics only emit for
completed audits, so a dashboard of successful metrics alone cannot show coverage.
Keep aggregate reports and fingerprints access-controlled too. This sample checker
is an operator signal after execution, **not release enforcement**: the SDK audit
filter reports failures and still transmits the update. Sites requiring approval
before transmission or model release need a separate fail-closed enforcement step
that checks coverage, provenance and policy before authorizing release. Neither
TRAP nor this example installs such a gate.
