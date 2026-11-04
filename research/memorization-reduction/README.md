# TRAP: experimental memorization reduction

This research entry point composes `PTTRAPLoss`, `FrozenReferenceScorer` and the
report-only memorization filters with a standard two-client `FedAvgRecipe`.
It uses a tiny synthetic token model for two rounds and needs no downloaded data
or checkpoints. It demonstrates API wiring; its scores are **not a benchmark or
proof of reduced leakage**. TRAP provides no differential privacy guarantee.

The TRAP method is introduced by Muhammed Ustaomeroglu et al. (2026) in
[TRAP: Understanding and Mitigating Privacy Memorization in Language Models](https://arxiv.org/abs/2609.32293),
arXiv:2609.32293. This example integrates the preprint's reference-based penalty
into federated fine-tuning with NVFlare.

From a clean virtual environment at the repository root:

```bash
python -m pip install -e .
python -m pip install -r research/memorization-reduction/requirements.txt
cd research/memorization-reduction
python job.py
```

The run prints the simulation artifact location. Each client's job run directory
contains `memorization_audits/*.json`. With successful execution, expect two
`completed` reports per client, one per round, each with 32 member and 32 nonmember
records for the `likelihood` score. These local audit records do not participate in
server model selection. `ce` measures the received model on the local held-out
cohort and selects checkpoints with `key_metric_mode="min"`.

To inspect the exported job instead of running it:

```bash
python job.py --export --export-dir ./jobs
```

`client.py` trains a separate reference on fixed public synthetic records before
freezing it. Only the target model is sent. The application checks reference/member
overlap. Real deployments must establish reference provenance, use representative
cohorts, choose attack scores and budget memory/compute. The example's random-token
data and short training are intentionally unsuitable for scientific conclusions.

## Project layout and shared components

`job.py`, `client.py`, and `model.py` contain the experiment-specific workflow under
`research/memorization-reduction/`. Reusable TRAP loss, reference scoring, filters,
auditors, and reporting components remain in the `nvflare/app_opt/pt` and
`nvflare/app_common` packages. Their API contracts are documented in the programming
guide and design document linked below.

`model.py` provides the model, deterministic local cohorts and a likelihood auditor.
`job.py` registers a matching incoming snapshot and outgoing audit for `train` tasks.
The auditor reconstructs complete differences when used in a compatible FedOpt job;
this example itself uses full weights. Reference scoring and audit model copies add
memory and runtime costs. Audits are cooperative and report-only: failures do not
stop model transmission, and missing/unsupported measurements are never zero leakage.
No summary analytics are relayed by this example.

See the [programming guide](../../docs/programming_guide/memorization_reduction.rst)
and [design](../../docs/design/federated_memorization_reduction.md). Full paired
benchmarking results and conclusions will be added separately.

## Functional validation

On November 3, 2026, the workflow was run from this research directory on macOS
arm64 using CPU, Python 3.12.9, PyTorch 2.11.0, and NVFlare 2.10.0dev0. The SDK
revision was [`70f1d4acd`](https://github.com/holgerroth/NVFlare/commit/70f1d4acddece1c4a0e2e7ca807c8e23f83a0c3e);
`job.py`, `client.py`, `model.py`, and `requirements.txt` were byte-identical to
that revision's implementation, relocated into `research/memorization-reduction/`.

Running `python job.py` completed two FedAvg rounds on both clients. All four
local audit reports had `completed` status, matching site/round identities,
baseline fingerprints, and 32 member plus 32 nonmember records. Job export with
`python job.py --export --export-dir <output-directory>` also succeeded. These
checks validate packaging and functional integration on CPU; full benchmarking
results and GPU validation remain separate work.

This research integration is distributed under the repository's [Apache 2.0 license](../../LICENSE).
