# NVFLARE Agent Skill Authoring Design

## Document Control

| Field | Value |
| --- | --- |
| Created date | 2026-05-26 |
| Status | Ready for Implementation |
| Parent design | [Agent Integration](agent_integration.md) |
| Related designs | [Agent Integration: Skill Evaluation](agent_integration.md#skill-evaluation), [Agent Skill Publication Handoff Checklist](agent_publication_handoff_checklist.md) |
| Current owner | NVFLARE product/docs maintainers |
| Review scope | Skill-writing rules, guide-compatible file structure, minimal metadata, helper scripts, examples, eval input files, and maintenance policy |

## Table of Contents

- [Document Control](#document-control)
- [Scope](#scope)
- [Agent Design Principles](#agent-design-principles)
- [Skills Are Workflow Playbooks, Not Command Wrappers](#skills-are-workflow-playbooks-not-command-wrappers)
- [Skill Granularity and Naming](#skill-granularity-and-naming)
- [Frontmatter and Product Metadata](#frontmatter-and-product-metadata)
- [Trigger and Handoff Contract](#trigger-and-handoff-contract)
- [Initial Customer Helper Scripts](#initial-customer-helper-scripts)
- [Skill File Requirements](#skill-file-requirements)
- [Example and Scaffold Conventions](#example-and-scaffold-conventions)
- [Authoring Workstreams and Deliverables](#authoring-workstreams-and-deliverables)

## Scope

This document owns how NVFLARE product skills are authored and maintained. The
runtime command contract, install model, and production safety boundaries live
in [Agent Integration](agent_integration.md). Evaluation and publication handoff scoring
live in [Agent Integration: Skill Evaluation](agent_integration.md#skill-evaluation).

Authoring rules should not restate runtime contracts differently. When a skill
needs to know how agents trigger or hand off work in the initial implementation, refer to the
integration design. Durable workflow state, receipts, provenance, transcript
replay, workspace cleanup, compatibility shims, and large lifecycle metadata
are deferred until a concrete product requirement promotes them into scope.

## Agent Design Principles

FLARE agent readiness should treat an LLM as a domain expert, not as a
deterministic workflow engine. The design should let the model reason about
ambiguous FLARE choices while code, schemas, tests, and tools enforce mechanical
constraints.

Principles:

- Let the expert focus on expert work. Skills should ask the LLM to reason
  about integration lane, failure cause, safety boundary, and user intent.
  Deterministic work such as command validation, export checks, manifest
  validation, schema generation, path scanning, and policy gates should live in
  CLI code or helper tools.
- Focus attention on what matters. `SKILL.md` should contain only trigger
  context, decision logic, workflow steps, and safety constraints. Large
  examples, command details, troubleshooting catalogs, and framework-specific
  deep dives should live in `references/` and be loaded only when needed.
- Learn from experience. Each skill should carry known failure cases and drift
  notes in `dev_tools/agent/skill_evals/<skill>/` or references. Repeated
  failures should become new tests, diagnosis patterns, guardrails, or
  reference guidance instead of one-off prompt edits.
- Make work observable and independently checked. Agent workflows should produce
  command logs, artifact manifests, findings, and benchmark reports. Evaluation
  should include an independent critique or grading step rather than relying on
  the same agent run to judge its own output.

Prompt instructions are not enough for production-grade behavior. Any mandatory
"must do" or "must not do" requirement should have a matching code guardrail,
test, schema constraint, or approval checkpoint where practical.

## Skills Are Workflow Playbooks, Not Command Wrappers

Skills should teach agents how to combine product surfaces:

- which FLARE integration surface to use for a customer workflow;
- which files to inspect;
- which edits are safe;
- which commands to run;
- how to validate before submission;
- how to diagnose common failures;
- when to stop and ask the user for approval.

Skills should not be one wrapper per CLI command. The CLI schema is the command
contract; skills are procedural product knowledge. When a sequence is mandatory
for correctness, prefer bundling the deterministic checks into a CLI command or
helper tool rather than relying on the LLM to remember every step.

Skills that edit user files must make recovery explicit:

- copy every file to be modified into `.nvflare_bak/<timestamp>/` before the
  first write.
- include `backup_path` in the final user-facing summary when files were
  changed.
- be idempotent: re-running the skill must not duplicate imports,
  `flare.init()`, `flare.receive()`, `flare.send()`, or generated config blocks.
- support a dry-run plan for mutating workflows that lists planned commands,
  files to create/change, approval checkpoints, mutating steps, and estimated
  duration.
- do not delete `.nvflare_bak/<timestamp>/` backups automatically after a
  successful run.

## Skill Granularity and Naming

FLARE skills should be workflow-oriented and organized by product category.
They should not be one giant general-purpose FLARE skill, and they should not
be one skill per CLI command.

The target granularity is an intent-level workflow that an agent can complete
end to end. Create a separate skill when the workflow has materially different:

- user intent;
- files to inspect or edit;
- safety boundary;
- validation commands;
- success criteria;
- evaluation dataset.

Keep commands together when they are steps in one user goal. For example,
`job submit`, `job monitor`, `job logs`, `job stats`, and `job download` belong
in `nvflare-job-lifecycle`; they should not become separate command-wrapper
skills. Conversion skills should be named for the customer's goal or framework,
not for the internal FLARE API used to implement the conversion. Do not ship a
broad `nvflare-federate-training-code` skill in the seed bundle if it
overlaps with framework-specific conversion skills. General routing should live
in `nvflare-orient`, `nvflare agent inspect`, and conversion reference material.
Once the framework is known, the framework-specific skill should own the
workflow. PyTorch is singled out only because it is the first detailed
framework-specific conversion path; it is not separate from training-code
conversion conceptually.

A generic conversion skill can be added later only if it has a clear non-overlap
boundary, such as custom Python, NumPy, or otherwise unrecognized training
loops. It must include negative trigger evals proving it does not steal PyTorch,
Lightning, TensorFlow, XGBoost, or sklearn prompts.
`nvflare-convert-numpy` is the deliberate first fallback for NumPy and simple
custom loops only. It is not a broad "federate anything" skill, and its negative
trigger evals must prove that framework-specific prompts route elsewhere.

Conversion skill families should follow the way customers describe their code,
not FLARE internals. This table is the canonical conversion-skill scope source;
the product catalog in the integration design is a summary view and should be
linted against this table or generated from skill metadata. The roadmap should
be derived from current FLARE examples, recipes, and optional app modules, then
revisited each release as the examples change.
Tier values match the product catalog: `seed` for the initial seed skills,
`m8` for the Milestone 8 base conversion expansion, `next` for follow-on
customer-journey skills, and `later` for later catalog expansion.

| Code Family | Skill | Scope | Current Repo Evidence | Tier |
| --- | --- | --- | --- | --- |
| PyTorch | `nvflare-convert-pytorch` | `torch.nn.Module`, `state_dict`, PyTorch data loaders, checkpoints, and metrics | `examples/hello-world/hello-pt`, `examples/advanced/cifar10/pt`, `nvflare.app_opt.pt`, recipe metadata | `seed` |
| PyTorch Lightning | `nvflare-convert-lightning` | `LightningModule`, `Trainer`, callbacks, checkpointing, multi-GPU, and logging patterns | `examples/hello-world/hello-lightning`, `examples/advanced/multi-gpu/lightning` | `m8` |
| TensorFlow/Keras | `nvflare-convert-tensorflow` | `tf.keras`, `model.fit`, weight exchange, callbacks, GPU memory behavior, and saved model patterns | `examples/hello-world/hello-tf`, `examples/hello-world/hello-cyclic`, `examples/advanced/cifar10/tf`, `nvflare.app_opt.tf` | `next` |
| Hugging Face / LLM / VLM | `nvflare-convert-huggingface` | `Trainer`, `AutoModel`, tokenizers, datasets, PEFT/LoRA, adapter checkpoints, and transformer/VLM fine-tuning | `examples/advanced/llm_hf`, `examples/advanced/medgemma`, `examples/advanced/qwen3-vl`, `research/auto-fl-research/tasks/vlm_med` | `next` |
| XGBoost | `nvflare-convert-xgboost` | `Booster`, `DMatrix`, horizontal histogram, bagging, vertical XGBoost, and secure variants | `examples/advanced/xgboost`, `nvflare.app_opt.xgboost.recipes`, recipe metadata | `next` |
| scikit-learn traditional ML | `nvflare-convert-sklearn` | estimators and pipelines such as logistic regression, SVM, k-means, preprocessing, `fit`, `predict`, and metrics | `examples/advanced/sklearn-linear`, `examples/advanced/sklearn-svm`, `examples/advanced/sklearn-kmeans`, `nvflare.app_opt.sklearn.recipes`, recipe metadata | `next` |
| Survival analysis | `nvflare-convert-survival-analysis` | Kaplan-Meier, Cox-style workflows, censored labels, concordance metrics, time-to-event data, and healthcare validation patterns | `examples/advanced/kaplan-meier-he`, self-paced survival analysis tutorial | `next` |
| MONAI / medical imaging | `nvflare-convert-monai` | medical image loaders, transforms, segmentation/classification loops, and healthcare evaluation artifacts | `examples/advanced/monai` | `later` |
| JAX | `nvflare-convert-jax` | JAX/NumPy parameter trees, `.npy` caches, and custom training loops | `examples/hello-world/hello-jax` | `later` |
| Flower applications | `nvflare-integrate-flower` | Running existing Flower PyTorch apps under FLARE job orchestration and tracking | `examples/hello-world/hello-flower` | `later` |
| GNN | `nvflare-convert-gnn` | graph data, node/edge task patterns, and GNN-specific model/data exchange | `examples/advanced/gnn` | `later` |
| NumPy/custom loops | `nvflare-convert-numpy` | custom Python or NumPy training loops when no stronger framework family is detected | `examples/hello-world/hello-numpy`, `examples/hello-world/ml-to-fl/np`, `nvflare.app_common.np`, recipe metadata | `later` |

Do not create one skill per algorithm unless the workflow is genuinely
different. Logistic regression, SVM, and k-means can start as references under
`nvflare-convert-sklearn` because their customer workflow is usually "convert my
sklearn estimator or pipeline." Survival analysis may deserve a separate skill
because censored labels, evaluation metrics, and clinical data conventions are
substantially different.

The roadmap should not blindly turn every example directory into a public
skill. A new conversion skill needs a distinct trigger surface, distinct edit
pattern, and enough examples or recipes to evaluate it. Otherwise, keep the
content as references under the closest existing skill.

Each `SKILL.md` should stay concise and trigger-focused:

- when to use;
- when not to use;
- decision tree;
- high-level workflow;
- approval checkpoints;
- validation checklist;
- links to deeper `references/` files.

Detailed examples, command walkthroughs, diagnosis patterns, framework
conversion details, and evaluation notes should live under `references/` so the
agent loads them only when needed.

## Frontmatter and Product Metadata

Each published skill must include the agentskills.io required frontmatter:

```yaml
---
name: nvflare-convert-pytorch
description: "Convert PyTorch training code into an NVFLARE federated training job and validate it with SimEnv and exported FLARE jobs."
min_flare_version: "2.8.0"
blast_radius: edits_files
category: Conversion
---
```

`name` and `description` are the public trigger surface. The description should
say when the skill should trigger and include enough negative boundary language
that nearby skills do not steal the prompt. FLARE-specific initial metadata should
stay minimal:

| Field | Required | Notes |
| --- | --- | --- |
| `name` | yes | must match the skill directory |
| `description` | yes | concise trigger description for the agent |
| `min_flare_version` | yes | first NVFLARE release that supports the skill; lint/documentation metadata initially because bundled skills ship with FLARE |
| `blast_radius` | yes | `read_only`, `edits_files`, `runs_simulator`, `submits_poc`, or `submits_production`; informational initially and the hook for future enforcement |
| `category` | yes for public skills | high-level product grouping for catalog and publication; draft, internal, and private skills are exempt |
| `skill_version` | optional | useful for changelog and debugging inside a FLARE release, but not a separate distribution cadence |

Do not add a broad `references/contract.yaml` requirement for the initial implementation. Rich sidecar
metadata such as allowed tools, allowed commands, compatibility shims, budgets,
agent profiles, `obsoletes`, lifecycle states, and durable handoff keys is
deferred to the roadmap.

Shared skill references under `skills/_shared/` are versioned content, not
untracked include files. Install manifests should record the shared-reference
hash or version used by an installed skill whenever shared files are copied into
the target agent skill directory. Updating shared reference content should be
treated as a behavior-affecting skill change for every public skill that
consumes it, with release notes or benchmark evidence when the shared guidance
changes observable agent behavior.

## Trigger and Handoff Contract

The authoritative runtime contract for skill triggering and one-lead plus
supporting-skill composition lives in
[Agent Integration](agent_integration.md#6-how-skills-are-triggered-and-used).
For the initial implementation, skill authors should assume handoff happens through visible command
outputs, generated artifacts, and the active agent session. Durable handoff
files, receipts, and provenance sidecars are deferred roadmap items.

## Initial Customer Helper Scripts

Scripts are useful when the work is deterministic and customer-visible, but not
yet stable enough to be a product CLI command. They should be skill-local,
JSON-producing helpers under `scripts/` and should be promoted into `nvflare`
CLI commands once their behavior becomes a product contract.

Recommended initial helper scripts:

| Script | Used By | Purpose | Promotion Path |
| --- | --- | --- | --- |
| `scripts/scan_project.py` | `nvflare-orient`, framework conversion skills | Static scan for frameworks, entry points, FLARE integration usage, `job.py`, SimEnv usage, export support, and absolute data paths | `nvflare agent inspect` |
| `scripts/validate_flare_integration.py` | framework conversion skills | AST-based check for FLARE receive/send patterns, `FLModel`, and incomplete conversion patterns without exposing API details as customer intent | `nvflare agent inspect` or a future conversion lint command |
| `scripts/validate_exported_job.py` | `nvflare-job-lifecycle`, framework conversion skills | Check required job files and obvious packaging mistakes; use `_export_manifest.json` or fingerprint metadata when the future export API provides them | `nvflare agent inspect` and `nvflare job submit` preflight |
| `scripts/extract_metrics_summary.py` | `nvflare-job-lifecycle`, framework conversion skills | Collect simple metrics from known JSON artifacts, logs, TensorBoard pointers, or result files so the agent can report training progress | future `nvflare job metrics` |
| `scripts/collect_job_evidence.py` | `nvflare-diagnose-job` | Run the standard evidence commands and save a support bundle with job meta, logs, stats, system status, and resources | future `nvflare job diagnose` |
| `scripts/match_failure_patterns.py` | `nvflare-diagnose-job` | Match collected logs and metadata against known patterns such as CUDA OOM, import errors, auth failures, timeouts, and expired startup kits | future `nvflare job diagnose` |
| `scripts/redact_support_bundle.py` | `nvflare-diagnose-job`, `nvflare-job-lifecycle` | Redact secrets, local usernames, tokens, and private paths before an evidence bundle is shared | future support-bundle command |

Script requirements:

- deterministic behavior only; no hidden LLM calls;
- one CLI-compatible JSON envelope on stdout and diagnostics on stderr. The
  envelope should use the same top-level shape as agent-facing CLI commands,
  including `schema_version`, `status`, `code`, `message`, and `data`;
- explicit path or job ID arguments; no broad home-directory scanning;
- no importing or executing user training modules;
- no reading private key contents;
- no mutation unless the script name and skill workflow clearly say it writes
  generated files;
- stable enough inputs and outputs to be covered by tests;
- easy removal or promotion when the equivalent product CLI command exists.

When a helper script is promoted to a product CLI command, leave a short
transition record in the skill reference, for example
`promoted_to: nvflare job diagnose`, and update skills to call the product
command before removing the script. Skill lint should fail when a public skill
continues to call a promoted script after the replacement CLI command is
available for the target FLARE version.

These scripts should reduce LLM burden. They should not become a second product
API surface that diverges from the CLI.



## Skill File Requirements

Each public FLARE skill should include:

- `SKILL.md` with valid frontmatter, an H1 heading, and concise workflow
  instructions;
- a repo-only eval suite at `dev_tools/agent/skill_evals/<skill>/evals.json`
  with guide-compatible prompts, expected outputs, files, and assertions
  (grading-oracle data is not shipped inside the skill);
- `dev_tools/agent/skill_evals/<skill>/files/` for input fixtures or minimized
  expected artifacts when a deterministic eval needs files;
- `references/` for detailed examples, command contracts, and product-specific
  guidance that should not bloat `SKILL.md`;
- `scripts/` and `assets/` only when needed by the stated workflow;
- publication handoff artifacts or externally generated skill-card/signature
  artifacts when the separate publication process produces them.

Global-negative evals that expect no FLARE skill to trigger may set
`nvflare.negative_for` to the literal string `"*"`. Adjacent-negative evals
against a specific competing skill should name that skill explicitly.

`SKILL.md` should stay compact enough for frequent loading. The initial lint uses
200 lines as the hard gate. Roughly 2,000 tokens is an advisory authoring
target, reported as a warning by a simple whitespace-based estimate unless a
project tokenizer is later standardized. Longer examples, walkthroughs,
command tables, and troubleshooting catalogs should move to `references/` and
be loaded only when needed.

Packaged skills must not include runtime-evaluator hooks. `SKILL.md` files
should not instruct agents to check evaluator environment variables, enable an
eval-on mode, or run an evaluator command before the final response. Keep
user-facing evidence requirements in the workflow, requirements, and output
sections. Preserve non-runtime skill metadata, eval cases, benchmark summaries,
and read-only performance reporting when they help skill development.

Packaged runtime skill artifacts must not point agents at `docs/design/`.
Design documents are review artifacts, not installed runtime instructions. Move
any reusable guidance into `SKILL.md`, `references/`, or `skills/_shared/`.

Recommended structure:

```text
skills/nvflare-convert-pytorch/          # shipped runtime content
  SKILL.md
  references/
    pytorch-client-api-conversion.md
    job-export-contract.md
  scripts/
    validate_exported_job.py

dev_tools/agent/skill_evals/nvflare-convert-pytorch/   # repo-only eval suite
  evals.json
  files/
    hello-pt/
```

This follows the public skill-guide structure: `SKILL.md` is the portable
instruction file, `references/`, `scripts/`, and `assets/` are loaded only when
needed, the eval suite at `dev_tools/agent/skill_evals/<skill>/evals.json` is
the hand-authored eval definition (kept out of the shipped skill), and generated
benchmark outputs live outside the skill source tree. FLARE-specific
requirements should
be represented as fields, assertions, references, or generated reports inside
that structure, not as a separate public skill layout.

Copyright and license text must appear below the YAML frontmatter and initial
H1 heading, not above the frontmatter.

License choice:

- default to dual Apache 2.0 plus CC-BY 4.0 when a skill mixes instructions,
  examples, helper scripts, or generated-code snippets;
- use Apache 2.0 only for skills that are primarily executable helper code;
- use CC-BY 4.0 only for skills that are purely instructional and contain no
  meaningful executable code snippets or helper scripts.

Examples and skills have separate source-of-truth roles:

- `examples/` remains the source for runnable product examples.
- skills should reference example paths and may include short snippets, but they
  should not copy whole examples into `references/`.
- eval input files may copy a minimized snapshot of an example under
  `dev_tools/agent/skill_evals/<skill>/files/` only when needed for
  deterministic testing; the fixture should record the source example path and
  source hash.
- release checks should verify that referenced example paths still exist and
  that eval input-file hashes are intentionally updated when examples change.

### Skills Maintenance Policy

Packaged skills are part of the NVFLARE user contract and need explicit
ownership:

- each public skill declares a precise trigger description, minimum FLARE
  version, and blast radius in frontmatter.
- public skills may declare `skill_version`; skill changes can follow semantic
  versioning at the skill level for debugging and review, even when bundled into
  an NVFLARE release.
- skill updates are versioned independently with `skill_version`, but official
  packaged distribution follows the NVFLARE release cycle. Out-of-cycle official
  skill fixes require an NVFLARE patch release; user-side workarounds can use
  `nvflare agent skills install --local` until an official release is
  available. This avoids implying that the native wheel installer fetches
  out-of-band catalog updates.
- breaking CLI schema changes require a skill compatibility audit before
  release.
- community PRs can add or update framework-specific skills when they include
  examples, trigger evals, behavior IDs, and the initial checks.
- community skill contributions follow the same project CLA, code-of-conduct,
  review, and release rules as other NVFLARE contributions. FLARE maintainers
  arbitrate trigger overlap, naming conflicts, lifecycle status, and whether a
  community change can alter an NVIDIA-owned public skill or must ship as a new
  skill/reference first.
- breaking changes to NVIDIA-owned public skills require a migration note and
  benchmark comparison before release; minor community improvements can merge
  when they preserve trigger boundaries and pass the admission gate.
- local user overrides are supported through `nvflare agent skills install
  --local`, but packaged skills remain the release source of truth.

Skill semantic-versioning rules:

| Bump | Triggers |
| --- | --- |
| Major | breaking trigger change, removed mandatory step, dropped FLARE version support, or incompatible output expectation |
| Minor | new optional behavior, new reference, new eval input or expected artifact, or expanded framework scope |
| Patch | wording, typo, benchmark update, eval input hash refresh, non-behavioral reference cleanup, or metadata correction that does not change routing |

Full deprecation lifecycle behavior, `obsoletes`, changelog commands, and
catalog migration rules are deferred roadmap items. Until then, removing or
renaming a shipped skill should require a release-note entry and explicit
maintainer approval.



## Example and Scaffold Conventions

New agent-ready examples should use a predictable layout:

```text
<example_or_job>/
  README.md
  job.py
  client.py
  model.py
  prepare_data.py
  download_data.py
  requirements.txt
```

Not every file is required for every example, but these names should be used
when the concepts exist.

README files should include:

- what the example does;
- file structure;
- setup and dependency installation;
- data preparation;
- job recipe overview;
- local SimEnv run command;
- export command;
- POC or production submit command;
- expected outputs and artifact locations;
- troubleshooting notes.

## Authoring Workstreams and Deliverables

- Add repo-root `skills/` and keep it as the source for public FLARE skills.
- Add the initial seed skill set defined in the integration design.
- Keep each `SKILL.md` concise and move deep reference material into
  `references/`.
- Add the initial helper scripts that support customer workflows:
  `scan_project.py`, `validate_flare_integration.py`,
  `validate_exported_job.py`, `extract_metrics_summary.py`,
  `collect_job_evidence.py`, `match_failure_patterns.py`, and
  `redact_support_bundle.py`.
- Add skill lint checks for name, description, directory matching, command
  references, script references, release compatibility, prohibited primary
  paths, runtime artifact references, trigger overlap, and guide-compatible eval
  shape.
- Add script contract tests for JSON-envelope output, explicit arguments, no
  user-code imports, no secret leakage, and expected failure handling.
- Package released customer-facing skills in the Python wheel from the same
  repo-root source.
