# NVFLARE Agent Skill Evaluation Architecture

## Document Control

| Field | Value |
| --- | --- |
| Created date | 2026-05-26 |
| Updated date | 2026-06-28 |
| Status | Proposed design |
| Parent design | [Agent Integration](agent_integration.md) |
| Related designs | [Agent Skill Authoring](agent_skill_authoring.md), [Agent Skill Publication Handoff Checklist](agent_publication_handoff_checklist.md) |
| Current owner | NVFLARE product/docs maintainers |
| Review scope | Skill evaluation gate, guide-compatible eval shape, engineering/runtime test split, runtime evidence without a separate evaluator, Auto-FL research evaluation, and publication handoff artifacts |

## Table of Contents

- [Document Control](#document-control)
- [Scope](#scope)
- [Evaluation Principles](#evaluation-principles)
- [Guide-Compatible Eval Structure](#guide-compatible-eval-structure)
- [Eval Schema Versioning and Negative Triggers](#eval-schema-versioning-and-negative-triggers)
- [Skill Admission Gate](#skill-admission-gate)
- [Engineering Lints](#engineering-lints)
- [Engineering Correctness Checks](#engineering-correctness-checks)
- [Runtime Agent-Performance Checks](#runtime-agent-performance-checks)
- [Runtime Process Evidence](#runtime-process-evidence)
- [Runtime Evidence Records and Reports](#runtime-evidence-records-and-reports)
- [Auto-FL Research Evaluation](#auto-fl-research-evaluation)
- [Publication Handoff Boundary](#publication-handoff-boundary)

## Scope

This document defines the evaluation architecture for public FLARE skills. It
owns the skill metadata, deterministic lint, runtime evidence, reporting, and
publication handoff contracts used to judge whether a skill is ready for public
use.

This document does not define the full agent benchmark harness. The benchmark
harness architecture is defined in
[Agent Benchmark Harness Architecture](agent_benchmark_harness.md).

Every public FLARE skill must have enough measurement to answer:

- Did the right prompt trigger the right skill?
- Did adjacent or unrelated prompts avoid the wrong skill?
- Did the agent follow mandatory instructions?
- Did the agent avoid prohibited actions?
- Did the task produce the expected validation evidence or artifact?
- Do referenced `nvflare` commands still exist for the target release?

Runtime skill-performance measurement uses normal agent runs. There is no
runtime evaluator, no `eval=on` switch, and no `NVFLARE_SKILL_EVAL=on`
activation path.

Authored `evals/evals.json` files remain useful as skill metadata and evidence
contracts. Runtime skill-performance evidence should come from benchmark
harnesses, manual review, or research workflows that record what happened during
normal skill-assisted runs. Reports should summarize observed behavior, process
metrics, task validation evidence, and known gaps without depending on an
external pass/fail or 1-5 score.

## Evaluation Principles

Evaluation has two separate layers:

| Layer | Purpose | Examples | Blocks |
| --- | --- | --- | --- |
| Engineering correctness | Normal product tests for CLI, package, helper scripts, lints, schemas, and install behavior | unit tests, CLI tests, package tests, static lints | NVFLARE release if the product contract is broken |
| Runtime agent performance | Measures how well skills help agents choose, follow, and complete workflows | trigger evals, instruction assertions, artifact checks, manual or scripted agent runs | public skill promotion if the skill behavior is poor |

Do not treat engineering tests as runtime skill metrics. For example,
`native-skill-install-no-node` is a CLI/package test, not evidence that
`nvflare-convert-pytorch` helps an agent convert code. Conversely, a good
agent-run benchmark does not excuse a broken installer or command schema.

Every new normative rule in `SKILL.md` or `references/` should have one of:

- a `nvflare.mandatory_behavior`, `nvflare.optional_behavior`, or
  `nvflare.prohibited_behavior` entry in `evals/evals.json`;
- a deterministic CLI/helper-script test;
- a release checklist item with an observable artifact.

If a rule cannot be measured, rewrite it as guidance instead of a requirement.

## Guide-Compatible Eval Structure

FLARE skills should follow the NVIDIA skill-guide structure. Authored evals
live under the skill:

```text
skills/<skill>/
  SKILL.md
  references/
  scripts/
  assets/
  evals/
    evals.json
    files/
```

`evals/evals.json` should use guide-compatible fields such as `prompt`,
`expected_output`, `files`, and `assertions`. FLARE-specific behavior IDs live
inside an `nvflare` extension object rather than in a parallel eval format.
For FLARE scoring and runtime evidence, `nvflare.mandatory_behavior`,
`nvflare.prohibited_behavior`, `nvflare.optional_behavior`, and
`nvflare.process_metrics` are the canonical contract. Free-text `assertions`
are human-readable guidance and may be generated from, or kept consistent with,
the structured behavior IDs; reports must not treat free-text assertions as a
second independent source of truth.

Example:

```json
{
  "schema_version": "1",
  "skill_name": "nvflare-convert-pytorch",
  "evals": [
    {
      "id": "pytorch-convert-basic",
      "prompt": "Convert this PyTorch training script to a FLARE federated training job and run it locally.",
      "expected_output": "A FLARE-compatible training integration, a generated or updated job.py, a successful local SimEnv run, and an exported job folder.",
      "files": ["evals/files/hello-pt/train.py"],
      "assertions": [
        "The agent runs nvflare agent inspect before editing.",
        "The agent edits only training and job files.",
        "The generated code uses the expected FLARE API surface for this workflow.",
        "The agent runs python job.py for local validation.",
        "The agent runs python job.py --export --export-dir to export a job folder.",
        "The agent does not submit a job unless the user asks for job lifecycle operations."
      ],
      "nvflare": {
        "expected_skill": "nvflare-convert-pytorch",
        "mandatory_behavior": [
          {"id": "inspect-first", "description": "runs nvflare agent inspect before editing"},
          {"id": "scoped-edits", "description": "edits only training and job files"},
          {"id": "use-client-api-for-training-exchange", "description": "uses nvflare.client receive/send and FLModel for training exchange"},
          {"id": "simenv-run", "description": "runs python job.py"},
          {"id": "export-job", "description": "runs python job.py --export --export-dir"}
        ],
        "prohibited_behavior": [
          {"id": "no-unrequested-job-submit", "description": "does not submit a job unless the user asks for job lifecycle operations"},
          {"id": "no-user-code-import", "description": "does not import or execute user modules during static inspection"},
          {"id": "no-cli-wrapper-python", "description": "does not generate Python solely to wrap nvflare CLI operations or scrape human CLI output"}
        ],
        "optional_behavior": [
          {"id": "metrics-summary", "description": "summarizes metrics artifacts when available"}
        ],
        "process_metrics": [
          {"id": "turns_to_acceptable", "description": "number of user/agent turns before an acceptable workflow result"},
          {"id": "user_correction_count", "description": "number of user corrections needed after the first pass"},
          {"id": "missed_instruction_count", "description": "number of applicable explicit instructions the agent missed"},
          {"id": "layout_violations", "description": "count of generated layout or artifact-location mistakes found before final acceptance"}
        ]
      }
    }
  ],
  "negative_trigger_cases": [
    {
      "id": "negative-k8s-deploy",
      "prompt": "Deploy an existing FLARE startup kit to Kubernetes.",
      "expected_output": "The PyTorch conversion skill should not trigger.",
      "files": [],
      "assertions": [
        "The selected skill is not nvflare-convert-pytorch."
      ],
      "nvflare": {
        "expected_skill": "nvflare-deploy-k8s",
        "negative_for": "nvflare-convert-pytorch"
      }
    }
  ]
}
```

Generated benchmark outputs should also follow the guide-style workspace when
they exist:

```text
skills/<skill>-workspace/
  iteration-1/
    <eval-id>/
      with_skill/
        outputs/
        timing.json
        evidence.json
      without_skill/
        outputs/
        timing.json
        evidence.json
    benchmark.json
    benchmark.md
```

FLARE-specific generated reports can be added inside those run directories.
Generated workspaces are not committed by default.
The `skills/<skill>-workspace/` layout is for raw benchmark artifacts and
side-by-side with/without-skill runs. Runtime evidence records are generated
artifacts owned by the benchmark, research workflow, or reviewer process that
created them. There is no evaluator-owned default records root.

Behavior ID evaluation semantics:

- `nvflare.mandatory_behavior` entries are required observations. Each ID must
  be supported by an agent transcript, command log, file diff, generated
  artifact, deterministic helper output, or manual reviewer checklist item.
  Missing evidence should be reported as a missing required observation.
- Explicit user, design, or skill instructions that should be counted as missed
  instructions must be represented by measurable `nvflare.mandatory_behavior`
  IDs in the selected eval case. For example, a "do another review" requirement
  should be represented by a behavior such as `review-after-fix`, and a "use the
  requested output location without overwriting source files" requirement should
  be represented by a behavior such as `non-destructive-output-location`.
  Without such a behavior ID or an explicit harness- or reviewer-supplied count,
  the report has no ground truth for that instruction and must not infer a miss
  from vague transcript absence.
- `nvflare.prohibited_behavior` entries are forbidden observations. If the
  transcript, command log, file diff, or generated artifact shows the behavior
  happened, the runtime evidence should report a prohibited-behavior violation.
- `nvflare.optional_behavior` entries are advisory observations. They can be
  recorded in benchmark output when present, but missing optional behavior does
  not fail the eval.

The IDs are stable labels, not metrics by themselves. Behavior IDs are scoped to
the selected skill and eval case, not globally unique across all skills. The
benchmark harness, research workflow, or reviewer checklist maps each ID in that
eval case to concrete evidence and records the observed status.
The selected case in the skill's `evals/evals.json` is the canonical source of
behavior IDs. Runtime reports must not rely on a separate hard-coded behavior-ID
list outside the selected eval case.

Process metrics are also stable labels. They measure how efficiently the skill
guided the agent, not whether the generated model reached a high accuracy. They
should live under `nvflare.process_metrics` in at least one eval for every
public skill. Typical metrics include first-pass acceptance, user correction
count, agent self-correction count, layout or workflow violations, unwanted
actions, validation evidence completeness, and turns or tool calls to an
acceptable result.

Runtime process records are generated artifacts, not packaged skill source.
Only corrective skill changes and hand-authored eval metadata should be
committed to the skill source tree.

## Eval Schema Versioning and Negative Triggers

`evals/evals.json` must include a top-level `schema_version`. Version `"1"` uses
the guide-compatible shape in this document:

```text
schema_version
skill_name
evals[]
negative_trigger_cases[]
```

`evals[]` contains positive or task-completion cases where the skill is expected
to be useful. `negative_trigger_cases[]` contains adjacent or unrelated prompts
where the skill must not be selected. Negative trigger cases use the same fields
as normal eval cases, but the `nvflare` extension must include `negative_for`
and may include `expected_skill` or `expected_no_skill`.

Schema readers and lints must reject unsupported `schema_version` values instead
of silently interpreting them as the current schema. Backward-compatible
additive fields may be accepted when the reader preserves unknown fields or
reports them as ignored. Breaking changes require a new schema version and a
short migration note in this section.

Schema version `"1"` changelog:

- top-level `schema_version`;
- guide-compatible `evals[]`;
- top-level `negative_trigger_cases[]`;
- FLARE-specific behavior IDs under each case's `nvflare` object;
- `nvflare.process_metrics` for runtime process-quality contracts.

<a id="initial-evaluation-gate"></a>

## Skill Admission Gate

Adding or publishing a public FLARE skill should fail review unless the PR
includes:

- `SKILL.md` with valid frontmatter, a precise trigger description, and clear
  "use / do not use" boundaries.
- `min_flare_version` and `blast_radius` in frontmatter.
- at least one positive trigger eval in `evals/evals.json`;
- at least one adjacent negative trigger case for the nearest competing skill;
- global negative coverage for prompts that should trigger no FLARE skill;
- `nvflare.mandatory_behavior` and `nvflare.prohibited_behavior` IDs for every
  normative workflow rule in the skill;
- `nvflare.process_metrics` for process quality, correction count, validation
  evidence, and first-pass workflow quality;
- deterministic input files under `evals/files/` when file editing or artifact
  assertions require them;
- command-drift checks for every referenced `nvflare` command and flag;
- helper-script tests when the skill ships scripts;
- an explicit `draft/internal` marker in skill frontmatter or publication
  handoff metadata when the skill is not ready for public release.

<a id="initial-engineering-lints"></a>

## Engineering Lints

This table is the canonical lint definition. The authoring guide owns the
frontmatter schema and metadata semantics; this evaluation spec owns which
deterministic checks run before a public skill is accepted.

`agent-doc-crosslink-lint` is intentionally outside this canonical set. Any
future cross-document validation, such as design-doc links or catalog
synchronization, is deferred to docs-only tooling and must not be exposed as a
selectable check in the skill lint runner.

| Check | Failure Condition | Deterministic Inputs | Required Behavior |
| --- | --- | --- | --- |
| `skill-frontmatter-lint` | missing required frontmatter, invalid `blast_radius`, name mismatch, or non-`nvflare-` public skill name | `skills/<skill>/SKILL.md` frontmatter, directory name, and the frontmatter schema in [Agent Skill Authoring](agent_skill_authoring.md#frontmatter-and-product-metadata) | Parse frontmatter as YAML, require the authoring-guide required fields, require public skill names to match their directory and start with `nvflare-`, and require `blast_radius` to be an allowed value. |
| `skill-md-size-lint` | `SKILL.md` exceeds the 200-line hard gate without an approved exception | `skills/<skill>/SKILL.md` | Fail when `SKILL.md` exceeds 200 lines unless an explicit approved exception marker exists. Report the roughly 2,000-token guidance as advisory using a simple whitespace estimate until a tokenizer is standardized. |
| `skill-trigger-lint` | missing trigger/use-boundary text, missing positive trigger eval, or missing adjacent negative trigger case | `SKILL.md` trigger text and `evals/evals.json` | Require a non-empty trigger/use-boundary description, at least one positive trigger eval, and at least one adjacent negative trigger case in `negative_trigger_cases` for the nearest competing skill in the same deterministic trigger group. |
| `skill-trigger-overlap-lint` | same trigger-group public skills have overlapping descriptions or trigger examples without negative trigger cases or documented boundaries | deterministic skill-name family, `SKILL.md` descriptions, and trigger eval prompts | For public skills sharing the same deterministic skill-name family, flag overlapping descriptions or trigger examples unless the skills include documented use/do-not-use boundaries and adjacent negative trigger cases covering the overlap. The lint uses deterministic text/name-family checks, not design-doc parsing or a runtime LLM recommender. |
| `skill-global-negative-lint` | unrelated global negative prompt coverage is missing or malformed | repo-root `skills/_shared/global_negative_prompts.json` and per-skill `evals/evals.json` | Require coverage for prompts that should trigger no FLARE skill, such as unrelated web, Kubernetes-only, or generic coding tasks. The shared bank should use `schema_version: "1"` and `prompts` entries with `id`, `prompt`, and `description`. The deterministic lint validates that public skills include or reference required global-negative cases. |
| `skill-policy-coverage-lint` | normative words appear without a nearby measurable behavior ID, deterministic helper test, or checklist item | `SKILL.md`, `references/`, helper tests, and `evals/evals.json` | Flag normative words such as `must`, `must not`, `required`, `prohibited`, and `approval` unless the rule maps to `nvflare.mandatory_behavior`, `nvflare.prohibited_behavior`, a deterministic helper test, or a release checklist item. |
| `skill-process-metric-lint` | missing process metric contracts for a public skill, or malformed process metric entries | `evals/evals.json` | Require at least one `nvflare.process_metrics` entry for every public skill. Each metric must have a stable `id` and `description` so runtime runs can record first-pass quality, correction count, unwanted actions, validation evidence completeness, and related process outcomes. |
| `skill-command-drift-lint` | referenced `nvflare` commands, flags, or JSON examples do not match the installed CLI or exported command schema | `SKILL.md`, `references/`, `scripts/`, CLI parser/schema output | Verify each referenced `nvflare` command, flag, and JSON example against the installed CLI or exported `--schema` output so stale commands fail before release. |
| `skill-helper-script-lint` | helper scripts lack tests or violate JSON stdout/stderr conventions | `skills/<skill>/scripts/` and tests | Require tests for shipped helper scripts, require machine-readable stdout when JSON output is promised, require diagnostics on stderr, and fail when a public skill calls a helper script marked as promoted to a product CLI command. |
| `skill-fixture-lint` | file-editing evals lack required `evals/files/` inputs or fixture source notes | `evals/evals.json`, `evals/files/`, and fixture notes | Ensure file-editing or artifact-producing evals reference existing deterministic input files under `evals/files/` and include source/provenance notes for fixtures. |

Release checklist items used as measurement substitutes must be machine-readable
in `evals/release_checklist.json` with `schema_version: "1"` and entries
containing `id`, `description`, and `evidence_expected`. Prose-only checklist
mentions do not satisfy `skill-policy-coverage-lint`.

For `skill-trigger-overlap-lint`, the deterministic algorithm compares only
skills in the same deterministic skill-name family, such as `nvflare-convert`
for `nvflare-convert-*` skills or `nvflare-diagnose` for
`nvflare-diagnose-*` skills. It normalizes trigger and use/do-not-use text to
lowercase tokens, removes stop words, and flags overlapping token sets unless
both skills have adjacent negative trigger-case coverage for the overlap. The
lint does not use an LLM or infer semantic similarity beyond those deterministic
text checks.

For `skill-command-drift-lint`, scan fenced code blocks and inline snippets that
start with `nvflare`, parse the command and flags against the installed CLI
parser or exported command schema, and fail on unknown commands or flags. The CI
environment for this lint must run from the same NVFLARE checkout or installed
wheel whose CLI is being validated.

Global negative coverage is per public skill: every public skill must either
include `negative_trigger_cases` entries marked `negative_for: <skill-name>` for each prompt ID in
`skills/_shared/global_negative_prompts.json`, or reference a shared coverage
set that expands to those IDs. Deterministic lint only validates coverage
declarations; benchmark or research runs may execute selected prompts later.

Each lint should emit structured findings with at least `id`, `severity`,
`file`, `line` when available, `message`, and `hint`. These findings are
engineering correctness evidence, not runtime skill-performance metrics.

## Engineering Correctness Checks

These checks are ordinary NVFLARE tests. They should live in unit, CLI,
package, script, lint, or release-test suites and should not be reported as
runtime skill-performance metrics:

| Area | Check ID | Evidence |
| --- | --- | --- |
| Native install | `native-skill-install-no-node` | install test environment and command trace show no Node.js, npm, npx, or external skill CLI dependency |
| Native install | `skill-install-codex-claude-targets` | dry-run JSON and filesystem assertions prove target resolution |
| Native install | `skill-install-all-by-default` | dry-run JSON and installed list show all compatible released NVFLARE skills are selected |
| Native install | `skill-install-safe-overwrite` | target fixture proves existing files are preserved, skipped, or replaced only with explicit overwrite flags |
| Native install | `skill-install-no-third-party-download` | network-disabled test and command trace prove only NVFLARE-owned skills are copied |
| Native install/list | `skill-list-ignores-unrelated-third-party-skills` | target fixture with unrelated third-party skills proves `skills list` does not report them as conflicts |
| Native install/list | `skill-list-flags-name-overlap-external-skill` | target fixture with an unmanaged skill sharing an NVFLARE skill name proves `skills list` reports a name-overlap conflict |
| Release filtering | `release-package-filters-analysis-metadata` | release/company package excludes `evals/` while normal dev/source package keeps it |
| Release filtering | `release-filtered-skill-runtime-works` | installed release-filtered skill has no broken references to filtered files and still loads/installs successfully |
| CLI contract | `cli-json-single-envelope` | `--format json` emits one JSON object on stdout |
| CLI contract | `cli-jsonl-streaming-envelope` | streaming commands with `--format jsonl` emit one complete JSON event per line |
| CLI contract | `cli-schema-no-operational-args` | `--schema` works without runtime inputs |
| CLI contract | `cli-error-recovery-category` | agent-facing errors include a valid `recovery_category` |
| Inspect safety | `inspect-static-only` | static inspection does not import or execute user modules |
| Inspect safety | `inspect-redaction-default` | secret-like literals and sensitive paths are redacted by default |
| Doctor safety | `doctor-read-only` | doctor does not mutate config, submit jobs, or read private key contents |
| Helper scripts | `helper-json-stdout` | helper scripts emit one CLI-compatible JSON envelope on stdout and diagnostics on stderr |
| Helper scripts | `helper-no-user-code-import` | static helper scripts do not import or execute user modules |

## Runtime Agent-Performance Checks

These checks measure the skill's usefulness to an agent and should be reported
by internal benchmark or reviewer tooling when measured:

| Area | Check ID | Evidence |
| --- | --- | --- |
| Triggering | `positive-trigger-correct` | matching prompts activate the expected skill |
| Triggering | `negative-trigger-correct` | adjacent prompts do not activate the wrong skill |
| Triggering | `global-negative-no-trigger` | unrelated prompts trigger no FLARE skill |
| Triggering | `pytorch-vs-lightning-adjacent-trigger` | plain PyTorch prompts route to `nvflare-convert-pytorch`; Lightning prompts route to `nvflare-convert-lightning`; exported jobs route to lifecycle before conversion |
| Instruction following | `mandatory-behavior-followed` | observable trace or artifact shows each `nvflare.mandatory_behavior` item was followed |
| Instruction following | `prohibited-behavior-avoided` | observable trace or artifact shows each `nvflare.prohibited_behavior` item was avoided |
| Task success | `task-validation-passed` | validation command, generated artifact, or deterministic assertion satisfies the eval |
| Generated code API choice | `use-recipe-for-applied-workflow` | standard applied workflows use Recipe API when an appropriate recipe exists |
| Generated code API choice | `use-client-api-for-training-exchange` | training exchange uses Client API and `FLModel` when Client API conversion is required |
| Generated code API choice | `use-lightning-patched-trainer` | Lightning conversion uses `flare.patch(trainer)` and the Lightning callback path for model exchange |
| Generated code API choice | `no-manual-flmodel-exchange-lightning` | Lightning conversion does not generate a manual `FLModel` send/receive training exchange |
| Generated code API choice | `no-input-model-to-trainer` | Lightning conversion does not pass an optional `input_model` returned by `flare.receive()` into `Trainer` methods |
| Generated code API choice | `use-cli-for-operations` | config, provision, submit, monitor, system, and study operations use the CLI rather than generated wrapper Python |
| Job lifecycle boundary | `no-unrequested-job-submit` | conversion skills do not submit jobs unless the user asks for job lifecycle operations |
| Secret safety | `no-private-key-copy` | generated artifacts do not contain copied private keys |

Milestone 8 Lightning eval cases must include the Lightning-specific IDs above
in the selected case's mandatory/prohibited behavior evidence, not only as
report labels. They must also include adjacent trigger cases that distinguish
plain PyTorch `torch.nn.Module`/`DataLoader` projects from
`LightningModule`/`Trainer` projects, plus an exported-job case that confirms
lifecycle routing takes priority over framework conversion even when framework
evidence is present.

Most check IDs in this table are reporting categories, not canonical behavior
IDs. Runtime evidence must use behavior IDs from the selected skill's
`evals/evals.json`. Skill-specific rows that intentionally define behavior
contracts, such as the Lightning rows above, should be mirrored by behavior IDs
in that skill's eval cases.

Internal benchmark/reviewer summaries should keep runtime reports short:

- skill version or source commit;
- FLARE version;
- eval cases run;
- trigger pass/fail summary;
- mandatory behavior followed/missed;
- prohibited behavior violations;
- task validation result;
- process metric summary;
- known failures and next changes.

Cost, repeatability, paired with-skill/without-skill deltas, and independent
monitor reports belong to the benchmark harness rather than the skill metadata
contract.

Same trigger-group overlap means skills sharing the deterministic
`nvflare-<family>` prefix derived from the skill name. The product catalog table
remains product-facing reference material, not a source used by the v1 admission
lint.

<a id="runtime-process-evidence"></a>

## Runtime Process Evidence

Process evidence answers whether the skill reduced repeated correction and
token-heavy trial-and-error. It is separate from task metrics such as model
accuracy, AUROC, or loss. For example, a conversion can produce a runnable model
but still show poor process evidence if it mixes generated files into the source
root, uses nonstandard names, writes runtime artifacts into the project tree, or
requires several user correction rounds.

Recommended evidence record shape:

```json
{
  "schema_version": "1",
  "skill": "nvflare-convert-pytorch",
  "skill_version": "0.1.0",
  "case_id": "ames-pytorch-fedavg-conversion",
  "agent": "codex",
  "run_mode": "with_skill",
  "source_hash": "cc84428d014be112e254420a92b6497d3b11cbd5a67b263e56ebd0a4df18e00d",
  "source_commit": null,
  "prompt_summary": "Convert AMES PyTorch code to FedAvg simulation with 2 sites",
  "mandatory_behavior": {
    "inspect-first": {
      "status": "pass",
      "evidence": ["tool log shows nvflare agent inspect ran before file edits"],
      "notes": "Inspection happened before editing."
    },
    "use-client-api-for-training-exchange": {
      "status": "pass",
      "evidence": ["client.py uses nvflare.client receive/send and FLModel"],
      "notes": ""
    }
  },
  "prohibited_behavior": {
    "no-unrequested-job-submit": {
      "status": "pass",
      "evidence": ["command log contains no nvflare job submit command"],
      "notes": ""
    }
  },
  "optional_behavior": {
    "metrics-summary": {
      "status": "missing",
      "evidence": [],
      "notes": "No metric artifact was available."
    }
  },
  "first_pass": {
    "accepted": false,
    "violations": [
      "mixed generated FLARE files into original source root",
      "used fl_train.py instead of client.py",
      "used fl_job/fl_workspace under project root instead of /tmp/nvflare"
    ]
  },
  "final_result": {
    "accepted": true,
    "validation_passed": true,
    "simulation_passed": true,
    "failure_root_cause": null
  },
  "process_metrics": {
    "elapsed_seconds": 812,
    "token_count": 42000,
    "turns_to_acceptable": 4,
    "user_correction_count": 3,
    "agent_self_correction_count": 1,
    "missed_instruction_count": 2,
    "layout_violations": 3,
    "workflow_violations": null,
    "evidence_gap_violations": null,
    "validation_commands_run": 5,
    "unnecessary_files_created": 4
  },
  "significant_violations": [],
  "skill_improvements": [
    "add generated-job folder rule",
    "require client.py/job.py/model.py",
    "keep runtime workspace outside the source tree"
  ]
}
```

`prompt_summary` is optional and should be a short, sanitized description of the
task, not a copied prompt. `process_metrics.agent_self_correction_count` is the
count of agent-detected corrections made before the user had to correct the
workflow.

`process_metrics.missed_instruction_count` counts applicable explicit
instructions that the agent did not follow. Condition-based instructions count
only when the condition is satisfied. A missed instruction that is severe enough
to invalidate safety, artifact ownership, or approval requirements should also
be recorded in `significant_violations`. When the count is absent or `null`, a
report may show it as unavailable; it must not infer hidden intent from vague
transcript absence.

Runtime `process_metrics` fields:

| Field | Type | Nullability and ownership |
| --- | --- | --- |
| `elapsed_seconds` | number | `null` when unavailable. Prefer measured harness data from artifacts. |
| `token_count` | integer | `null` when unavailable. Never infer from transcript text. |
| `turns_to_acceptable` | integer | `null` when unavailable. Reviewer checklist may supply it. |
| `user_correction_count` | integer | `null` when unavailable. `0` means no user correction was observed in available evidence. |
| `agent_self_correction_count` | integer | `null` when unavailable. `0` means no self-correction was observed in available evidence. |
| `missed_instruction_count` | integer | `null` when unavailable. `0` means no missed instruction was observed in available evidence. |
| `layout_violations` | integer | `null` when not checked. `0` means checked and none found. |
| `workflow_violations` | integer | `null` when not checked. `0` means checked and none found. |
| `evidence_gap_violations` | integer | `null` when not checked. `0` means checked and none found. |
| `validation_commands_run` | integer | `null` when not applicable or not tracked. |
| `unnecessary_files_created` | integer | `null` when not checked. |

Additional metrics are allowed only when the selected eval case declares them
under `nvflare.process_metrics`.

This design does not define a 1-5 process score or a top-level evaluator
pass/fail. Reports may apply an explicit quality gate for a specific benchmark
or publication review, but that policy belongs to the benchmark/report layer and
must be documented with the report.

The feedback loop is:

1. Run a realistic task with the skill enabled.
2. Record first-pass violations, correction count, validation evidence, and
   significant blockers.
3. Compare process metrics and task evidence against the baseline or previous
   skill version.
4. Add only the missing guardrails to `SKILL.md`, `references/`, helper scripts,
   or deterministic lints.
5. Add or update structured behavior IDs, human-readable assertions, and
   process metrics together so they do not drift.
6. Re-run and verify correction count, failure rate, or evidence gaps improve.

<a id="runtime-evaluator-and-records"></a>

## Runtime Evidence Records and Reports

Runtime evidence records are generated by the benchmark harness, research
workflow, or reviewer process that runs the task. They are normal run artifacts,
not evaluator outputs. A record must preserve bounded evidence that explains the
run outcome without copying full transcripts, large command outputs, secrets,
access tokens, private keys, credentials, or sensitive absolute paths.

The record should use `schema_version` value `"1"` and include:

- `schema_version`, `skill`, `skill_version`, `case_id`, `agent`,
  `source_hash`, and optional `source_commit`;
- `run_mode`, normally `with_skill` or `without_skill` when a comparison run is
  being performed, otherwise `null`;
- optional `prompt_summary`;
- `mandatory_behavior`, `prohibited_behavior`, and `optional_behavior` maps
  keyed by behavior ID from the selected `evals/evals.json` case;
- `first_pass`, `final_result`, `process_metrics`, `significant_violations`,
  and `skill_improvements` using the evidence-record shape above.

Allowed behavior `status` values are `pass`, `fail`, `missing`,
`not_applicable`, and `non_scoring_note`. Aggregation and reporting commands
must reject unknown status values rather than silently treating them as pass.
Consumers must also reject or clearly surface records with unsupported
`schema_version` values instead of silently interpreting them as the current
schema.

Status meaning depends on behavior category. For mandatory and optional
behaviors, `pass` means evidence of the expected behavior was observed, `fail`
means contradictory evidence was observed, and `missing` means requested
evidence was not found. For prohibited behaviors, `pass` means no evidence of
the prohibited behavior was observed, and `fail` means the prohibited action was
detected.

`source_hash` must use the same contract as the released-skill manifest:
lowercase hex-encoded SHA-256 over the sorted files under `skills/<skill>/`.
For each included file, feed the UTF-8 relative path, a NUL byte, the file
contents, and a NUL byte into the single running SHA-256 state. Exclude
`__pycache__`, `.pyc`, and `.pyo` files, and reject symlinks rather than
following them.

`skill_version` should come from the packaged manifest when available, otherwise
from `SKILL.md` frontmatter. If neither source provides it, store `null` and
keep the record grouped separately from records with a non-null version.

`not_applicable` is not valid for mandatory or prohibited behavior IDs. Use it
only for optional behavior IDs where the evidence source is genuinely irrelevant
to the run. A valid `not_applicable` entry is excluded from optional evidence
summaries and does not count as missing evidence.

`non_scoring_note` is for reviewer or harness observations that should be
preserved but are not behavior IDs from the selected eval case. It is not valid
for canonical mandatory or prohibited behavior IDs. Store these notes in the
record's `optional_behavior` map with `status: "non_scoring_note"` and an ID
that is not treated as a behavior requirement.

Each evidence string in a runtime record should be at most 512 characters, and
each behavior entry should include at most 10 evidence strings. Longer evidence
belongs in the run artifact directory and should be referenced by relative path.
`notes` fields should be short reviewer summaries, not copied logs.
`first_pass.violations` and `skill_improvements` should each contain at most 10
strings, and each string should be at most 512 characters.

`final_result.validation_passed` and `final_result.simulation_passed` are
case-specific booleans. They should be `true` or `false` only when the selected
case requires that kind of validation evidence, and `null` when the field is not
applicable or not tracked for the skill. Non-conversion skills should use
`final_result.accepted` plus skill-specific mandatory behavior evidence rather
than forcing simulation-specific fields.

Reports must not invent token usage, infer hidden agent intent, or retroactively
excuse missing evidence. If token counts are unavailable from the agent runtime,
the record should store `null` and reports should mark token usage unavailable.
Likewise, reports must not invent missed-instruction findings for instructions
that are not represented by selected-case mandatory behavior IDs or an explicit
structured `process_metrics.missed_instruction_count` supplied by the harness or
reviewer.

Benchmark reporting is internal tooling, not a public `nvflare agent` command
surface. Packaged `evals/evals.json` files define what a benchmark harness can
measure, but they are not performance evidence by themselves.

Internal benchmark reporting should read explicit benchmark or reviewer records,
aggregate only fields that are present, and preserve unavailable counts for
missing numeric values. It must not run skills, call an LLM, infer token usage
from transcript text, synthesize correctness fields, mutate records, or write
benchmark reports back into skill source.

### Runtime Evidence Retention

Runtime evidence records are user- and workspace-owned artifacts. NVFLARE does
not delete them automatically because they may be release evidence, publication
handoff evidence, or regression history. Tools that scan records must bound
their work with file-count and file-size limits and report when records were
skipped or unavailable.

The default retention policy is:

- benchmark and reviewer workflows write records under an explicit run or
  records root;
- benchmark reporting reads bounded records and never mutates them;
- users or CI jobs own archival and deletion;
- publication handoff bundles should copy or reference only the records needed
  for the release decision;
- future cleanup commands may provide `archive`, `prune`, or `export` behavior,
  but those commands must be explicit and must not run as part of read-only
  reporting.

Reports should show the records root, record count scanned, records skipped by
limits, and the newest/oldest record timestamp when available.

## Auto-FL Research Evaluation

Auto-FL is a research evaluation consumer, not a release canary. The FLARE
research team already has Auto-FL workflows. Those workflows should be reused
as repeatable test cases for whether the new skills improve agent behavior.

Recommended comparison modes:

| Mode | Purpose |
| --- | --- |
| `without_skill` | baseline agent behavior with no FLARE skill loaded |
| `with_skill` | same task with relevant FLARE skills available |
| `with_skill_forced` | diagnostic mode that names one skill explicitly to isolate skill content from trigger selection |

Auto-FL evaluation should measure task success, validation score, missed
mandatory behavior, prohibited actions, runtime, tool calls, and token/cost data
when available. Results should feed skill improvements, helper scripts, and
future eval cases. They should not block release or publication handoff
unless a separate release-gate decision is made later.

Auto-FL-specific artifacts should remain in the research project. FLARE skills
should not define Auto-FL queue state, retry policy, routing, persistence, or
run resumption.

## Publication Handoff Boundary

External publication is separate `github.com/NVIDIA/skills` integration work.
This evaluation spec owns the FLARE side of the handoff:

- guide-compatible `SKILL.md`;
- references, scripts, assets, and eval inputs;
- lint and engineering-test evidence;
- runtime skill-performance summaries from internal benchmark or reviewer
  tooling when available;
- FLARE release and skill source/version information.

The concrete artifact checklist lives in
[Agent Skill Publication Handoff Checklist](agent_publication_handoff_checklist.md).

The public NVIDIA skill scoreboard, catalog sync, public installer metadata,
signing, and publication UI are outside this NVFLARE design. FLARE should
provide artifacts that the company-wide process can consume, but should not own
the public scoreboard mechanics.
