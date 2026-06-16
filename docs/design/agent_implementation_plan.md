# NVFLARE Agent Skills Initial Implementation Plan

## Document Control

| Field | Value |
| --- | --- |
| Created date | 2026-05-26 |
| Updated date | 2026-06-04 |
| Status | Ready for Implementation |
| Sources | [Agent Integration](agent_integration.md), [Agent Skill Authoring](agent_skill_authoring.md), [Agent Skill Evaluation](agent_skill_evaluation.md), and the temporary deferred-roadmap planning note |
| Scope | First implementation cut for native NVFLARE agent skills |
| Out of scope | Public NVIDIA skills catalog mechanics, public scoreboard ownership, Node/npm/npx installer dependency, Auto-FL product roadmap, and deferred roadmap items |

## Intent

This plan implements the simplified initial design. Deferred mechanisms such as
receipts, provenance, durable workflow state, transcript replay, workspace
cleanup, full lifecycle commands, compatibility shims, PR-bot automation, and
the large policy catalog are intentionally not part of this plan. They live in
the temporary deferred-roadmap planning note.

The core test split is:

- Engineering correctness: unit, CLI, package, helper-script, lint, and release
  tests that can block NVFLARE CI/release.
- Runtime agent performance: trigger correctness, instruction following,
  prohibited-action avoidance, task validation, and benchmark evidence that can
  block public skill promotion.

Do not report normal engineering tests as runtime skill-performance metrics.

## First Implementation Cut

The first useful slice is:

- `nvflare agent skills install --agent codex|claude [--skill <name>]
  [--dry-run] [--format json]`;
- `nvflare agent skills list --agent codex|claude --format json`;
- packaged skills copied from repo-root `skills/` into the NVFLARE wheel;
- minimal released-skill manifest with name, version/source hash, and
  FLARE-version compatibility;
- `nvflare agent inspect <path> --format json`;
- `nvflare agent doctor [--online] --format json`;
- guide-compatible skill layout under `skills/<skill>/`;
- initial lints for frontmatter, size, trigger boundaries, trigger overlap,
  catalog category, global negatives, policy coverage, command drift, helper
  scripts, fixtures, and doc crosslinks;
- at least one public-candidate skill, preferably `nvflare-orient` or
  `nvflare-convert-pytorch`, with `evals/evals.json`.

## Milestone Summary

| Milestone | Theme | Blocks Native Package Mechanics | Required Before First Agent-Skills Release |
| --- | --- | --- | --- |
| 0 | CLI envelope and `nvflare agent` command group | Yes | Yes |
| 1 | Skill source layout and minimal frontmatter validator | Yes | Yes |
| 2 | Wheel manifest and packaging | Yes | Yes |
| 3 | Native skill install/list | Yes | Yes |
| 4 | Read-only inspect and doctor | Yes | Yes |
| 5 | Initial skill lint and admission gate | Yes | Yes |
| 6 | Seed skill bundle | Yes | Yes |
| 7 | Skill benchmark reporting and optional Auto-FL research runs | Internal skill-quality gate, not NVFLARE package mechanics | Yes |
| 8 | Base conversion expansion and customer lifecycle skill wave | No | Yes |
| 9 | Framework conversion skill wave | No | Yes |
| 10 | Specialized workflow skill wave | No | Yes |
| 11 | PET and security skill wave | No | Yes |
| 12 | Export/manifest and publication handoff | Yes | Yes; final handoff gate |

The first native agent-skills release is not complete until skill benchmark
reporting and follow-on skill waves are done. Benchmark evidence follows the
seed bundle and gates later skill waves so one or a small number of skills can
be corrected before the same mistakes are copied across the catalog. It is an
internal quality gate, not an external publication step. Export manifest,
manifest-aware consumers, and publication handoff are combined in Milestone 12
because they package and validate the reviewed release artifacts.

## Initial PR Sequence

The first implementation PRs should stay small and map directly to the
milestones:

1. `nvflare/tool/agent/` skeleton plus reusable JSON envelope helper.
2. Repo-root `skills/` directory plus minimal `SKILL.md` frontmatter validator.
3. Wheel skill manifest plus packaging changes that include released skills in
   the built wheel.
4. Native `nvflare agent skills install/list` for `codex` and `claude`.
5. Read-only `nvflare agent inspect` and `nvflare agent doctor`.

This sequence combines the milestone breakdown with the review recommendation
to avoid bundling source layout, validation, manifest generation, and wheel
packaging into one large PR.

Benchmark performance reporting is not part of the public `nvflare agent`
command surface. Runtime evidence summaries and benchmark drafts belong to the
benchmark tooling until there is enough evidence and UX clarity to promote a
separate supported CLI contract.

## Milestone 0: CLI Envelope

Deliverables:

- Add `nvflare agent` command registration.
- Add shared JSON envelope helper with `schema_version`, `status`, `code`,
  `message`, `hint`, optional `recovery_category`, optional `suggested_skill`,
  and command-specific `data`.
- Define initial agent error codes: `INVALID_ARGS`, `CASE_REQUIRED`,
  `UNKNOWN_SKILL`, `UNKNOWN_CASE`, `EVIDENCE_REQUIRED`,
  `CHECKLIST_SCHEMA_INVALID`, `CHECKLIST_MISMATCH`, `INVALID_BEHAVIOR_ID`,
  `INVALID_STATUS`, `CONFLICTING_EVIDENCE`, `ARTIFACT_NOT_FOUND`,
  `RECORD_WRITE_FAILED`, and `UNSUPPORTED_SCHEMA_VERSION`.
- Add `--format json`, `--format jsonl` for streaming commands where needed,
  and `--schema` conventions for agent-facing commands.

Engineering tests:

- JSON success/error envelope tests.
- stdout/stderr separation tests.
- `--schema` tests that do not require operational arguments.
- non-interactive command tests.

## Milestone 1: Skill Source and Frontmatter

Deliverables:

- Add repo-root `skills/` with guide-compatible structure.
- Add minimal frontmatter validation for `name`, `description`,
  `min_flare_version`, and `blast_radius`.
- Add at least one fixture skill for validator tests.

Engineering tests:

- frontmatter parsing;
- directory-name and skill-name matching;
- required-field failures;
- invalid `blast_radius` fixture.

## Milestone 2: Wheel Manifest and Packaging

Deliverables:

- Add a minimal released-skill manifest with skill name, source hash or version,
  and FLARE-version compatibility.
- Define the source-hash contract for released skills: lowercase hex-encoded
  SHA-256 over sorted files under `skills/<skill>/`. For each included file,
  feed the UTF-8 relative path, a NUL byte, file contents, and a NUL byte into
  the single running SHA-256 state. Exclude `__pycache__`, `.pyc`, and `.pyo`
  files, and reject symlinks rather than following them.
- Update the active build backend configuration, such as `pyproject.toml`,
  `setup.py`, or package-data rules, so the wheel actually includes released
  skill files and the manifest.
- Support a build-time no-skills control wheel for A/B evaluation by honoring
  `NVFLARE_PACKAGE_AGENT_SKILLS=0`. The default remains skill bundling enabled;
  the disabled build should add a `no_skills` wheel build tag, such as
  `1no_skills`, and write an empty bundled-skill manifest rather than omitting
  the bundle package.

Engineering tests:

- source-vs-wheel manifest checks;
- source hash/manifest validation;
- package contents check.
- empty bundle manifest and stale bundled-content cleanup when skill packaging
  is disabled.
- no-skills wheel filename includes the `no_skills` build tag when the wheel
  command is available.

## Milestone 3: Native Install/List

Deliverables:

- Implement `nvflare agent skills install --agent codex|claude`.
- Implement `--skill`, `--dry-run`, `--format json`, and conflict reporting.
- Resolve `codex` to `$CODEX_HOME/skills` or `~/.codex/skills`.
- Resolve `claude` to `~/.claude/skills`.
- If `$CODEX_HOME` is set but the path does not exist, create only the
  `$CODEX_HOME/skills` path needed for installation after normal dry-run and
  conflict checks; report the resolved path in JSON output.
- Implement `nvflare agent skills list --agent codex|claude --format json`.
- Limit `skills list` conflicts to name-overlap conflicts with available
  NVFLARE-source skills or managed NVFLARE installs. Unrelated third-party
  skills already present in the target directory must not appear as conflicts.
- Ensure install does not require Node.js, npm, npx, network access, or an
  external skill CLI.

Engineering tests:

- `native-skill-install-no-node`;
- `skill-install-codex-claude-targets`;
- `skill-install-all-by-default`;
- `skill-install-safe-overwrite`;
- `skill-install-no-third-party-download`;
- `skill-list-ignores-unrelated-third-party-skills`;
- `skill-list-flags-name-overlap-external-skill`.

## Milestone 4: Inspect and Doctor

Deliverables:

- Implement read-only `nvflare agent inspect <path> --format json`.
- Implement static framework and FLARE-integration detection without importing
  or executing user code.
- Implement read-only `nvflare agent doctor --format json`.
- Add optional `doctor --online` bounded read-only checks through the active
  startup-kit context.

Engineering tests:

- static-only inspect fixtures with import side effects;
- redaction fixtures for secrets and sensitive paths;
- symlink and traversal cap fixtures;
- doctor read-only before/after checks.

## Milestone 5: Initial Lints and Admission Gate

Deliverables:

- Add `tools/agent_skill_checks/` or equivalent local entry point.
- Implement lints:
  `skill-frontmatter-lint`, `skill-md-size-lint`, `skill-trigger-lint`,
  `skill-trigger-overlap-lint`, `skill-catalog-category-lint`,
  `skill-global-negative-lint`, `skill-policy-coverage-lint`,
  `skill-command-drift-lint`, `skill-helper-script-lint`,
  `skill-fixture-lint`, and `agent-doc-crosslink-lint`.
- Add the shared global negative prompt bank at
  `skills/_shared/global_negative_prompts.json` with `schema_version: "1"` and
  `prompts` entries containing `id`, `prompt`, and `description`.
- Treat [Agent Skill Evaluation](agent_skill_evaluation.md#engineering-lints)
  as the canonical lint behavior definition; the implementation plan should not
  restate each lint's inputs and pass/fail semantics.
- Implement release checklist coverage only from machine-readable
  `evals/release_checklist.json`; prose-only checklist mentions do not satisfy
  policy coverage lint.
- Implement command-drift lint by parsing `nvflare` snippets from skill docs and
  checking commands/flags against the installed CLI parser or exported command
  schema in the same checkout or wheel.
- Keep `evals/evals.json` guide-compatible and put FLARE-specific IDs under
  `nvflare.mandatory_behavior`, `nvflare.optional_behavior`, and
  `nvflare.prohibited_behavior`.
- Follow the example and behavior-ID pass/fail semantics in
  [Agent Skill Evaluation](agent_skill_evaluation.md#guide-compatible-eval-structure):
  mandatory IDs require evidence, prohibited IDs fail when observed, and
  optional IDs are recorded but non-blocking.
- Treat the 200-line `SKILL.md` limit as the hard initial lint. The roughly
  2,000-token target is advisory and can be reported with a simple
  whitespace-based estimate unless a tokenizer is standardized later.
- Emit lint findings with a shared shape: `id`, `severity`, `file`, `line`
  when available, `message`, and `hint`.

Engineering tests:

- lint fixtures for pass/fail cases;
- trigger-overlap lint fixtures with same-category overlap, adjacent-negative
  coverage, and unrelated-category non-overlap cases;
- CLI command-schema drift fixtures;
- global negative bank schema fixtures and per-skill global-negative coverage
  fixtures;
- doc crosslink fixtures for valid links, stale anchors, stale lint IDs, and
  stale command names.

## Milestone 6: Seed Skill Bundle

Deliverables:

- Add the first hand-vetted public-candidate skills.
- Keep each `SKILL.md` under the 200-line or roughly 2,000-token target.
- Move long content into `references/`.
- Add `evals/evals.json` and minimal `evals/files/` when needed.
- Do not add public runtime skill-evaluation, skill-performance, or
  skill-benchmarking subcommands. Packaged `evals/evals.json` describes what
  benchmark tooling may measure, but it is not useful runtime performance
  evidence by itself.

Recommended first skills:

- `nvflare-orient`;
- `nvflare-convert-pytorch`;
- `nvflare-diagnose-job`.

Exit criteria:

- each public-candidate skill passes the initial admission gate;
- commands referenced by skills match the installed CLI;
- each skill has at least one positive trigger eval and one adjacent negative
  trigger case;
- public-candidate skills pass global negative trigger checks.
- `nvflare agent info`, `doctor`, and command schemas advertise only the
  supported agent commands and no runtime benchmark/evaluator command surface.

## Milestone 7: Skill Benchmark Reporting and Auto-FL Research

Deliverables:

- Do not add a runtime skill evaluator, an eval-on switch, or a public
  evaluator command. Packaged skills must not instruct agents to create
  evaluator-only artifacts before the final response.
- Preserve `skills/<skill>/evals/evals.json` as non-runtime skill metadata for
  trigger coverage, behavior IDs, fixtures, and benchmark/reporting context.
  This metadata is useful for authoring and benchmark analysis, but it does not
  imply that agents should run an evaluator during normal skill use.
- Preserve the source-vs-release distinction: source/dev skills keep `evals/`,
  release/company skill packages can filter it out, and released runtime
  instructions must not reference filtered files.
- Keep user-facing evidence requirements in each skill's workflow,
  requirements, and output sections. Agents should collect only evidence needed
  to complete and explain the user's task.
- Keep performance records as explicit inputs produced by benchmark harnesses,
  reviewer workflows, or other external measurement systems. Benchmark
  reporting may summarize fields that are present, but it must not synthesize
  missing correctness fields or infer token usage from transcript text.
- Keep benchmark summaries and generated benchmark drafts in an explicitly
  internal benchmark tool. Do not expose benchmark reporting through the public
  `nvflare agent` CLI surface, and do not commit generated benchmark reports as
  skill source.
- Measure positive trigger, negative trigger, mandatory behavior,
  prohibited behavior, and task validation for the seed skills before expanding
  to additional skill waves.
- Reuse Auto-FL research workflows as advisory evaluation scenarios after the
  relevant skills exist.

Engineering tests:

- `agent info` and `agent skills --schema` advertise only supported agent
  command contracts;
- `agent skills --schema` lists only supported skills subcommands;
- packaged `SKILL.md` files contain no runtime evaluator hook, eval-on switch,
  or evaluator command instruction;
- the v1 lint id set uses process-metric coverage, not evaluator coverage;
- benchmark tooling consumes supplied records without creating or modifying
  them and keeps any report-generation commands outside the public
  `nvflare agent` CLI surface.

This milestone evaluates the seed skill set before additional skill waves are
implemented. Benchmark evidence is required before a skill is used as a template
for broader catalog expansion. It does not perform external publication or
handoff. Auto-FL remains an advisory research test case: run
existing Auto-FL tasks without skills, with skills available, and optionally
with a skill forced to isolate skill content.

## Milestone 8: Base Conversion Expansion and Customer Lifecycle Skill Wave

The seed bundle plus benchmark reporting prove the package, install, lint,
authoring, and measurement mechanics. The rest of the skill roadmap should be
implemented as follow-on skill-development waves, not left as an unowned
candidate list. Every new skill in these waves must use the authoring and
benchmark contract:

- `SKILL.md` with required frontmatter, trigger boundaries, negative trigger
  guidance, safety boundaries, and validation checklist;
- `references/` for long examples, framework details, diagnosis patterns, and
  command walkthroughs;
- optional `scripts/` only for deterministic JSON-producing helpers that are
  candidates for later promotion into `nvflare agent` commands;
- `evals/evals.json` and fixtures under `evals/files/` when needed;
- admission through the initial skill lints, command-drift checks, trigger
  overlap checks, global negative checks, and doc crosslink checks.

Each wave must include benchmark or reviewer evidence for each new public skill
before that wave is considered complete. A wave can carry a documented
draft/internal exception, but that skill cannot be used as a template for later
waves or included in publication handoff until Milestone 7 benchmark evidence is
available for that skill.

Deliverables:

- Add `nvflare-convert-lightning` as the second base conversion skill after
  PyTorch. Implement it before lifecycle skills so PyTorch and Lightning
  conversion effects can be measured without POC or job-lifecycle context.
- Add `nvflare-poc-workflow` and `nvflare-job-lifecycle`.
- Cover POC lifecycle, job validation, submission, monitoring, log/stat
  download, and target identity/config checks.
- Keep local FLARE installation and virtual-environment repair out of the skill
  catalog. A general-purpose agent can handle "install NVFLARE locally" without
  a FLARE-specific skill, and that workflow cannot depend on NVFLARE already
  being installed.

Skill coverage:

- `nvflare-convert-lightning`
  - Use cases: convert PyTorch Lightning code to an NVFLARE job while
    preserving `LightningModule`, `LightningDataModule`, `Trainer`, callbacks,
    loggers, checkpoints, and multi-GPU/DDP patterns where possible.
  - Primary users: data scientists and developers with Lightning training code
    who need the same PyTorch-family FLARE recipe selection as plain PyTorch
    but a different Client API integration path.
  - What we add: `skills/nvflare-convert-lightning/SKILL.md`, references for
    Lightning detection, Lightning conversion, Lightning validation, and
    Lightning DDP/tracking, plus evals. The skill reuses PyTorch recipe
    discovery and shared PyTorch-family state-dict/tensor exchange guidance;
    it does not introduce a Lightning-only recipe family.
    The canonical patched-trainer loop uses `flare.patch(trainer)`, optional
    `flare.receive()` only for FL task progression and metadata, and leaves
    model load/send ownership to the Lightning callback rather than generating
    manual `FLModel` exchange code.
    Single-GPU or ordinary single-process PyTorch-family training may use
    in-process execution; PyTorch distributed training, including plain PyTorch
    DDP and Lightning DDP, should use external process launch such as
    `launch_external_process=True`. TensorFlow distributed/multi-GPU execution
    is not settled in this milestone and must be verified separately before
    encoding framework guidance. Lightning eval fixtures must enforce the
    patched-loop contract by checking that generated code does not create a
    manual `FLModel` send/receive path or pass `input_model` into `Trainer`.
  - Non-triggers: plain PyTorch without Lightning, exported jobs, failed job
    diagnosis, POC start/stop/cleanup, and job submit/monitor/download.

Milestone 8 Stage 5 checkpoint:

- Run the private deterministic checkpoint with:

  ```bash
  python -m dev_tools.agent.skills.checks.milestone8_checkpoint \
    --repo-root . \
    --benchmark-evidence <stage5-benchmark-evidence.json> \
    --format json
  ```

- The checkpoint validates skill lint/admission, PyTorch/Lightning
  `nvflare agent inspect` routing, install/list behavior, and dev/release
  packaging for the two conversion skills.
- The checkpoint does not run Codex or Claude. It requires separately produced
  benchmark evidence for Codex/PyTorch, Codex/Lightning, Claude/PyTorch, and
  Claude/Lightning conversion runs before Stage 5 can be considered complete.
- Required benchmark evidence per run: correctness, runtime seconds,
  dependency behavior, generated structure, token usage, metric evidence, and
  artifact location. Missing benchmark evidence leaves the checkpoint
  `incomplete`, not `ok`.

- `nvflare-poc-workflow`
  - Use cases: bridge from generated-job simulation (`python job.py` with
    SimEnv) to a local FLARE system with separate server and client processes;
    prepare, start, verify, stop, or clean that local system; confirm
    server/client readiness before a job is submitted; recover orphaned local
    server/client processes when workspace tracking is broken; hand off to job
    lifecycle once the local system is running; protect users from starting over
    an active workspace or cleaning up a workspace with active jobs.
  - Primary users: data scientists who have validated a generated job in
    simulation and want the next local system-level step, developers
    testing examples against real local FLARE processes, and agents that
    finished conversion and need a localhost server/client system before job
    submit.
  - Example prompts: "I ran `python job.py`; what is the next local system
    test?", "Run this exported job with separate local server and client
    processes", "Show me how this job behaves in a local FLARE system instead
    of SimEnv", "I want to submit, monitor, and download this job locally
    before moving to a remote system", "Start a local server and clients for
    this exported job", "Check whether my local FLARE system is running", "Stop
    the local FLARE system", "Clean up the local FLARE system from the last
    run", and "The POC workspace was overwritten; find and stop leftover FLARE
    processes."
  - Evidence triggers: user asks for the next step after SimEnv simulation,
    local server/client processes, localhost system-level testing, POC
    prepare/start/status/stop/cleanup, orphaned process recovery, or a local
    submit target; `nvflare agent doctor` reports no active local system while
    the user wants local submit/monitor/download; process evidence indicates
    leftover local FLARE server/client processes from a POC workspace that can
    no longer track them; or a conversion skill has exported a job and the user
    asks to run it in a local FLARE system rather than only SimEnv.
  - What we add: `skills/nvflare-poc-workflow/SKILL.md`, plus references for
    local POC lifecycle, POC readiness, and cleanup/restore. The skill should
    identify the intended workspace and site count, run existing POC CLI
    commands, verify with `nvflare agent doctor --online --format json` when a
    system should be running, stop through the normal POC CLI path first, and
    use bounded orphan-process discovery only when the workspace can no longer
    track started processes. Killing leftover processes requires clear evidence
    and explicit user confirmation. Before start or cleanup, it should check
    whether the target workspace already has running server/client processes or
    active jobs. Starting a different POC workspace while another local POC is
    running is allowed only with an explicit user choice and a report of the
    active workspace/ports/processes. If Milestone 8 needs a smaller first
    delivery, bounded orphan-process recovery can be deferred without blocking
    Lightning conversion or normal POC lifecycle guidance.
  - Context-loading rule: keep the main `SKILL.md` short. It should route to
    `references/poc-lifecycle.md` for prepare/start/status, to
    `references/poc-readiness.md` only for readiness/status verification, and
    to `references/poc-cleanup-and-restore.md` only when the user asks to stop,
    clean, overwrite, recover orphaned processes, or when evidence shows an
    active/conflicting workspace. This is a shared lifecycle skill for any
    exported FLARE job, regardless of framework or recipe. Framework conversion
    skills must hand off to `nvflare-poc-workflow` instead of duplicating or
    loading these POC references directly.
  - Non-triggers: installing FLARE, repairing the local Python environment,
    converting training code, generating job source, selecting recipes,
    remote startup-kit operation, cloud or multi-machine deployment, or
    leading job submit/monitor/download after the system is already running.
    Environment setup remains ordinary agent assistance and product docs, not a
    lifecycle skill.

- `nvflare-job-lifecycle`
  - Use cases: validate an existing or exported FLARE job, submit it to a
    running FLARE system, monitor status, collect bounded logs/stats, download
    results, and summarize metrics/artifacts from observed outputs.
  - Primary users: data scientists with an exported job folder, users ready to
    submit a generated job, and agents that receive `nvflare agent inspect`
    output with `conversion_state == "exported_job"`.
  - Example prompts: "Validate this job before I submit it", "Submit this
    exported FLARE job", "Monitor job 123 until it finishes", "Download the
    result for the last job", "Show logs and stats for this failed job", "Run
    this job on my POC system", and "Submit this job to the remote FLARE system."
  - Evidence triggers: `nvflare agent inspect <path> --format json` reports an
    exported job, the user points to a FLARE job folder, a POC workflow reports
    a running local system and the next action is submit/monitor/download, or a
    conversion skill has exported a job and the user asks to run it outside
    SimEnv.
  - What we add: `skills/nvflare-job-lifecycle/SKILL.md`, plus references for
    job validation and submit/monitor/download. Result-summary guidance belongs
    in the submit/monitor/download reference rather than a separate metrics
    reference. The skill should inspect the job path, validate structure and
    exported metadata when present, check startup-kit identity/config context,
    use existing `nvflare job` commands, and report job ID/status,
    logs/stats/result paths, metrics, artifacts, and unresolved blockers.
  - `references/job-validation.md` should be loaded only for explicit
    validation or pre-submit checks. It validates the job root, recognized
    `meta.*` file, `deploy_map`, app directories, server/client config files,
    config parseability, obvious missing referenced source files, launcher
    resource hints, exported manifest/fingerprint metadata when present, and
    path/symlink/secret risks. It reports blockers versus warnings with exact
    file evidence. It does not import user code, train, submit, diagnose
    runtime failures, select recipes, or judge model quality.
  - `references/submit-monitor-download.md` should be loaded only when the user
    asks to submit, monitor, inspect status/logs/stats, or download results. It
    must preflight the target FLARE system before submit: no active target
    system, unreachable server, wrong/missing startup kit path,
    mismatched role/identity, or insufficient study authorization are blockers,
    not conditions to silently work around. It should report the active kit,
    target server, user/site role, study context, and exact command that was or
    was not run. It should route user intent to current `nvflare job`
    commands: `submit`, `list`, `meta`, `wait`, `monitor`, `stats`, `logs`, and
    `download`; `abort`, `delete`, and `clone` are explicit-request or
    user-requested paths. Prefer JSON/JSONL output where supported and point to
    `nvflare job <subcommand> --schema` for full command details instead of
    copying the CLI manual into the skill.
  - Non-triggers: raw training-code conversion, POC prepare/start/stop/cleanup,
    local FLARE install/repair, framework recipe generation, cloud/Kubernetes
    deployment, deep root-cause diagnosis where `nvflare-diagnose-job` is the
    lead task, or system-level shutdown/restart flows. Environment
    setup remains ordinary agent assistance and product docs, not a lifecycle
    skill.

Framework conversion skills own generated `job.py`, client code, recipe
selection, SimEnv execution, and export for their framework-specific conversion
requests. For example, a PyTorch request that asks for SCAFFOLD should route to
`nvflare-convert-pytorch`, use `nvflare recipe list` and
`nvflare recipe show scaffold-pt` to select the PyTorch SCAFFOLD recipe, and
generate the converted PyTorch job there. This is the same ownership pattern
already used by `nvflare-convert-pytorch` for FedAvg: the framework conversion
skill discovers the recipe, explains the selection when needed, creates
`client.py` and `job.py`, validates with `python job.py`, and exports with
`python job.py --export --export-dir <exported_job_root>`, where
`exported_job_root` is the exact submit-ready directory later passed to
`nvflare job submit -j <exported_job_root>`.
TensorFlow SCAFFOLD should route to `nvflare-convert-tensorflow` and use the
TensorFlow recipe.

Do not add standalone `nvflare-local-validation` or
`nvflare-identity-and-config` public skills in this milestone. Local validation
should be shared guidance used by conversion skills and
`nvflare-job-lifecycle`. Identity/config checks are part of
`nvflare-job-lifecycle`, not separate skills, until those workflows grow enough
distinct evidence, files, or policy checks to justify their own skill.

## Milestone 9: Framework Conversion Skill Wave

Deliverables:

- Add `nvflare-convert-tensorflow`, `nvflare-convert-huggingface`,
  `nvflare-convert-xgboost`, `nvflare-convert-sklearn`, and
  `nvflare-convert-survival-analysis`.
- Scope each skill to the framework-specific edit pattern, examples, recipes,
  validation commands, and negative triggers defined in the authoring design's
  conversion table.

## Milestone 10: Specialized Workflow Skill Wave

Deliverables:

- Add `nvflare-experiment-tracking`, `nvflare-site-specific-training`, and
  `nvflare-collaborative-etl`.
- Cover TensorBoard/MLflow instrumentation, heterogeneous site scripts or
  app/config folders, federated ETL, preprocessing, feature validation,
  data-quality checks, and safe handoff into training or statistics workflows.

## Milestone 11: PET and Security Skill Wave

Deliverables:

- Start with `nvflare-run-private-set-intersection` for PSI/private set
  intersection.
- Add DP, HE, and privacy-policy-filter skills only after their evidence,
  validation fixtures, safety boundaries, and operational contracts are
  ready.

Each wave should update the product skill catalog in
[Agent Integration](agent_integration.md#product-skill-catalog), the source
tables in [Agent Skill Authoring](agent_skill_authoring.md), and any deferred
roadmap entries that are promoted into current scope. A skill is not considered
implemented just because its name appears in the catalog; implementation means
the full authoring package, engineering lint coverage, and evaluation evidence
exist in the repo.

## Milestone 12: Export/Manifest and Publication Handoff

Deliverables:

- Add `_export_manifest.json` to exported job folders with required files,
  source path/hash, timestamp, NVFLARE version, exporter, and validation status.
- Add a nested `fingerprint` section in `_export_manifest.json` with FLARE,
  Python, recipe, framework dependency, and source-hash metadata.
- Prefer one manifest file with a nested fingerprint unless separate consumers
  need a separate `job_fingerprint.json`.
- Make `nvflare agent inspect` consume `_export_manifest.json` and nested
  fingerprint metadata when present.
- Keep `nvflare agent inspect` compatible with current exported jobs that lack
  the manifest.
- Make future submit preflight consume the same manifest/fingerprint contract
  when that submit-preflight surface is promoted into current scope.
- Tie released skill content to the NVFLARE release that ships it.
- Provide guide-compatible skill files and initial evaluation evidence.
- Use the [Agent Skill Publication Handoff Checklist](agent_publication_handoff_checklist.md)
  when preparing an external catalog handoff.
- Keep external catalog registration, signing, public installer metadata, and
  public scoreboard mechanics outside this implementation plan.
- Do not hand off a skill as public-ready until Milestone 7 runtime evaluation
  passes for that skill.

Engineering tests:

- exported job manifest content and schema tests;
- manifest source-hash and required-file validation tests;
- backward-compatible export tests for jobs that do not request manifest-aware
  behavior;
- inspect tests for exported jobs with and without `_export_manifest.json`;
- stale manifest, missing required file, and source-hash mismatch fixtures;
- preflight compatibility tests when submit preflight is implemented;
- publication handoff checks that released skill content is tied to the NVFLARE
  release and has runtime evaluation evidence.

## Agent Benchmark Harness Architecture Catch-Up

The benchmark harness design is ahead of the current implementation. Before a
second agent adapter is added or the harness is treated as production benchmark
infrastructure, complete these catch-up items:

- move Codex-specific home, model, auth, launch, and compatibility behavior out
  of shared host/container modules and into a `CodexAdapter`;
- replace structural adapter typing with the documented abstract base class and
  registration validation;
- implement `harness/scenarios.py` so direct CLI runs and scenario YAMLs compile
  into one validated `run_plan.json`;
- implement replay mode for record/report development without live agent
  invocation;
- extract container progress and skill-exposure setup into
  `container/progress.py` and `container/skills.py`;
- implement `reports/structure_tree.py` or keep `structure_quality_signal`
  explicitly unavailable until it exists;
- make flat harness modules explicit re-export shims or remove them;
- surface `prompt_hash`, stable failure categories, and
  unavailable `structure_quality_signal` in run summaries;
- add `KNOWN_PENDING_BENCHMARK_AGENTS` behavior for planned but unsupported
  adapters;
- add a named display-record limit and total/displayed counts for skill
  benchmark summaries.

## Deferred Work

Do not implement these in the initial implementation unless a separate scope decision promotes them:

- receipts, provenance, and durable workflow state;
- transcript record/replay;
- workspace cleanup;
- full lifecycle commands beyond install/list;
- compatibility shims, `obsoletes`, and changelog commands;
- PR-bot automation;
- large policy catalog;
- full paired harness, instruction-monitor service, and cost-accounting system;
- public scoreboard mechanics.
