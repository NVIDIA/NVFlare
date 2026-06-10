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
  prohibited-action avoidance, task validation, and benchmark notes that can
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
  `nvflare-convert-pytorch`, with `evals/evals.json` and a short
  `BENCHMARK.md`.

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
| 8 | Customer lifecycle skill wave | No | Yes |
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
- Add `evals/evals.json`, minimal `evals/files/` when needed, and a
  hand-authored `BENCHMARK.md` summarizing manual trigger checks, mandatory
  behavior checks, prohibited behavior checks, and known gaps.
- Do not add public `agent skills performance` or `agent skills benchmark`
  subcommands. Packaged `evals/evals.json`
  describes what benchmark tooling may measure, but it is not useful runtime
  performance evidence by itself.

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
- `BENCHMARK.md` clearly states whether the skill is public-ready based on
  manual initial checks or still draft/internal pending Milestone 7 evaluation.
- `nvflare agent info`, `doctor`, and command schemas do not advertise
  `agent skills performance`, `agent skills benchmark`, or
  `agent skills evaluate`.

## Milestone 7: Skill Benchmark Reporting and Auto-FL Research

Deliverables:

- Do not add a runtime skill evaluator, an eval-on switch, or a
  `skills evaluate` command. Packaged skills must not instruct agents to create
  evaluator-only artifacts before the final response.
- Preserve `skills/<skill>/evals/evals.json` as non-runtime skill metadata for
  trigger coverage, behavior IDs, fixtures, and benchmark/reporting context.
  This metadata is useful for authoring and benchmark analysis, but it does not
  imply that agents should run an evaluator during normal skill use.
- Keep user-facing evidence requirements in each skill's workflow,
  requirements, and output sections. Agents should collect only evidence needed
  to complete and explain the user's task.
- Keep performance records as explicit inputs produced by benchmark harnesses,
  reviewer workflows, or other external measurement systems. Benchmark
  reporting may summarize fields that are present, but it must not synthesize
  missing correctness fields or infer token usage from transcript text.
- Keep benchmark summaries and generated benchmark drafts in
  `assist_tools/skills_benchmark` or another explicitly internal benchmark
  tool. Do not expose them as `agent skills performance` or
  `agent skills benchmark` until the benchmark workflow itself is
  stable and the command value is clear without prior hidden setup.
- Use internal benchmark tooling to upgrade `BENCHMARK.md` from manual initial
  summaries to benchmark summaries when automated or repeated evidence exists.
  The rendered file is a publication/review draft; runtime records remain the
  raw evidence.
- Measure positive trigger, negative trigger, mandatory behavior,
  prohibited behavior, and task validation for the seed skills before expanding
  to additional skill waves.
- Reuse Auto-FL research workflows as advisory evaluation scenarios after the
  relevant skills exist.

Engineering tests:

- `agent info` and `agent skills --schema` do not advertise `skills evaluate`,
  `skills performance`, or `skills benchmark` commands;
- invoking the removed `agent skills evaluate`, `agent skills performance`, or
  `agent skills benchmark` command forms returns a structured invalid-args
  error and lists only supported skills subcommands;
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

## Milestone 8: Customer Lifecycle Skill Wave

The seed bundle plus benchmark reporting prove the package, install, lint,
authoring, and measurement mechanics. The rest of the skill roadmap should be
implemented as follow-on skill-development waves, not left as an unowned
candidate list. Every new skill in these waves must use the authoring and
benchmark contract:

- `SKILL.md` with required frontmatter, trigger boundaries, negative trigger
  guidance, approval checkpoints, and validation checklist;
- `references/` for long examples, framework details, diagnosis patterns, and
  command walkthroughs;
- optional `scripts/` only for deterministic JSON-producing helpers that are
  candidates for later promotion into `nvflare agent` commands;
- `evals/evals.json`, fixtures under `evals/files/` when needed, and
  `BENCHMARK.md` with trigger checks, mandatory behavior checks, prohibited
  behavior checks, and known gaps;
- admission through the initial skill lints, command-drift checks, trigger
  overlap checks, global negative checks, and doc crosslink checks.

Each wave must include benchmark or reviewer evidence for each new public skill
before that wave is considered complete. A wave can carry a documented
draft/internal exception, but that skill cannot be used as a template for later
waves or included in publication handoff until Milestone 7 benchmark evidence is
available for that skill.

Deliverables:

- Add `nvflare-setup-local`, `nvflare-poc-workflow`, and
  `nvflare-job-lifecycle`.
- Cover local readiness, POC lifecycle, job validation, submission, monitoring,
  log/stat download, identity/config checks, and production approval boundaries.

Skill coverage:

- `nvflare-setup-local`
  - Covers optional local bootstrap and readiness. Use it when the user wants to
    prepare or repair the local FLARE environment itself, or when another
    workflow detects that FLARE readiness is missing or broken before it can
    continue.
  - Covers fresh-machine or container bootstrap, editable or wheel installation
    guidance, Python and virtual-environment sanity, optional dependency
    guidance, CLI availability for `nvflare`, `nvflare agent`,
    `nvflare recipe`, `nvflare job`, and `nvflare poc`, bundled-skill
    install/list readiness, and final verification through
    `nvflare agent doctor`.
  - Does not cover POC prepare/start/stop/cleanup, POC or production job
    submission, framework conversion edits, generated job source, or routine
    validation that a conversion skill can do for its own output.
  - Does not cover cloud, Kubernetes, Docker production deployment,
    firewall/security-group setup, cert distribution, or multi-machine
    provisioning. Route those to later deployment or provisioning workflows.

- `nvflare-poc-workflow`
  - Covers local POC system lifecycle: prepare, start, verify server/client
    readiness, report current POC workspace and kit state, stop, cleanup, and
    restore the previous kit/config when supported by the CLI.
  - Covers readiness handoff into `nvflare-job-lifecycle` once a local POC
    system is running and the user wants to submit or monitor a job.
  - Does not cover installing FLARE, converting training code, generating job
    source, selecting framework recipes, submitting or monitoring jobs as the
    lead workflow, production startup-kit operation, or cloud/multi-machine
    deployment.

- `nvflare-job-lifecycle`
  - Covers job operations against an already running FLARE system, including POC
    and production/startup-kit based systems: validate an existing or exported
    job before submit, submit, monitor, inspect metadata, collect logs and
    stats, download outputs, summarize metrics/results, and report evidence.
  - Covers SimEnv and exported-job validation only for existing jobs or as an
    independent pre-submit pass. Framework conversion skills own the first local
    validation for jobs they generated.
  - Covers startup-kit identity/config checks, selected server/user/site
    context, explicit production approval prompts, and bounded submit summaries
    before production submission.
  - Does not cover local environment bootstrap, POC prepare/start/stop/cleanup,
    framework conversion edits, recipe-specific generated source, cloud or
    Kubernetes deployment, deep failure diagnosis, or bypassing approval gates
    for production submit/shutdown/delete/restart flows.

Framework conversion skills own generated `job.py`, client code, recipe
selection, SimEnv execution, and export for their framework-specific conversion
requests. For example, a PyTorch request that asks for SCAFFOLD should route to
`nvflare-convert-pytorch`, use `nvflare recipe list` and
`nvflare recipe show scaffold-pt` to select the PyTorch SCAFFOLD recipe, and
generate the converted PyTorch job there. This is the same ownership pattern
already used by `nvflare-convert-pytorch` for FedAvg: the framework conversion
skill discovers the recipe, explains the selection when needed, creates
`client.py` and `job.py`, validates with `python job.py`, and exports with
`python job.py --export --export-dir /tmp/nvflare/job_config/<job_name>`.
TensorFlow SCAFFOLD should route to `nvflare-convert-tensorflow` and use the
TensorFlow recipe.

Do not add standalone `nvflare-local-validation`,
`nvflare-identity-and-config`, or `nvflare-production-submit` public skills in
this milestone. Local validation should be shared guidance used by conversion
skills and `nvflare-job-lifecycle`. Identity/config checks and production-submit
approval are safety gates inside `nvflare-job-lifecycle`, not separate skills,
until those workflows grow enough distinct evidence, files, approval records, or
policy checks to justify their own skill.

## Milestone 9: Framework Conversion Skill Wave

Deliverables:

- Add `nvflare-convert-lightning`, `nvflare-convert-tensorflow`,
  `nvflare-convert-huggingface`, `nvflare-convert-xgboost`,
  `nvflare-convert-sklearn`, and `nvflare-convert-survival-analysis`.
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
  validation fixtures, approval checkpoints, and production safety contracts are
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
