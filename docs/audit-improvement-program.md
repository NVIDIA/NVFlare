# Audit improvement program

This document tracks incremental improvements to NVFlare audit generation. It is
an engineering roadmap, not a description of features already available or a
claim of regulatory compliance. For current behavior and its limitations, see
[Auditing](user_guide/admin_guide/security/auditing.rst).

## Delivery and merge contract

Only one implementation PR is active at a time. Future stages remain plans,
without dependent implementation branches. Every merged PR must leave NVFlare
working and deployable without a later stage. Tests reduce risk; they do not
guarantee the absence of defects.

Before each stage:

1. Refresh canonical `NVIDIA/NVFlare` `main` and verify the predecessor has merged.
2. Read its final diff, review decisions, and relevant CI results. Recheck the
   actual APIs, configuration, file layout, event semantics, and failure behavior.
3. Update downstream assumptions and split or reorder stages when required.
4. Branch from refreshed `main` and define the next PR's behavior, preserved
   defaults, compatibility tests, and rollback procedure.

During review, update tests, documentation, and this roadmap when a decision
changes a shared assumption. After merge, record the PR and merge commit here
in the next stage's PR; an open or approved PR is not a stable dependency.
Fix defects in merged prerequisites through a separate PR before dependent work.
Do not rely on a later stage to restore functionality or complete retrieval.

Each PR requires targeted tests, affected integration tests with verified CI
selection, project style checks, and independent review of the final diff against
its base. Resolve review findings before publication. Include configuration,
reader, peer, download, and rollback checks whenever those boundaries change.
Preserve evidence during rollback and keep deployment activation opt-in.

## Stage ledger

Stage boundaries are provisional. A stage can be split; its complete increment
must still be independently deployable. Status values are planned, active,
merged, and superseded. PR/commit entries are filled from verified GitHub state,
never inferred from a local branch or approval.

| Stage | Status | Deliverable and compatibility boundary | PR / merge commit |
| --- | --- | --- | --- |
| 1 | Active | Characterization tests, current coverage map, and this program document; no production changes. | Pending publication / not merged |
| 2 | Planned | Safe escaping, bounded fields, secret-safe command metadata, and restrictive permissions for new files; retain legacy record structure. | — |
| 3 | Planned | Internal typed event contract and compatibility adapters; retain current API signatures, legacy output, and wire headers. | — |
| 4 | Planned | Optional versioned JSONL, readers for both formats, and site configuration loaded before auditing starts in every process; legacy stays default and formats use separate files. | — |
| 5 | Planned | Continue-and-alert failure handling, health, bounded recovery, persistence status, and optional synchronization; audit failure cannot unexpectedly alter task outcomes. | — |
| 6 | Planned | Process ownership, unique secondary files, job sub-worker placement, and complete authorized retrieval; writer and inventory changes ship together. | — |
| 7 | Planned | Authentication, authorization, command rejection, dispatch, and actual completion events; preserve security and command semantics. | — |
| 8 | Planned | Workflow/task-attempt correlation with distinct preparation, transmission, receipt, acceptance, and completion events; older peers continue working with unknown metadata. | — |
| 9 | Planned | Job/artifact lifecycle events at authoritative boundaries; preserve idempotency and distinguish preparation from confirmed transfer. | — |
| 10 | Planned | Single-owner rotation, safe reopen, and segment retrieval; disabled by default with no automatic deletion. | — |
| 11 | Planned | Optional authenticated collection, bounded persistent buffering, deduplication, and export health; no mandatory service or SDK. | — |
| 12 | Planned | Explicit archival/retention policy, acknowledgement, and protected-storage integration; automatic deletion disabled by default. | — |
| 13 | Planned | Optional durable intent before classified sensitive administrative mutations; continue-and-alert stays default, and completion failure never causes replay. | — |
| 14 | Planned | Regulated deployment profile, failure drills, performance measurements, monitoring, and recovery guidance, assembled from working capabilities. | — |

## Accepted direction and constraints

These are intended future requirements, not settings supported by Stage 1:

- Preserve existing path precedence, default legacy text output, and the
  simulator's default no-op auditor. Make JSONL, rotation, collection, deletion,
  and strict admission opt-in.
- Target regulated deployments while continuing operation and alerting on audit
  failures by default. This permits evidence gaps and must not be advertised as
  lossless auditing.
- Introduce administrator-controlled audit configuration in site
  `local/resources.json`, loaded before auditor creation in every runtime. Job
  configuration must not weaken site policy. Current child startup initializes
  auditing before its normal resource configuration; Stage 4 must address this.
- Distinguish generated event identity from successful configured persistence,
  export acknowledgement, failure, and uncertainty. Recovery summaries count
  known failures, not exact losses across crashes.
- Include UTC time, site/role, process incarnation and sequence, actor provenance,
  action/phase/outcome, and applicable job, workflow, task-attempt, and resource
  identifiers in the structured contract. Keep parent event IDs separate from
  resource IDs; peer-provided metadata is not independently verified identity.
- Preserve existing cross-site event references. Additional fields are optional
  for mixed-version peers. Do not infer success from missing evidence or impose
  a global event ordering from timestamps alone.
- Give secondary writers process-incarnation UUID filenames, with primary
  ownership enforced by a crash-released OS lock. Inventory and authorized
  retrieval must include secondary and rotated files, including external roots.
- Redact before serialization or fallback diagnostics. Exclude credentials,
  raw command arguments, model contents, and private dataset locations by
  default. Record allow-listed identifiers and existing artifact digests.
- Strict admission requires successful local synchronization of an intent before
  a classified sensitive mutation. Audit and business changes are not atomic:
  unmatched intents have unknown outcomes; completion-write failure after commit
  alerts without replay. Retention duration is deployment-specific.

## Baseline and validation evidence

Stage 1 starts from canonical `main` commit
`f787dd919bee1f98e1f0f36ecb9b4c37b6f66154`. Its tests characterize existing writer,
service, and workspace behavior; the coverage map is source-traced. These are
not end-to-end federation, collection, or compliance qualification claims.

Use the repository-local `.venv`. Targeted suites are
`tests/unit_test/fuel/sec/audit_test.py` and
`tests/unit_test/apis/workspace_audit_test.py`, together with existing workspace
tests. They use temporary files and controlled in-process boundaries, not real
child runtimes. The premerge unit command in `.github/workflows/premerge.yml`
collects `tests/unit_test` through `runtest.sh`; no new test runner is needed.

Later stages must add fast integration coverage for actual child startup,
ownership, transport, rotation, retrieval, and collectors as those features ship.
Follow [test placement and review rules](../tests/README.md). A focused passing
suite does not establish a full test-suite audit. Record actual commands and
results in each PR's Testing section, including unavailable checks.

## Industry references

- [OWASP Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html): safe security event contents, sensitive-data exclusion, injection prevention, and failure testing.
- [OpenTelemetry log model](https://opentelemetry.io/docs/specs/otel/logs/data-model/): structured timestamps, source identity, and correlation; no mandatory SDK is implied.
- [NIST SP 800-92](https://csrc.nist.gov/pubs/sp/800/92/final): log management throughout collection, access, retention, and disposal.

No particular retention duration or certification follows from these references.
