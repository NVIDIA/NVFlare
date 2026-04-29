# Job Submit Token Implementation Plan

## Goal

Add retry-safe job submission with `--submit-token` without changing job `meta.json`
semantics. The token is server-owned submission metadata, not an auth token and not
job-defined execution metadata.

## Scope

- `nvflare job submit --submit-token <token>`
- `nvflare job list --submit-token <token>`
- Server-side study-scoped submit records
- Python API updates
- Tests and documentation

Do not add `--submit-token` to `monitor`, `download`, `abort`, `delete`, or `clone`.

## Design Rules

- Job `meta.json` remains job-owned metadata for FLARE execution, such as
  `deploy_map`, `resource_spec`, `min_clients`, and launcher configuration.
- `submit_token` is stored only as server-owned submission metadata.
- Submit-token scope is `(server/project context, study, submitter identity, submit_token)`.
- Same scope + same token + same job content returns the existing `job_id`.
- Same scope + same token + different job content returns `SUBMIT_TOKEN_CONFLICT`.
- Same submitter and same study without `--submit-token` keeps existing behavior and
  creates a new job for each submit.
- Same submitter and same study with different `--submit-token` values creates separate
  jobs, even when content is identical.
- Persistent job storage is the source of truth. Any in-memory lookup is only a cache.

## 1. CLI Commands

Affected commands:

- `nvflare job submit`
- `nvflare job list`

Changes:

- Add `--submit-token <token>` to submit.
- Add `--submit-token <token>` to list.
- Validate token syntax: non-empty, at most 128 characters, matching
  `^[A-Za-z0-9._:-]{1,128}$`.
- Keep current behavior when the flag is absent.

## 2. Python API

Affected APIs:

- `Session.submit_job`
- `Session.list_jobs`
- `SessionSpec`

Changes:

- Extend `submit_job(job_definition_path, submit_token=None)`.
- Extend `list_jobs(..., submit_token=None)`.
- Preserve compatibility: `submit_job()` still returns `job_id`.

## 3. File Transfer

Current `push_folder` rebuilds the server command as only `submit_job <folder>`,
which drops flags after the job folder.

Change:

- Preserve extra submit args after the folder name so `--submit-token` reaches the
  server.

## 4. Server Admin Command

Affected:

- `JobCommandModule.submit_job`
- Server-side submit command parser/usage

Changes:

- Parse `submit_job <folder> [--submit-token token]`.
- Validate token syntax on the server.
- Return `SUBMIT_TOKEN_CONFLICT` when the same scoped token is reused for different
  content.

## 5. Job Store / Submit Record Storage

Add server-side submit records in the same job store, but outside the normal job
namespace.

Recommended namespace:

```text
job_submit_records/<study_hash>/<submitter_hash>/<submit_token_hash>
```

Record payload:

```json
{
  "schema_version": 1,
  "state": "creating|created",
  "submit_token": "TOKEN",
  "job_id": "pre-generated-job-id",
  "study": "cancer",
  "submitter_name": "...",
  "submitter_org": "...",
  "submitter_role": "...",
  "job_name": "hello-numpy",
  "job_folder_name": "hello-numpy",
  "job_content_hash": "sha256:...",
  "submit_time": "2026-04-29T10:00:00-07:00"
}
```

Do not store `submit_token` in job `meta.json`.

## 6. JobDefManager API

Avoid putting submit-record storage logic directly in `job_cmds.py`.

Add public manager methods, for example:

- `get_submit_record(study, submitter, submit_token, fl_ctx)`
- `create_submit_record(record, fl_ctx)`
- `update_submit_record(record, fl_ctx)`
- `get_job_by_submit_token(study, submitter, submit_token, fl_ctx)`

Implement them in `SimpleJobDefManager`. Use a manager lock or no-overwrite storage
semantics around the same-scope token creation path.

## 7. Submit Flow

With `--submit-token`:

1. Validate uploaded job as today.
2. Compute canonical job content hash.
3. Build scoped submit-record key.
4. If record exists:
   - same hash + job exists: return existing `job_id`;
   - same hash + job missing + `state: creating`: retry creation with recorded `job_id`;
   - different hash: return `SUBMIT_TOKEN_CONFLICT`.
5. If record does not exist:
   - pre-generate `job_id`;
   - create submit record with `state: creating`;
   - call existing job creation with that `job_id`;
   - update submit record to `state: created`;
   - return `job_id`.

## 8. Job List Flow

With `--submit-token`:

1. Resolve the submit record under the selected study and current submitter.
2. Fetch the referenced `job_id` from the job store.
3. Return the normal job list shape for that job.
4. If no record exists, return an empty list.

There is no reserved `all` study selector. Callers either omit `--study` for the default
study or specify one concrete study name.

Do not scan job `meta.json` for `submit_token`.

## 9. Content Hashing

Use a canonical hash, not a raw uploaded zip hash.

Reason:

- job folders may be signed during file transfer;
- volatile signing artifacts can change between submits.

Hash sorted relative paths and bytes of real job content, excluding volatile signing
artifacts such as:

- `.__nvfl_sig.json`
- submitter certificate file, if applicable

## 10. Concurrency / Crash Recovery

Requirements:

- Two concurrent submits with the same scope and token must not create duplicate jobs.
- A retry after client timeout must resolve to the accepted job.
- A retry after server restart must still work from persistent storage.

Approach:

- Submit-record creation uses no-overwrite semantics or an equivalent manager lock.
- Create the submit record before the job object with `state: creating`.
- Create the job using the pre-generated `job_id`.
- Update the submit record to `state: created`.

Recovery cases:

- record exists, job exists: return existing job.
- record exists, job missing: recreate with recorded `job_id`.
- record missing: create fresh record and job.

## 11. Error Contract

Add CLI/server error:

- `SUBMIT_TOKEN_CONFLICT`

Meaning:

- Same scoped token was already used for different job content.

Response should include safe context:

- conflict code;
- existing `job_id`, if available;
- hint to use a new token for a new job.

## 12. Documentation

Update:

- `docs/design/nvflare_cli.md`
- `docs/user_guide/nvflare_cli/job_cli.rst`
- command help/schema examples if applicable

Clarify:

- token is not auth;
- token is study-scoped;
- token is not job `meta.json`;
- no token preserves existing behavior;
- `job list --submit-token` is the recovery path.

## 13. Tests

Unit tests:

- submit parser accepts/rejects token.
- list parser accepts token.
- file transfer preserves submit args.
- token validation.
- same token + same content returns same job.
- same token + different content conflicts.
- same token in different study is independent.
- no token creates duplicate jobs as today.
- submit token is not present in job `meta.json`.
- canonical hash ignores signature artifacts.
- list by token returns expected job.

Integration-style tests:

- submit with token, simulate timeout/retry, confirm one job.
- restart/recreate manager from persisted store, list by token still works.

## Agent Work Split

Suggested parallel work ownership:

- CLI/API implementer: tasks 1, 2, and command help/schema.
- File-transfer/server parser implementer: tasks 3 and 4.
- Storage implementer: tasks 5, 6, 7, and 10.
- Job-list/query implementer: task 8.
- Hash/error implementer: tasks 9 and 11.
- Docs implementer: task 12.
- Test implementer: task 13.
- Drift monitor: verifies implementation follows this plan and does not move
  `submit_token` into job `meta.json`.
- Architecture/security/performance reviewer: checks persistence semantics,
  authorization scope, token hashing, concurrency, and storage scans.
- Coordinator: tracks dependencies, merge order, test coverage, and remaining gaps.
