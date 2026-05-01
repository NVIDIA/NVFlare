# Job Download Artifact Contract Test Plan

## Scope

Validate the `nvflare job download --format json` contract for local artifact
discovery after the existing download operation completes. Human output stays
concise and reports only the final download location.

The command must return local CLI-machine paths only. It must not expose server
workspace, job-store, or transfer temporary paths.

## Contract

For `nvflare job download <job_id> --output-dir <path> --format json`, the JSON
envelope `data` must include:

- `job_id`: requested job ID.
- `download_path`: actual final local directory returned by the download API.
- `path`: backward-compatible alias for `download_path`.
- `artifacts`: discovered local artifact paths under `download_path`.
- `missing_artifacts`: expected artifact categories not found locally.

When `--output-dir` is omitted, the CLI should produce a job-specific local final
destination equivalent to `./<job_id>`. The current download API treats its
`destination` argument as a parent directory, so the CLI requests the current
working directory as the parent and reports the actual path returned by the
transfer layer. That path may differ if collision handling adds a suffix.

## Unit Tests

1. JSON success response includes `job_id`, `download_path`, `path`, `artifacts`,
   and `missing_artifacts`.
2. `path` equals `download_path` for backward compatibility.
3. Default output directory passed to `download_job_result()` is the absolute
   current directory parent, preventing a nested `./<job_id>/<job_id>` result
   while preserving the final `./<job_id>` default behavior.
4. Explicit `--output-dir` is passed through as an absolute destination.
5. Artifact discovery reports a global model when a common model file is present,
   for example `FL_global_model.pt`, `global_model.pt`, or `global_model.pth`.
6. Artifact discovery reports `metrics_summary` when `metrics_summary.json` is
   present.
7. Artifact discovery reports `client_logs` as a mapping from site name to local
   `log.txt` path, excluding server logs when identifiable.
8. Missing expected artifact categories are listed in `missing_artifacts` without
   failing the command.
9. Empty or nonexistent download paths produce a successful response with an
   empty `artifacts` object and relevant `missing_artifacts`.
10. Artifact discovery skips symlinked files or directories that could resolve
    outside `download_path`.
11. Existing error behavior remains unchanged for `JobNotFound`,
    `AuthenticationError`, and `NoConnection`.

## Documentation Checks

- CLI docs describe `download_path` as a local path on the CLI machine.
- CLI docs instruct agents to use `data.artifacts.*` as the source of truth.
- CLI docs state that missing artifacts do not make the download fail.
- Design docs state that the server download protocol is unchanged.

## Non-Goals

- No server-side path disclosure.
- No server protocol changes.
- No attempt to guarantee every framework-specific model naming convention.
- No command failure solely because model, metrics, or logs are absent.
