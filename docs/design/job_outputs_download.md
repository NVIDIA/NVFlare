# Job Outputs Download

## Summary

Downloading a finished job today returns the whole job: the submitted job
(`job.zip`), its metadata (`meta.json`) and the complete server workspace
(`workspace.zip`). Most users only want the job's result: the global model,
the statistics, the metrics or the evaluation report.

This design lets a job declare its output files in `meta.json` and adds an
outputs-only download. It is opt-in in both directions:

- A job that declares no outputs behaves exactly as today.
- A caller that does not ask for an outputs-only download receives the whole job,
  as today.

No persistor, controller or job layout changes are needed. Output files keep
the same relative paths as in a full download, so code that reads them works
with either download.

## Current State

| Step | Behavior |
| --- | --- |
| Job end | `JobRunner._save_workspace` archives the run, result, log and audit folders into the `workspace` job-store component. Each folder's files are stored relative to that folder, so all of them sit at the archive root. By default the four folders are the same run folder. |
| Admin command | `download_job <job_id>` copies the `data`, `meta` and `workspace` components into a transfer folder. The admin client downloads them and unzips them into `<job_id>/job/`, `<job_id>/meta.json` and `<job_id>/workspace/`. |
| Client API | `Session.download_job_result(job_id, destination=None)` sends `download_job`. Recipe `Run.get_result()` and `nvflare job download` call it. |

The server app folder is `app_server/` in the run folder. The common results are
already at stable, known paths:

| Usage pattern | Result | Path in the server workspace |
| --- | --- | --- |
| PyTorch training | Global model, best model | `app_server/FL_global_model.pt`, `app_server/best_FL_global_model.pt` |
| TensorFlow training | Global model | `app_server/tf_model.weights.h5` |
| Scikit-learn, XGBoost bagging | Global model | `app_server/<persistor file name>` |
| Training metrics | Metrics artifact | `metrics/metrics_summary.json`, `metrics/round_metrics.jsonl` |
| Federated statistics | Statistics | The recipe's `stats_output_path`, relative to the run folder |
| Cross-site evaluation | Evaluation report | `cross_site_val/cross_val_results.json` |

All these file names are set by recipe or component arguments, so the code that
configures the job already knows them. Histogram-based and vertical XGBoost
save their models on the clients, so they have no server-side model.

## Goals

- A job can declare which server workspace files are its result.
- A caller can download only those files, at the same paths as a full download.
- No existing job, script, execution environment or deployment changes behavior
  unless it opts in.
- Failed jobs still give the caller what they need to debug.

## Non-Goals

- No change to where persistors or controllers write files.
- No change to the stored `workspace` archive or to the existing `download_job`
  command.
- No result files from client sites. Only the server workspace is covered.
- No change to the default download. Making outputs-only the default is a
  separate, later decision.

## Design

### Job metadata: `job_outputs`

A new optional key in `meta.json`:

```json
{
  "job_outputs": [
    "app_server/FL_global_model.pt",
    "app_server/best_FL_global_model.pt",
    "metrics/"
  ]
}
```

- Each entry is a path relative to the root of the server workspace archive.
  That root is the job's run folder. When separate result, log or audit roots
  are configured, their contents are merged at the same root.
- An entry ending in `/` selects that folder and everything under it.
- Entries are exact paths. There are no wildcards.
- `JobMetaKey.JOB_OUTPUTS = "job_outputs"`.
- The existing `JobMetaKey.RESULT_LOCATION` and `JobDefManager.set_results_uri`
  are removed. Nothing calls `set_results_uri`, it is not part of
  `JobDefManagerSpec`, and no stored job has `result_location`. Removing them
  leaves one key for job results.
- `JOB_OUTPUTS` is added to `CLONED_META_KEYS`, so a cloned job keeps its
  declaration.

### Validation

The server's `JobMetaValidator` checks `job_outputs` at submission and rejects
the job when the value is invalid. Errors appear at submit time, not after a
long run.

- The value is a list of at most 100 non-empty strings.
- Each entry uses `/` separators, is relative, and contains no `..`, `.`, empty
  segment, backslash or drive prefix.
- Duplicates are removed.

### Setting the key

- `JOB_OUTPUTS` is added to `USER_SETTABLE_JOB_META_KEYS`, so
  `set_recipe_meta(recipe, JobMetaKey.JOB_OUTPUTS, [...])` works and checks the
  same rules on the client. A user's value replaces the built-in list.
- Built-in recipes set it from the file names they pass to their components.
  They normalize each name with `posixpath.normpath` first, so a documented
  input such as `stats_output_path="./output"` is declared as `output`. A recipe
  never fails because of its own declaration.
- Utilities that add a result-producing component after the recipe is built,
  such as `add_cross_site_evaluation`, append to the existing list rather than
  replacing it.
- Hand-written jobs may add the key to `meta.json` or leave it out.

### Admin command: `download_job` with an `outputs_only` option

The existing `download_job <job_id>` command gains an optional `outputs_only`
option. There is no new command, so authorization, the permission category and
the command list are unchanged.

The option travels as a command property, not as a command argument:

- The admin protocol already sends optional per-command properties with a
  command, and the server places them on the connection. `app_command` passes
  its data this way.
- An older server ignores a property it does not use and returns the full
  download. A command argument would instead be rejected: the server's
  `authorize_job` validates extra arguments as targets.
- The admin client's folder pull already uses a second argument as the local
  folder name. The only client change is that it forwards the command's
  properties to the server.

The client sends `{"outputs_only": true}`. Without it, `download_job` is
unchanged. With it, the server:

1. Requires the job to be finished, as today.
2. Prepares the full download when:
   - the job did not finish successfully, or
   - the job's metadata has no `job_outputs`.
3. Otherwise copies the `workspace` component into the transfer folder, and
   writes a new `workspace.zip` there that holds only the declared entries.
   Then it removes the copied archive. It uses the job store's existing
   `get_storage_for_download`, so it works with any `StorageSpec` without
   loading the archive into memory. The admin client unzips the filtered
   archive into `workspace/` as it does today, and a large folder entry is still
   one transfer.
4. Always writes `outputs_manifest.json` with:
   - `outputs_only`: whether the download was filtered;
   - `reason`: why the download was not filtered, when it was not;
   - `found` and `missing`: the declared entries, when it was filtered.

   A missing entry is not an error. For example, the best model exists only
   when the workflow tracked one.
5. Includes `meta.json`, as today, so the caller still has the job status and
   timing.

The download mode travels in the manifest, not in the command reply, because
the admin client's folder pull returns only the local location to the caller.
A download that asked for outputs only and has no manifest came from an older
server.

Extraction matches each archive member's normalized name against the declared
entries and rejects symlink entries. It never writes outside the transfer
folder.

A completed job whose workspace could not be archived gets the same internal
error with or without `outputs_only`.

Outputs-only layout, with the same relative paths as a full download:

```text
<job_id>/
  meta.json
  outputs_manifest.json
  workspace/
    app_server/FL_global_model.pt
    app_server/best_FL_global_model.pt
    metrics/metrics_summary.json
    metrics/round_metrics.jsonl
```

### Client API

| API | Change |
| --- | --- |
| `Session.download_job_result(job_id, destination=None, outputs_only=False)` | When `outputs_only=True`, sends `download_job` with the `outputs_only` property. When the download has no `outputs_manifest.json`, the server is older: it logs one line that the whole job was downloaded. |
| `ExecEnv.get_job_result(job_id, timeout=0.0, outputs_only=False)` | POC and production environments pass `outputs_only` to the session. The simulator runs locally and returns its workspace unchanged. |
| `Run.get_result(timeout=0.0, clean_up=True, outputs_only=False)` | Passes `outputs_only` to the environment only when it is `True`, so existing custom `ExecEnv` subclasses with the two-argument signature keep working. |
| `nvflare job download <job_id> --outputs-only` | Passes `outputs_only=True`. JSON output adds `outputs_only` and `missing_job_outputs` from the manifest. `artifacts` and `missing_artifacts` describe what was downloaded. For example, client logs are listed as missing in an outputs-only download. |

The returned path is the downloaded `<job_id>` folder in both modes. Result
files are at the same relative path under `workspace/` in both modes, so the run
summary and the CLI's artifact discovery find the model and metrics either way.

## Compatibility

| Case | Behavior |
| --- | --- |
| Existing jobs and job folders | No `job_outputs`. Every download is unchanged. |
| Existing scripts | `outputs_only` defaults to `False`. Unchanged. |
| Existing custom `ExecEnv` subclasses | `Run` does not pass the new argument unless the caller asks for an outputs-only download. |
| New client, old server | The server ignores the property and returns the whole job. The client sees no manifest and logs a notice. |
| Old client, new server | The old client never sends the property. Unchanged. |
| Code that references `JobMetaKey.RESULT_LOCATION` or calls `set_results_uri` | Must drop the reference. Neither ever held or produced a value. |
| A job with `job_outputs` on an old server | The old server stores the key and ignores it. Downloads are full. |
| Failed or aborted job with `outputs_only=True` | The whole job, including logs. The manifest says it was not filtered and why. |

## Security

- The command, its authorization and its permission category are unchanged,
  so no new data becomes reachable. Outputs-only download returns a subset of what
  `download_job` already returns.
- The server validates paths at submission and again at extraction, and
  rejects symlink entries in the archive.
- `job_outputs` names files, not contents. It must not contain secrets, the
  same as other job metadata.

## Testing

Unit tests:

- `job_outputs` validation: valid entries, folder entries, and rejection of
  absolute paths, `..`, `.`, backslashes, empty segments, non-strings and
  oversized lists.
- `set_recipe_meta` accepts the key and applies the same checks. Recipe
  declarations are normalized, and `add_cross_site_evaluation` appends to them.
- Cloned jobs keep `job_outputs`.
- `download_job` with `outputs_only`:
  - writes a filtered `workspace.zip` with only the declared entries, at their
    original paths;
  - reports missing entries in the manifest;
  - rejects symlink entries;
  - falls back to the whole job for undeclared or failed jobs, with the reason
    in the manifest;
  - is unchanged without the property.
- The admin client forwards command properties on a folder pull.
- `Session.download_job_result(outputs_only=True)` reports a full download when
  the server returns no manifest.
- `Run.get_result()` works with an `ExecEnv` that implements the two-argument
  `get_job_result`.
- Built-in recipes declare files that match the names they pass to their
  components.

Integration test: run a PyTorch FedAvg job in POC, download it with
`outputs_only=True`, and check that the model file is present at its usual path
and other workspace files are absent.

## Rollout

1. Metadata key, validation, cloning and `set_recipe_meta` support, and removal
   of `result_location` and `set_results_uri`.
2. `download_job` option, client API, recipe `Run` and CLI flag.
3. Built-in recipe declarations, starting with PyTorch FedAvg, federated
   statistics and cross-site evaluation.

Each step is usable alone. Until a recipe declares its files, an outputs-only
request for its jobs returns the whole job.
