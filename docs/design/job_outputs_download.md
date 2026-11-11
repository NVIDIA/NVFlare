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

No persistor, controller or job layout changes are needed. When a job that
declares outputs ends, the server stores those outputs apart from the rest of
the workspace, so each file is stored once. An outputs-only download sends the
outputs archive, and a full download sends both archives, unzipped into the
same `workspace/` folder. Either way the server replies at once. Output files
keep the same relative paths as in a full download, so code that reads them
works with either download.

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
| Scikit-learn, XGBoost bagging | Global model | `app_server/<persistor file name>`, such as `app_server/model_param.joblib` |
| NumPy training | Global model | `models/server.npy` (the persistor writes it under the result root, which is merged at the archive root) |
| FedAvg without a persistor | Global model | The controller's `save_filename`, relative to the run folder |
| Training metrics | Metrics artifact | `metrics/metrics_summary.json`, `metrics/round_metrics.jsonl` |
| Federated statistics | Statistics | The recipe's `stats_output_path`, relative to the run folder |
| Cross-site evaluation | Evaluation report | `cross_site_val/cross_val_results.json` |

All these file names are set by component arguments, so the code that
configures each component (the recipe, its model wrapper or its job class)
already knows them. Histogram-based and vertical XGBoost
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
  command's results. A job that declares outputs stores them apart from the
  rest of its workspace, and a full download returns the same files as before.
- No result files from client sites. Only the server workspace is covered.
- No change to the default download. Making outputs-only the default is a
  separate, later decision.

## Using It from Recipes

### Built-in recipes: nothing to declare

Built-in recipes declare their outputs. The caller only asks for them:

```python
from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import ProdEnv, add_cross_site_evaluation

recipe = FedAvgRecipe(
    name="hello-pt",
    min_clients=2,
    num_rounds=5,
    model=SimpleNetwork(),
    train_script="client.py",
)
# Declares app_server/FL_global_model.pt, app_server/best_FL_global_model.pt
# and metrics/. Cross-site evaluation adds cross_site_val/.
add_cross_site_evaluation(recipe)

env = ProdEnv(startup_kit_location="/path/to/admin@nvidia.com")
run = recipe.execute(env)

path = run.get_result(outputs_only=True)
model_path = os.path.join(path, "workspace", "app_server", "FL_global_model.pt")
```

Federated statistics work the same way. The recipe declares its
`stats_output_path`, normalized:

```python
from nvflare.recipe.fedstats import FedStatsRecipe

recipe = FedStatsRecipe(
    name="stats",
    stats_output_path="./output",  # declared as "output"
    sites=["site-1", "site-2"],
    statistic_configs={"count": {}, "mean": {}, "stddev": {}},
    stats_generator=my_stats_generator,
)
run = recipe.execute(env)
path = run.get_result(outputs_only=True)
stats_path = os.path.join(path, "workspace", "output")
```

### Choosing the outputs

`set_recipe_meta` replaces the built-in list. For example, to keep only the
final model:

```python
from nvflare.apis.job_def import JobMetaKey
from nvflare.recipe import set_recipe_meta

set_recipe_meta(recipe, JobMetaKey.JOB_OUTPUTS, ["app_server/FL_global_model.pt"])
```

A recipe declares the files of the components it creates. With a persistor the
user passes as `model_persistor`, the recipe does not know the model's file
name, so it declares nothing, and the job downloads whole until its outputs
are declared. A job that writes its own files declares them the same way.
Paths are relative to the server run folder:

```python
set_recipe_meta(recipe, JobMetaKey.JOB_OUTPUTS, ["app_server/report.json", "plots/"])
```

### When the whole job comes back

The server sends the whole job when the job did not complete, declares no
outputs, or has no outputs archive, for example because it produced none of its
declared outputs. The caller then sees one line that says why, for example:

```text
Downloaded the whole job 3f2a... because the job has no archived outputs: none of its declared outputs were produced, or it finished before outputs were archived
```

The downloaded folder has the same layout either way, so code that reads
`workspace/app_server/FL_global_model.pt` works with both.

### Default behavior

Without `outputs_only=True`, nothing changes:

```python
path = run.get_result()  # whole job, as today
```

With `SimEnv`, the job runs locally and `get_result(outputs_only=True)` returns
the local workspace unchanged.

### Outside recipes

```python
from nvflare.fuel.flare_api.flare_api import new_secure_session

sess = new_secure_session("admin@nvidia.com", "/path/to/admin@nvidia.com")
path = sess.download_job_result(job_id, outputs_only=True)
```

```shell
nvflare job download <job_id> --outputs-only
```

A hand-written job adds `job_outputs` to its `meta.json`, as shown below.

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
- The code that adds a result-producing server component records that
  component's files on the job, in the component's framework module, through
  helpers in `nvflare/job_config/job_outputs.py`. No component class changes.

  | Where | Records |
  | --- | --- |
  | PyTorch model wrapper (`PTModel`) and PyTorch FedOpt recipe | `app_server/<global_model_file_name>`, `app_server/<best_global_model_file_name>` of the `PTFileModelPersistor` they create |
  | TensorFlow model wrapper (`TFModel`) | `app_server/<save_name>` of its `TFModelPersistor` |
  | Scikit-learn recipes | `app_server/<save_name>` of the `JoblibModelParamPersistor` they create |
  | XGBoost bagging recipe | `app_server/<save_name>` of its `XGBModelPersistor` |
  | NumPy FedAvg and logistic regression recipes | `<model_dir>/<model_name>` of their NumPy persistor, with `.npy` added when missing, as `np.save` does |
  | Base `FedAvgRecipe` without a persistor | `<save_filename>`, which the `FedAvg` controller saves itself, relative to the run folder |
  | `BaseFedJob` and the recipes that add a `MetricsArtifactWriter` | `<results_dir>/`, or the summary and round files by name when `results_dir` is the run folder itself, as supporting outputs |
  | `StatsJob` (federated statistics) | `<output_path>` |
  | `add_cross_site_evaluation`, `add_final_global_evaluation`, NumPy cross-site evaluation recipe | `cross_site_val/` |

- Only the exact component class the code knows is recorded, and the
  persistor it records is marked as accounted for. A persistor the user passes
  as `model_persistor`, a subclass, or another metrics component may save
  elsewhere, so it marks the job's outputs unknown. A recipe that creates a
  persistor and passes it on as `model_persistor` (the scikit-learn recipes)
  tags it with a function that names its files; the function runs when the
  persistor is added, so later setting changes are reflected, and a copy of
  the persistor carries no tag.
- Metrics are supporting outputs. A job whose record holds only metrics has a
  main result, such as a model, that nothing recorded.
- `Recipe.export` and `Recipe.run` turn the record into `job_outputs`, so a
  component added after the recipe is built, such as by
  `add_cross_site_evaluation`, is included. Each path is normalized with
  `posixpath.normpath`, so `stats_output_path="./output"` is declared as
  `output`. A path that would not pass validation is skipped, and a file inside
  a declared folder is not listed again. A recipe never fails because of its
  own declaration.
- Nothing is declared, and the job downloads whole, when:
  - some outputs are unknown, or only supporting outputs were recorded;
  - a model persistor on the server was added without a declaration of its
    files. This check runs at export over the job's server components and
    uses only the framework-neutral `ModelPersistor` base class, so a missed
    declaration fails safe;
  - a recorded path is invalid, such as an absolute folder;
  - the job ships a `config/environment.json` for the server app, which can
    move where components write (for example a PyTorch checkpoint folder); or
  - there are more than `MAX_JOB_OUTPUTS` entries.

  A declaration without the model would leave it out of an outputs-only
  download.
- A value set with `set_recipe_meta` replaces the recorded list and is kept as
  set, including when components are added later.
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

### Storing the outputs at job end

When a job ends, `JobRunner._save_workspace` archives the run folder (with any
separate result, log and audit roots) into the `workspace` job-store component
and then deletes the folder. For a job that declares outputs, it first stores
them as their own `outputs` component:

1. **Stage.** Move the declared output files out of the workspace folders into
   a staging folder beside them, at their workspace paths. On the same
   filesystem a move is a rename, so nothing is copied. Symlinks are skipped.
   A later workspace folder's file replaces an earlier one's at the same path,
   as unzipping the workspace does; the earlier file stays in the workspace.
   The log files the server's log commands read (`log.txt`, `log.json`,
   `error_log.txt`) are copied rather than moved, so those commands keep
   working when a job declares them.
2. **Store the outputs.** `JobDefManagerSpec.save_job_outputs(job_id,
   [staging_folder])` archives the staging folder as the `outputs` component,
   with the same folder contract `save_workspace` uses, so any job store that
   archives a workspace archives the outputs the same way.
3. **Store the rest.** `save_workspace` archives the workspace folders, which
   no longer contain the declared outputs.
4. Delete the staging folder and the workspace folders, as today.

Each file is read and written once, as before this change, so job end takes no
longer and nothing is stored twice.

When nothing declared was produced, nothing is staged and the workspace holds
everything, as before. When the outputs are not stored, because
`save_job_outputs` fails or returns None (its default, so job-store
implementations without it keep working), the staged files are moved back
before the workspace is archived, so nothing is lost.

**Why at job end.** By the time anyone downloads, the result files exist only
inside stored archives; the run folder is gone. A download replies at once
because the server only links stored archives into the transfer folder and
streams them after replying. Building a subset on request would mean copying
the declared entries out of the workspace archive before replying, which for a
multi-GB model can exceed how long the admin client waits for a reply (5
seconds unless a command timeout is set). Job end is the last moment the
outputs exist as files, so they can be separated without copying.

### Full download

A full download of a job with an `outputs` component sends `workspace.zip` and
`outputs.zip`, both linked from the store, so the server still replies at
once. Its reply carries `unzip_into: {"outputs.zip": "workspace"}`, and the
admin client unzips `outputs.zip` into `workspace/` after `workspace.zip`. The
result is the same `workspace/` folder as before this change. The client
accepts only a plain folder name there, never a path. A job without an
`outputs` component downloads exactly as before.

### Serving an outputs-only download

The client sends `{"outputs_only": true}`. Without it, `download_job` is
unchanged. With it, the server:

1. Requires the job to be finished, as today.
2. Prepares the full download when:
   - the job did not finish successfully,
   - the job's metadata has no `job_outputs`, or
   - the job has no `outputs` component: none of its declared outputs were
     produced, it finished before this change, or storing them failed.
3. Otherwise checks the job's component list, so a job store that raises
   for a missing component is never asked for one, and prepares `meta.json`
   and the `outputs` component as `workspace.zip`, with the job store's existing `get_storage_for_download`,
   exactly as a full download prepares the `workspace` component.
   `FilesystemStorage` links the stored archive, so the server replies at once
   and streams the file afterwards. If preparing the outputs fails, the server
   prepares the whole job instead. The admin client unzips it into
   `workspace/` as usual.
4. Reports the result in its reply to `download_job`, beside the file list:
   - `outputs_only`: whether only the declared outputs were sent;
   - `outputs_only_reason`: why not, when not.

   A missing entry is not an error. For example, the best model exists only
   when the workflow tracked one.
5. Includes `meta.json`, as today, so the caller still has the job status and
   timing.

The admin client's folder pull passes these two reply fields through to its
caller, next to the local location. A reply to an outputs-only request without
`outputs_only` came from an older server. No file is added to the download.

The `outputs` component is a system component: it is not listed or downloaded
as an additional job component.

Outputs-only layout, with the same relative paths as a full download:

```text
<job_id>/
  meta.json
  workspace/
    app_server/FL_global_model.pt
    app_server/best_FL_global_model.pt
    metrics/metrics_summary.json
    metrics/round_metrics.jsonl
```

### Client API

| API | Change |
| --- | --- |
| `Session.download_job_result(job_id, destination=None, outputs_only=False)` | When `outputs_only=True`, sends `download_job` with the `outputs_only` property. When the whole job came back, it logs one line that says so and why: the reply's `outputs_only_reason`, or, when the reply has no `outputs_only`, that the server is older. |
| `ExecEnv.get_job_result(job_id, timeout=0.0, outputs_only=False)` | POC and production environments pass `outputs_only` to the session and print the same notice when the whole job came back. The simulator runs locally and returns its workspace unchanged. |
| `Run.get_result(timeout=0.0, clean_up=True, outputs_only=False)` | Passes `outputs_only` to the environment only when it is `True`, so existing custom `ExecEnv` subclasses with the two-argument signature keep working. |
| `nvflare job download <job_id> --outputs-only` | Passes `outputs_only=True`. Text output prints the notice when the whole job came back. JSON output adds `outputs_only` (whether the download was filtered) and `outputs_only_notice`. `artifacts` and `missing_artifacts` describe what was downloaded. For example, client logs are listed as missing in an outputs-only download. |

The returned path is the downloaded `<job_id>` folder in both modes. Result
files are at the same relative path under `workspace/` in both modes, so the run
summary and the CLI's artifact discovery find the model and metrics either way.

## Alternatives Considered

### Producing the outputs-only download

| Option | Outcome |
| --- | --- |
| Filter the stored workspace archive on each request: copy the declared entries into a new archive, then reply | Rejected. The copy runs before the server replies, and for a multi-GB model it can exceed the admin client's reply wait (5 seconds unless a command timeout is set). The download fails on exactly the large-model jobs it is for, and the path differs from a full download. |
| Filter on request, but make the client wait longer for this one reply | Rejected. It hides the cost rather than removing it, the copy still grows with the model, and outputs-only downloads still take a different path from full downloads. |
| Reply first, then prepare the files | Rejected. It needs a rework of how folder downloads start, for one command. |
| Download the whole job and filter on the client | Rejected. It saves nothing on transfer, which is the point of the feature. |
| Archive the declared outputs at job end as a second copy, keeping the workspace archive whole | Rejected. Downloads reply at once, but every declared output, usually the model, is stored twice, and job end reads and writes it a second time. For large models that doubles their storage. |
| **Store the declared outputs at job end apart from the rest of the workspace** | **Chosen.** The outputs are moved, not copied, out of the run folder and archived on their own; the workspace archive holds everything else. Each file is stored once and written once, as before. An outputs-only download links the outputs archive, and a full download links both, so both reply at once. The admin client merges the two archives into `workspace/`. |

The server gathers the outputs into a staging folder and hands the folder to the
job store, the same folder contract `save_workspace` uses. Each store archives
it the way it archives a workspace, so a filesystem store and an object store
such as S3 need no new content type, and the server builds no zip of its own.

### Where the declaration comes from

| Option | Outcome |
| --- | --- |
| A table in the common recipe module mapping component class names to their file rules | Rejected. Framework knowledge, such as PyTorch's checkpoint folder, would live in a framework-neutral module and drift silently when a class is renamed or moved. |
| A `get_job_outputs` method on each result-producing component | Rejected. It changes the components' interfaces, and a subclass that saves elsewhere inherits a declaration naming the base class's files, so its real model would be dropped. |
| **The code that creates each component records that component's files, in its framework module** | **Chosen.** No component interface changes. The knowledge sits where the file names are chosen, only the exact component class is recorded, and anything unknown leaves the job undeclared. A record is opt-in, so a missed declaration could leave a file out; the export-time check for unaccounted model persistors, and treating metrics alone as no result, make that fail safe. |

### When some outputs cannot be known

Declare nothing, so the job downloads whole, rather than declare a partial
list. A partial list, such as metrics without a user persistor's model, would
filter the model out silently. The same applies to a shipped server
`environment.json`, which can move where components write, and to a list
longer than `MAX_JOB_OUTPUTS`.

### Telling the caller what it received

| Option | Outcome |
| --- | --- |
| An `outputs_manifest.json` file in the download, listing found and missing outputs | Rejected. It adds a file to the user's result that nobody acts on. The only actionable case is a whole-job fallback, and its reason is enough. |
| **Two fields in the `download_job` reply, passed through by the folder pull** | **Chosen.** No extra file. The client turns a whole-job result into one line with the reason, and a missing field identifies an older server. |

### Requesting the download

| Option | Outcome |
| --- | --- |
| A new admin command | Rejected. An older server would reject it, and it would need its own authorization and permission category. |
| An extra `download_job` argument | Rejected. The server validates extra arguments as targets, so an older server would reject it. |
| **A `download_job` command property** | **Chosen.** An older server ignores it and returns the whole job, and the command's authorization is unchanged. |

### When nothing declared was produced

Return the whole job with a reason, rather than an empty download: an empty
`workspace/` helps no one, and the reason tells the user to check the job or
its declaration.

## Compatibility

| Case | Behavior |
| --- | --- |
| Existing jobs and job folders | No `job_outputs`. Every download is unchanged. |
| Existing scripts | `outputs_only` defaults to `False`. Unchanged. |
| Existing custom `ExecEnv` subclasses | `Run` does not pass the new argument unless the caller asks for an outputs-only download. |
| New client, old server | The server ignores the property and returns the whole job. The reply has no `outputs_only`, and the client shows a notice. |
| Old client, new server | Not supported: the admin client and the server run the same version. An older client's full download of a job with an `outputs` component would unzip `outputs.zip` into its own folder rather than into `workspace/`. |
| Code that references `JobMetaKey.RESULT_LOCATION` or calls `set_results_uri` | Must drop the reference. Neither ever held or produced a value. |
| A job with `job_outputs` on an old server | The old server stores the key and ignores it. Downloads are full. |
| Failed or aborted job with `outputs_only=True` | The whole job, including logs. The client shows why the whole job came back. |
| Completed job that produced none of its declared outputs | The whole job. The client shows why. |
| Job that finished before this change | No `outputs` component: its workspace archive holds everything. Full downloads are unchanged, and outputs-only downloads return the whole job, with the reason. |
| Job store implementation without `save_job_outputs` | The default saves nothing, so the declared outputs stay in the workspace archive. Outputs-only downloads return the whole job. |
| Tools that read the stored `workspace` component directly | A job that declares outputs has them in the `outputs` component, not in `workspace`. The server's log commands are unaffected: log files are copied, not moved. |

## Security

- The command, its authorization and its permission category are unchanged,
  so no new data becomes reachable. Outputs-only download returns a subset of what
  `download_job` already returns.
- The server validates paths at submission and again before staging, moves
  only files under the job's workspace folders, and skips symlinks.
- The admin client unzips a downloaded archive only into a plain folder name
  from the server's reply, never a path.
- `job_outputs` names files, not contents. It must not contain secrets, the
  same as other job metadata.

## Testing

Unit tests:

- `job_outputs` validation: valid entries, folder entries, and rejection of
  absolute paths, `..`, `.`, backslashes, empty segments, non-strings and
  oversized lists.
- `set_recipe_meta` accepts the key and applies the same checks, and its value
  replaces the derived declaration.
- Each declaration point records the files of the component it creates; a
  persistor the user passes, a subclass, or a shipped server
  `environment.json` leaves the job undeclared; paths are normalized and
  invalid ones skipped. A component added by `add_cross_site_evaluation`, even
  after an earlier export, is declared.
- Cloned jobs keep `job_outputs`.
- At job end, the declared outputs are moved out of the workspace at their
  paths, without symlinks, with log files copied, and a later workspace
  folder's file replacing an earlier one's; they are stored as `outputs` and
  the workspace archive holds the rest; nothing is staged when none was
  produced; when `save_job_outputs` fails or saves nothing, the files are put
  back and the workspace archive holds everything.
- `download_job` with `outputs_only`:
  - links the stored `outputs` archive as `workspace.zip`, as a full download
    links the workspace;
  - falls back to the whole job for undeclared or failed jobs, or a job
    without an outputs archive, and reports the reason in its reply;
  - is unchanged without the property for a job without an `outputs`
    component.
- A full download of a job with an `outputs` component links both archives
  and tells the client to unzip `outputs.zip` into `workspace/`; the client
  merges them and accepts only plain folder names.
- The `outputs` component is a valid storage component and is not listed as
  an additional job component.
- The admin client forwards command properties on a folder pull and passes the
  reply's `outputs_only` fields back.
- `Session.download_job_result(outputs_only=True)` reports a whole-job download
  with its reason, and an older server whose reply has no `outputs_only`.
- `Run.get_result()` works with an `ExecEnv` that implements the two-argument
  `get_job_result`.
- The PyTorch FedAvg recipe exports the expected `job_outputs`.

Integration test, to follow: run a PyTorch FedAvg job in POC, download it with
`outputs_only=True`, and check that the model file is present at its usual path
and other workspace files are absent.

## Implementation

| Part | Location |
| --- | --- |
| Key, validation, constants | `nvflare/apis/job_def.py` (`JobMetaKey.JOB_OUTPUTS`, `validate_job_outputs`, `job_output_matches`) |
| Submission check | `JobMetaValidator._validate_job_outputs` |
| Cloning | `CLONED_META_KEYS` in `job_cmds.py` |
| Recipe declaration | Helpers in `nvflare/job_config/job_outputs.py`, called where each component is created; `nvflare/recipe/_job_outputs.py` applies the record in `Recipe.export` and `Recipe.run` |
| Outputs storage | `JobRunner._save_job_outputs`, `stage_job_outputs` in `nvflare/private/fed/server/job_outputs.py`, `JobDefManagerSpec.save_job_outputs`, storage component `outputs` |
| Full download | `JobCommandModule._get_default_job_components` (adds `outputs.zip`), reply key `unzip_into`, `FileTransferModule.pull_folder` |
| Outputs-only download | `JobCommandModule.download_job` |
| Client | `FileTransferModule.pull_folder`, `Session.download_job_result`, `ExecEnv.get_job_result`, `Run.get_result`, `nvflare job download --outputs-only` |
