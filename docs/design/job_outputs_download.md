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
| Scikit-learn, XGBoost bagging | Global model | `app_server/<persistor file name>`, such as `app_server/model_param.joblib` |
| NumPy training | Global model | `models/server.npy` (the persistor writes it under the result root, which is merged at the archive root) |
| FedAvg without a persistor | Global model | The controller's `save_filename`, relative to the run folder |
| Training metrics | Metrics artifact | `metrics/metrics_summary.json`, `metrics/round_metrics.jsonl` |
| Federated statistics | Statistics | The recipe's `stats_output_path`, relative to the run folder |
| Cross-site evaluation | Evaluation report | `cross_site_val/cross_val_results.json` |

All these file names are set by component arguments, so the components a
recipe configures already name them. Histogram-based and vertical XGBoost
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

A recipe declares the files of the built-in components it configures, including
a built-in persistor passed as `model_persistor`. With a custom persistor class
the recipe does not know the model's file name, so it declares nothing, and the
job downloads whole until its outputs are declared. A job that writes its own
files declares them the same way. Paths are relative to the server run folder:

```python
set_recipe_meta(recipe, JobMetaKey.JOB_OUTPUTS, ["app_server/report.json", "plots/"])
```

### When the whole job comes back

The server sends the whole job when the job did not complete, declares no
outputs, or produced none of them. The caller then sees one line that says why,
for example:

```text
Downloaded the whole job 3f2a... because none of the job's declared outputs were found: app_server/report.json
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
- Recipes derive the key from their server components when they are exported
  or run (`Recipe.export` and `Recipe.run`), so no recipe has its own
  declaration code. Each result-producing component reports the files it
  writes through one method, kept next to the code that writes them:

  ```python
  def get_job_outputs(self, app_env: Optional[dict], component_ids: Set[str]) -> Optional[List[str]]:
  ```

  `app_env` is the server app's `config/environment.json` shipped with the job
  (or None), and `component_ids` are the server app's component IDs. The method
  returns paths relative to the run folder, or None when the component cannot
  tell. The recipe module only collects the server components, calls the
  method where it exists, and normalizes the result; it holds no framework
  knowledge. The built-in components declare:

  | Component | Declared output |
  | --- | --- |
  | `PTFileModelPersistor` | `app_server/<global_model_file_name>`, `app_server/<best_global_model_file_name>`; under `app_server/<APP_CKPT_DIR>/` when the app's `environment.json` sets it |
  | `TFModelPersistor`, `JoblibModelParamPersistor`, `XGBModelPersistor` | `app_server/<save_name>` |
  | `NPModelPersistor` | `<model_dir>/<model_name>`, with `.npy` added when missing, as `np.save` does |
  | `FedAvg` whose `persistor_id` names no component | `<save_filename>` |
  | `MetricsArtifactWriter` | `<results_dir>/`, or its summary and round files by name when `results_dir` is the run folder itself |
  | `JsonStatsFileWriter` | `<output_path>` |
  | `CrossSiteModelEval`, `CrossSiteEval` | `<cross_val_dir>/` |

  Each path is normalized with `posixpath.normpath`, so
  `stats_output_path="./output"` is declared as `output`. A path that would not
  pass validation is skipped, and a file inside a declared folder is not listed
  again. A recipe never fails because of its own declaration.
- `ModelPersistor.get_job_outputs` returns None by default, so a job with a
  persistor that does not override it declares nothing. So does any component
  that returns None, such as `PTFileModelPersistor` with a checkpoint folder
  outside the app, and a job whose `environment.json` cannot be read. A
  declaration without the model would leave it out of an outputs-only
  download.
- Because the declaration is made at export or run time, a component added
  after the recipe is built, such as by `add_cross_site_evaluation`, is
  included.
- A value set with `set_recipe_meta` replaces the derived list and is kept as
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

The client sends `{"outputs_only": true}`. Without it, `download_job` is
unchanged. With it, the server:

1. Requires the job to be finished, as today.
2. Prepares the full download when:
   - the job did not finish successfully,
   - the job's metadata has no `job_outputs`, or
   - none of the declared outputs exist in the workspace archive (checked
     after filtering in step 3).
3. Otherwise prepares the `workspace` component in the transfer folder under a
   temporary name, with the job store's existing `get_storage_for_download`.
   It works with any `StorageSpec` without loading the archive into memory.
   The prepared file may not be a copy: `FilesystemStorage` creates a symlink
   to the stored archive. So the server:
   - reads the prepared file and writes the declared entries, stored without
     recompression, to a separate temporary file;
   - unlinks the prepared file, which never modifies the stored archive;
   - renames the filtered file to `workspace.zip`.

   The stored archive is never written, so later full downloads are
   unaffected. The admin client unzips the filtered archive into `workspace/`
   as it does today, and a large folder entry is still one transfer.
4. Reports the result in its reply to `download_job`, beside the file list:
   - `outputs_only`: whether the download was filtered;
   - `outputs_only_reason`: why it was not, when it was not.

   A missing entry is not an error. For example, the best model exists only
   when the workflow tracked one. When every entry is missing, the download
   falls back to the whole job, as in step 2.
5. Includes `meta.json`, as today, so the caller still has the job status and
   timing.

The admin client's folder pull passes these two reply fields through to its
caller, next to the local location. A reply to an outputs-only request without
`outputs_only` came from an older server. No file is added to the download.

Extraction matches each archive member's normalized name against the declared
entries and rejects symlink entries. It never writes outside the transfer
folder.

A completed job that has no workspace archive gets the whole job, as a full
download does, and the reply gives the reason.

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

## Compatibility

| Case | Behavior |
| --- | --- |
| Existing jobs and job folders | No `job_outputs`. Every download is unchanged. |
| Existing scripts | `outputs_only` defaults to `False`. Unchanged. |
| Existing custom `ExecEnv` subclasses | `Run` does not pass the new argument unless the caller asks for an outputs-only download. |
| New client, old server | The server ignores the property and returns the whole job. The reply has no `outputs_only`, and the client shows a notice. |
| Old client, new server | The old client never sends the property. Unchanged. |
| Code that references `JobMetaKey.RESULT_LOCATION` or calls `set_results_uri` | Must drop the reference. Neither ever held or produced a value. |
| A job with `job_outputs` on an old server | The old server stores the key and ignores it. Downloads are full. |
| Failed or aborted job with `outputs_only=True` | The whole job, including logs. The client shows why the whole job came back. |
| Completed job that produced none of its declared outputs | The whole job. The client shows why. |

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
- `set_recipe_meta` accepts the key and applies the same checks, and its value
  replaces the derived declaration.
- Each built-in component declares the paths its arguments name; paths are
  normalized, invalid ones skipped. A component added by
  `add_cross_site_evaluation`, even after an earlier export, is declared.
- Cloned jobs keep `job_outputs`.
- `download_job` with `outputs_only`:
  - writes a filtered `workspace.zip` with only the declared entries, at their
    original paths;
  - leaves the stored archive unchanged when the prepared file is a symlink to
    it, so a later full download still has every file;
  - rejects symlink entries;
  - falls back to the whole job for undeclared or failed jobs, or when no
    declared output exists, and reports the reason in its reply;
  - is unchanged without the property.
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
| Recipe declaration | `get_job_outputs` on each result-producing component (default on `ModelPersistor`); `nvflare/recipe/_job_outputs.py` collects them, called by `Recipe.export` and `Recipe.run` |
| Outputs-only download | `JobCommandModule.download_job` and `nvflare/private/fed/server/job_outputs.py` |
| Client | `FileTransferModule.pull_folder`, `Session.download_job_result`, `ExecEnv.get_job_result`, `Run.get_result`, `nvflare job download --outputs-only` |
