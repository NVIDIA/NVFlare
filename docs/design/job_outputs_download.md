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

No change to how persistors or controllers work, or to the job layout. When a job that
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
| FLARE API, Recipe API and CLI | The FLARE API's `Session.download_job_result(job_id, destination=None)` sends `download_job`. The Recipe API's `Run.get_result()` and the `nvflare job download` command call it. |

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

All these file names are set by each component's own arguments, so each
component class can name them from an instance. Histogram-based and vertical XGBoost
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

A job built from built-in components declares its outputs: when its
`meta.json` is generated, each server component names the files it writes. The
caller only asks for them:

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

Federated statistics work the same way. The statistics writer declares the
recipe's `stats_output_path`, normalized:

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

`set_recipe_meta` replaces the list the job's components name. For example, to keep only the
final model:

```python
from nvflare.apis.job_def import JobMetaKey
from nvflare.recipe import set_recipe_meta

set_recipe_meta(recipe, JobMetaKey.JOB_OUTPUTS, ["app_server/FL_global_model.pt"])
```

A built-in persistor passed as `model_persistor` is declared from its own
settings. A persistor class of the user's own, or a subclass of a built-in one,
may save anywhere, so the job declares nothing and downloads whole until its
outputs are declared. The user lists them with `set_recipe_meta`, or names the
class's files once, beside the class:

```python
from nvflare.apis.job_outputs import register_output_resolver

register_output_resolver(MyPersistor, lambda persistor: [f"app_server/{persistor.file_name}"])
```

A job that writes other files of its own lists them with `set_recipe_meta`.
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

A `FedJob` built without a recipe declares its outputs the same way when it is
exported. A hand-written job adds `job_outputs` to its `meta.json`, as shown
below.

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
  same rules before submission. A user's value replaces the list the job's components name.
- Each result-producing server component class names its files with a
  resolver registered beside the class, in the class's own module, through
  `register_output_resolver` in `nvflare/apis/job_outputs.py`. A resolver is a
  function of an instance that returns its paths from the instance's own
  arguments. No component class or interface changes. The registry depends
  only on `nvflare.apis`, so a component at any layer can register.

  | Class | Names |
  | --- | --- |
  | `PTFileModelPersistor` | `app_server/<global_model_file_name>`, `app_server/<best_global_model_file_name>` |
  | `DiskFedAvgPersistor` | `app_server/FL_global_model.safetensors` |
  | `TFModelPersistor`, `JoblibModelParamPersistor`, `XGBModelPersistor` | `app_server/<save_name>` |
  | `NPModelPersistor`, `LRModelPersistor` | `<model_dir>/<model_name>`, with `.npy` added when missing, as `np.save` does |
  | `FedAvg`, and `PTFedAvg` and the PyTorch and TensorFlow `FedOpt`, which save the same way | `<save_filename>`, which the controller saves itself, only when it has no persistor |
  | `CrossSiteModelEval` | `<cross_val_dir>/` |
  | `JsonStatsFileWriter` | `<output_path>` |
  | `MetricsArtifactWriter` | `<results_dir>/`, or the summary and round files by name when `results_dir` is the run folder itself, as supporting outputs |

- When `meta.json` is generated, `FedJobConfig` calls `resolve_job_outputs`
  in `nvflare/job_config/job_outputs.py`. It walks the server app's components
  and workflow controllers and looks up each one's exact class. Every recipe
  and every `FedJob` export goes through it, so a component added at any time,
  such as by `add_cross_site_evaluation`, and a setting changed after a
  component was added are both reflected. Each path is normalized with
  `posixpath.normpath`, so `stats_output_path="./output"` is declared as
  `output`, and a file inside a declared folder is not listed again. A job
  never fails to export because of its own declaration.
- Nothing is declared, and the job downloads whole, when:
  - a model persistor on the server has no resolver, such as a user's own
    persistor;
  - a component's class has no resolver but a base class has one: the
    subclass may write elsewhere than its base;
  - only supporting outputs, such as metrics, were named: the main result is
    unknown;
  - a resolver raises;
  - a path would not pass validation, such as an absolute folder;
  - the job ships a `config/environment.json` for the server app, which can
    move where components write (for example a PyTorch checkpoint folder); or
  - there are more than `MAX_JOB_OUTPUTS` entries.

  A declaration without the model would leave it out of an outputs-only
  download.
- A value set with `set_recipe_meta`, or in a `FedJob`'s `meta_props`, is kept
  as set, and the walk is skipped.
- A unit test lists every `ModelPersistor` subclass in `nvflare` and requires
  each to register a resolver or to appear on a short list of persistors that
  declare nothing on purpose. A new built-in persistor without one fails the
  test instead of silently making jobs download whole.
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
and then deletes the folder. For a job that declares outputs, it stores them as
their own `outputs` component and the rest as `workspace`:

1. **Link the outputs.** Beside each workspace folder, build a folder of hard
   links naming the declared output files at their relative paths. A hard link
   is another name for the same file, so nothing is copied and the workspace
   folders are not changed. A symlink is never an output.
2. **Store the outputs.** `JobDefManagerSpec.save_job_outputs(job_id,
   output_link_folders)` archives them as the `outputs` component, with the
   same folder contract `save_workspace` uses, so any job store that archives
   a workspace archives the outputs the same way.
3. **Link the rest.** From a fresh walk, build a second folder of hard links
   naming everything but the files linked as outputs, so a file created while
   the outputs were stored is archived as it is now. If a stored output was
   replaced, written to or removed meanwhile, or a declared output was created
   meanwhile, the outputs archive would be out of date, so the workspace is
   stored whole instead.
   The workspace folders share archive paths, and the later folder's copy
   wins. The client unzips the outputs archive first, so a path that is an
   output in one folder and not in another (for example a symlink) could change
   which copy a full download keeps; the workspace is then stored whole too.
   The log files the server's log commands read (`log.txt`, `log.json`,
   `error_log.txt`) are named here too when a job declares them, so those
   commands keep working; like any other output, a declared log that changed
   after it was stored makes the workspace stored whole. Symlinks are
   recreated pointing at what they resolve to in the untouched workspace, so a
   link to a declared output still reads it, and a linked folder stays a link.
4. **Record the split.** Set `job_outputs_stored: true` in the job's meta.
   Downloads rely on this record, not on listing the store's components.
5. **Store the rest.** `save_workspace` archives the rest link folders.
6. Remove the link folders, then delete the workspace folders, as today. A
   link folder that cannot be removed is logged.

Each file is read and written once, as before this change, so job end takes no
longer and nothing is stored twice.

The workspace folders stay intact until both archives are stored:

- A retry after a failed workspace save rebuilds the links from the same files
  and stores the same outputs again.
- A crash leaves only link folders, which are extra names for the same files;
  the next attempt replaces them.
- The workspace folders are archived whole, as before this change, when the
  job store does not implement `save_job_outputs` (checked before any walk),
  nothing declared was produced, a link cannot be made (for example a link
  folder on a different filesystem), a folder cannot be read, or the outputs
  are not stored. If an earlier attempt had recorded the split, the record is
  cleared, so downloads do not use that older outputs archive. If clearing the
  record fails, the archival fails and is retried, rather than leave the older
  outputs archive authoritative.

`job_outputs_stored` is written only by the server; a submitted job's value is
removed at submission.

**Why at job end.** By the time anyone downloads, the result files exist only
inside stored archives; the run folder is gone. A download replies at once
because the server only links stored archives into the transfer folder and
streams them after replying. Building a subset on request would mean copying
the declared entries out of the workspace archive before replying, which for a
multi-GB model can exceed how long the admin client waits for a reply (5
seconds unless a command timeout is set). Job end is the last moment the
outputs exist as files, so they can be separated without copying.

### Full download

A full download of a job whose meta records `job_outputs_stored` sends
`workspace.zip` and `outputs.zip`, both linked from the store, so the server
still replies at once. If the outputs archive cannot be read from the store,
the rest of the job is still downloaded, and the reply's `outputs_note` says
the workspace lacks the declared outputs; the FLARE API logs it as a warning
("Job ...: the job's outputs archive cannot be read ...").

The reply also carries `unzip_into: {"outputs.zip": "workspace"}`, and the
admin client unzips `outputs.zip` into `workspace/` first and `workspace.zip`
after it. The only files in both archives are server logs a job declares as
outputs, and the workspace archive's copy is the newer one. The result is the
same `workspace/` folder as before this change. The admin client
accepts only a plain folder name there, never a path. A job without the
record downloads exactly as before.

### Serving an outputs-only download

The FLARE API sends `{"outputs_only": true}`. Without it, `download_job` is
unchanged. With it, the server:

1. Requires the job to be finished, as today.
2. Prepares the full download when:
   - the job did not finish successfully,
   - the job's metadata has no `job_outputs`, or
   - the job's meta does not record `job_outputs_stored`: none of its
     declared outputs were produced, it finished before this change, or they
     were kept in its workspace archive.
3. Otherwise prepares `meta.json` and the `outputs` component as
   `workspace.zip`, with the job store's existing `get_storage_for_download`,
   exactly as a full download prepares the `workspace` component.
   `FilesystemStorage` links the stored archive, so the server replies at once
   and streams the file afterwards. If the outputs archive cannot be read, the
   server prepares the whole job instead, with that reason and with
   saying the whole job lacks them too. The admin client
   unzips it into
   `workspace/` as usual.
4. When the download is not what was asked for, says why in one reply field
   beside the file list, `outputs_note`: why an outputs-only request returned
   the whole job, and that stored outputs are missing, when they are. When the
   download is what was asked for, the reply has no note. An outputs-only
   download also sets `outputs_only: true` in the reply.

   A missing entry is not an error. For example, the best model exists only
   when the workflow tracked one.
5. Includes `meta.json`, as today, so the caller still has the job status and
   timing.

The admin client's folder pull passes `outputs_only` and `outputs_note`
through to its caller, next to the local location, and the FLARE API shows the
note as `Job <job_id>: <note>.` No file is added to the download. A server
without this feature ignores the `outputs_only` property and sends neither
field, so the FLARE API reports an outputs-only request whose reply lacks
`outputs_only: true` as a whole-job download, and never assumes the filter was
applied.

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

### FLARE API, Recipe API and CLI

These are the admin-side APIs for downloading a job's result. They are separate
from the Client API (`flare.init()`, `flare.receive()`, `flare.send()`) that
training code uses on the sites, which this design does not change.

| API | Change |
| --- | --- |
| FLARE API: `Session.download_job_result(job_id, destination=None, outputs_only=False)` | When `outputs_only=True`, sends `download_job` with the `outputs_only` property. When the reply carries an `outputs_note`, in either mode, logs it as a warning. |
| Recipe API: `ExecEnv.get_job_result(job_id, timeout=0.0, outputs_only=False)` | POC and production environments pass `outputs_only` to the session and print the same notice when the whole job came back. The simulator runs locally and returns its workspace unchanged. |
| Recipe API: `Run.get_result(timeout=0.0, clean_up=True, outputs_only=False)` | Passes `outputs_only` to the environment only when it is `True`, so existing custom `ExecEnv` subclasses with the two-argument signature keep working. |
| CLI: `nvflare job download <job_id> --outputs-only` | Passes `outputs_only=True`. Text output prints the notice when the whole job came back. JSON output adds `outputs_only` (whether only the declared outputs were downloaded) and `outputs_note`. `artifacts` and `missing_artifacts` describe what was downloaded. For example, client logs are listed as missing in an outputs-only download. |

The returned path is the downloaded `<job_id>` folder in both modes. Result
files are at the same relative path under `workspace/` in both modes, so the run
summary and the CLI's artifact discovery find the model and metrics either way.

## Alternatives Considered

### Producing the outputs-only download

| Option | Outcome |
| --- | --- |
| Filter the stored workspace archive on each request: copy the declared entries into a new archive, then reply | Rejected. The copy runs before the server replies, and for a multi-GB model it can exceed the admin client's reply wait (5 seconds unless a command timeout is set). The download fails on exactly the large-model jobs it is for, and the path differs from a full download. |
| Filter on request, but make the admin client wait longer for this one reply | Rejected. It hides the cost rather than removing it, the copy still grows with the model, and outputs-only downloads still take a different path from full downloads. |
| Reply first, then prepare the files | Rejected. It needs a rework of how folder downloads start, for one command. |
| Download the whole job and filter it on the admin side | Rejected. It saves nothing on transfer, which is the point of the feature. |
| Archive the declared outputs at job end as a second copy, keeping the workspace archive whole | Rejected. Downloads reply at once, but every declared output, usually the model, is stored twice, and job end reads and writes it a second time. For large models that doubles their storage. |
| Move the declared outputs out of the run folder, archive them, then archive the rest | Rejected. Between the move and the workspace archive, a staging folder holds the only copy of the model: a retry after a failed workspace save, a crash, a failed move back, or a copy across filesystems could lose it. |
| **Store the declared outputs at job end apart from the rest, through two folders of hard links** | **Chosen.** Hard links name the outputs and the rest without copying or moving anything, and the run folder stays intact until both archives are stored, so a retry or crash loses nothing. Each file is stored once and written once, as before. An outputs-only download links the outputs archive, and a full download links both, so both reply at once. The admin client merges the two archives into `workspace/`. When links cannot be made, the workspace is stored whole. |

The server names the outputs and the rest with folders of hard links and hands
them to the job store, the same folder contract `save_workspace` uses. Each store archives
it the way it archives a workspace, so a filesystem store and an object store
such as S3 need no new content type, and the server builds no zip of its own.

### Where the declaration comes from

| Option | Outcome |
| --- | --- |
| The code that adds each component (a recipe, a model wrapper, a job class) records the component's files on the job | Rejected. The same knowledge is applied at every place a class is used, so each new recipe must remember it; extra state is needed to catch a forgotten record; paths are captured when the component is added; and a `FedJob` built without that code declares nothing. |
| A `get_job_outputs` method on each result-producing component | Rejected. It changes the components' interface, and a subclass that saves elsewhere inherits a declaration naming the base class's files, so its real model would be dropped. |
| A table of class rules in one framework-neutral module, or a rule file per framework | Rejected. A rule kept apart from its class drifts when an argument is renamed; a neutral table holds framework knowledge, and loading per-framework rule files can import optional frameworks such as PyTorch. |
| An explicit list only, written by the user or by each recipe | Rejected as the only source. Users would need internal paths, and a wrong list silently filters the model out. It stays as the override. |
| Components report the files they write while the job runs | Rejected. It changes runtime component code, and the outputs would be known only at job end, not in `meta.json`. |
| **A resolver registered beside each component class, applied to the job's server components when `meta.json` is generated** | **Chosen.** One rule per class, next to the arguments it reads, applied to every instance with its final settings, for recipes and plain `FedJob`s alike. No interface changes, and exact-class lookup means a subclass or an unknown persistor declares nothing. |

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
| Separate reply fields for the mode, the whole-job reason and missing outputs | Rejected. The caller knows the mode it asked for, so echoing it back adds nothing, and the FLARE API only ever shows the reasons as one line. |
| **One `outputs_note` field in the `download_job` reply, passed through by the folder pull** | **Chosen.** No extra file. The note is present only when the download is not what was asked for, and the FLARE API shows it as is. |

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
| Jobs built with `FedJob` | A job whose server components all name their files now gets `job_outputs` in its `meta.json`, and its outputs are stored apart at job end. Full downloads return the same files. |
| Existing custom `ExecEnv` subclasses | `Run` does not pass the new argument unless the caller asks for an outputs-only download. |
| New client, old server | Not supported: the admin client and the server run the same version. An older server ignores the property and returns the whole job; the FLARE API reports it as a whole-job download. |
| Old client, new server | Not supported: the admin client and the server run the same version. An older client's full download of a job with an `outputs` component would unzip `outputs.zip` into its own folder rather than into `workspace/`. |
| Code that references `JobMetaKey.RESULT_LOCATION` or calls `set_results_uri` | Must drop the reference. Neither ever held or produced a value. |
| A job with `job_outputs` on an old server | The old server stores the key and ignores it. Downloads are full. |
| Failed or aborted job with `outputs_only=True` | The whole job, including logs. The FLARE API, Recipe API and CLI show why the whole job came back. |
| Completed job that produced none of its declared outputs | The whole job, with the reason shown. |
| Job that finished before this change | No `outputs` component: its workspace archive holds everything. Full downloads are unchanged, and outputs-only downloads return the whole job, with the reason. |
| Job store implementation without `save_job_outputs` | The default saves nothing, so the declared outputs stay in the workspace archive. Outputs-only downloads return the whole job. |
| Tools that read the stored `workspace` component directly | A job that declares outputs has them in the `outputs` component, not in `workspace`. The server's log commands are unaffected: log files stay in `workspace` too. |

Known limitations:

- A declared output that is replaced, written to or removed between storing
  the outputs and linking the rest makes the server store the workspace whole,
  as before this change. Results published into the workspace by a container
  job launcher are the case this guards. The check compares each file's
  identity, size and modification time, so an overwrite that keeps all three
  is not detected. Declared log files get the same check, so a job that
  declares a log the server still writes at job end is stored whole.

## Security

- The command, its authorization and its permission category are unchanged,
  so no new data becomes reachable. Outputs-only download returns a subset of what
  `download_job` already returns.
- The server validates paths at submission and again before splitting, links
  only files under the job's workspace folders, and never treats a symlink as
  an output.
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
- Each registered resolver names its class's files from non-default
  arguments; resolution is by exact class.
- The walk collects main and supporting outputs from components and
  controllers and reflects settings changed after a component was added; a
  persistor without a resolver, a subclass of a resolved class, metrics alone,
  an invalid path or a shipped server `environment.json` declares nothing; a
  user's own registered persistor is declared; paths are normalized.
- Every built-in `ModelPersistor` registers a resolver or is listed as
  declaring nothing on purpose.
- Built-in recipes, including one given a built-in persistor and one with a
  component added by `add_cross_site_evaluation` after an earlier export,
  export the expected `job_outputs`, as does a plain `FedJob`.
- Cloned jobs keep `job_outputs`.
- At job end, hard-link folders name the declared outputs and the rest at
  their paths, beside each workspace folder, without changing it: the same
  file, log files in both, symlinks with the rest. The outputs are stored as
  `outputs` and the rest as `workspace`. The workspace is archived whole when
  nothing declared was produced, links cannot be made, or `save_job_outputs`
  fails or saves nothing. A retry after a failed workspace save stores the same
  outputs again, and leftover links from an interrupted attempt are replaced.
- The rest is linked from a fresh walk after the outputs are stored; relative
  symlinks to outputs still resolve and linked folders stay links; an
  unreadable folder stores the workspace whole; the split is recorded in the
  job's meta, cleared by a later fallback, and skipped for job stores without
  `save_job_outputs`; a submitted record is removed.
- Downloads decide from the meta record; a split job whose outputs archive
  cannot be read downloads the rest with an `outputs_note`, and an
  outputs-only request returns the whole job with that reason.
- A declared file written after the outputs are linked, or a stored output
  replaced or removed meanwhile, stores the workspace whole; a failed record clear fails the archival so it is retried.
- `download_job` with `outputs_only`:
  - links the stored `outputs` archive as `workspace.zip`, as a full download
    links the workspace;
  - falls back to the whole job for undeclared or failed jobs, or a job
    without an outputs archive, and reports the reason in its reply;
  - is unchanged without the property for a job without an `outputs`
    component.
- A full download of a job with an `outputs` component links both archives
  and tells the admin client to unzip `outputs.zip` into `workspace/`; the admin client
  merges them and accepts only plain folder names.
- The `outputs` component is a valid storage component and is not listed as
  an additional job component.
- The admin client forwards command properties on a folder pull and passes the
  reply's `outputs_only` and `outputs_note` back.
- `Session.download_job_result` shows an `outputs_note` in either mode and
  says nothing when there is none; an outputs-only reply without
  `outputs_only: true` (an older server) is reported as the whole job.
- `Run.get_result()` works with an `ExecEnv` that implements the two-argument
  `get_job_result`.

Integration test, to follow: run a PyTorch FedAvg job in POC, download it with
`outputs_only=True`, and check that the model file is present at its usual path
and other workspace files are absent.

## Implementation

| Part | Location |
| --- | --- |
| Key, validation, constants | `nvflare/apis/job_def.py` (`JobMetaKey.JOB_OUTPUTS`, `validate_job_outputs`, `job_output_matches`) |
| Submission check | `JobMetaValidator._validate_job_outputs` |
| Cloning | `CLONED_META_KEYS` in `job_cmds.py` |
| Declaration | `register_output_resolver` in `nvflare/apis/job_outputs.py`, called beside each component class; `resolve_job_outputs` in `nvflare/job_config/job_outputs.py`, called by `FedJobConfig` when `meta.json` is generated |
| Outputs storage | `JobRunner._save_job_outputs`, `split_job_outputs` in `nvflare/private/fed/server/job_outputs.py`, `JobDefManagerSpec.save_job_outputs`, storage component `outputs` |
| Full download | `JobCommandModule._get_default_job_components` (adds `outputs.zip`), reply key `unzip_into`, `FileTransferModule.pull_folder` |
| Outputs-only download | `JobCommandModule.download_job` |
| Admin client and APIs | Admin client: `FileTransferModule.pull_folder`. FLARE API: `Session.download_job_result`. Recipe API: `ExecEnv.get_job_result`, `Run.get_result`. CLI: `nvflare job download --outputs-only` |
