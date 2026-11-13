# Job Outputs Download

## Summary

Downloading a finished job today returns the whole job: the submitted job
(`job.zip`), its metadata (`meta.json`) and the complete server workspace
(`workspace.zip`). Most users only want the job's result: the global model,
the statistics, the metrics or the evaluation report.

This design adds an outputs-only download. A job's outputs are the files it
produced on the server: everything in its server workspace except what was
deployed with it (its server app's config and custom code, and its job files)
and the server's own logs and system files. Nothing has to be declared: built-in
and custom components, recipes, plain `FedJob`s and hand-written jobs all work
the same way.

When a job ends, the server stores its outputs apart from the rest of the
workspace, so each file is stored once. An outputs-only download sends the
outputs archive, and a full download sends both archives, unzipped into the
same `workspace/` folder. Either way the server replies at once. Output files
keep the same relative paths as in a full download, so code that reads them
works with either download. A caller that does not ask for an outputs-only
download receives the same files as today.

## Current State

| Step | Behavior |
| --- | --- |
| Job end | `JobRunner._save_workspace` archives the run, result, log and audit folders into the `workspace` job-store component. Each folder's files are stored relative to that folder, so all of them sit at the archive root. By default the four folders are the same run folder. |
| Admin command | `download_job <job_id>` copies the `data`, `meta` and `workspace` components into a transfer folder. The admin client downloads them and unzips them into `<job_id>/job/`, `<job_id>/meta.json` and `<job_id>/workspace/`. |
| FLARE API, Recipe API and CLI | The FLARE API's `Session.download_job_result(job_id, destination=None)` sends `download_job`. The Recipe API's `Run.get_result()` and the `nvflare job download` command call it. |

A server run folder holds three kinds of files:

| Kind | Examples | Written by |
| --- | --- | --- |
| Deployed with the job | `app_server/config/config_fed_server.json`, `app_server/custom/net.py`, `meta.json`, `fl_app.txt` | The server, when it deploys the job's server app |
| Server logs and system files | `log.txt`, `error_log.txt`, `log.json`, `log_fl.txt`, `stats_pool_summary.json`, `audit.log`, `job_cert/` | The server's logging and runtime |
| Produced by the job | `app_server/FL_global_model.pt`, `models/server.npy`, `metrics/`, `cross_site_val/`, `tb_events/`, a custom persistor's file | The job's components: persistors, controllers, writers, receivers |

The common results are at stable paths, but they are set by each component's
own arguments, and custom components write wherever they choose. Histogram-based
and vertical XGBoost save their models on the clients, so they have no
server-side model.

## Goals

- A caller can download only the files the job produced, at the same paths as
  a full download.
- It works for every job without declaring anything, including jobs with custom
  components and hand-written jobs.
- No existing script, execution environment or deployment changes behavior
  unless it asks for an outputs-only download.
- Failed jobs still give the caller what they need to debug.

## Non-Goals

- No change to where persistors or controllers write files.
- No change to the files a full `download_job` returns. A job's outputs are
  stored apart from the rest of its workspace, and a full download returns the
  same files as before.
- No result files from client sites. Only the server workspace is covered.
- No change to the default download. Making outputs-only the default is a
  separate, later decision.

## Using It

### Nothing to declare

Ask for the outputs:

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
add_cross_site_evaluation(recipe)

env = ProdEnv(startup_kit_location="/path/to/admin@nvidia.com")
run = recipe.execute(env)

# app_server/FL_global_model.pt, app_server/best_FL_global_model.pt, metrics/,
# cross_site_val/ and anything else the job wrote; not its config, custom code or logs
path = run.get_result(outputs_only=True)
model_path = os.path.join(path, "workspace", "app_server", "FL_global_model.pt")
```

The same holds for a custom persistor, a custom report or metrics writer,
federated statistics, a plain `FedJob` and a hand-written job: whatever files
the job writes on the server are its outputs.

### When the whole job comes back

The server sends the whole job when the job did not complete, or has no
outputs archive: it produced no output files, or it finished before this
change. The caller then sees one line that says why, for example:

```text
Job 3f2a...: the whole job was downloaded because the job did not complete successfully (status FINISHED:EXECUTION_EXCEPTION).
```

The downloaded folder has the same layout either way, so code that reads
`workspace/app_server/FL_global_model.pt` works with both.

### What an outputs-only download leaves out

- The submitted job (`job.zip`) and the server app's config and custom code.
- The server's logs. Use the full download when you need the logs; a failed
  job always downloads whole, logs included.
- Client files, which are not part of either download.

### Default behavior

Without `outputs_only=True`, the download returns the same files as today:

```python
path = run.get_result()  # whole job, including the logs
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

## Design

### Which files are outputs

At job end, the server walks the job's workspace folders. A file is an output
unless it is:

- **deployed with the job and unchanged**: a file of the server app deployed
  from the submitted job (`app_server/<path in the app>`, read from the stored
  job) whose size and CRC-32 still match the deployed copy, or `meta.json` and
  `fl_app.txt`, which deployment writes beside it;
- **a server log**: `log.txt`, `error_log.txt`, `log.json` or `log_fl.txt`,
  including a task worker's `<prefix>_` copy and rotated `.<n>` copies;
- **a system file**: the stats-pool summary and records (with any prefix),
  `audit.log`, the job signature file, or anything in a `__pycache__` or
  `job_cert` folder; or
- **a symlink**, which stays with the rest, as before.

A file the job creates inside `app_server/custom/` is an output, and so is a
deployed file the job rewrote, such as a seed checkpoint replaced by the
trained model. If the deployed app cannot be read from the store, the job's own
files cannot be told apart, so the workspace is stored whole.

Logs written under non-default file names by a custom logging configuration
count as outputs: the outputs-only download then holds more than needed, never
less.

### Job metadata: `job_outputs_stored`

The server records `job_outputs_stored: true` in a job's meta when it stores
the job's outputs apart (`JobMetaKey.JOB_OUTPUTS_STORED`). Downloads decide
from this record. It is written only by the server; a submitted job's value is
removed at submission. A job adds nothing to its `meta.json` for this feature.

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
and then deletes the folder. It now stores the job's outputs as their own
`outputs` component and the rest as `workspace`:

1. **Link the outputs.** Beside each workspace folder, build a folder of hard
   links naming the output files at their relative paths. A hard link is
   another name for the same file, so nothing is copied and the workspace
   folders are not changed. Each file's identity is read from its link, so a
   file replaced right after it was linked is seen as changed in step 3.
2. **Store the outputs.** `JobDefManagerSpec.save_job_outputs(job_id,
   output_link_folders)` archives them as the `outputs` component, with the
   same folder contract `save_workspace` uses, so any job store that archives
   a workspace archives the outputs the same way.
3. **Link the rest.** From a fresh walk, build a second folder of hard links
   naming everything but the files linked as outputs, so a server log written
   while the outputs were stored is archived as it is now. If a stored output
   was replaced, written to or removed meanwhile, or a new output appeared
   meanwhile, the outputs archive would be out of date, so the workspace is
   stored whole instead.
   The workspace folders share archive paths, and the later folder's copy
   wins. The client unzips the outputs archive first, so a path that is an
   output in one folder and not in another (for example a symlink) could change
   which copy a full download keeps; the workspace is then stored whole too.
   Logs are never outputs, so they stay in the workspace archive, where the
   server's log commands read them. Symlinks are
   recreated pointing at what they resolve to in the untouched workspace, so a
   link to an output still reads it, and a linked folder stays a link.
4. **Record the split.** Set `job_outputs_stored: true` in the job's meta.
   Downloads rely on this record, not on listing the store's components.
5. **Store the rest.** `save_workspace` archives the rest link folders.
6. Remove the link folders, then delete the workspace folders, as today. A
   link folder that cannot be removed is logged.

The store zips each list of folders when it saves them, so a file is copied
into exactly one archive: the model is in `outputs`, not in `workspace`. Each
file is read and written once, as before this change, so job end takes no
longer and nothing is stored twice.

The workspace folders stay intact until both archives are stored:

- A retry after a failed workspace save rebuilds the links from the same files
  and stores the same outputs again.
- A crash leaves only link folders, which are extra names for the same files;
  the next attempt replaces them.
- The workspace folders are archived whole, as before this change, when the
  job store does not implement `save_job_outputs` (checked before any walk),
  the job produced no output files, the job or its deployed app cannot be
  read, a link cannot be made (for
  example a link folder on a different filesystem), a folder cannot be read, or
  the outputs are not stored. If an earlier attempt had recorded the split, the
  record is cleared, so downloads do not use that older outputs archive. If
  clearing the record fails, the archival fails and is retried, rather than
  leave the older outputs archive authoritative.

**Why at job end.** By the time anyone downloads, the result files exist only
inside stored archives; the run folder is gone. A download replies at once
because the server only links stored archives into the transfer folder and
streams them after replying. Building a subset on request would mean copying
entries out of the workspace archive before replying, which for a multi-GB
model can exceed how long the admin client waits for a reply (5 seconds unless
a command timeout is set). Job end is the last moment the outputs exist as
files, so they can be separated without copying.

### Full download

A full download of a job whose meta records `job_outputs_stored` sends
`workspace.zip` and `outputs.zip`, both linked from the store, so the server
still replies at once. If the outputs archive cannot be read from the store,
the rest of the job is still downloaded, and the reply's `outputs_note` says
the workspace lacks the outputs; the FLARE API logs it as a warning
("Job ...: the job's outputs archive cannot be read ...").

The reply also carries `unzip_into: {"outputs.zip": "workspace"}`, and the
admin client unzips `outputs.zip` into `workspace/` first and `workspace.zip`
after it. No file is in both archives. The result is the same `workspace/`
folder as before this change. The admin client accepts only a
plain folder name there, never a path. A job without the record downloads
exactly as before.

### Serving an outputs-only download

The FLARE API sends `{"outputs_only": true}`. Without it, `download_job` is
unchanged. With it, the server:

1. Requires the job to be finished, as today.
2. Prepares the full download when:
   - the job did not finish successfully, or
   - the job's meta does not record `job_outputs_stored`: it produced no
     output files, it finished before this change, or they were kept in its
     workspace archive.
3. Otherwise prepares `meta.json` and the `outputs` component as
   `workspace.zip`, with the job store's existing `get_storage_for_download`,
   exactly as a full download prepares the `workspace` component.
   `FilesystemStorage` links the stored archive, so the server replies at once
   and streams the file afterwards. If the outputs archive cannot be read, the
   server prepares the whole job instead, with that reason. The admin client
   unzips it into `workspace/` as usual.
4. When the download is not what was asked for, says why in one reply field
   beside the file list, `outputs_note`: why an outputs-only request returned
   the whole job, and that stored outputs are missing, when they are. When the
   download is what was asked for, the reply has no note. An outputs-only
   download also sets `outputs_only: true` in the reply.
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
    cross_site_val/cross_val_results.json
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
| CLI: `nvflare job download <job_id> --outputs-only` | Passes `outputs_only=True`. Text output prints the notice when the whole job came back. JSON output adds `outputs_only` (whether only the job's outputs were downloaded) and `outputs_note`. `artifacts` and `missing_artifacts` describe what was downloaded. For example, client logs are listed as missing in an outputs-only download. |

The returned path is the downloaded `<job_id>` folder in both modes. Result
files are at the same relative path under `workspace/` in both modes, so the run
summary and the CLI's artifact discovery find the model and metrics either way.

## Alternatives Considered

### Producing the outputs-only download

| Option | Outcome |
| --- | --- |
| Filter the stored workspace archive on each request: copy the selected entries into a new archive, then reply | Rejected. The copy runs before the server replies, and for a multi-GB model it can exceed the admin client's reply wait (5 seconds unless a command timeout is set). The download fails on exactly the large-model jobs it is for, and the path differs from a full download. |
| Filter on request, but make the admin client wait longer for this one reply | Rejected. It hides the cost rather than removing it, the copy still grows with the model, and outputs-only downloads still take a different path from full downloads. |
| Reply first, then prepare the files | Rejected. It needs a rework of how folder downloads start, for one command. |
| Download the whole job and filter it on the admin side | Rejected. It saves nothing on transfer, which is the point of the feature. |
| Archive the outputs at job end as a second copy, keeping the workspace archive whole | Rejected. Downloads reply at once, but every output, usually the model, is stored twice, and job end reads and writes it a second time. For large models that doubles their storage. |
| Move the outputs out of the run folder, archive them, then archive the rest | Rejected. Between the move and the workspace archive, a staging folder holds the only copy of the model: a retry after a failed workspace save, a crash, a failed move back, or a copy across filesystems could lose it. |
| **Store the outputs at job end apart from the rest, through two folders of hard links** | **Chosen.** Hard links name the outputs and the rest without copying or moving anything, and the run folder stays intact until both archives are stored, so a retry or crash loses nothing. Each file is stored once and written once, as before. An outputs-only download links the outputs archive, and a full download links both, so both reply at once. The admin client merges the two archives into `workspace/`. When links cannot be made, the workspace is stored whole. |

The server names the outputs and the rest with folders of hard links and hands
them to the job store, the same folder contract `save_workspace` uses. Each
store archives it the way it archives a workspace, so a filesystem store and an
object store such as S3 need no new content type, and the server builds no zip
of its own.

### Which files are outputs

| Option | Outcome |
| --- | --- |
| A declared list, written by the user or by each recipe | Rejected. Users would need to know internal paths, a custom component's files are left out until someone lists them, and a wrong list silently filters the model out. |
| The code that adds each component (a recipe, a model wrapper, a job class) records the component's files | Rejected. The same knowledge is applied at every place a class is used, each new recipe must remember it, and custom components and plain `FedJob`s declare nothing. |
| A resolver registered beside each component class, applied when `meta.json` is generated | Rejected. Exact for built-in components, but a custom persistor's job downloads whole until its author learns to register it, and a custom writer that is not a persistor is left out silently. Every new component needs a resolver. |
| A dedicated outputs folder that components save into | Rejected. It moves where every built-in component saves today, breaking existing paths; keeping the old paths means saving the model twice; and a custom component that saves elsewhere is left out silently. |
| **Every file the job produced: the workspace minus what was deployed with the job and left unchanged, and the server's logs and system files** | **Chosen.** Nothing to declare or register for any component, built-in or custom, recipe or hand-written job. Nothing the job produced is left out silently: the worst case is a few more files than needed, never a missing model. Files stay where they are written, so paths are the same in both downloads. |

### Telling the caller what it received

| Option | Outcome |
| --- | --- |
| An `outputs_manifest.json` file in the download, listing what was included | Rejected. It adds a file to the user's result that nobody acts on. The only actionable case is a whole-job fallback, and its reason is enough. |
| Separate reply fields for the whole-job reason and missing outputs | Rejected. The FLARE API only ever shows the reasons as one line. |
| **One `outputs_note` field in the `download_job` reply, plus an `outputs_only` mark, passed through by the folder pull** | **Chosen.** No extra file. The note is present only when the download is not what was asked for, and the FLARE API shows it as is. The mark lets the FLARE API tell an older server's whole-job reply from a filtered one. |

### Requesting the download

| Option | Outcome |
| --- | --- |
| A new admin command | Rejected. An older server would reject it, and it would need its own authorization and permission category. |
| An extra `download_job` argument | Rejected. The server validates extra arguments as targets, so an older server would reject it. |
| **A `download_job` command property** | **Chosen.** An older server ignores it and returns the whole job, and the command's authorization is unchanged. |

### When the job produced no output files

Return the whole job with a reason, rather than an empty download: an empty
`workspace/` helps no one, and the reason tells the user to check the job.

## Compatibility

| Case | Behavior |
| --- | --- |
| Existing scripts | `outputs_only` defaults to `False`. Full downloads return the same files as before. |
| Jobs submitted after this change | Their outputs are stored in an `outputs` component at job end, and the rest in `workspace`. Full downloads merge both into `workspace/`. |
| Existing custom `ExecEnv` subclasses | `Run` does not pass the new argument unless the caller asks for an outputs-only download. |
| New client, old server | Not supported: the admin client and the server run the same version. An older server ignores the property and returns the whole job; the FLARE API reports it as a whole-job download. |
| Old client, new server | Not supported: the admin client and the server run the same version. An older client's full download of a job with an `outputs` component would unzip `outputs.zip` into its own folder rather than into `workspace/`. |
| Failed or aborted job with `outputs_only=True` | The whole job, including logs. The FLARE API, Recipe API and CLI show why the whole job came back. |
| Completed job that produced no output files | The whole job, with the reason shown. |
| Job that finished before this change | No `outputs` component: its workspace archive holds everything. Full downloads are unchanged, and outputs-only downloads return the whole job, with the reason. |
| Job store implementation without `save_job_outputs` | The default saves nothing, so the outputs stay in the workspace archive. Outputs-only downloads return the whole job. |
| Tools that read the stored `workspace` component directly | A job's outputs are in the `outputs` component, not in `workspace`. The server's log commands are unaffected: log files stay in `workspace`. |

Known limitations:

- An output that is replaced, written to or removed between storing the
  outputs and linking the rest, or a new output that appears meanwhile, makes
  the server store the workspace whole, as before this change. Results
  published into the workspace by a container job launcher are the case this
  guards. The check compares each file's identity, size and modification time,
  so an overwrite that keeps all three is not detected.
- Files a component writes outside the job's workspace folders, such as an
  absolute path or a remote tracking server, are in neither download.

## Security

- The command, its authorization and its permission category are unchanged,
  so no new data becomes reachable. Outputs-only download returns a subset of what
  `download_job` already returns.
- The server links only files under the job's workspace folders, and never
  treats a symlink as an output.
- The job's credentials (`job_cert/`) are removed before archiving, as today,
  and are never treated as outputs.
- The admin client unzips a downloaded archive only into a plain folder name
  from the server's reply, never a path.

## Testing

Unit tests:

- Produced files: models, metrics, evaluation results, TensorBoard events and
  a new file under `custom/` and a deployed file the job rewrote are outputs;
  `meta.json`, `fl_app.txt`, server logs (prefixed and rotated), stats-pool
  files, `audit.log`, the signature file, `__pycache__`, `job_cert/` and the
  deployed app's unchanged files are not.
- The deployed app is read from the stored job for a `server` or `@ALL`
  deployment; when it cannot be read, the workspace is stored whole. A retry that cannot read the job clears a record
  an earlier attempt left.
- At job end, hard-link folders name the outputs and the rest at their paths,
  beside each workspace folder, without changing it: the same file, logs and
  symlinks with the rest. The outputs are stored as
  `outputs` and the rest as `workspace`. The workspace is archived whole when
  nothing was produced, links cannot be made, or `save_job_outputs` fails or
  saves nothing. A retry after a failed workspace save stores the same outputs
  again, and leftover links from an interrupted attempt are replaced.
- The rest is linked from a fresh walk after the outputs are stored; relative
  symlinks to outputs still resolve and linked folders stay links; an
  unreadable folder stores the workspace whole; the split is recorded in the
  job's meta, cleared by a later fallback, and skipped for job stores without
  `save_job_outputs`; a submitted record is removed.
- A source replaced right after its link was made, a new output written after
  the outputs are linked, or a stored output replaced or removed meanwhile,
  stores the workspace whole; a server log written meanwhile does not; a failed
  record clear fails the archival so it is retried.
- Downloads decide from the meta record; a split job whose outputs archive
  cannot be read downloads the rest with an `outputs_note`, and an
  outputs-only request returns the whole job with that reason.
- `download_job` with `outputs_only`:
  - links the stored `outputs` archive as `workspace.zip`, as a full download
    links the workspace;
  - falls back to the whole job for failed jobs or a job without an outputs
    archive, and reports the reason in its reply;
  - is unchanged without the property for a job without an `outputs`
    component.
- A full download of a job with an `outputs` component links both archives
  and tells the admin client to unzip `outputs.zip` into `workspace/`; the
  admin client merges them and accepts only plain folder names.
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
and the job's config, custom code and logs are absent.

## Implementation

| Part | Location |
| --- | --- |
| Meta record and command property | `nvflare/apis/job_def.py` (`JobMetaKey.JOB_OUTPUTS_STORED`, `OUTPUTS_ONLY_PROP`) |
| Selecting the outputs | `is_produced` and `link_job_outputs` in `nvflare/private/fed/server/job_outputs.py`; `JobRunner._deployed_files` reads the deployed server app from the stored job |
| Outputs storage | `JobRunner._save_job_outputs`, `SplitWorkspace` in `nvflare/private/fed/server/job_outputs.py`, `JobDefManagerSpec.save_job_outputs`, storage component `outputs` |
| Full download | `JobCommandModule._get_default_job_components` (adds `outputs.zip`), reply key `unzip_into`, `FileTransferModule.pull_folder` |
| Outputs-only download | `JobCommandModule.download_job` |
| Admin client and APIs | Admin client: `FileTransferModule.pull_folder`. FLARE API: `Session.download_job_result`. Recipe API: `ExecEnv.get_job_result`, `Run.get_result`. CLI: `nvflare job download --outputs-only` |
