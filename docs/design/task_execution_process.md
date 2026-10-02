# Task execution lifetime: CPU Process support

Client application execution can be job-based or task-based.
The default lifetime is `job`. With `task` lifetime, the Client Job (CJ) remains
alive for communication, input/result filters and publication, while a fresh
worker runs the application's Executor for each assignment.

This increment supports ordinary Executors and the exact `ClientAPIExecutor`
class with `execution_mode="in_process"`, on CPU with the Process backend. It
does not add Docker, Kubernetes or Slurm task launchers, GPU admission, automatic
application-state transfer, external-process/attach Client API adapters, retries
or restart recovery.

## Job configuration

Select task lifetime through the Job API:

```python
from nvflare.job_config.api import FedJob

job = FedJob(name="my-job", min_clients=2, execution_lifetime="task")
```

Or use the common recipe setting before export or execution:

```python
recipe.set_execution_lifetime("task")
```

For a hand-written client application configuration, set the top-level field
`"execution_lifetime": "task"`. Omit it or use `"job"` for existing behavior.
This setting is separate from `ClientAPIExecutor.execution_mode`.

The submitted/exported job retains the original Executor specifications and
filters. It does not name the internal supervisor or select a launch backend.
The CJ authorizes the original application specifications and constructs the
private framework supervisor; the worker repeats component authorization before
importing or constructing application code. The default class allow-list is not
expanded, and a job cannot select the supervisor explicitly.

An in-process Client API script uses the same `client.py` for either lifetime.
In task mode it runs on the worker's main thread, with one assignment per worker.
Every worker starts with fresh Python state: applications requiring persistent
optimizer or other task-local state must save and reload that state explicitly.

Component dependencies use constructor arguments named `*_id` / `*_ids` and
their transitive references. For nonstandard/dynamic wiring, declare a
`component_dependencies` list on the relevant component specification (outside
`args`). Ordinary string values are not dependency declarations. A component
referenced by both the worker graph and a CJ filter, widget, or other retained
component is rejected rather than silently removed from the CJ.

Nonstandard references such as `source_model` require an explicit dependency
declaration. Task placement cannot infer whether an arbitrary string is a
component ID or ordinary application data without importing application code.
For example, an Executor that looks up `engine.get_component(self.source_model)`
declares that reference on its specification:

```json
{
  "path": "custom.Trainer",
  "component_dependencies": ["model"],
  "args": {"source_model": "model"}
}
```

The same rule applies to nonstandard references in filters, retained CJ
components and transitive worker dependencies. Nested argument dictionaries
remain ordinary data unless they are actual component specifications.

Task-lifetime jobs require client runtimes advertising `task_execution_process_v1`.
The server rejects deployment to clients without that capability; it does not
silently fall back to job lifetime. The server and client must both support this
feature. Default job-lifetime jobs keep the existing launch arguments.

## Site-owned launch backend

The site, not the job, selects the TaskLauncher. An optional top-level entry in
the site's `resources.json` configures it:

```json
{
  "task_launcher": {
    "path": "nvflare.app_common.task_launcher.process_launcher.ProcessTaskLauncher",
    "args": {
      "stop_grace_period": 2.0,
      "descendant_settle_timeout": 0.25,
      "poll_interval": 0.05
    },
    "environment_variables": ["SITE_DATA_API_KEY"]
  },
  "task_execution": {
    "artifact_cleanup": "job"
  }
}
```

Policy is read only from the canonical site resources file, including its
`resources.json.default` fallback, never from job-supplied resources files.
Omitting the entry selects the Process launcher. Custom site launchers implement
`nvflare.apis.task_launcher_spec.TaskLauncherSpec` and declare `launch_mode`.
The CJ validates that this mode matches the actual selected JobLauncher before
injecting the launcher into its framework supervisors. The initial supported
deployment is Process Client Parent (CP), Process CJ and Process task worker;
mixed-mode chains are rejected. The simulator uses its own Process startup path.

`environment_variables` is an optional site-owned allow-list of environment
variable names to forward from the CJ to workers. It supports script arguments
such as `${secret:SITE_DATA_API_KEY}` without placing secret values in submitted
jobs, bootstrap files or diagnostics. Jobs cannot expand this policy. Federation
bootstrap credentials, Client API bootstrap paths and GPU visibility variables
cannot be forwarded through it; unapproved variables remain excluded.

TaskLauncher/TaskHandle contracts are public extension points in `nvflare.apis`.
The Process backend lives in `nvflare.app_common.task_launcher`. The supervisor
in `nvflare.private.fed.client.task_worker_executor` is internal runtime code,
not an application customization or subclassing point. Future backends can use
the same supervisor through site/runtime injection.

The legacy `MultiProcessExecutor`/`PTMultiProcessExecutor` stack and its rank
sub-worker runtime have been removed, without compatibility aliases. Old job
configurations selecting those classes must migrate to the job-based Client API
with external-process `torchrun`, as in `examples/advanced/multi-gpu/pt`.
That distributed training path is not part of this CPU task-worker profile.
Task-worker component construction now lives in the private runtime utility
`nvflare.private.fed.utils.worker_component_builder`. It and the CJ configurator
share the authorization-tree walker; each retains its runtime's site-policy wiring.
The shared walker preserves existing traversal semantics: recognized component
specifications expose nested components through `args`, not arbitrary metadata.

The worker's private `TaskRuntime` owns its local context manager, compute graph,
event dispatch and abort signal. It does not inherit `ClientEngineSpec` or
`ServerEngineSpec`. Client metadata and Client API backend injection live in
`nvflare.private.fed.client.task_worker_client_api`, outside the local runtime
and compute pipeline. The Client API adapters share metadata, API/script setup
and owned DataBus cleanup with the job-based in-process backend; only their
transport and execution lifetimes differ. Server-side worker integration and
parent/job engine-interface restructuring remain future work.

## Lifecycle and support boundaries

Worker argv and environments omit federation bootstrap credentials; workers do
not create a federation connection. This is not credential filesystem isolation:
Process workers run under the site's UID and share its workspace, including
credential files such as `job.key`. Startup clears credential environment variables defensively,
before importing application code, in case a launcher forwarded them incorrectly.
Workers receive inert bootstrap configuration and eager local FOBS input artifacts.
Result artifacts are immutable and validated against the attempt identity and digest.
For Client API scripts, `flare.send()` durably stages `script_result.fobs`. The
runtime-injected Client API backend returns a Shareable through the same Executor
pipeline as ordinary Executors, including task hooks and finalization. The final
`result.fobs` and completion are written only after that pipeline finishes.
A failure after `send()` therefore cannot publish a successful result.
Likewise, `system_panic()` triggers the attempt's abort signal and fails the
attempt even if its Executor returns a Shareable. Compute finalizers still run;
the worker does not write successful completion after a fatal event.
The script's `result_wait_timeout` starts at its first `flare.receive()` and ends
at durable `flare.send()`: process startup, imports before receive, and finalizers
are not charged to that result-wait budget. An optional `worker_timeout` on the
executor entry (next to `tasks` and `executor`) bounds the entire attempt,
including startup and finalization; its default is `None`.

The worker uses the site's logging configuration with an attempt-specific file
prefix. `NVFLARE_SECURE_LOGGING` and `FL_LOG_LEVEL` are inherited as framework
logging settings. Client API logs and ordinary Executor analytics on the standard
`analytix_log_stats` channel are staged
locally and replayed by the CJ only after successful settlement and finalization.

The Process launcher owns a POSIX process group, observes descendants, and
escalates cancellation from SIGTERM to SIGKILL. Zombie-only groups are settled
because their members cannot execute or hold compute resources; a live or
uninspectable member still prevents settlement. The CJ requires confirmed
settlement before reading completion, forwarding analytics or returning a result
to ClientRunner. A worker must not detach descendants into another POSIX session;
process-group containment is not a hostile-code sandbox.
A small guardian in the same group cleans up on CJ loss or worker exit, including
remaining descendants. The CLI bypasses Python's unbounded exit-time thread and
child joins only after compute finalization and durable completion. Log flushing
is attempted with a two-second bound so a stray thread holding a logging lock
cannot prevent process exit.
Unconfirmed settlement fails the job and retains the handle and artifacts.
`torchrun` modes that detach ranks into separate sessions are unsupported by this
task-worker profile; job-based external-process multi-GPU execution is unchanged.

The site controls bulky input, staged/final result, analytics and bootstrap
retention with `task_execution.artifact_cleanup` in `resources.json`:

- `job` (default): retain payloads during the job; release owned attempts at
  `END_RUN`, after worker settlement and artifact reads finish. This does not
  depend on server acceptance and includes settled failed/unaccepted attempts.
- `accepted`: release each settled attempt after both a successful result send
  and explicit server acknowledgement of successful workflow result processing
  (or a previously accepted matching client task). Server result-filter and
  result-callback failures are not acknowledged as accepted, including retries.
  Failed/unaccepted attempts remain.
- `retain`: do not automatically release task payloads, even at job end.

All policies retain completion/failure records and lifecycle diagnostics under
`<run_dir>/.nvflare/task-execution/`. Jobs cannot override site retention policy.
No policy deletes payloads while a worker is live or settlement is unconfirmed.
Abrupt CJ termination can leave artifacts for later site-managed cleanup; this
slice does not add crash recovery or a background retention service. `retain`
does not prevent an administrator from removing the whole job workspace.

Acceptance is successful workflow result processing, not a durable aggregation/checkpoint guarantee.
Transport OK without an admission acknowledgement (including replies from older
servers) does not trigger the optional `accepted` policy. It does not affect
the default `job` policy.

This launcher does not reserve CPU, memory or GPU resources and rejects explicit
resource requests it cannot honor. CPU task execution must not inherit a job-long
GPU reservation. Component placement follows serialized component-ID references
and their transitive dependencies; components shared between workers and CJ
filters or retained components are rejected. Dynamic component lookup and additional component/event
combinations need explicit qualification.

## Integration checks

The NumPy Process simulator tests cover job-based and task execution, Job API
export, CJ-owned filters, process settlement and publication cleanup:

```bash
python -m pytest -q tests/integration_test/fast/task_worker_process_e2e_test.py
```

On Linux, the hello-pt matrix runs the unchanged example script in job-based and
task mode in SimEnv, PocEnv and ProdEnv:

```bash
python -m pytest -q tests/integration_test/fast/hello_pt_task_lifetime_test.py
```

ProdEnv requires a running provisioned Process deployment, with
`NVFLARE_E2E_PROD_ADMIN_KIT` pointing to its admin kit and
`NVFLARE_E2E_PROD_WORKSPACE` to its participant workspace root. Missing deployment
configuration skips those rows; skipped rows are not production qualification.
PocEnv/ProdEnv use ProcessJobLauncher and task rows use ProcessTaskLauncher.
These checks qualify the CPU Process profile, not GPU resource admission or
other launch backends.
