# Task execution lifetime: CPU Process support

Client application execution can be resident for the job or disposable per task.
The default remains `resident`. With `task` lifetime, the Client Job (CJ) stays
resident for communication, input/result filters and publication, while a fresh
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
`"execution_lifetime": "task"`. Omit it or use `"resident"` for existing behavior.
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
  }
}
```

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

## Lifecycle and support boundaries

Workers receive credential-stripped bootstrap data and eager local FOBS input
artifacts. Result artifacts are immutable and validated against the attempt
identity and digest. For Client API scripts, `flare.send()` stages the result;
successful completion is committed only after the script and finalization
finish. A failure after `send()` therefore cannot publish a successful result.
The script's `result_wait_timeout` becomes the worker timeout.

The Process launcher owns a POSIX process group, observes descendants, and
escalates cancellation from SIGTERM to SIGKILL. Zombie-only groups are settled
because their members cannot execute or hold compute resources; a live or
uninspectable member still prevents settlement. The CJ requires confirmed
settlement before reading completion, forwarding analytics or returning a result
to ClientRunner. A worker must not detach descendants into another POSIX session;
process-group containment is not a hostile-code sandbox.

An explicit server acknowledgement of workflow admission (or a previously
received matching client task) releases the attempt's bulky input/result/bootstrap
payloads. Completion metadata and lifecycle diagnostics remain at
`<run_dir>/.nvflare/task-execution/diagnostics.jsonl`. Failed or unaccepted
attempts retain payloads for an explicit later retention decision. Transport OK
without an admission acknowledgement, including replies from older servers,
does not authorize cleanup.

This launcher does not reserve CPU, memory or GPU resources and rejects explicit
resource requests it cannot honor. CPU task execution must not inherit a job-long
GPU reservation. Component placement follows serialized component-ID references
and their transitive dependencies; components shared between workers and CJ
filters are rejected. Dynamic component lookup and additional component/event
combinations need explicit qualification.

## Integration checks

The NumPy Process simulator tests cover resident and task execution, Job API
export, CJ-owned filters, process settlement and publication cleanup:

```bash
python -m pytest -q tests/integration_test/fast/task_worker_process_e2e_test.py
```

On Linux, the hello-pt matrix runs the unchanged example script in resident and
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
