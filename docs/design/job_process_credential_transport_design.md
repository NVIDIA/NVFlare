# Job-Process Credential Transport

Status: implemented (this PR covers the framework side: parsers, utils, Process/Docker/K8s
launchers; the Slurm ride-along lands with the Slurm launcher PR). Scope: transport only —
per-job/short-lived credentials are a separate future PR; this design is the delivery
mechanism they will reuse.

## Problem

CJ/SJ bootstrap credentials (auth token, token signature, session ID) are passed as
command-line arguments. Command lines are world-readable on Linux hosts (`ps`,
`/proc/*/cmdline`, including containerized processes seen from the host), stored in the K8s
API as pod `args`, and routinely captured into logs by monitoring tools. Slurm is already
fixed via its `worker_entry` adapter; the other launchers are not.

## Design

**Worker contract.** Three env names next to `JobProcessArgs`:

```python
class JobProcessEnv:
    AUTH_TOKEN = "NVFLARE_JOB_AUTH_TOKEN"
    TOKEN_SIGNATURE = "NVFLARE_JOB_TOKEN_SIGNATURE"
    SSID = "NVFLARE_JOB_SSID"
```

In the CJ and SJ arg parsers, per credential:

```python
value = os.environ.pop(JobProcessEnv.AUTH_TOKEN, None)
parser.add_argument("--token", default=value, required=value is None)
```

CLI wins if supplied; missing-both keeps today's error; the env var is always removed, so
job-spawned children never inherit it. Verify the CJ's multi-GPU `sub_worker_process` spawn
does not re-emit credentials in argv; if it does, apply the same treatment.

**Launchers** stop rendering credential flags into commands and deliver values as:

| Launcher | Transport | Notes |
| --- | --- | --- |
| Process | child environment | |
| Docker | container environment | not a secret store: `docker inspect` shows it to daemon users — documented, accepted (Docker has no per-container secret primitive) |
| K8s | per-job Secret, `env[].valueFrom.secretKeyRef` | literal env in the pod object would be as API-visible as args; the pre-existing workspace-transfer token rides the same Secret for the same reason. Lifecycle: create Secret → create pod → patch Secret with an `ownerReference` to the pod → delete Secret when the job handle reaches a terminal state (completed pods are not deleted, so GC alone would never fire on the success path). A launch failure before pod creation deletes the Secret immediately; the ownerReference makes Kubernetes GC the backstop when the parent dies before observing a terminal state. Extends the startup-kit Secret RBAC with the `patch` and `delete` verbs (helm role templates updated). |
| Slurm | unchanged `secret.env` file, now exporting the generic names | batch script starts the real worker module directly; **delete `worker_entry.py`** |

## Compatibility

No fallback machinery. Docker/K8s job images must contain an NVFlare release with the env
parser; an older image fails immediately with the worker's existing required-argument error —
loud, diagnosable, release-noted. Direct CLI invocation keeps working (CLI path is retained
in the parsers). Custom launchers that render commands from `generate_*_command` /
`JOB_PROCESS_ARGS` and implement `launch_job` directly must now also deliver
`get_credential_env(job_args)` via the child environment — release-noted; the launcher
design doc's step tables document the new step.

## Threat notes

Protects against command-line inspection and log/telemetry capture. Does not protect against:
root/admins, code inside the job process, same-UID readers of `/proc/<pid>/environ` (the
exec-time snapshot persists for the process lifetime regardless of `os.environ.pop`), or
credential reuse after theft (the per-job-credentials PR's job).

## Code touch

`job_launcher_spec.py` (+5), CJ/SJ parsers (~+10 each), `job_launcher_utils.py` (split
credential args from module args, ~+15), the four launchers (Process/Docker ~10 each; K8s
~40 for the Secret + ownerReference; Slurm ~−40 net including `worker_entry` deletion),
Slurm rev-11 doc/QA ride-along (batch-script contract, code-touch list, SEC-04).

## Tests

Parser: env-only accepted, CLI wins, missing-both errors, env removed after parse.
Per launcher: no credential value in argv/container command/pod object/batch text; K8s Secret
carries an ownerReference to its pod; end-to-end authentication succeeds on all four.
Live acceptance: inspect real process metadata, not rendered config.
