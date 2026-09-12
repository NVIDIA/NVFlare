# Hello PyTorch execution environments

This advanced continuation moves the [Hello PyTorch](../../hello-world/hello-pt/README.md) application from local
simulation to a local POC federation and then to a provisioned production system. It reuses the beginner example's
actual `client.py`, `model.py`, and `prepare_data.py`; only the Recipe options and execution environment change.
Keep this directory in a full NVFlare checkout alongside `examples/hello-world/hello-pt`; copying this directory alone
does not include the shared application.

Complete the beginner simulation first so you already understand its training, evaluation, and artifacts:

```bash
cd examples/hello-world/hello-pt
python job.py
```

Then enter this directory and install the matching NVFlare 2.10 dependencies:

```bash
cd ../../advanced/hello-pt-environments
python -m pip install -r requirements.txt
```

## Compare the three environments

| Stage | What changes | Command |
| --- | --- | --- |
| Simulation | Server and clients run through the local simulator. | `python job.py` |
| POC | `PocEnv` provisions, starts, and stops a local system for this job. | `python job.py --env poc` |
| Production | `ProdEnv` connects to an already-running system through an authorized admin startup kit. | `python job.py --env prod --startup-kit <admin-kit> --username <admin-identity>` |

The Recipe, model, client script, data code, and local training loop stay the same across all three stages. With the
fixed seeds, local runs currently report 75% final accuracy on site-1 and 77% on site-2; these are observations,
not benchmark claims or exact CI thresholds. Premerge CI checks that simulation and POC produce identical per-site
final accuracies. Unit tests cover production argument handling, environment construction, and job export; they do
not connect to a live production federation.

## Run a job-scoped local POC

```bash
python job.py --env poc
```

`PocEnv` provisions a local federation, starts separate server and client processes, submits the same application,
downloads the result, and stops the services. It needs permission to start processes and bind the standard NVFlare
POC ports. The POC lifecycle belongs to this invocation, so provisioning and process startup occur for every job and
make this deliberately slower than simulation.

Each invocation uses a unique workspace beside the configured CLI POC workspace, with a `.recipe-<UUID>` suffix.
On success, the command prints and retains that workspace so the result and service logs remain available across
later runs. The configured CLI workspace and earlier Recipe results are preserved. Stop any running CLI POC with
`nvflare poc stop` before starting this example, since the services still need the same local ports.

If a job returns a downloaded result but its status is unsuccessful or unavailable, the command stops the services,
retains the result and workspace, prints the result path and `poc_console.log` locations, and exits nonzero. This keeps
the client errors available for diagnosis. Monitoring errors or interruptions before a result is obtained trigger
service shutdown and workspace cleanup. `PocEnv` handles deployment failures itself. If shutdown cannot be verified,
it preserves the workspace and reports recovery instructions.

If `NVFLARE_HOME` is set, POC admin transfers use `$NVFLARE_HOME/examples`, so downloaded results can be outside the
retained POC workspace. The printed result path is authoritative. To keep downloads inside the run workspace,
unset `NVFLARE_HOME` before running this example. Removing a retained workspace does not remove externally downloaded
results.

### Stop an interrupted run and remove retained artifacts

An interrupted Recipe run has its own workspace. Target that exact path to stop its services; an ordinary
`nvflare poc stop` targets the separate CLI workspace. For example, substitute the workspace printed by your run:

```bash
NVFLARE_POC_WORKSPACE="/tmp/nvflare/poc.recipe-<UUID>" nvflare poc stop
```

If the process was killed before it printed the workspace, locate the run's `.recipe-<UUID>` directory beside the
configured CLI POC workspace and inspect its service logs. After confirming its services have stopped and saving any
artifacts you need, remove that specific directory:

```bash
rm -rf "/tmp/nvflare/poc.recipe-<UUID>"
```

Remove any downloaded result outside that workspace separately, using the printed result path. Each invocation
retains its own directory, so repeat this for the particular old runs you no longer need.

## Connect to an existing production system

A production submission requires a running provisioned NVFlare system, network connectivity, and an authorized admin
startup kit. `--username` must match the identity represented by that kit; it defaults to `admin@nvidia.com`.

```bash
python job.py --env prod \
    --startup-kit /path/to/admin/startup-kit \
    --username researcher@example.com
```

This integrated Recipe path constructs the job, submits it, waits for completion, downloads the result, and prints
its location. `ProdEnv` does not start or stop the provisioned system; the server and clients must already be running
and ready for connections.

## Export for CLI-managed submission

Export the environment-independent job when you want to inspect or edit its generated configuration and submit it to
an already-running POC or production system with the NVFlare CLI:

```bash
python job.py --export --export-dir /tmp/nvflare/jobs
```

For a reusable local POC, prepare it once, start it, and submit as many jobs as needed before stopping it:

```bash
nvflare poc prepare -n 2
nvflare poc start
nvflare job submit -j /tmp/nvflare/jobs/hello-pt

# Replace JOB_ID with the ID printed by the submit command.
nvflare job monitor JOB_ID
nvflare job download JOB_ID -o /tmp/nvflare/hello-pt-results

# Keep the POC running for more jobs, then stop it when finished.
nvflare poc stop
```

For production, the provisioned system must already be running. Register and activate its admin startup kit, then use
the same job commands:

```bash
nvflare config add hello-pt-admin /path/to/admin/startup-kit
nvflare config use hello-pt-admin
nvflare job submit -j /tmp/nvflare/jobs/hello-pt

# Replace JOB_ID with the ID printed by the submit command.
nvflare job monitor JOB_ID
nvflare job download JOB_ID -o /tmp/nvflare/hello-pt-results
```

Unlike `PocEnv`, `nvflare job submit` does not own the system lifecycle: it assumes the selected POC or production
system is already running and leaves it running after the job. The command itself returns without waiting. The active
`nvflare config` selection supplies the startup kit and its admin identity to subsequent CLI commands; the Recipe
script's `--username` option is only for the integrated `python job.py --env prod` path.

Export alone verifies construction of the deployable application. It does not prove connectivity, authorization, or
successful execution on an external system. See the [Job CLI guide](../../../docs/user_guide/nvflare_cli/job_cli.rst)
for startup-kit selection and job lifecycle commands, and the
[deployment guide](../../../docs/user_guide/admin_guide/deployment/index.rst) for provisioning and production
operations.

## Advanced Recipe controls

Run `python job.py --help` for all example and Recipe export options. The most useful combinations are:

```bash
# Persist client metrics through a server-side TensorBoard receiver.
python -m pip install tensorboard
python job.py --experiment_tracking tensorboard

# Evaluate the final client models as well as the server models.
python job.py --evaluation cross-site

# Run the shared client script out of process and stream its logs.
python job.py --launch_external_process --enable_log_streaming

# Periodically release client model parameters and run garbage collection.
python job.py --client_memory_gc_rounds 1

# Override selected local-training controls; omitted values remain client-owned defaults.
python job.py --epochs 2 --batch_size 16 --learning_rate 0.05 --num_workers 0
```

The shared client defaults to one local epoch, batch size 32, and no data-loader worker processes. Its SGD learning
rate is 0.1 for synthetic images and 0.01 for CIFAR-10 unless `--learning_rate` overrides it.

`--evaluation none` produces a plain FedAvg job without post-training model evaluation. The default remains
`--evaluation final`, matching the beginner quickstart. `--evaluation cross-site` additionally collects each client's
latest local model and evaluates all submitted client and server models.

## Optional CIFAR-10 path

All simulated or local POC clients run on the same machine, so they can share one cache and the same logical CIFAR-10
datasets. Prepare both splits once before any clients start, and pass the same client-local path to the job:

```bash
python ../../hello-world/hello-pt/prepare_data.py --data_root "/data/cifar cache"
python job.py --dataset cifar10 --data_root "/data/cifar cache"
```

Each client checks for missing or empty cache files when loading data and reports the preparation command in its
error log. This happens after simulation or POC starts; `job.py` does not validate the cache, and exports do not
require local data. Failed POC jobs retain their results and service logs as described above. Use an absolute
client-local `--data_root` path to avoid depending on a client process's working directory. Clients open the cache
with downloads disabled, so concurrent processes do not race while writing it. The client checks file presence and
nonzero size; torchvision performs its own integrity checks when loading. This option is useful for experimentation
but is not a federated data partition.

For simulation and POC, relative `--data_root` paths are resolved from the submission working directory before
running or exporting the recipe. This only normalizes the path; it does not validate or download the cache.

For production, `--data_root` is preserved as a path on each client—not on the admin machine running `job.py`. Every site operator
must prepare CIFAR-10 at that same local path before the job is submitted. Running `prepare_data.py` beside the admin
startup kit does not populate remote clients. Use the synthetic default unless client-side data has been staged.
