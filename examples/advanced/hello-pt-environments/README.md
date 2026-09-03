# Hello PyTorch execution environments

This advanced continuation moves the [Hello PyTorch](../../hello-world/hello-pt/README.md) application from local
simulation to a local POC federation and then to a provisioned production system. It reuses the beginner example's
actual `client.py`, `model.py`, and `prepare_data.py`; only the Recipe options and execution environment change.

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
fixed defaults, both simulation and POC report 75% final accuracy on site-1 and 77% on site-2. Premerge CI enforces
that equivalence in addition to production construction/export; it does not claim that ordinary CI owns a live
provisioned production federation.

## Run a job-scoped local POC

```bash
python job.py --env poc
```

`PocEnv` provisions a local federation, starts separate server and client processes, submits the same application,
downloads the result, and stops the services. It needs permission to start processes and bind the standard NVFlare
POC ports. The POC lifecycle belongs to this invocation, so provisioning and process startup occur for every job and
make this deliberately slower than simulation.

On success, the command retains the POC workspace so the printed result path and service logs remain available. Copy
results you want to keep before another POC run replaces the workspace. If provisioning, submission, or monitoring
fails after this invocation begins its POC lifecycle, the command stops the processes, removes that failed workspace,
and exits nonzero. Before provisioning, a stopped retained workspace is moved aside atomically. If provisioning fails,
the partial replacement is removed and the retained workspace is restored with its prior results and logs intact.

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
nvflare job download JOB_ID

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
nvflare job download JOB_ID
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

Clients open that cache with downloads disabled, so concurrent processes do not race while writing it. This option is
useful for experimentation but is not a federated data partition.

For production, `--data_root` is a path on each client—not on the admin machine running `job.py`. Every site operator
must prepare CIFAR-10 at that same local path before the job is submitted. Running `prepare_data.py` beside the admin
startup kit does not populate remote clients. Use the synthetic default unless client-side data has been staged.
