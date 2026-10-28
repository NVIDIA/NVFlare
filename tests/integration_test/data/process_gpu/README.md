# Process GPU qualification fixture

Use an exact frozen candidate and existing administrator-approved capacity.
Check lease ownership, active workloads, duplicate jobs and imported source before
running. These tests do not schedule or reserve GPUs and do not exclude other
processes. Keep unrelated workloads untouched and retain earlier evidence.

## Site setup and normal controls

Use the static gpu_devices configuration from the
[profile](../../../../docs/design/process_task_gpu_profile.md). Give each site its
explicit one-device assignment. Existing ResourceManager components need no new
API. The standard empty ListResourceManager fixture remains sufficient for these
task controls. Set a finite task worker_timeout so hung workers have a bounded
execution budget. Use real YAML for one task control and HOCON for another to
exercise actual CP startup and CJ policy, authorization and logging readers.

The exporter uses the existing FedAvgRecipe and sets only `execution_lifetime`
for the lifecycle choice. Its default Client API `in_process` script runs directly
inside the disposable task worker; no extra external trainer or internal executor
configuration is needed. The site still chooses Process. Shared validation rejects
task lifetime with `external_process`/`attach` on every backend; job-lifetime
`external_process` with `launch_once=False` remains the existing fresh-trainer
option. See the profile's mode matrix for its different publication/lifetime
guarantees and the startup/model/optimizer reload costs.

Export and submit these two-client, two-round jobs serially through the existing
provisioned admin connection:

```bash
python tests/integration_test/data/process_gpu/export_job.py --device cpu --lifetime job --output /tmp/process-gpu-jobs
python tests/integration_test/data/process_gpu/export_job.py --device cpu --lifetime task --output /tmp/process-gpu-jobs
python tests/integration_test/data/process_gpu/export_job.py --device cuda --lifetime task --output /tmp/process-gpu-jobs
python tests/integration_test/data/process_gpu/export_job.py --device cuda --lifetime task --zero-gpu-site site-2 --output /tmp/process-gpu-jobs
python tests/integration_test/data/process_gpu/export_job.py --device cuda --lifetime job --output /tmp/process-gpu-jobs
```

Run both CPU controls and both CUDA/task controls with the task visibility
profile. The CUDA/task export uses resource_spec.@default.num_of_gpus=1 and
independent empty CJ budgets normalized from the client task configurations. The
mixed control adds a site-2 zero-GPU override: site-1 must perform real CUDA
computation while site-2 trains on CPU with no visible CUDA devices. Check that
server-stored and deployed metadata preserve @default, the site override and the
CJ/compute split. Task CJs must have empty CUDA visibility in every task control.
CPU workers require zero visible CUDA devices; CUDA workers require actual
allocated CUDA bytes and kernels. For the resident CUDA/job control, after owned tasks settle, stop only
owned idle participants, remove gpu_devices and restore the normal job resource
manager/consumer configuration with the administrator-approved per-site assignment.
Run this control serially; an empty job GPU inventory is not a passing control.

Each normal job must finish successfully, produce FL_global_model.pt with weight
approximately [[2, -3]] (absolute tolerance 0.001), and cross_val_results.json with
both sites' final MSE below 0.0001. Preserve job IDs, terminal states, both rounds,
actual imports/configuration, process_gpu_compute records and final models.

## OOM and cleanup recovery

Export the real CJ/worker fault case:

```bash
python tests/integration_test/data/process_gpu/export_job.py --device cuda --lifetime task --gpu-oom --output /tmp/process-gpu-jobs
```

Site-1 training requests one tensor larger than the device's total VRAM. This
forces a real CUDA OOM without filling the GPU with repeated allocations. Confirm
an OutOfMemoryError failure artifact, no successful completion for that attempt,
a bounded terminal job outcome with the site failure reported according to controller policy, and no surviving owned worker/descendant or
CUDA context. Then submit a normal CUDA/task job on the same devices and confirm
successful training/evaluation. Do not reset the GPU or restart the whole site to
make recovery pass. Preserve failure and subsequent success evidence.

For focused real-CUDA process-scope tests, save an approved one-device assignment
as JSON, for example {"gpu_devices": ["GPU-<approved-UUID>"]}:

```bash
NVFLARE_PROCESS_GPU_DEVICES=/approved/path/devices.json python -m pytest -v \
  --basetemp=/approved/path/process-gpu-evidence \
  tests/integration_test/slow/process_task_gpu_test.py
```

The suite covers successful reuse before publication finishes; descendant cleanup
on cancellation and wait timeout; and exception, abrupt exit and actual CUDA OOM,
each followed by a successful fresh CUDA worker. Framework supervision tests also
verify that settled OOM failure clears active ownership and an uncertain cleanup
retains the handle and rejects new work. A controlled observation fault in unit
tests is not evidence of a real CUDA/driver fault.

## Actual CJ publication and receipt retry

Run another CUDA/task job with the explicit test-only hook on owned idle site-1:
set NVFLARE_PROCESS_GPU_DELAY_ROOT to a fresh evidence directory,
NVFLARE_PROCESS_GPU_DELAY_SITE=site-1 and NVFLARE_PROCESS_GPU_RETRY_ONCE=1, and add
only this fixture's cj_hooks directory to that fixture's PYTHONPATH. Other sites
run normally. Never install the hook globally or copy a private kit.

The real CJ pauses its first send after its supervisor returns a settled result.
The hook requires an empty CJ CUDA environment, records the task identity and
waits for a resume marker; it also drops one receipt observation to force a real
saved-result retry. While it waits, run:

```bash
python tests/integration_test/data/process_gpu/reuse_during_upload.py \
  --devices-config /approved/path/devices.json \
  --runtime-root /owned/site/job/.nvflare/task-execution \
  --compute-root /owned/site/job/process-gpu-models \
  --pause-root /approved/path/cj-pause \
  --output /approved/path/reuse-evidence
```

Use the original task's assigned device. The helper correlates the original worker's CUDA compute record, completion and
supervisor settlement diagnostics, runs another real CUDA worker on that device,
then resumes publication. It verifies settlement < CJ pause < new CUDA compute <
original publication finished and that receipt retry reaches RECEIVED. Confirm
the original worker did not rerun, and saved payload bytes remain intact.

Remove the test hook environment from owned idle participants afterward. Capture
cleanup showing no campaign workers, descendants, tested listeners or GPU compute
contexts remain. Retain old ledgers as historical evidence; the revised product
never opens them. Do not delete records, alter driver/security settings, reset a
GPU, interrupt unrelated work or create new leases to complete the campaign.
