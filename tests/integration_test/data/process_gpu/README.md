# Process GPU qualification fixture

These commands prepare tests; they do not claim any Colossus result. Use the
exact frozen candidate, approved existing capacity and a verified Process
CP → Process CJ → Process worker topology. Revalidate lease ownership, idle
capacity, live imports, and device ownership before changing the fixture.

## Normal controls

From the candidate repository, export four two-site, two-round jobs:

```bash
python tests/integration_test/data/process_gpu/export_job.py --device cpu --lifetime job --output /tmp/process-gpu-jobs
python tests/integration_test/data/process_gpu/export_job.py --device cpu --lifetime task --output /tmp/process-gpu-jobs
python tests/integration_test/data/process_gpu/export_job.py --device cuda --lifetime task --output /tmp/process-gpu-jobs
python tests/integration_test/data/process_gpu/export_job.py --device cuda --lifetime job --output /tmp/process-gpu-jobs
```

Run the first three serially on a task-exclusive site configuration from
[the profile](../../../../docs/design/process_task_gpu_profile.md). Each site's
GPU pool must contain sufficient approved devices. Sites sharing devices must
use the same canonical ledger. Both CPU rows assert zero CUDA devices; the GPU
task row asserts one and executes actual CUDA training and evaluation.

For the GPU job row, after all task workers and reservations are settled, use
the supported serial job-mode configuration described in the profile. Each site
must have its own approved GPU capacity. Remove the task partition and enable
the ordinary CP GPUResourceManager/GPUResourceConsumer; do not submit this row
to the task-only empty job pool. Submit each exported directory through the
existing provisioned admin connection and retain job IDs and terminal states.

Each passing job must produce `FL_global_model.pt` with weight approximately
`[[2, -3]]` (absolute tolerance 0.001), and `cross_val_results.json` with site-1
and site-2 results and final MSE below 0.0001. Capture both clients' two training
rounds and evaluation logs. `process_gpu_compute` records contain PID/PPID,
site, round, CUDA visibility, device name, allocated CUDA bytes and loss.
GPU rows require actual CUDA bytes and kernels, not visibility alone.

For task rows, correlate each completion/diagnostic identity with the ledger's
physical execution ID, UUIDs and released record. Confirm no live worker,
descendant or GPU compute context remains. CPU CJs must not own GPU contexts.

## Real GPU failure and reuse scope tests

On the approved host, save only the site's `gpu_partition` object as a test
configuration JSON file. It must name the SAME persistent ledger used by its
participating CJs, not a new directory. Schedule these tests serially after the
normal task jobs; no unrelated GPU workload may use the tested partition.

```bash
NVFLARE_PROCESS_GPU_PARTITION=/approved/path/partition.json python -m pytest -v \
  --basetemp=/approved/path/process-gpu-evidence \
  tests/integration_test/slow/process_task_gpu_test.py
```

The tests run CUDA, train and save a model, check contention without overcommit,
cancel a process group containing a descendant, and exercise worker exception
and abrupt exit. A controlled unavailable scope observation tests reservation
retention; CUDA itself remains real. These are launcher scope tests, separate
from the two-site federation controls.

The scope reuse case holds a synthetic publisher blocked after first-worker settlement
and runs another CUDA worker on the same UUID before releasing that publisher.
It verifies saved model bytes remain readable. This is a deterministic local
publication-stage harness; it does not by itself qualify a delayed upload
through the resident CJ. Use the additional case below for that boundary.

## Real CJ delayed publication and receipt retry

After the normal matrix, schedule an additional CUDA task job on the task-only
profile. Restart only the owned, idle site-1 fixture with these explicit test
environment settings; do not install the hook globally or copy a private kit:

```bash
NVFLARE_PROCESS_GPU_DELAY_ROOT=/approved/path/fresh-cj-traces \
NVFLARE_PROCESS_GPU_DELAY_SITE=site-1 \
NVFLARE_PROCESS_GPU_RETRY_ONCE=1 \
PYTHONPATH="$CANDIDATE/tests/integration_test/data/process_gpu/cj_hooks:$CANDIDATE:$PYTHONPATH" \
  bash /owned/site-1/startup/start.sh
```

Use an absolute verified candidate path and a fresh, test-owned trace directory.
Both sites and the server must import the frozen candidate. The test-only
`sitecustomize.py` wraps the existing CJ publication methods. It pauses the first
publication for site-1 for at most 180 seconds, asserts empty CJ CUDA visibility,
then uses the original transport and readiness/receipt code. Once the server
owns the result, it deliberately drops one receipt observation to exercise the
real retry path. Record the hook's hash and environment along with source identity.
No authentication, resource, class policy or server receipt check is disabled.

While the CUDA task job starts, run:

```bash
python tests/integration_test/data/process_gpu/reuse_during_upload.py \
  --partition /approved/path/partition.json \
  --pause-root /approved/path/fresh-cj-traces \
  --output /approved/path/fresh-reuse-evidence
```

This waits for the real CJ pause, finds that exact assignment in the shared
ledger, and requires its reservation already released. It starts an independent
real-CUDA worker on the same UUID and ledger, trains/saves a model, confirms
whole-scope settlement, and only then resumes the CJ. It requires a successful
real `RECEIVED` receipt after the injected loss and writes `timeline.json` proving
release < CJ pause < new CUDA compute < prior CJ publication finished. Preserve
the original job's completions and ledger rows to prove the receipt retry did
not rerun its physical attempt. Complete the job and verify its final artifacts
and residue just as for the normal matrix. Remove the test environment from
owned participants before other work resumes. A missing hook marker, timeout,
failed receipt, or missing final artifact is a failed/incomplete case.

Keep logs, models, JSON facts, exact source/tree/config hashes, ledger snapshots
and initial failed attempts. Test collection without the explicit partition
environment skips GPU cases; a skip is not a GPU pass. Never export private kits
or credentials as evidence. An acknowledgment or active chat is not a running job.
