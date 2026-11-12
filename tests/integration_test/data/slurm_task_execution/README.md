# Slurm qualification fixture

This fixture exports a bounded two-client, two-round compute probe. It does not
submit jobs. Configure the site using the Slurm deployment guides, install the
candidate and application dependencies on the participating hosts, then submit
through the normal NVFlare job CLI.

From the repository root, export CPU and CUDA cases in each lifetime:

```bash
PYTHONPATH="$PWD" python tests/integration_test/data/slurm_task_execution/export_probe.py --device cpu --clients 2 --lifetime job --output /tmp/slurm-probes
PYTHONPATH="$PWD" python tests/integration_test/data/slurm_task_execution/export_probe.py --device cpu --clients 2 --lifetime task --output /tmp/slurm-probes
PYTHONPATH="$PWD" python tests/integration_test/data/slurm_task_execution/export_probe.py --device cuda --gpu-count 1 --clients 2 --lifetime job --output /tmp/slurm-probes
PYTHONPATH="$PWD" python tests/integration_test/data/slurm_task_execution/export_probe.py --device cuda --gpu-count 1 --clients 2 --lifetime task --output /tmp/slurm-probes
```

Repeat with `--resource-format portable` to exercise portable compute and
independent CJ budgets. Preserve native and portable cases separately.

`client.yaml` and `server.yaml` are deployment inputs for a local, single-node
fixture. Replace their environment paths and size the parent allocations for the
complete campaign. Their loopback networking is suitable only when all
allocations share a host. Production qualification needs routable cross-host
worker connections, verified storage semantics and the intended transport security.

Run cases serially and retain:

- Candidate, script and configuration hashes; NVFlare and dependency versions.
- Server-stored submitted metadata, including literal `@default`/`@ALL` keys,
  compute requirements and the independent CJ budget.
- Job/site/task/attempt identities joined to native allocation and step IDs.
- Requested and allocated CPU, memory and GPU TRES. Explain scheduler rounding.
- Compute checksums, returned metrics and the persisted server model. CPU must
  observe no CUDA devices; CUDA must execute kernels and synchronize on each site.
- Application result, cleanup confirmation and GPU capacity reuse as separate
  outcomes. A successful result remains successful if cleanup is unconfirmed.
  Claim resource release only with scheduler evidence; demonstrate capacity reuse
  with another successful allocation on the released capacity.

Record blocked, failed and unconfirmed cases honestly. A local mock, exported
metadata, low GPU utilization or a cancellation acknowledgement does not establish
resource release. Retain control records and allocation IDs for native Slurm
investigation. Include queued/running cancellation, delayed submission, node loss
and accounting outages when qualifying failure behavior.

Targeted local checks:

```bash
python -m pytest -q tests/unit_test/app_opt/job_launcher/slurm_task_launcher_test.py tests/unit_test/app_opt/job_launcher/slurm_task_service_test.py tests/unit_test/app_opt/job_launcher/slurm_task_integration_test.py tests/unit_test/app_opt/job_launcher/slurm_task_qualification_test.py
python -m pytest -q tests/integration_test/fast/slurm_task_worker_test.py tests/integration_test/fast/slurm_task_session_test.py tests/integration_test/fast/task_execution_process_test.py tests/integration_test/fast/task_supervisor_process_test.py
```
