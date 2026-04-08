# Minimal K8s Job-Level Monitoring Validation

This example submits a minimal NumPy FedAvg job to a production NVFLARE deployment and verifies that job-level metrics reach the Kubernetes monitoring stack.

It was validated against a MicroK8s deployment with:

- `statsd-exporter`, Prometheus, and Grafana in namespace `nvflare-monitoring`
- an NVFLARE server pod and one connected client pod (`site-2`)
- `StatsDReporter` pointed at `statsd-exporter.nvflare-monitoring.svc.cluster.local:9125`

This example is intentionally lightweight so the validation focuses on monitoring and deployment wiring rather than dataset setup.

## Prerequisites

- Follow [../../k8s/README.md](../../k8s/README.md) to deploy the monitoring stack.
- Make sure the NVFLARE runtime image used by the server and client includes `datadog`.
- Start the NVFLARE server and at least one client pod in Kubernetes.
- Keep the admin startup kit signed and unchanged. Do not edit `fed_admin.json` in place.

## Installation Note

If you need the latest monitoring export behavior from `main`, install NVFLARE from this repo instead of relying on the latest PyPI release.

For example, from the repo root:

```bash
python3 -m pip install -e .
```

## Submit the Job

From this directory in the repository, run:

```bash
python3 job.py \
  --startup_kit_location /path/to/prod_00/admin@nvidia.com \
  --client_sites site-2 \
  --statsd_host statsd-exporter.nvflare-monitoring.svc.cluster.local \
  --statsd_port 9125
```

Set `--startup_kit_location` to your provisioned admin startup kit and `--client_sites` to the site name(s) that will run the job.

The default `--statsd_host` is correct when the NVFLARE **server and clients** run in the **same Kubernetes cluster** as the monitoring stack. If not, set it to a hostname or IP those processes can reach on port 9125.

## What the Example Does

- creates a minimal NumPy FedAvg job with one train script: [client.py](./client.py)
- adds `JobMetricsCollector` on server and client
- adds `StatsDReporter` on server and client
- exports the job before submission
- ensures the exported job config keeps the explicit StatsD endpoint
- submits the job with the admin startup kit and waits for completion

## Validated Metrics

The validated run emitted job-level metrics with `env="k8s"` and `job_id=<job-id>` tags, including:

- `_before_process_submission_count`
- `_after_process_submission_count`
- `_before_process_task_request_count`
- `_after_process_task_request_count`
- `_before_pull_task_count`
- `_after_pull_task_count`
- `_before_task_execution_count`
- `_after_task_execution_count`
- `_round_started_count`
- `_round_time_taken`
- `_run_time_taken`
- `_task_execution_time_taken`

Both server and client-side metrics were observed in `statsd-exporter`.
