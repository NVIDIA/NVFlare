# NVFLARE Monitoring with StatsD, Prometheus, and Grafana

NVFLARE monitoring publishes FL system metrics through `StatsDReporter`, converts them with `statsd-exporter`, stores them in Prometheus, and visualizes them in Grafana.

This guide explains the monitoring architecture, supported topologies, and how to choose the right deployment path.

## Choose a Guide

Use the guide that matches your environment:

| If you want to... | Start here | Notes |
|-------------------|------------|-------|
| run the original local or POC monitoring walkthroughs for setup 1 and setup 2 | [jobs/README.md](jobs/README.md) | Best for learning the existing monitoring model end to end on one machine or in the classic POC layout. |
| deploy the monitoring stack in Kubernetes | [k8s/README.md](k8s/README.md) | Covers the monitoring stack manifests, in-cluster validation, and hybrid deployment guidance. |
| submit a minimal monitored job to a production-style Kubernetes deployment | [jobs/k8s_hello_numpy/README.md](jobs/k8s_hello_numpy/README.md) | Validated against MicroK8s with job-level metrics visible in `statsd-exporter`. |

Common issues (Grafana datasource, StatsD reachability, submitter host, `datadog`) are covered in [Troubleshooting](#troubleshooting) below.

## Architecture

Today the metrics pipeline is:

```text
NVFLARE (StatsDReporter)
  -> StatsD or DogStatsD on port 9125
  -> statsd-exporter
  -> Prometheus scrape on port 9102
  -> Grafana dashboards backed by Prometheus
```

StatsD is still part of the supported path. Moving to Kubernetes changes how you expose and secure the services, but it does not remove `statsd-exporter`.

## Recommended Paths

### Local Docker Compose

For local validation, use the Compose stack in [setup/docker-compose.yml](setup/docker-compose.yml).

This tracked example now:

- binds published ports to `127.0.0.1`
- reads the Grafana password from `setup/.env`
- pins image tags for reproducibility

Create the environment file first:

```bash
cd examples/advanced/monitoring/setup
cp .env.example .env
# edit .env and set a non-empty GRAFANA_ADMIN_PASSWORD
docker compose up -d
```

`docker-compose.yml` uses required variable interpolation for the Grafana password; **`docker compose up` fails** if `GRAFANA_ADMIN_PASSWORD` is unset or empty (Docker Compose **v2.24+**). Upgrade Compose if you see a substitution error.

Default local URLs:

- Grafana: `http://127.0.0.1:3000`
- Prometheus: `http://127.0.0.1:9090`
- statsd-exporter metrics: `http://127.0.0.1:9102/metrics`

### Kubernetes Reference Deployment

For a Kubernetes deployment of the monitoring stack, use the manifests and guidance in [k8s/README.md](k8s/README.md).

The Kubernetes reference stack includes:

- `statsd-exporter` as an internal ClusterIP service on `9125` UDP and TCP
- Prometheus scraping `statsd-exporter:9102`
- Grafana with a Secret-backed admin password
- optional Grafana Ingress and example NetworkPolicy

This path is intended for development, validation, and documentation. It is not a full production bundle.

## Setup Types

NVFLARE supports three monitoring topologies.

### 1. Shared Monitoring System for All Sites

All sites send metrics to one shared monitoring stack.

![setup-1](figures/setup-1.png)

Use this when all clients and the server can reach the same `statsd-exporter` endpoint.

### 2. Clients Stream Metrics to the Server Site

Clients stream metrics to the server site, and the server site publishes them to the monitoring stack.

![setup-2](figures/setup-2.png)

This is usually the best fit when clients are remote and you do not want to expose the StatsD port at every site.

It is also the recommended pattern when the NVFLARE server runs outside Kubernetes and one or more clients run inside Kubernetes. See the hybrid deployment notes in [k8s/README.md](k8s/README.md).

### 3. Individual Monitoring System for Each Site

Each site runs its own monitoring stack.

![setup-3](figures/setup-3.png)

This is useful when sites are isolated and cannot share a central monitoring endpoint.

## Monitoring Components

The main components are:

1. `StatsDReporter`: sends metrics to `statsd-exporter`
2. `JobMetricsCollector`: emits job-level metrics on server and client workflows
3. `SysMetricsCollector`: emits parent-process system metrics from local site configuration
4. `RemoteMetricsReceiver`: receives federated metrics streamed to the server
5. `ConvertToFedEvent`: converts local metrics events for transport to the server

## Configuration Overview

### Setup 1: Shared Monitoring System

In setup 1, server and clients all send directly to the same StatsD endpoint.

Job-level components can be generated through the Job API as shown in [jobs/README.md](jobs/README.md), or configured manually. The essential `StatsDReporter` settings are:

```json
{
  "id": "statsd_reporter",
  "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
  "args": {
    "host": "<statsd_exporter_host>",
    "port": 9125
  }
}
```

For system metrics, add `SysMetricsCollector` and `StatsDReporter` to the local site resources file:

```text
<startup>/<site-name>/local/resources.json
```

Example:

```json
{
  "id": "sys_metrics_collector",
  "path": "nvflare.metrics.sys_metrics_collector.SysMetricsCollector",
  "args": {
    "tags": {
      "site": "<site>",
      "env": "dev"
    }
  }
},
{
  "id": "statsd_reporter",
  "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
  "args": {
    "host": "<statsd_exporter_host>",
    "port": 9125
  }
}
```

### Setup 2: Clients Stream Metrics to the Server

In setup 2, clients do not send directly to `statsd-exporter`. Instead, they collect metrics locally and stream them to the server.

Client-side components:

- `JobMetricsCollector`
- `SysMetricsCollector`
- `ConvertToFedEvent`

Server-side components:

- `JobMetricsCollector`
- `SysMetricsCollector`
- `StatsDReporter`
- `RemoteMetricsReceiver`

Client-side example:

```json
{
  "id": "job_metrics_collector",
  "path": "nvflare.metrics.job_metrics_collector.JobMetricsCollector",
  "args": {
    "tags": {
      "site": "site-1",
      "env": "dev"
    },
    "streaming_to_server": true
  }
},
{
  "id": "event_converter",
  "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
  "args": {
    "events_to_convert": ["metrics_event"]
  }
}
```

Server-side example:

```json
{
  "id": "statsd_reporter",
  "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
  "args": {
    "host": "<statsd_exporter_host>",
    "port": 9125
  }
},
{
  "id": "remote_metrics_receiver",
  "path": "nvflare.metrics.remote_metrics_receiver.RemoteMetricsReceiver",
  "args": {
    "events": ["fed.metrics_event"]
  }
}
```

### Setup 3: One Monitoring Stack per Site

Setup 3 uses the same component pattern as setup 1, but each site points `StatsDReporter` at its own local `statsd-exporter`.

## Kubernetes-Specific Notes

When NVFLARE runs in Kubernetes, point `StatsDReporter` at the internal service DNS name for the monitoring namespace. For the reference manifests in [k8s/README.md](k8s/README.md), the service endpoint is:

```text
statsd-exporter.nvflare-monitoring.svc.cluster.local:9125
```

If NVFLARE runs outside the cluster and the monitoring stack runs inside Kubernetes, do not use HTTP Ingress for StatsD traffic. Use one of these patterns instead:

- VPN or private connectivity to the cluster
- an internal LoadBalancer or NodePort restricted to trusted NVFLARE nodes
- setup 2 so only the server site needs to reach StatsD

## Troubleshooting

- **Grafana cannot query Prometheus (`lookup … no such host`, or empty data).** If Grafana runs **inside Docker or Kubernetes**, set the Prometheus datasource URL to **`http://prometheus:9090`** (the Compose/K8s service name), **not** `http://127.0.0.1:9090`. From inside the Grafana container, `127.0.0.1` is the Grafana pod/container itself. Check for typos such as **`127.0.0.01`**. After changing a UI-saved datasource, you may need to remove the override so file-based provisioning applies again, or fix the URL in the UI to match the service name.

- **Prometheus has no NVFLARE metrics.** Metrics only appear after **StatsD** traffic hits **`statsd-exporter` on port 9125** (UDP and/or TCP). Ensure `JobMetricsCollector` / `SysMetricsCollector` and `StatsDReporter` are configured, and that `StatsDReporter` `host`/`port` are reachable from the process or pod that runs them. On the same host as Compose, you can sanity-check with: `echo "demo.test:1|c" | nc -u -w1 127.0.0.1 9125` (then look for series on `http://127.0.0.1:9102/metrics` or in Prometheus after a scrape).

- **Prometheus target `statsd` is DOWN.** Confirm the `statsd-exporter` container or pod is running, the scrape target matches your network (e.g. `statsd-exporter:9102` on the Docker/K8s network), and nothing blocks port **9102** between Prometheus and `statsd-exporter`.

- **NVFLARE on your laptop, monitoring stack on a remote server with ports bound to `127.0.0.1` on the server.** Other machines cannot send StatsD to that bind address. Either run FL on the server, publish **9125** on an address the clients can reach (with firewall rules), use **setup 2** so only the server sends StatsD, or use private networking.

- **SSH local port forward for StatsD from a laptop.** OpenSSH `ssh -L` forwards **TCP** by default. The Datadog-based `StatsDReporter` typically sends **UDP**. A plain TCP tunnel is not enough; use a **UDP→TCP bridge** (e.g. `socat`) on the laptop, expose **9125** properly, or run the submitter/workloads where cluster DNS and UDP/TCP to StatsD work.

- **`ModuleNotFoundError: datadog` or failure when loading `StatsDReporter`.** Install **`datadog`** in the Python environment (see [jobs/requirements.txt](jobs/requirements.txt)). Container images that use `StatsDReporter` must include `datadog` in the image (see [k8s/README.md](k8s/README.md#runtime-requirement)).

- **Grafana dashboards.** The minimal Compose and K8s examples provision a **Prometheus datasource only**; they do not ship curated NVFLARE JSON dashboards. Use **Explore** or build/import your own panels.

## Event Reference

The following event-to-metric mapping is the reference for the current monitoring path:

| Event | Metric Count | Metric Time Taken |
|-------|--------------|-------------------|
| `SYSTEM_START` | `_system_start_count` | |
| `SYSTEM_END` | `_system_end_count` | `_system_time_taken` |
| `ABOUT_TO_START_RUN` | `_about_to_start_run_count` | |
| `START_RUN` | `_start_run_count` | |
| `ABOUT_TO_END_RUN` | `_about_to_end_run_count` | |
| `END_RUN` | `_end_run_count` | `_run_time_taken` |
| `CHECK_END_RUN_READINESS` | `_check_end_run_readiness_count` | |
| `SWAP_IN` | `_swap_in_count` | |
| `SWAP_OUT` | `_swap_out_count` | |
| `START_WORKFLOW` | `_start_workflow_count` | |
| `END_WORKFLOW` | `_end_workflow_count` | `_workflow_time_taken` |
| `ABORT_TASK` | `_abort_task_count` | |
| `FATAL_SYSTEM_ERROR` | `_fatal_system_error_count` | |
| `JOB_DEPLOYED` | `_job_deployed_count` | |
| `JOB_STARTED` | `_job_started_count` | |
| `JOB_COMPLETED` | `_job_completed_count` | `_job_time_taken` |
| `JOB_ABORTED` | `_job_aborted_count` | |
| `JOB_CANCELLED` | `_job_cancelled_count` | |
| `CLIENT_DISCONNECTED` | `_client_disconnected_count` | |
| `CLIENT_RECONNECTED` | `_client_reconnected_count` | |
| `BEFORE_PULL_TASK` | `_before_pull_task_count` | |
| `AFTER_PULL_TASK` | `_after_pull_task_count` | `_pull_task_time_taken` |
| `BEFORE_PROCESS_TASK_REQUEST` | `_before_process_task_request_count` | |
| `AFTER_PROCESS_TASK_REQUEST` | `_after_process_task_request_count` | `_process_task_request_time_taken` |
| `BEFORE_PROCESS_SUBMISSION` | `_before_process_submission_count` | |
| `AFTER_PROCESS_SUBMISSION` | `_after_process_submission_count` | `_process_submission_time_taken` |
| `BEFORE_TASK_DATA_FILTER` | `_before_task_data_filter_count` | |
| `AFTER_TASK_DATA_FILTER` | `_after_task_data_filter_count` | `_data_filter_time_taken` |
| `BEFORE_TASK_RESULT_FILTER` | `_before_task_result_filter_count` | |
| `AFTER_TASK_RESULT_FILTER` | `_after_task_result_filter_count` | `_result_filter_time_taken` |
| `BEFORE_TASK_EXECUTION` | `_before_task_execution_count` | |
| `AFTER_TASK_EXECUTION` | `_after_task_execution_count` | `_task_execution_time_taken` |
| `BEFORE_SEND_TASK_RESULT` | `_before_send_task_result_count` | |
| `AFTER_SEND_TASK_RESULT` | `_after_send_task_result_count` | `_send_task_result_time_taken` |
| `BEFORE_PROCESS_RESULT_OF_UNKNOWN_TASK` | `_before_process_result_of_unknown_task_count` | |
| `AFTER_PROCESS_RESULT_OF_UNKNOWN_TASK` | `_after_process_result_of_unknown_task_count` | `_process_result_of_unknown_task_time_taken` |
| `PRE_RUN_RESULT_AVAILABLE` | `_pre_run_result_available_count` | |
| `BEFORE_CHECK_CLIENT_RESOURCES` | `_before_check_client_resources_count` | |
| `AFTER_CHECK_CLIENT_RESOURCES` | `_after_check_client_resources_count` | `_check_client_resources_time_taken` |
| `SUBMIT_JOB` | `_submit_job_count` | |
| `DEPLOY_JOB_TO_SERVER` | `_deploy_job_to_server_count` | |
| `DEPLOY_JOB_TO_CLIENT` | `_deploy_job_to_client_count` | |
| `BEFORE_CHECK_RESOURCE_MANAGER` | `_before_check_resource_manager_count` | |
| `BEFORE_SEND_ADMIN_COMMAND` | `_before_send_admin_command_count` | |
| `BEFORE_CLIENT_REGISTER` | `_before_client_register_count` | |
| `AFTER_CLIENT_REGISTER` | `_after_client_register_count` | `client_register_time_taken` |
| `CLIENT_REGISTER_RECEIVED` | `_client_register_received_count` | |
| `CLIENT_REGISTER_PROCESSED` | `_client_register_processed_count` | |
| `CLIENT_QUIT` | `_client_quit_count` | |
| `SYSTEM_BOOTSTRAP` | `_system_bootstrap_count` | |
| `BEFORE_AGGREGATION` | `_before_aggregation_count` | |
| `END_AGGREGATION` | `_end_aggregation_count` | `_aggregation_time_taken` |
| `RECEIVE_BEST_MODEL` | `_receive_best_model_count` | |
| `BEFORE_TRAIN` | `_before_train_count` | |
| `AFTER_TRAIN` | `_after_train_count` | `_train_time_taken` |
| `TRAIN_DONE` | `_train_done_count` | |
| `TRAINING_STARTED` | `_training_count` | |
| `TRAINING_FINISHED` | `_training_count` | `_training_time_taken` |
| `ROUND_STARTED` | `_round_started_count` | |
| `ROUND_DONE` | `_round_done_count` | `_round_time_taken` |

These metrics can be separated into Job Metrics and System Metrics. System Metrics are associated with the client and server parent processes, while Job Metrics are associated with each job.

## Job Example

Use [jobs/README.md](jobs/README.md) for the hello-pt walkthrough covering:

- setup 1 with a shared monitoring system
- setup 2 with client metrics streamed to the server
- example dashboard screenshots and job submission flow
