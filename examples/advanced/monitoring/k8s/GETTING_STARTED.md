# NVFLARE Monitoring Deployment Paths

This guide collects the Kubernetes and mixed-environment guidance that should not live in the legacy top-level monitoring README.

## Choose a Guide

Use the guide that matches your environment:

| If you want to... | Start here | Notes |
|-------------------|------------|-------|
| learn the original local or POC monitoring model for setup 1 and setup 2 | [../README.md](../README.md) and [../jobs/README.md](../jobs/README.md) | Best when the server and clients run in the classic local or POC layout. |
| deploy the monitoring stack in Kubernetes | [README.md](README.md) | Covers the manifests, in-cluster validation, hybrid guidance, and example NetworkPolicy. |
| submit a minimal monitored job to a production-style in-cluster deployment | [../jobs/k8s_hello_numpy/README.md](../jobs/k8s_hello_numpy/README.md) | Validated against MicroK8s with job-level metrics visible in `statsd-exporter`. |
| run a hardened local monitoring stack without changing the legacy Compose example | [../setup/SECURE_LOCAL.md](../setup/SECURE_LOCAL.md) | Uses a separate secure-local Compose file and an env example for Grafana credentials. |

## Typical Paths

If you want the shortest path from zero to working monitoring, follow one of these:

1. Local or POC learning path:
   Start the original local monitoring stack or the separate secure-local variant, follow [../jobs/README.md](../jobs/README.md) setup 1 or setup 2, submit the example job, then inspect metrics in `statsd-exporter`, Prometheus, or Grafana.

2. In-cluster Kubernetes path:
   Deploy the monitoring stack from [README.md](README.md), start the NVFLARE server and client pods in the cluster, then use [../jobs/k8s_hello_numpy/README.md](../jobs/k8s_hello_numpy/README.md) to submit a minimal monitored job and verify job-level metrics.

3. Hybrid path:
   Keep the NVFLARE server outside Kubernetes, use setup 2 so Kubernetes clients stream metrics to the server, make sure the server can reach the monitoring stack, then follow the hybrid guidance in [README.md](README.md).

## Architecture Reminder

The current metrics pipeline is:

```text
NVFLARE (StatsDReporter)
  -> StatsD or DogStatsD on port 9125
  -> statsd-exporter
  -> Prometheus scrape on port 9102
  -> Grafana dashboards backed by Prometheus
```

StatsD remains part of the supported path. Kubernetes changes how you expose and secure the services, but it does not remove `statsd-exporter`.

## Troubleshooting

- **Grafana cannot query Prometheus (`lookup ... no such host`, or empty data).**
  If Grafana runs inside Docker or Kubernetes, set the Prometheus datasource URL to the service name such as `http://prometheus:9090`, not `http://127.0.0.1:9090`. From inside the Grafana container or pod, `127.0.0.1` is Grafana itself.

- **Prometheus has no NVFLARE metrics.**
  Metrics only appear after StatsD traffic reaches `statsd-exporter` on port `9125`. Confirm that `JobMetricsCollector` or `SysMetricsCollector` and `StatsDReporter` are configured, and that the `StatsDReporter` host and port are reachable from the process or pod that sends metrics.

- **Prometheus target `statsd` is DOWN.**
  Confirm the `statsd-exporter` container or pod is running, the scrape target is correct for the current network, and nothing blocks port `9102` between Prometheus and `statsd-exporter`.

- **NVFLARE is outside the cluster, but the monitoring stack is inside Kubernetes.**
  Standard HTTP Ingress does not carry StatsD traffic. Use private routing, a restricted internal LoadBalancer or NodePort, or setup 2 so only the server site needs to reach the StatsD endpoint.

- **SSH local forwarding for StatsD does not work.**
  OpenSSH `ssh -L` forwards TCP by default. The Datadog-based `StatsDReporter` typically sends UDP. A plain TCP tunnel is not enough; use a UDP-aware bridge or run the workloads where cluster DNS and StatsD connectivity work natively.

- **`ModuleNotFoundError: datadog` when `StatsDReporter` processes or emits metrics.**
  Because `StatsDReporter` uses a lazy import, NVFLARE can start successfully and only fail later when metrics are sent. Install `datadog` in the Python environment, or bake it into the container image. See [README.md](README.md) for the container runtime requirement.

- **You need a safer local Compose example without changing the legacy one.**
  Use [../setup/SECURE_LOCAL.md](../setup/SECURE_LOCAL.md), which keeps the old Compose example untouched and provides a separate secure-local stack.

- **Grafana dashboards are empty even though data exists.**
  The example stacks provision a Prometheus datasource but do not ship curated NVFLARE dashboards. Use Grafana Explore first, then build or import dashboards once you confirm the metrics exist.
