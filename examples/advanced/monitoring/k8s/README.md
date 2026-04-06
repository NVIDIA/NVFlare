# Minimal Kubernetes manifests for NVFLARE monitoring

This directory contains a reference Kubernetes deployment for the current NVFLARE monitoring stack:

- `statsd-exporter`
- Prometheus
- Grafana

It is intended for development, validation, and documentation. It is not a complete production bundle.

## Runtime Requirement

If you enable `StatsDReporter` inside NVFLARE server or client containers, the runtime image must include the Python `datadog` package in addition to `nvflare`.

For example:

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir nvflare datadog

ENV PYTHONUNBUFFERED=1
```

Without `datadog`, NVFLARE fails during config loading when it tries to instantiate `nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter`.

## What It Deploys

- namespace `nvflare-monitoring`
- `statsd-exporter` as a ClusterIP service on `9125` UDP and TCP plus `9102` for Prometheus scraping
- Prometheus scraping `statsd-exporter:9102`
- Grafana with a Secret-backed admin password and a provisioned Prometheus datasource
- optional Grafana Ingress and example NetworkPolicy

## Prerequisites

- Kubernetes 1.24+
- a cluster with working DNS
- NetworkPolicy-capable CNI only if you apply `50-networkpolicy-statsd-example.yaml`

## 1. Create the Grafana admin secret

Create the namespace and the Grafana password Secret:

```bash
kubectl create namespace nvflare-monitoring --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic grafana-admin \
  -n nvflare-monitoring \
  --from-literal=password='REPLACE_WITH_STRONG_PASSWORD'
```

## 2. Apply the manifests

From this directory:

```bash
kubectl apply -f 00-namespace.yaml
kubectl apply -f 10-statsd-exporter-deployment.yaml
kubectl apply -f 11-statsd-exporter-service.yaml
kubectl apply -f 20-prometheus-configmap.yaml
kubectl apply -f 21-prometheus-deployment.yaml
kubectl apply -f 22-prometheus-service.yaml
kubectl apply -f 30-grafana-configmap.yaml
kubectl apply -f 31-grafana-deployment.yaml
kubectl apply -f 32-grafana-service.yaml
```

Optional:

```bash
# edit the hostname and TLS secret first
kubectl apply -f 40-ingress-grafana.yaml

# apply only after adjusting labels to match your NVFLARE namespaces
# kubectl apply -f 50-networkpolicy-statsd-example.yaml
```

## 3. Verify

```bash
kubectl get pods,svc -n nvflare-monitoring
```

Port-forward Grafana:

```bash
kubectl port-forward -n nvflare-monitoring svc/grafana 3000:3000
```

Then open `http://127.0.0.1:3000`.

Port-forward Prometheus:

```bash
kubectl port-forward -n nvflare-monitoring svc/prometheus 9090:9090
```

Then open `http://127.0.0.1:9090/targets` and confirm the `statsd` target is `UP`.

## 4. Point NVFLARE at the in-cluster StatsD endpoint

If NVFLARE runs in the cluster, configure `StatsDReporter` to send to:

- host: `statsd-exporter.nvflare-monitoring.svc.cluster.local`
- port: `9125`

Pods in the same namespace can also use the short service name `statsd-exporter`.

## 4.1 Validated End-to-End K8s Flow

The following path was validated against a MicroK8s cluster:

1. Deploy the monitoring stack in namespace `nvflare-monitoring`.
2. Run the NVFLARE server in Kubernetes using an image that includes both `nvflare` and `datadog`.
3. Add `SysMetricsCollector` and `StatsDReporter` to the server `local/resources.json` and point `StatsDReporter` to `statsd-exporter.nvflare-monitoring.svc.cluster.local:9125`.
4. Add `SysMetricsCollector` and `StatsDReporter` to the client `local/resources.json` and point it at the same in-cluster StatsD service.
5. Keep the signed startup kit files unchanged when possible. In particular, do not edit `fed_client.json` directly in a provisioned kit unless you also regenerate the kit signature.
6. If the provisioned client kit still expects the original external hostname, preserve the signed `fed_client.json` and instead route that hostname in the client pod manifest. In the validated flow, the client pod used `hostAliases` so `flaredevserv` resolved to the current `server` service ClusterIP.
7. Start the server and client pods, then verify metrics in `statsd-exporter` and Prometheus.

The validated metrics included:

- `_system_start_count` for both server and client
- `_before_client_register_count` and `_after_client_register_count` for the client
- `_client_register_received_count` and `_client_register_processed_count` on the server
- `_before_client_heartbeat_count` and `_after_client_heartbeat_count` for the client
- `_client_heartbeat_received_count` and `_client_heartbeat_processed_count` on the server

## 4.2 Validated Job-Level K8s Flow

Job-level monitoring was also validated against the same MicroK8s deployment using the minimal example in [../jobs/k8s_hello_numpy/README.md](../jobs/k8s_hello_numpy/README.md).

The validated production-style submission path was:

1. Keep the signed admin startup kit unchanged.
2. Submit the job from [../jobs/k8s_hello_numpy/job.py](../jobs/k8s_hello_numpy/job.py).
3. If the admin hostname is not resolvable from the submitter, add hostname mapping outside the startup kit instead of editing `fed_admin.json`.
4. Verify the emitted metrics in `statsd-exporter` or Prometheus by filtering on `env="k8s"` and the job ID.

The validated job-level metrics included:

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

Example client pod pattern:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nvflare-client-site2
  namespace: default
spec:
  restartPolicy: Never
  hostAliases:
    - ip: "<server-service-cluster-ip>"
      hostnames:
        - "flaredevserv"
  containers:
    - name: client
      image: localhost:32000/nvfl-min:0.0.2
      workingDir: /workspace
      command: ["/bin/bash", "-lc"]
      args:
        - |
          export PYTHONUNBUFFERED=1
          exec /workspace/startup/sub_start.sh
      volumeMounts:
        - name: kit
          mountPath: /workspace
  volumes:
    - name: kit
      hostPath:
        path: /home/flaredevserv/prod_00/site-2
        type: Directory
```

This approach lets you validate the K8s monitoring path without modifying signed startup-kit files inside the mounted client kit.

## 5. Exposing StatsD for external NVFLARE

Standard HTTP Ingress does not carry StatsD traffic. If NVFLARE runs outside the cluster, use one of these patterns:

- VPN or private routing into the cluster
- internal LoadBalancer or NodePort restricted to trusted NVFLARE hosts
- setup 2 from [../README.md](../README.md), so only the server site needs to reach StatsD

Do not expose Prometheus `:9090` or raw `statsd-exporter` metrics `:9102` publicly without authentication and network controls.

## 6. Production Notes

These manifests are intentionally minimal.

- Prometheus uses `emptyDir`, so historical metrics are lost if the pod is recreated.
- Image tags are pinned, but you should still review upgrades deliberately.
- Grafana Ingress is optional and should be protected with TLS and your normal access controls.
- If you use `StatsDReporter` in containerized NVFLARE workloads, make sure the image includes `datadog`.
- Avoid editing signed startup-kit files such as `fed_client.json` in place. Prefer changing pod or service routing around the signed kit.

## Files

| File | Purpose |
|------|---------|
| `00-namespace.yaml` | namespace |
| `10-statsd-exporter-deployment.yaml` | `statsd-exporter` deployment |
| `11-statsd-exporter-service.yaml` | `statsd-exporter` service |
| `20-prometheus-configmap.yaml` | Prometheus scrape configuration |
| `21-prometheus-deployment.yaml` | Prometheus deployment |
| `22-prometheus-service.yaml` | Prometheus service |
| `30-grafana-configmap.yaml` | Grafana datasource config |
| `31-grafana-deployment.yaml` | Grafana deployment |
| `32-grafana-service.yaml` | Grafana service |
| `40-ingress-grafana.yaml` | optional Grafana Ingress |
| `50-networkpolicy-statsd-example.yaml` | optional example NetworkPolicy |
