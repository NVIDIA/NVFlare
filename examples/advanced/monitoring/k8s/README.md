# Minimal Kubernetes manifests for NVFLARE monitoring

This directory contains a reference Kubernetes deployment for the current NVFLARE monitoring stack:

- `statsd-exporter`
- Prometheus
- Grafana

It is intended for development, validation, and documentation. It is not a complete production bundle.

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
