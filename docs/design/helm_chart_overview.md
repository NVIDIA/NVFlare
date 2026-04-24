# NVFlare Helm Chart — Overview

**Version:** 2.7.x / Helm chart 0.1.0
**Date:** 2026-04-10

---

## What It Is

NVFlare's provisioning system generates ready-to-deploy Helm charts for the FL server and all FL clients as part of `nvflare provision`. Operators run a single `helm install` per participant to bring up the federated-learning deployment on Kubernetes.

The target environment is **microk8s running on an AWS EC2 instance**.

---

## What Gets Generated

Running `nvflare provision -p project.yml` produces one Helm chart per participant alongside the startup kit (certs, config files):

```
prod_00/
├── server1/
│   ├── startup/          ← certs, fed_server.json
│   └── helm_chart/       ← server Helm chart
├── site-1/
│   ├── startup/
│   └── helm_chart/       ← client Helm chart
└── site-2/
    └── helm_chart/
```

Each `helm_chart/` directory is a self-contained Helm chart installable with:

```bash
helm install server1  prod_00/server1/helm_chart/
helm install site-1   prod_00/site-1/helm_chart/
helm install site-2   prod_00/site-2/helm_chart/
```

---

## Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    microk8s on AWS EC2                               │
│                                                                      │
│  namespace: ingress                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  nginx ingress controller  (microk8s enable ingress)           │  │
│  │  TCP passthrough: fedLearnPort → nvflare-server:fedLearnPort   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  namespace: default                                                  │
│  ┌─────────────────────────────┐  ┌──────────────────────────────┐  │
│  │  FL Server                  │  │  FL Client (one per site)    │  │
│  │  Deployment: nvflare-server │  │  Pod: site-1                 │  │
│  │  Service:    nvflare-server │  │  Service: site-1             │  │
│  │  hostPort → fedLearnPort    │  │                              │  │
│  └─────────────────────────────┘  └──────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Job Pods  (launched dynamically by K8sJobLauncher)            │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
         ▲
  External FL clients
  <ec2-public-ip>:fedLearnPort
```

---

## Key Design Decisions

**Single builder for server and clients.** `HelmChartBuilder` generates both the server chart and all client charts in one provisioning pass, ensuring consistent PVC names, mount paths, and port values across all participants.

**Template files as package data.** Kubernetes manifest templates live under `nvflare/lighter/templates/helm/{server,client}/` and are copied verbatim at provision time. This keeps Helm `{{ }}` syntax out of Python strings and makes templates independently editable.

**`hostPort` for external access on EC2.** The server Deployment binds `fedLearnPort` (and `adminPort` when distinct) directly to the EC2 host network via `hostPort`. External FL clients connect to `<ec2-public-ip>:fedLearnPort` — no LoadBalancer or NodePort remapping needed.

**Fixed service name `nvflare-server`.** The server Kubernetes Service uses a fixed name so the microk8s TCP passthrough ConfigMap can reference it reliably, and so client `comm_config.json` files can be generated with a known hostname.

**`adminPort` suppressed when equal to `fedLearnPort`.** When the provisioning context does not set a separate admin port, it defaults to the same value as `fedLearnPort`. The builder omits `adminPort` from `values.yaml` in this case, and Helm guards prevent duplicate port entries.

**`uid=` injected by the pod template.** The client site name appears once in `values.yaml` as `name` and is appended as `uid={{ .Values.name }}` by the pod template. This means a single `helm install --set name=<site>` correctly updates the pod name, service selector, and the NVFlare process argument simultaneously.

**`comm_config.json` alignment.** During provisioning, `HelmChartBuilder` writes Kubernetes-appropriate host and port values into each participant's `COMM_CONFIG_ARGS`. `StaticFileBuilder.finalize()` (which runs after all builders) uses these to emit `comm_config.json`, ensuring that job pods launched by `K8sJobLauncher` can reach the parent client process at the correct Kubernetes Service name and port.

---

## Prerequisites

Before running `helm install`:

1. `microk8s enable ingress` on the cluster node.
2. PVC `nvflws` (workspace) created in the target namespace.
3. Startup kit contents copied into the workspace PVC under `<site-name>/startup/`.
4. Container image pushed to a registry accessible from the cluster.

**AWS Security Group inbound rules:**

| Port | Source | Purpose |
|------|--------|---------|
| `fedLearnPort` | FL client IPs | FL client connections |
| `adminPort` | Admin IPs | Admin console (if distinct) |
| 22 | Operator IP | SSH |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `nvflare/lighter/impl/helm_chart.py` | `HelmChartBuilder` — generates charts during `nvflare provision` |
| `nvflare/lighter/templates/helm/server/` | Server Kubernetes manifest templates |
| `nvflare/lighter/templates/helm/client/` | Client Kubernetes manifest templates |
| `nvflare/app_opt/job_launcher/k8s_launcher.py` | `K8sJobLauncher` — launches per-job Pods from within a running client |

For implementation details see [helm_chart_design.md](helm_chart_design.md).
