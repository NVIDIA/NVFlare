# NVFlare Helm Chart — Technical Design

**Version:** Helm chart 0.1.0
**Date:** 2026-04-10

For a high-level overview see [helm_chart_overview.md](helm_chart_overview.md).

---

## Table of Contents

1. [HelmChartBuilder](#1-helmchartbuilder)
2. [Server Chart](#2-server-chart)
3. [Client Chart](#3-client-chart)
4. [Provisioning Pipeline](#4-provisioning-pipeline)
5. [Runtime Flow](#5-runtime-flow)
6. [Security](#6-security)
7. [Open Gaps](#7-open-gaps)

---

## 1. HelmChartBuilder

`HelmChartBuilder` (`nvflare/lighter/impl/helm_chart.py`) is a single `Builder` subclass that generates the server chart and all client charts in one `build()` call.

### Constructor Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `docker_image` | required | `repo:tag` for all participants; tag becomes `appVersion` |
| `parent_port` | `8102` | Port job pods use to talk back to the client or server process |
| `workspace_pvc` | `nvflws` | PVC claim name for the runtime workspace volume |
| `workspace_mount_path` | `/var/tmp/nvflare/workspace` | In-container mount path for workspace PVC |

### Template File Resolution

Helm template files are stored as package data and resolved at runtime:

```python
_HELM_TEMPLATES_DIR = os.path.join(os.path.dirname(prov.__file__), "templates", "helm")

def _helm_src(role: str, filename: str) -> str:
    return os.path.join(_HELM_TEMPLATES_DIR, role, filename)
```

`Chart.yaml` and `values.yaml` are written via `yaml.dump`. All Kubernetes manifest templates are copied verbatim via `shutil.copy` to avoid YAML-parser interference with Helm `{{ }}` syntax.

---

## 2. Server Chart

**Output directory:** `<wip>/<server-name>/helm_chart/`

### Chart Structure

```
<server-name>/helm_chart/
├── Chart.yaml
├── values.yaml
└── templates/
    ├── _helpers.tpl
    ├── server-deployment.yaml
    ├── server-service.yaml
    └── server-tcp-services.yaml
```

### values.yaml Reference

| Key | Description |
|-----|-------------|
| `name` | Server participant name |
| `image.repository` | Container image repository |
| `image.tag` | Container image tag |
| `image.pullPolicy` | `IfNotPresent` |
| `persistence.workspace.claimName` | PVC name for workspace |
| `persistence.workspace.friendlyName` | Display name for the workspace PVC (same as `claimName`) |
| `persistence.workspace.mountPath` | Mount path for workspace PVC |
| `fedLearnPort` | FL client connection port (from `CtxKey.FED_LEARN_PORT`) |
| `adminPort` | Admin console port; `null` when equal to `fedLearnPort` |
| `parentPort` | Port job pods use to talk back to the server process |
| `command` | `[/usr/local/bin/python3]` |
| `args` | Arguments to `nvflare.private.fed.app.server.server_train` |

### Port Handling

`fedLearnPort` and `adminPort` are read from `CtxKey.FED_LEARN_PORT` and `CtxKey.ADMIN_PORT` in the provision context. `adminPort` is set to `None` when it equals `fedLearnPort`; Helm `{{- if .Values.adminPort }}` guards suppress the duplicate entry from the Deployment and Service.

### Template Details

**`server/service.yaml`** — ClusterIP Service with the fixed name `nvflare-server`:

```yaml
metadata:
  name: nvflare-server       # fixed, not .Values.name
spec:
  ports:
    - name: fl-port
      port: {{ .Values.fedLearnPort }}
    {{- if .Values.adminPort }}
    - name: admin-port
      port: {{ .Values.adminPort }}
    {{- end }}
    - name: nvflare-server
      port: {{ .Values.parentPort }}
```

The fixed name allows the `tcp-services` ConfigMap and client `comm_config.json` to reference the server by a stable, known DNS name.

**`server/deployment.yaml`** — `hostPort` binds `fedLearnPort` and `adminPort` directly to the EC2 host network so external FL clients can connect to `<ec2-public-ip>:fedLearnPort` without a LoadBalancer:

```yaml
ports:
  - containerPort: {{ .Values.fedLearnPort }}
    hostPort: {{ .Values.fedLearnPort }}
    protocol: TCP
  {{- if .Values.adminPort }}
  - containerPort: {{ .Values.adminPort }}
    hostPort: {{ .Values.adminPort }}
    protocol: TCP
  {{- end }}
  - containerPort: {{ .Values.parentPort }}
    protocol: TCP
```

**`server/tcp-services.yaml`** — ConfigMap watched by the microk8s nginx ingress controller for raw TCP passthrough on `fedLearnPort`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-ingress-microk8s-conf
  namespace: ingress
data:
  {{ .Values.fedLearnPort | quote }}: {{ printf "%s/nvflare-server:%v" .Release.Namespace .Values.fedLearnPort | quote }}
```

Requires `microk8s enable ingress`. The ingress controller DaemonSet must also be patched to expose `fedLearnPort` and reference this ConfigMap via `--tcp-services-configmap`.

### comm_config.json Alignment

`_build_server_chart()` updates the server's `COMM_CONFIG_ARGS` (pre-seeded to `{}` by `StaticFileBuilder.initialize()`):

```python
{
    CommConfigArg.HOST: "nvflare-server",   # fixed Kubernetes Service name
    CommConfigArg.PORT: self.parent_port,
    CommConfigArg.SCHEME: "tcp",
    CommConfigArg.CONN_SEC: ConnSecurity.CLEAR,
}
```

`StaticFileBuilder.finalize()` reads these values to emit `comm_config.json`.

---

## 3. Client Chart

**Output directory:** `<wip>/<client-name>/helm_chart/`

### Chart Structure

```
<client-name>/helm_chart/
├── Chart.yaml
├── values.yaml
└── templates/
    ├── _helpers.tpl
    ├── client-deployment.yaml
    └── service.yaml
```

### values.yaml Reference

| Key | Description |
|-----|-------------|
| `name` | **Single source of truth**: pod name, service selector label, and `uid=` process arg |
| `image.repository` | Container image repository |
| `image.tag` | Container image tag |
| `image.pullPolicy` | `Always` |
| `persistence.workspace.claimName` | PVC name for workspace |
| `persistence.workspace.friendlyName` | Display name for the workspace PVC (same as `claimName`) |
| `persistence.workspace.mountPath` | Mount path for workspace PVC |
| `port` | Port job pods use to communicate back to this client process |
| `command` | `[/usr/local/bin/python3]` |
| `args` | Arguments to `nvflare.private.fed.app.client.client_train` (no `uid=`) |
| `restartPolicy` | `Never` |

### Template Details

**`client/deployment.yaml`** — `uid=` is appended by the template, not stored in `values.yaml`:

```yaml
args:
  {{- toYaml .Values.args | nindent 8 }}
  - uid={{ .Values.name }}
```

This ensures `helm install --set name=<site>` updates the pod name, service selector label, and `uid=` process argument in a single override.

**`client/service.yaml`** — ClusterIP Service named after the client with no `-svc` suffix, so its Kubernetes DNS name matches `internal.resources.host` in `comm_config.json`:

```yaml
metadata:
  name: {{ include "nvflare-client.name" . }}   # resolves to client.name
```

### comm_config.json Alignment

`_build_one_client_chart()` updates the client's `COMM_CONFIG_ARGS`:

```python
{
    CommConfigArg.HOST: client.name,     # matches Kubernetes Service name
    CommConfigArg.PORT: self.parent_port,
    CommConfigArg.SCHEME: "tcp",
    CommConfigArg.CONN_SEC: ConnSecurity.CLEAR,
}
```

Consistency requirements across chart and `comm_config.json`:

| Source | Field | Value |
|--------|-------|-------|
| `values.yaml` / `client-deployment.yaml` | `containerPort` | `parent_port` |
| `service.yaml` | `port` / `targetPort` | `parent_port` |
| `service.yaml` | `metadata.name` | `<client.name>` |
| `comm_config.json` | `internal.resources.port` | `parent_port` |
| `comm_config.json` | `internal.resources.host` | `<client.name>` |

### K8sJobLauncher

`ClientK8sJobLauncher.launch_job()` runs each FL job's executor in a dynamically-created Pod. The job image is read from `launcher_spec` in `meta.json`:

```json
{
  "launcher_spec": {
    "site-1": {
      "k8s": { "image": "myregistry/nvflare-client:2.7.0" }
    },
    "default": {
      "k8s": { "image": "myregistry/nvflare-job:2.7.0" }
    }
  }
}
```

If no site-specific `launcher_spec[site]["k8s"]` entry exists, the launcher can
use `launcher_spec["default"]["k8s"]` when present.

**`K8sJobLauncher`** responsibilities:
- Loads `kubeconfig` from a file path provided at construction time.
- Builds a Pod manifest with an ephemeral workspace volume, a startup-kit Secret mount, and an optional `nvfldata` PVC.
- Attaches `nvidia.com/gpu` resource limits when specified in job metadata.
- Converts the UUID job ID to an RFC 1123-compliant pod name via `uuid4_to_rfc1123()`.
- Registers itself as launcher on `BEFORE_JOB_LAUNCH` unconditionally (launcher type is site policy, not job config); `launch_job` raises `RuntimeError` if no image is found for this site.

**`K8sJobHandle`** responsibilities:
- Submits the pod via `core_v1_api.create_namespaced_pod()`.
- Polls pod phase every 1 second; terminates stuck-in-pending pods after `pending_timeout`.
- Maps Kubernetes phases to `JobReturnCode`:

| Kubernetes Phase | `JobState` | `JobReturnCode` |
|-----------------|-----------|----------------|
| Pending | STARTING | UNKNOWN |
| Running | RUNNING | UNKNOWN |
| Succeeded | SUCCEEDED | SUCCESS (0) |
| Failed | TERMINATED | ABORTED (9) |
| Unknown | UNKNOWN | UNKNOWN (127) |

---

## 4. Provisioning Pipeline

### Builder Sequence

```
project.yml → Provisioner.provision()
    │
    ├─► initialize()  WorkspaceBuilder, CertBuilder, StaticFileBuilder, HelmChartBuilder, ...
    ├─► build()       WorkspaceBuilder, CertBuilder, StaticFileBuilder, HelmChartBuilder, ...
    └─► finalize()    ... HelmChartBuilder, StaticFileBuilder, CertBuilder, WorkspaceBuilder
```

**Ordering requirement:** `HelmChartBuilder` must appear **after** `StaticFileBuilder` in `project.yml`. `StaticFileBuilder.initialize()` pre-seeds `COMM_CONFIG_ARGS = {}` on each participant; `HelmChartBuilder.build()` overwrites it with Kubernetes values; `StaticFileBuilder.finalize()` reads those values to emit `comm_config.json`.

### Full Provisioning Output

```
prod_NN/
├── <server-name>/
│   ├── startup/                       ← certs, fed_server.json
│   └── helm_chart/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── _helpers.tpl
│           ├── server-deployment.yaml
│           ├── server-service.yaml
│           └── server-tcp-services.yaml
├── site-1/
│   ├── startup/
│   └── helm_chart/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── _helpers.tpl
│           ├── client-deployment.yaml
│           └── service.yaml
└── site-2/
    └── helm_chart/  ...
```

---

## 5. Runtime Flow

### Server Startup

```
helm install nvflare-server <server-chart-dir>
    │
    ▼
Deployment scheduled → image pulled → PVCs mounted
    │
    ▼
python3 -u -m nvflare.private.fed.app.server.server_train
    ├─ reads fed_server.json from workspace PVC
    ├─ binds fedLearnPort (also on EC2 host via hostPort)
    └─ waits for FL client connections
```

### Client Startup

```
helm install site-1 <client-chart-dir>
    │
    ▼
Pod scheduled → image pulled → PVCs mounted
    │
    ▼
python3 -u -m nvflare.private.fed.app.client.client_train
    ├─ uid=site-1  (injected by pod template from .Values.name)
    ├─ reads startup kit from workspace/startup
    ├─ connects to nvflare-server:fedLearnPort (ClusterIP within cluster)
    └─ enters event loop, waiting for tasks
```

### Job Execution

```
Server submits job → ClientEngine.fire_event(BEFORE_JOB_LAUNCH)
    │
    ▼
ClientK8sJobLauncher.handle_event()  — registers unconditionally (site policy)
    │
    ▼
ClientK8sJobLauncher.launch_job(job_meta, fl_ctx)
    ├─ Pod manifest: name=rfc1123(job_id), image from launcher_spec, volumes: emptyDir workspace + startup Secret + optional nvfldata
    ├─ core_v1_api.create_namespaced_pod()
    ├─ K8sJobHandle.enter_states([RUNNING])  — polls every 1s, terminates if stuck in Pending
    └─ returns K8sJobHandle  →  poll() / wait() / terminate()
```

---

## 6. Security

### mTLS

All server↔client communication uses mutual TLS. Certificates are generated by `nvflare provision` (RSA, project root CA) and stored in each site's workspace `startup/` directory. `secure_train=true` in the pod args activates TLS on both sides.

### Networking

`hostPort` on the server Deployment binds `fedLearnPort` to the EC2 host interface. The `nvflare-server` Service is `ClusterIP`; all external access goes via `hostPort`.

The microk8s nginx ingress controller additionally accepts `fedLearnPort` for TCP passthrough via the `nginx-ingress-microk8s-conf` ConfigMap.

**AWS Security Group inbound rules:**

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| `fedLearnPort` | TCP | FL client IPs | FL client connections |
| `adminPort` | TCP | Admin IPs | Admin console (if distinct) |
| 22 | TCP | Operator IP | SSH |

### RBAC for K8sJobLauncher

```yaml
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["create", "delete", "get", "watch"]
```

The kubeconfig for this service account must be placed in the client startup kit so it lands in the site's workspace `startup/` directory during deployment.

---

## 7. Open Gaps

| # | Gap | Proposed Fix |
|---|-----|-------------|
| 1 | `imagePullSecrets` not in server Deployment or client Pod templates | Add optional `imagePullSecrets` to `values.yaml`; accept `image_pull_secrets` list in `HelmChartBuilder` constructor |
| 2 | No CPU/memory resource limits on server or client pods | Add `resources:` section to `values.yaml` (default empty); accept `resources` constructor parameter |
| 3 | No liveness/readiness probes on either pod | Add optional exec or HTTP probe against the admin API port |
| 4 | microk8s ingress DaemonSet requires manual patch to expose `fedLearnPort` | Add a Helm hook Job that patches the DaemonSet at install time |
| 5 | No Helm chart tests in either chart | Add `templates/tests/` pods that verify service reachability |
