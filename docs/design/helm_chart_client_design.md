# NVFlare Client Helm Chart — Design Document

**Version:** 2.7.x / Helm chart 0.1.0
**Date:** 2026-03-20
**Scope:** End-to-end design for generating, packaging, and deploying the NVFlare federated-learning client on Kubernetes via Helm.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Requirements](#2-requirements)
3. [Architecture](#3-architecture)
4. [Component Details](#4-component-details)
5. [Implementation](#5-implementation)
6. [Helm Chart Structure](#6-helm-chart-structure)
7. [Provisioning Pipeline](#7-provisioning-pipeline)
8. [Runtime Flow on Kubernetes](#8-runtime-flow-on-kubernetes)
9. [Security Model](#9-security-model)
10. [Gaps and Future Work](#10-gaps-and-future-work)

---

## 1. Overview

NVIDIA FLARE (NVFlare) is a federated-learning platform in which a central **FL Server** coordinates training rounds and one or more **FL Clients** (sites) train on local data. In a production Kubernetes deployment each client runs as a Pod. The `helm/nvflare-client` Helm chart (located at `helm/nvflare-client/`) is the packaging artifact that operators use to deploy a single NVFlare client site to a Kubernetes cluster.

This document describes:

- what the chart contains and why each field exists,
- how NVFlare's provisioning subsystem (`nvflare/lighter/`) generates per-client Helm artifacts via `ClientHelmChartBuilder`,
- how the `K8sJobLauncher` (`nvflare/app_opt/job_launcher/k8s_launcher.py`) dynamically launches per-job Pods from inside a running client, and
- the remaining open gaps.

---

## 2. Requirements

### 2.1 Functional Requirements

| # | Requirement |
|---|-------------|
| FR-1 | Deploy exactly one NVFlare client process per Helm release. |
| FR-2 | The provisioned client folder (certs, `fed_client.json`, config) is copied into the `nvflws` PVC before the chart is installed. |
| FR-3 | Mount a shared workspace PVC (`nvflws`) used by job processes. |
| FR-4 | Allow optional data PVC mounting for job execution (`nvfldata`). |
| FR-5 | Configure the client's site name (`uid`) via a single `values.yaml` field (`name`); no duplication with the args list. |
| FR-6 | Server hostname resolution is the cluster's responsibility (DNS). `hostAliases` is not supported; the client connects outbound to the server. |
| FR-7 | Expose `containerPort` / Service port so job pods launched by `K8sJobLauncher` can communicate back to the parent client process. |
| FR-8 | Restart policy `Never` — the pod exits when the FL session ends; restarts are operator-controlled. |
| FR-9 | Support GPU resource requests for job Pods spawned by `K8sJobLauncher`. |
| FR-10 | The chart must be renderable with a single `helm install` after PVCs and image exist. |
| FR-11 | `nvflare provision` must generate a ready-to-use client Helm chart for every client in the project. |
| FR-12 | `comm_config.json`'s `internal.resources.host` and `internal.resources.port` must equal the Kubernetes Service name and `containerPort` so that job pods launched by `K8sJobLauncher` can reach the parent client process. |

### 2.2 Non-Functional Requirements

| # | Requirement |
|---|-------------|
| NFR-1 | Helm chart `appVersion` is derived from the container image tag, not hard-coded separately. |
| NFR-2 | All sensitive material (certs, keys) is supplied via PVC, never baked into the image. |
| NFR-3 | Pod name and Service selector are derived from the same `name` value to avoid mis-routing. |
| NFR-4 | The chart is namespace-aware — all resources live in the target namespace. |
| NFR-5 | Resource limits for the main client container are not hard-coded; operators add them in `values.yaml`. |

### 2.3 Out of Scope (current version)

- Server-side Helm chart.
- Multi-client chart (one release = one site).
- Operator CRD or StatefulSet (a plain Pod is used for simplicity).
- Automatic certificate generation inside the chart.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                           │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  Helm Release: nvflare-client        │                            │
│  │                                     │                            │
│  │  ┌──────────────┐  ┌─────────────┐  │                            │
│  │  │  Pod: site-1  │  │  Svc:       │  │                            │
│  │  │               │  │  site-1     │  │                            │
│  │  │  nvflare      │  │  port 18002 │  │                            │
│  │  │  client_train │  └─────────────┘  │                            │
│  │  │               │                  │                            │
│  │  │  mounts:      │                  │                            │
│  │  │  nvfletc (RO) │── certs/config   │                            │
│  │  │  nvflws  (RW) │── workspace      │                            │
│  │  └──────────────┘                  │                            │
│  └─────────────────────────────────────┘                            │
│                                                                     │
│  ┌─────────────────────────────────────┐                            │
│  │  Job Pods (created by K8sJobLauncher)│                            │
│  │  Pod: <job-id>                      │                            │
│  │  mounts: nvfletc, nvflws, nvfldata  │                            │
│  │  GPU resource limit (optional)      │                            │
│  └─────────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
         │                              │
   gRPC/TLS 18002                  Admin API
         │                              │
┌────────┴──────────────────────────────┴──────┐
│              FL Server (external)            │
│  nvflare.private.fed.app.server.server_train │
└──────────────────────────────────────────────┘
```

### 3.1 Key Code Modules

| Module | Path | Role |
|--------|------|------|
| Hand-authored Helm chart | `helm/nvflare-client/` | Reusable chart operators can customise |
| `ClientHelmChartBuilder` | `nvflare/lighter/impl/client_helm_chart.py` | Generates one client chart per site during `nvflare provision` |
| `K8sJobLauncher` / `ClientK8sJobLauncher` | `nvflare/app_opt/job_launcher/k8s_launcher.py` | Launches per-job Pods from within the running client |
| `K8sJobHandle` | same file | Tracks pod lifecycle (poll/wait/terminate) |
| `JobLauncherSpec` / `JobHandleSpec` | `nvflare/apis/job_launcher_spec.py` | Abstract interfaces |
| Provisioner | `nvflare/lighter/provisioner.py` | Orchestrates builders that emit startup kits |
| Master template | `nvflare/lighter/templates/master_template.yml` | YAML templates for all generated artifacts |
| `client_train` entrypoint | `nvflare/private/fed/app/client/` | The Python module run inside the Pod |

---

## 4. Component Details

### 4.1 The `helm/nvflare-client` Chart

The chart has three templates and one values file:

```
helm/nvflare-client/
├── Chart.yaml              # chart metadata (name, version, appVersion)
├── values.yaml             # operator-facing configuration
└── templates/
    ├── _helpers.tpl        # name, labels, image helpers
    ├── client-pod.yaml     # Pod definition
    └── service.yaml        # ClusterIP Service
```

**`Chart.yaml`** declares `appVersion: "3.5.10"` which must be kept in sync with the container image tag used in `values.yaml`.

**`values.yaml`** exposes every operator-facing knob. `uid=` is **not** in the args list — it is appended by the pod template from `.Values.name`. The client connects outbound to the server so no `hostAliases` field is needed or supported:

```yaml
name: site-1                          # single source of truth: pod name, label, and uid= arg
image:
  repository: localhost:32000/nvflare
  tag: "3.5.10"
  pullPolicy: Always
persistence:
  etc:   { claimName: nvfletc, mountPath: /var/tmp/nvflare/etc }
  workspace: { claimName: nvflws,  mountPath: /var/tmp/nvflare/workspace }
port: 18002   # port job pods use to talk back to this client process
command: [/usr/local/bin/python3]
args:
  - -u
  - -m
  - nvflare.private.fed.app.client.client_train
  - -m
  - /var/tmp/nvflare/workspace
  - -s
  - fed_client.json
  - --set
  - secure_train=true
  - config_folder=config
  - org=nvidia
  # uid= is NOT here — injected as uid={{ .Values.name }} by client-pod.yaml
restartPolicy: Never
```

**`_helpers.tpl`** defines four named templates:

- `nvflare-client.name` — pod/container name from `.Values.name`, max 63 chars
- `nvflare-client.labels` — standard `app.kubernetes.io/*` labels
- `nvflare-client.selectorLabels` — selector used by the Service
- `nvflare-client.image` — `repository:tag` concatenation

**`client-pod.yaml`** binds two PVCs as volumes, renders the `args` list from `values.yaml`, and appends `uid={{ .Values.name }}` as the final `--set` argument:

```yaml
      args:
        {{- toYaml .Values.args | nindent 8 }}
        - uid={{ .Values.name }}
```

This ensures that `--set name=site-2` (or any `values.yaml` override) automatically updates all three places where the site name appears: the pod name, the service selector label, and the `uid=` process argument.

**`service.yaml`** creates a ClusterIP Service whose name is exactly `{{ include "nvflare-client.name" . }}` (i.e. `<client-name>`, no `-svc` suffix). This ensures the Kubernetes DNS name matches `internal.resources.host` in `comm_config.json` (see §7.4).

### 4.2 `ClientHelmChartBuilder` (client chart generation during provisioning)

`ClientHelmChartBuilder` in `nvflare/lighter/impl/client_helm_chart.py` plugs into the `nvflare provision` pipeline and generates a fully-populated Helm chart for every client participant in the project.

**Output directory:** `<wip>/<client-name>/nvflare_hc_clients/` — inside each client's own wip participant directory. After `WorkspaceBuilder.finalize()` moves `wip/` → `prod_NN/`, the final path is:

```
prod_NN/
├── site-1/
│   ├── startup/            # certs, fed_client.json (from other builders)
│   └── nvflare_hc_clients/
│       ├── Chart.yaml
│       ├── values.yaml     # pre-filled: name, image, org, PVCs
│       └── templates/
│           ├── _helpers.tpl
│           ├── client-pod.yaml  # uid={{ .Values.name }} — no duplication
│           └── service.yaml
└── site-2/
    └── nvflare_hc_clients/
        └── ...
```

**Constructor parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `docker_image` | required | `repo:tag` for the client container; tag is used as `appVersion` |
| `client_port` | `18002` | Port job pods use to communicate back to this client process |
| `workspace_pvc` | `nvflws` | PVC name for the runtime workspace |
| `etc_pvc` | `nvfletc` | PVC name for startup-kit (certs + config) |
| `workspace_mount_path` | `/var/tmp/nvflare/workspace` | Mount path inside the container |
| `etc_mount_path` | `/var/tmp/nvflare/etc` | Mount path inside the container |

**Builder lifecycle:**

- `build()` — calls `_build_client_chart()` for each client returned by `project.get_clients()`, writing output to `ctx.get_ws_dir(client)/nvflare_hc_clients/`.

**Key design decisions:**

1. `uid=` is absent from the generated `values.yaml` args list. It is appended by `client-pod.yaml` via `- uid={{ .Values.name }}`, satisfying FR-5 and fixing the duplication gap.
2. Template files (`_helpers.tpl`, `client-pod.yaml`, `service.yaml`) are written as raw strings rather than via `yaml.dump` because they contain Helm template syntax (`{{ }}`) that YAML parsers would reject.
3. `Chart.yaml` derives `appVersion` from the image tag portion of `docker_image`; if no tag is present it defaults to `"latest"`.
4. `hostAliases` is not supported. The NVFlare client connects outbound to the server; the cluster's DNS resolves the server hostname.
5. `containerPort` / `port` is the port job pods use to communicate back to the parent client process after being launched by `K8sJobLauncher`. It is unrelated to the outbound client-to-server connection.
6. **`comm_config.json` alignment**: during `build()`, `ClientHelmChartBuilder` updates `COMM_CONFIG_ARGS` on the client participant with `host=client.name` and `port=client_port`. `StaticFileBuilder.finalize()` (which runs after all `build()` calls) then emits `comm_config.json` with `internal.resources.host = <client.name>` and `internal.resources.port = <client_port>` — identical to the Kubernetes Service name and `containerPort`/`targetPort` in the Helm chart. `ClientHelmChartBuilder` must be listed after `StaticFileBuilder` in `project.yml` so that any `listening_host` values set by `StaticFileBuilder` are overridden correctly.

### 4.3 `K8sJobLauncher` and `K8sJobHandle`

Once the main client Pod is running and an FL job is submitted, the client engine calls `ClientK8sJobLauncher.launch_job()` to run the job's executor in its own Pod. This is distinct from the Helm chart which deploys the persistent client process.

**`K8sJobLauncher`** responsibilities:

- Loads `kubeconfig` from a file path provided at construction time.
- Builds a `Pod` manifest programmatically with three PVCs: `nvflws`, `nvfldata`, `nvfletc`.
- Attaches `nvidia.com/gpu` resource limits when the job metadata specifies GPU count.
- Converts the UUID job ID to an RFC 1123-compliant pod name via `uuid4_to_rfc1123()`.
- Listens to `BEFORE_JOB_LAUNCH` events; registers itself as launcher only when the job metadata contains an image for this site.

**`K8sJobHandle`** responsibilities:

- Submits the pod via `core_v1_api.create_namespaced_pod()`.
- Polls pod phase (`Pending` → `Running` → `Succeeded`/`Failed`).
- Detects stuck-in-pending pods (configurable `pending_timeout`) and terminates them.
- Maps Kubernetes phases to `JobReturnCode` (`SUCCESS`, `ABORTED`, `UNKNOWN`).
- `terminate()` calls `delete_namespaced_pod()` with `grace_period_seconds=0`.

**`ClientK8sJobLauncher`** and **`ServerK8sJobLauncher`** are concrete subclasses that supply the correct `module_args` for client vs. server job processes.

---

## 5. Implementation

### 5.1 Pre-requisites

Before deploying the Helm chart the operator must:

1. **Run `nvflare provision`** to generate startup kits and per-client Helm charts.
2. **Create the `nvflws` PVC** in the target namespace — the script copies the provisioned client folder into it.
3. **Push the NVFlare Docker image** to a registry accessible from the cluster.

```bash
# Example PVC manifest (adjust storageClassName to match your cluster)
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nvflws
  namespace: default
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 10Gi
EOF
```

### 5.2 Startup Kit Layout inside the `nvflws` PVC

The NVFlare client process runs with `-m /var/tmp/nvflare/workspace` as its workspace root. It expects the client's startup kit under `<workspace>/<client-name>/`:

```
/var/tmp/nvflare/workspace/          ← nvflws PVC mount
└── site-1/                          ← copied from prod_NN/site-1/
    ├── startup/
    │   ├── fed_client.json
    │   ├── rootCA.pem
    │   ├── client.crt
    │   └── client.key
    ├── transfer/
    └── local/
```

### 5.3 Deploy Script (`helm/deploy_client.sh`)

`helm/deploy_client.sh` automates the copy-then-deploy workflow in five steps:

1. Validates that the `nvflws` PVC exists in the target namespace.
2. Launches a temporary `busybox` helper pod that mounts the `nvflws` PVC at `/ws/`.
3. Copies `prod_NN/<client-name>/` into `/ws/<client-name>/` via `kubectl cp`.
4. Deletes the helper pod.
5. Runs `helm install` (or `helm upgrade` if the release already exists) from the chart at `prod_NN/<client-name>/nvflare_hc_clients/`.

**Usage:**

```bash
./helm/deploy_client.sh <client-name> <prod-dir> [namespace] [release-name]
```

| Argument | Description |
|----------|-------------|
| `client-name` | Site name matching the provisioned folder, e.g. `site-1` |
| `prod-dir` | Path to the `prod_NN` directory, e.g. `workspace/example_project/prod_00` |
| `namespace` | Kubernetes namespace (default: `default`) |
| `release-name` | Helm release name (default: `<client-name>`) |

**Example — deploy two sites:**

```bash
# Provision
nvflare provision -p project.yml

# Deploy each site (each call runs the 5-step copy + install)
./helm/deploy_client.sh site-1 workspace/example_project/prod_00
./helm/deploy_client.sh site-2 workspace/example_project/prod_00

# Verify
kubectl get pods -l app.kubernetes.io/managed-by=Helm
kubectl logs site-1
```

**What happens inside the cluster during the copy step:**

```
kubectl run nvflare-init-site-1 \        # helper pod mounts nvflws at /ws/
    --overrides='{ "volumes": [{"pvc": "nvflws"}], ... }'
kubectl cp prod_00/site-1/  nvflare-init-site-1:/ws/site-1
kubectl delete pod nvflare-init-site-1
```

After the copy the PVC contains `/ws/site-1/startup/fed_client.json` etc., which maps to `/var/tmp/nvflare/workspace/site-1/startup/` inside the client container.

### 5.4 Multi-site Deployment

Each site requires its own `nvflws` PVC (the PVC is per-site since it holds that site's workspace and startup kit). Use a distinct PVC name per site and pass it through the chart's `persistence.workspace.claimName` value:

```bash
# Create per-site PVCs
for SITE in site-1 site-2; do
  kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nvflws-${SITE}
  namespace: default
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 10Gi
EOF
done

# Deploy each site, overriding the PVC name
PROD=workspace/example_project/prod_00
./helm/deploy_client.sh site-1 "${PROD}" default site-1
./helm/deploy_client.sh site-2 "${PROD}" default site-2
# Note: deploy_client.sh uses PVC name 'nvflws' by default.
# For per-site PVCs, pass --set persistence.workspace.claimName=nvflws-<site>
# to the helm install/upgrade call, or edit the chart's values.yaml before running.
```

### 5.5 Configuring `K8sJobLauncher` in `fed_client.json`

To use dynamic per-job pod launching instead of subprocess launching, configure the client with:

```json
{
  "components": [
    {
      "id": "job_launcher",
      "path": "nvflare.app_opt.job_launcher.k8s_launcher.ClientK8sJobLauncher",
      "args": {
        "config_file_path": "/var/tmp/nvflare/workspace/site-1/startup/kubeconfig",
        "workspace_pvc": "nvflws",
        "etc_pvc": "nvflws",
        "data_pvc_file_path": "/var/tmp/nvflare/workspace/site-1/startup/data_pvc.yaml",
        "namespace": "nvflare",
        "pending_timeout": 30
      }
    }
  ]
}
```

The `data_pvc_file_path` YAML file maps a single PVC name to a mount path:

```yaml
my-data-pvc: /var/tmp/nvflare/data
```

---

## 6. Helm Chart Structure

### 6.1 Full Rendered Output (default values)

**Pod:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: site-1
  labels:
    app.kubernetes.io/name: site-1
    app.kubernetes.io/instance: <release>
    app.kubernetes.io/managed-by: Helm
spec:
  restartPolicy: Never
  volumes:
    - name: nvfletc
      persistentVolumeClaim: { claimName: nvfletc }
    - name: nvflws
      persistentVolumeClaim: { claimName: nvflws }
  containers:
    - name: site-1
      image: localhost:32000/nvflare:3.5.10
      imagePullPolicy: Always
      ports:
        - containerPort: 18002
      command: ["/usr/local/bin/python3"]
      args:
        - -u
        - -m
        - nvflare.private.fed.app.client.client_train
        - -m
        - /var/tmp/nvflare/workspace
        - -s
        - fed_client.json
        - --set
        - secure_train=true
        - config_folder=config
        - org=nvidia
        - uid=site-1          # appended by template from .Values.name; not in values.yaml args
      volumeMounts:
        - name: nvfletc  mountPath: /var/tmp/nvflare/etc
        - name: nvflws   mountPath: /var/tmp/nvflare/workspace
```

**Service:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: site-1          # equals client.name — matches comm_config.json internal.resources.host
spec:
  selector:
    app.kubernetes.io/name: site-1
  ports:
    - protocol: TCP
      port: 18002        # equals client_port — matches comm_config.json internal.resources.port
      targetPort: 18002
```

### 6.2 Values Reference

| Key | Default | Description |
|-----|---------|-------------|
| `name` | `site-1` | **Single source of truth**: pod name, container name, service selector label, and `uid=` process arg |
| `image.repository` | `localhost:32000/nvflare` | Container image repository |
| `image.tag` | `"3.5.10"` | Image tag |
| `image.pullPolicy` | `Always` | Kubernetes image pull policy |
| `persistence.etc.claimName` | `nvfletc` | PVC for startup kit (certs + config) |
| `persistence.etc.mountPath` | `/var/tmp/nvflare/etc` | Mount path inside container |
| `persistence.workspace.claimName` | `nvflws` | PVC for runtime workspace |
| `persistence.workspace.mountPath` | `/var/tmp/nvflare/workspace` | Mount path inside container |
| `port` | `18002` | Port job pods use to communicate back to this client process (not used for client-to-server traffic) |
| `command` | `[/usr/local/bin/python3]` | Container entrypoint |
| `args` | see §6.1 (without `uid=`) | Arguments to `client_train`; `uid=` is appended by the template |
| `restartPolicy` | `Never` | Kubernetes pod restart policy |

---

## 7. Provisioning Pipeline

### 7.1 How `nvflare provision` Works

```
project.yml
    │
    ▼
nvflare/lighter/provision.py  ─── parse project YAML
    │
    ▼
Provisioner.provision()
    │
    ├─► Builder.initialize()  (all builders, in order)
    ├─► Builder.build()       (all builders, in order)
    └─► Builder.finalize()    (all builders, reversed)
             │
             ├─ WorkspaceBuilder        → directory structure
             ├─ CertBuilder             → PKI (rootCA, server, client certs)
             ├─ StaticFileBuilder       → fed_server.json, fed_client.json
             ├─ DockerBuilder           → start.sh with docker run
             ├─ ClientHelmChartBuilder  → <name>/nvflare_hc_clients/ (per client)
             └─ ...
```

### 7.2 Helm Artifact Inventory

| Artifact | Builder | Target role |
|----------|---------|-------------|
| `prod_NN/<name>/nvflare_hc_clients/` | `ClientHelmChartBuilder` | One chart per client |
| `helm/nvflare-client/` | Hand-authored (source-controlled) | Reusable chart template |

### 7.3 `comm_config.json` Alignment with the Helm Chart

When a job is submitted, `K8sJobLauncher` creates a new Pod. That job pod must connect back to the parent NVFlare client process. NVFlare's internal CellNet transport reads `local/comm_config.json` to discover the host and port of the internal listener:

```json
{
  "allow_adhoc_conns": false,
  "backbone_conn_gen": 2,
  "internal": {
    "scheme": "tcp",
    "resources": {
      "host": "site-1",
      "port": 18002,
      "connection_security": "clear"
    }
  }
}
```

The three values that must be consistent across the Helm chart and `comm_config.json` are:

| Source | Field | Value |
|--------|-------|-------|
| `values.yaml` / `client-pod.yaml` | `containerPort` | `18002` |
| `service.yaml` | Service `port` / `targetPort` | `18002` |
| `service.yaml` | Service `metadata.name` | `site-1` |
| `comm_config.json` | `internal.resources.port` | `18002` |
| `comm_config.json` | `internal.resources.host` | `site-1` |

**How provisioning enforces this:** `ClientHelmChartBuilder.build()` updates the participant's `COMM_CONFIG_ARGS` dict (initialized to `{}` by `StaticFileBuilder.initialize()`) with:

```python
{
    CommConfigArg.HOST: client.name,   # → "site-1"
    CommConfigArg.PORT: self.client_port,  # → 18002
    CommConfigArg.SCHEME: "tcp",
    CommConfigArg.CONN_SEC: ConnSecurity.CLEAR,
}
```

`StaticFileBuilder.finalize()` then renders `comm_config.json` from these args. Because the Kubernetes Service is named `<client.name>` (no `-svc` suffix), within the same namespace a job pod can reach the client at `site-1:18002` — exactly what `comm_config.json` specifies.

**Builder ordering requirement:** `ClientHelmChartBuilder` must appear **after** `StaticFileBuilder` in the `builders:` list in `project.yml`. This guarantees that if `StaticFileBuilder.build()` sets a `listening_host`-derived host/port, `ClientHelmChartBuilder.build()` overrides it with the correct Kubernetes values.

### 7.4 Why Template Files Are Written as Raw Strings

`HelmChartBuilder` stores its templates in `master_template.yml` and emits plain Kubernetes YAML (no Helm template syntax needed because all values are baked in by the Python builder). `ClientHelmChartBuilder` takes a different approach: the pod template must contain `{{ .Values.name }}` so that `helm install --set name=<site>` works without rebuilding the chart. YAML parsers reject bare `{{ }}` blocks, so the three template files are stored as Python string literals in `client_helm_chart.py` and written verbatim to disk. `Chart.yaml` and `values.yaml` contain only plain data and continue to use `yaml.dump`.

---

## 8. Runtime Flow on Kubernetes

### 8.1 Client Pod Startup

```
helm install (or kubectl apply)
    │
    ▼
Pod scheduled → image pulled → PVCs mounted
    │
    ▼
python3 -u -m nvflare.private.fed.app.client.client_train
    │  args include uid=<.Values.name>
    ├─ reads /var/tmp/nvflare/workspace/fed_client.json
    ├─ reads /var/tmp/nvflare/workspace/<site-name>/startup/ (certs, fed_client.json)
    ├─ connects to FL Server over gRPC/TLS
    └─ enters event loop, waiting for tasks
```

### 8.2 Job Execution with `K8sJobLauncher`

```
Server submits job to Client
    │
    ▼
ClientEngine.fire_event(BEFORE_JOB_LAUNCH)
    │
    ▼
ClientK8sJobLauncher.handle_event()
    │  checks job_meta for image → registers self as launcher
    ▼
ClientK8sJobLauncher.launch_job(job_meta, fl_ctx)
    │
    ├─ build Pod manifest:
    │    name: <rfc1123(job_id)>
    │    image: <from job_meta>
    │    volumes: nvflws, nvfldata, nvfletc
    │    resources: nvidia.com/gpu: N  (if specified)
    │    args: nvflare.private.fed.app.client.client_train + job args
    │
    ├─ core_v1_api.create_namespaced_pod()
    │
    ├─ K8sJobHandle.enter_states([RUNNING])
    │    polls pod phase every 1s
    │    terminates if stuck in Pending > pending_timeout
    │
    └─ returns K8sJobHandle to ClientEngine
         │
         ├─ poll() → JobReturnCode
         ├─ wait() → blocks until SUCCEEDED/TERMINATED
         └─ terminate() → delete_namespaced_pod(grace=0)
```

### 8.3 Pod Phase → Job State Mapping

| Kubernetes Phase | `JobState` | `JobReturnCode` |
|-----------------|-----------|----------------|
| Pending | STARTING | UNKNOWN |
| Running | RUNNING | UNKNOWN |
| Succeeded | SUCCEEDED | SUCCESS (0) |
| Failed | TERMINATED | ABORTED (9) |
| Unknown | UNKNOWN | UNKNOWN (127) |

---

## 9. Security Model

### 9.1 TLS / mTLS

- All server↔client communication uses mutual TLS.
- Certificates are generated by `nvflare provision` (RSA, signed by project root CA).
- Certs are stored in the `nvflws` PVC under `<site-name>/startup/`: `rootCA.pem`, `client.crt`, `client.key`. They are placed there by `deploy_client.sh` before the chart is installed.
- The `secure_train=true` flag in the pod args activates TLS on both sides.

### 9.2 Kubernetes RBAC

`K8sJobLauncher` needs permissions to create and delete Pods in its namespace. The operator must create a `ServiceAccount` + `Role` + `RoleBinding`:

```yaml
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["create", "delete", "get", "watch"]
```

The kubeconfig file used by `K8sJobLauncher` must be bound to this service account. It is placed in the client's startup kit directory (e.g. `prod_NN/site-1/startup/kubeconfig`) so that `deploy_client.sh` copies it into the `nvflws` PVC alongside the other startup files.

### 9.3 Image Pull Secrets

For private registries, add to `values.yaml`:

```yaml
imagePullSecrets:
  - name: my-registry-secret
```

(The current chart templates do not include `imagePullSecrets` — this is a known gap, see §10.1.)

---

## 10. Open Gaps and Future Work

### 10.1 `imagePullSecrets` Not Supported

**Gap:** The `client-pod.yaml` template has no `imagePullSecrets` block.

**Proposed fix:** Add optional `imagePullSecrets` to `values.yaml` (default `[]`) and render them in the pod spec with a `{{- with .Values.imagePullSecrets }}` guard. `ClientHelmChartBuilder` should accept an `image_pull_secrets` list parameter and emit it into the generated `values.yaml`.

### 10.2 No Resource Limits on Client Pod

**Gap:** The main client pod has no CPU/memory limits. Only job pods (via `K8sJobLauncher`) can request GPU resources.

**Proposed fix:** Add a `resources:` section to `values.yaml` (default empty) and render it in `client-pod.yaml`. `ClientHelmChartBuilder` should accept `resources` as a constructor parameter.

### 10.3 No Liveness / Readiness Probes

**Gap:** The pod has no health checks. Kubernetes cannot distinguish a deadlocked client from a healthy one.

**Proposed fix:** Add an optional HTTP or exec probe against the NVFlare admin API port.

### 10.4 `K8sJobLauncher` Data PVC is Single-Entry

**Gap:** `data_pvc_file_path` only supports one PVC key; the rest of the dict is ignored.

**Proposed fix:** Support a list of `{pvc, mountPath}` entries.

### 10.5 No Helm Chart Tests

**Gap:** `helm/nvflare-client/` has no `templates/tests/` directory.

**Proposed fix:** Add a `test-connection` pod that verifies the client service is reachable.
