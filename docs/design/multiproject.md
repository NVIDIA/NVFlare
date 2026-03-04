# Multi-Project Support in Flare

## Introduction

Flare currently operates as a single-tenant system. All server and client processes run under the same Linux user, all jobs share a flat store (`jobs/<uuid>/`), and every authorized admin can see and act on every job. There is no data segregation between different collaborations running on the same infrastructure.

To achieve genuine multi-tenancy, we introduce a **project** concept as the primary tenant boundary. A project encapsulates a private dataset, a set of participants (users and sites), an authorization policy, and runtime isolation. This document specifies the required changes across the full Flare stack.

### Design Principles

1. **Least privilege by default** — users see nothing outside their project(s)
2. **Defense in depth** — logical access control (authz) + physical isolation (containers/PVs)
3. **Backward compatible** — a `default` project preserves current single-tenant behavior
4. **`scope` deprecated** — the existing `scope` data-governance concept is superseded by `project`; `scope` will be removed in a future release
5. **Feature-gated** — all multitenancy behavior gated on `api_version: 4` in `project.yml`; single-tenant deployments see zero behavior change


---

## Project Model

A project is a named, immutable tenant boundary with these properties:

| Property | Description |
|----------|-------------|
| `name` | Unique identifier (e.g., `cancer-research`) |
| `clients` | Set of FL client sites enrolled in this project |
| `users` | Set of admin users with per-project roles |
| `authorization` | Per-project authorization policy |

- Users are associated with one or more projects, each with an independent role.
- **Clients participate in all projects they are enrolled in simultaneously.** Data isolation on shared clients is achieved through the runtime environment: K8s jobs mount project-specific PVs, Docker jobs mount project-specific host directories. The Flare parent process on the client does not access project data directly.
- Jobs belong to exactly one project (immutable after submission).
- A `default` project exists for backward compatibility.

---

## User Experience

### Data Scientist (Recipe API)

The recipe is unchanged. The project is specified via `ProdEnv` or `PocEnv`:

```python
recipe = FedAvgRecipe(
    name="hello-pt",
    min_clients=n_clients,
    num_rounds=num_rounds,
    initial_model=SimpleNetwork(),
    train_script=args.train_script,
)

env = ProdEnv(
    startup_kit_location=args.startup_kit_location,
    project="cancer-research",
)
run = recipe.execute(env)
```

`PocEnv` supports the same parameter:

```python
env = PocEnv(
    poc_workspace=args.poc_workspace,
    project="cancer-research",
)
run = recipe.execute(env)
```

If `project` is omitted in either env, the `default` project is used.

### Admin (FLARE API / Admin Console)

The `Session` gains a project context:

```python
sess = new_secure_session(
    username="admin@org_a.com",
    startup_kit_location="./startup",
    project="cancer-research",      # new
)
# All subsequent operations scoped to this project
jobs = sess.list_jobs()             # only cancer-research jobs
sess.submit_job("./my_job")        # tagged to cancer-research
```

Admin console equivalent:

```
> set_project cancer-research
Project set to: cancer-research

> list_jobs
... only shows cancer-research jobs ...
```

A user with roles in multiple projects can switch context:

```
> set_project multiple-sclerosis
Project set to: multiple-sclerosis
```

### Platform Administrator

A new **platform admin** role (distinct from per-project `project_admin`) manages cross-project concerns:

- Create/archive projects
- Assign clients to projects
- Assign project admins
- View system-wide health (without seeing job data)

---

## Data Model Changes

### Job Metadata

`project` becomes a first-class, immutable field on every job. Set at submission time from the user's active project context. Cannot be changed after creation.

### Job Store Partitioning

New multitenant jobs are stored at `jobs/<project>/<uuid>/` (vs. current `jobs/<uuid>/`). No migration of existing jobs — they remain at `jobs/<uuid>/` and implicitly belong to the `default` project.

Physical partitioning enables:
- Filesystem-level isolation (different mount points per project in K8s)
- Simpler backup/restore per project
- Prevents cross-project data access via path traversal

### Project Registry

The server loads `project.yml` directly at startup for project/role lookup. No separate registry format or database needed.

---

## Access Control Changes

### Role Model

Roles are **per-project**, not global. A user can be `lead` in one project and `member` in another.

Today, the role is baked into the X.509 certificate (`UNSTRUCTURED_NAME` field). A single cert cannot encode multiple per-project roles.

**Layered resolution (no breaking change):**
1. If `ProjectRegistry` exists AND user has a mapping for the active project → use registry role
2. Otherwise → fall back to cert-embedded role (existing behavior)

The cert format is unchanged. Existing deployments with `api_version: 3` certs keep working. The cert role field is not removed or made vestigial in this version — it remains the primary source for single-tenant deployments.

### Admin Role Hierarchy

| Role | Scope | Capabilities |
|------|-------|-------------|
| `platform_admin` | Global | Create/delete projects, assign clients, system shutdown, view all sessions |
| `project_admin` | Per-project | All job ops within project, view project's clients (no client lifecycle control) |
| `org_admin` | Per-project | Manage own-org jobs, view own-org clients within project |
| `lead` | Per-project | Submit/manage own jobs, view own-org clients within project |
| `member` | Per-project | View-only within project |

### Command Authorization Matrix

Every command is scoped to the user's active project. Operations on resources outside the active project are denied.

#### Job Operations

| Command | project_admin | org_admin | lead | member |
|---------|:---:|:---:|:---:|:---:|
| `submit_job` | yes | no | yes | no |
| `list_jobs` | all in project | all in project | all in project | all in project |
| `get_job_meta` | all in project | own-org jobs | own jobs | all in project |
| `download_job` | all in project | own-org jobs | own jobs | no |
| `download_job_components` | all in project | own-org jobs | own jobs | no |
| `clone_job` | all in project | no | own jobs | no |
| `abort_job` | all in project | own-org jobs | own jobs | no |
| `delete_job` | all in project | own-org jobs | own jobs | no |
| `show_stats` | all in project | all in project | all in project | all in project |
| `show_errors` | all in project | all in project | all in project | all in project |
| `app_command` | all in project | own-org jobs | own jobs | no |
| `configure_job_log` | all in project | own-org jobs | own jobs | no |

**"all in project"** = any job within the active project.
**"own-org jobs"** = jobs submitted by a user in the same org, within the active project.
**"own jobs"** = jobs submitted by this user, within the active project.

#### Infrastructure Operations

Since clients are shared across projects, **only `platform_admin` can perform client lifecycle operations** (restart, shutdown, remove). Disrupting a client affects all projects running on it.

| Command | platform_admin | project_admin | org_admin | lead | member |
|---------|:---:|:---:|:---:|:---:|:---:|
| `check_status` | all clients | project's clients (view) | own-org + project (view) | own-org + project (view) | project's clients (view) |
| `restart` | all | no | no | no | no |
| `shutdown` | all | no | no | no | no |
| `shutdown_system` | yes | no | no | no | no |
| `remove_client` | all | no | no | no | no |
| `sys_info` | all | project's clients | own-org + project | own-org + project | no |
| `report_resources` | all | project's clients | own-org + project | own-org + project | no |
| `report_env` | all | project's clients | own-org + project | own-org + project | no |

#### Shell Commands

| Command | platform_admin | project_admin | org_admin | lead | member |
|---------|:---:|:---:|:---:|:---:|:---:|
| `pwd`, `ls`, `cat`, `head`, `tail`, `grep` | all | project's clients | own-org + project | own-org + project | no |

Shell commands must be **restricted to the project's workspace path** on the target site. See Unresolved Questions.

#### Session / Platform Commands

| Command | platform_admin | project_admin | org_admin | lead | member |
|---------|:---:|:---:|:---:|:---:|:---:|
| `list_sessions` | all | project's sessions | no | no | no |
| `set_project` | any project | assigned projects | assigned projects | assigned projects | assigned projects |
| `list_projects` | all | assigned only | assigned only | assigned only | assigned only |
| `dead` | yes | no | no | no | no |

---

## Authorization Enforcement

Two layers, evaluated in order:

1. **Project filter** (new): Is this resource in the user's active project? If no, invisible.
2. **RBAC policy** (existing): Does the user's project-role permit this operation on this resource?

The existing `authorization.json` policy format is largely unchanged — project scoping happens above it.

---

## Provisioning Changes

### project.yml

The v4 schema uses three top-level sections with a deliberate separation of concerns:

- **`sites`** — infrastructure participants (server, clients). Always present. Identity and trust are cert-based; these entries never go away.
- **`admins`** — human participants with per-platform and per-project roles. **Optional.** Omit entirely when using SSO (see [Future: SSO](#future-sso-for-human-users)); roles are then provided by IdP claims.
- **`projects`** — tenant definitions: which clients are enrolled, and (optionally) which admins have which roles. The `admins:` block inside each project is also omitted under SSO.

This separation is intentional: `sites` and `projects.clients` form the **permanent skeleton** of the file. The `admins` sections are an **optional overlay** that exists today but disappears when SSO is introduced — with no restructuring of the rest of the file.

```yaml
api_version: 4

# Infrastructure — always present, cert-based mTLS
sites:
  server1.example.com: { type: server, org: nvidia }
  hospital-a:          { type: client, org: org_a }
  hospital-b:          { type: client, org: org_a }
  hospital-c:          { type: client, org: org_b }

# Human admins — omit entirely when using SSO
admins:
  platform-admin@nvidia.com: { org: nvidia, role: platform_admin }
  trainer@org_a.com:         { org: org_a }
  viewer@org_b.com:          { org: org_b }

projects:
  cancer-research:
    clients: [hospital-a, hospital-b]
    # Omit when using SSO (roles come from IdP claims)
    admins:
      trainer@org_a.com: lead

  multiple-sclerosis:
    clients: [hospital-a, hospital-c]
    admins:
      trainer@org_a.com: member
      viewer@org_b.com:  lead
```

**SSO migration**: drop the top-level `admins:` block and the `admins:` entries inside each project. The rest of the file is unchanged:

```yaml
api_version: 4

sites:
  server1.example.com: { type: server, org: nvidia }
  hospital-a:          { type: client, org: org_a }
  hospital-b:          { type: client, org: org_a }
  hospital-c:          { type: client, org: org_b }

projects:
  cancer-research:
    clients: [hospital-a, hospital-b]
  multiple-sclerosis:
    clients: [hospital-a, hospital-c]
```

### Certificate Changes

Certs continue to encode identity (name, org) and role. **No change to cert format.** The `UNSTRUCTURED_NAME` role field remains populated and serves as the fallback for single-tenant mode.

In multitenant mode (`api_version: 4`), per-project roles are resolved from the `ProjectRegistry` loaded from `project.yml` at server startup. The cert role is only used when no registry mapping exists (backward compat).

### Startup Kit Changes

- **Server startup kit** includes `project.yml` — the authoritative source for project definitions, client enrollment, and user roles
- **Admin startup kits** are unchanged (cert for identity; project membership is server-side knowledge)

---

## Job Scheduler Changes

The scheduler becomes project-aware:

1. **Candidate filtering**: Only schedule jobs to clients enrolled in the job's project
2. **Validation**: `deploy_map` sites must be a subset of the project's enrolled clients
3. **Quota/priority**: Deferred. K8s-level resource quotas per namespace may suffice initially. Future option: route different projects to different K8s scheduling queues via pod labels/nodeSelectors.

---

## Runtime Isolation (ProdEnv)

The project becomes a property of the job, and ProdEnv prepares the corresponding isolated environment.

### Subprocess (Default — Single-Tenant Only)

- Job workspace isolated to `<workspace>/<project>/<job_id>/` (logical separation only)
- **No physical isolation**: same Linux user, shared `/tmp`, shared filesystem, shared GPU memory
- **Not suitable for multi-tenant deployments** — use K8s, Docker, or Slurm for cross-project isolation
- Retained for single-tenant and trusted environments (e.g., single org, development, POC)

### Docker

- Per-project volume mounts: each project's jobs mount a **different host directory** (e.g., `/data/<project>/`) as the workspace
- Per-container `/tmp`: each container gets its own tmpfs or bind mount — no shared host `/tmp`
- Per-project Docker network (no cross-project container communication)
- Container name includes project: `<project>-<client>-<job_id>`

### Kubernetes (Primary Target)

Clients participate in all their enrolled projects. **Data isolation is achieved by mounting different PersistentVolumes per project in each job pod.** The Flare client parent process runs in its own pod (or on the node) and does not mount project data PVs — it only orchestrates job pod creation.

| Concern | Mechanism |
|---------|-----------|
| Namespace isolation | One K8s namespace per project |
| Storage isolation | PersistentVolumeClaim per project per client (not hostPath) |
| Temp directory isolation | Each pod gets its own `/tmp` via `emptyDir` — no shared host `/tmp` |
| Network isolation | NetworkPolicy per namespace |
| Resource limits | ResourceQuota per namespace (deferred, see Scheduler) |
| Pod security | PodSecurityPolicy/Standards per namespace |

Job pods are created in the project's K8s namespace, mounting a pre-provisioned PVC (`<project>-workspace`) per project per client. Each pod also gets its own `/tmp` via `emptyDir` to prevent cross-project leakage via temporary files. This applies to both server and client job pods.

### Slurm

- Per-project Slurm accounts/partitions
- Per-project storage paths
- Job submission includes `--account=<project>`

---

## FLARE API Changes

- `Session` gains a `project` parameter (defaults to `"default"`) and `set_project()`/`list_projects()` methods
- `list_jobs` is automatically filtered to the active project (replaces the `-u` user-only filter)
- `get_system_info` returns only clients enrolled in the active project
- All job operations validate that the target job belongs to the active project

---

## Audit Trail

Every audit log entry gains a `project` field:

```
[2026-02-18 10:30:00] user=trainer@org_a.com project=cancer-research action=submit_job job_id=abc123
[2026-02-18 10:31:00] user=trainer@org_a.com project=cancer-research action=list_jobs
```

Audit logs should be queryable per project for compliance.

---

## Migration / Backward Compatibility

1. **Feature gate**: all multitenancy behavior gated on `project.yml` having `api_version: 4` with a `projects:` section. Without it, the system behaves identically to today.
2. **Default project**: all existing jobs, clients, and users are in the `default` project
3. **Cert role fallback**: if no project registry exists (or user has no registry mapping), fall back to cert-embedded role
4. **API compatibility**: `project` parameter defaults to `"default"` everywhere
5. **Config version**: `api_version: 4` in `project.yml` signals multi-project support; version 3 continues to work as single-tenant

---

## Design Decisions

| # | Question | Decision |
|---|----------|----------|
| D1 | Can clients participate in multiple projects? | **Yes.** Clients participate in all enrolled projects simultaneously. Data isolation is physical: K8s mounts different PVs per project; Docker mounts different host directories. The Flare parent process does not access project data. |
| D2 | Project lifecycle management? | **Deferred.** Projects are defined at provisioning time in `project.yml`. Runtime project CRUD is not in scope for v1. |
| D3 | Per-project quota management? | **Deferred.** Rely on K8s ResourceQuota per namespace for now. Future: route projects to different K8s scheduling queues via pod labels. |
| D4 | `check_status` information leakage? | **Server has global knowledge, filtering the response is sufficient.** The server parent process knows about all clients and jobs; it filters responses to only include resources in the user's active project. No architectural change needed. |
| D5 | Server-side job store isolation? | **Server job pods must only access their project's data.** The server job process (running in K8s/Docker) must not mount the entire job store — only the project-partitioned slice. Current `FilesystemStorage` will be replaced by a database or object store in the future, which will enforce project-scoped access natively. For v1 with filesystem: mount only `jobs/<project>/` into the server job pod. No migration of existing jobs — they stay at `jobs/<uuid>/` and belong to `default`. |
| D6 | Role storage: certs vs. server-side registry? | **Layered: registry overrides cert.** `project.yml` defines per-project roles; the server loads it at startup via `ProjectRegistry`. Certs continue to authenticate identity (name, org) and carry a role as fallback. No cert format change required. |
| D7 | How do shared clients know which project PV to mount? | **The launcher passes the project name to the client.** Job metadata carries the project; the server includes it when dispatching to clients. The client-side `K8sJobLauncher`/`DockerJobLauncher` uses the project name to select the correct PV/volume mount. |
| D8 | Cross-project isolation in subprocess mode? | **Subprocess mode is single-tenant/trusted only.** Only K8s, Docker, and Slurm launchers provide secure multi-tenant isolation (separate namespaces, volumes, `/tmp`). The default subprocess launcher offers no physical isolation and is only suitable for single-tenant or trusted environments. |

---

## Unresolved Questions

1. **Cross-project visibility**: Can a platform admin see job metadata across all projects (for debugging)? Should `list_jobs` have a `--all-projects` flag for platform admins?

2. **Existing `scope` concept**: The `scope` concept will be deprecated in favor of `project`. The `project` boundary subsumes data-governance scoping; existing `scope` usage will be migrated to `project`.

3. **External IdP integration**: SSO is a follow-on (see Future: SSO section), but should the `ProjectRegistry` interface be designed now to accommodate an IdP backend later? What claims/attributes should the IdP provide (project membership, role, org)?

4. **Shell commands (pwd, ls, cat, head, tail, grep)**: These allow direct filesystem access on server/client sites. In a multi-tenant environment:
   - How do we restrict file access to the active project's workspace? Current implementation does basic path validation (no `..`, no absolute paths) but has no project awareness.
   - In K8s, project data lives on per-project PVs that are only mounted into job pods — the client parent process does not have them mounted. Shell commands executed on the parent process have **no access** to project data at all.
   - Options: (a) disable shell commands in multi-tenant mode, (b) replace with a project-scoped log/artifact download API that retrieves data from the job store, (c) route shell commands to a running job pod (requires the job to be active), (d) launch an ephemeral "debug pod" in the project's namespace with the project PV mounted.
   - The current `cat log.txt` pattern assumes a single workspace. With per-project workspaces, the working directory concept needs redefinition.
   - **This is a significant UX change** — today admins rely heavily on shell commands for debugging. Need a clear alternative.

5. **Provisioning at scale**: With N projects and M users, the current "one provision run per project" model means M*N startup kits in the worst case. Is a shared-CA model with a single startup kit per user viable?

---

## Future: SSO for Human Users

The current design separates two kinds of participants that today are both managed via X.509 certs:

- **Sites** (server, clients, relays) — infrastructure with stable identity, long-lived
- **Humans** (admins) — users who change roles, join/leave projects, need MFA

In a future version, human authentication moves to a standard SSO system (OIDC/SAML) with short-lived tokens, while sites continue using mutual TLS with provisioned certs.

| | Sites (v1 and future) | Humans (v1) | Humans (future) |
|---|---|---|---|
| **Authentication** | mTLS certs | mTLS certs | SSO (OIDC/SAML) tokens |
| **Identity source** | Cert CN + org | Cert CN + org | IdP claims |
| **Role source** | N/A | `project.yml` registry (cert fallback) | IdP claims or `project.yml` |
| **Lifecycle** | Provisioned, long-lived | Provisioned, long-lived | IdP-managed, dynamic |
| **Startup kit** | Yes (certs, config) | Yes (certs, config) | No — just a login URL |

**Why this matters for v1 design decisions:**

The server-side `ProjectRegistry` (loaded from `project.yml`) is the right abstraction because it decouples role resolution from the cert. Today the registry overrides the cert role; in the future, the registry (or IdP) replaces the cert entirely for humans. The same `ProjectRegistry` interface can be backed by `project.yml` now and by an IdP adapter later.

This also means per-project startup kits for humans (alternative approach considered) would be a dead end — SSO eliminates admin certs entirely, so building around per-project certs for humans would be throwaway work.

The v4 `project.yml` schema is designed with this migration in mind: the `admins:` section (top-level and per-project) is explicitly optional. A deployment using SSO simply omits it; the `sites:` and `projects:` skeleton is identical in both modes. No schema version bump or file restructuring is needed when migrating to SSO.

---

## Implementation

See [multiproject_implementation.md](multiproject_implementation.md) for the full implementation plan.

---

## Phase 1: Minimal Project Plumbing

Phase 1 delivers no access control, no job store partitioning, and no cert/registry changes. The sole goal is to thread the `project` name from user-facing APIs into the runtime launchers so K8s and Docker can mount the correct volume/directory.

### Scope

1. Add `project: str = "default"` parameter to `ProdEnv` and `PocEnv`.
2. Pass `project` through to the job metadata at submission time.
3. `K8sJobLauncher` reads `project` from job metadata and selects the corresponding PVC (`<project>-workspace`).
4. `DockerJobLauncher` reads `project` from job metadata and mounts `/data/<project>/` as the workspace volume.
5. No changes to authorization, job store paths, `project.yml`, scheduler, or any other component.

### What this enables

- Data scientists can tag jobs with a project and get physical data isolation on K8s/Docker immediately.
- Lays the plumbing for all subsequent phases without requiring a full multitenancy deployment.

### What this does NOT do

- No access control — any user can submit to any project name.
- No job store partitioning (`jobs/<uuid>/` path unchanged).
- No `project.yml` parsing or `ProjectRegistry`.
- No `set_project` / `list_projects` admin commands.
- Subprocess launcher unchanged (single-tenant/trusted only).
