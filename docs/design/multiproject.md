# Multi-Project Support in Flare

## Revision History

| Version | Notes |
|---------|-------|
| 1 | Initial version |
| 2 | Incorporate feedback and Mayo discussion |

## Introduction

Flare currently operates as a single-tenant system. All server and client processes run under the same Linux user, all jobs share a flat store (`jobs/<uuid>/`), and every authorized admin can see and act on every job. There is no data segregation between different collaborations running on the same infrastructure.

To achieve genuine multi-tenancy, we introduce a **project** concept as the primary tenant boundary. A project encapsulates a private dataset, a set of participants (users and sites), an authorization policy, and runtime isolation. This document specifies the required changes across the full Flare stack.

### Design Principles

1. **Least privilege by default** — users see nothing outside their project(s)
2. **Defense in depth** — logical access control (authz) + physical isolation (containers/PVs)
3. **Backward compatible** — a `default` project preserves current single-tenant behavior
4. **`scope` deprecated** — the existing `scope` data-governance concept is superseded by `project`; `scope` will be removed in a future release
5. **Phased rollout** — Phase 1 project plumbing is available without `api_version: 4`; full multitenancy enforcement is gated on `api_version: 4` in `project.yml`


---

## Project Model

A project is a named, immutable tenant boundary with these properties:

| Property | Description |
|----------|-------------|
| `name` | Unique identifier (e.g., `cancer-research`) |
| `sites` | Set of FL sites enrolled in this project (must reference client-type site entries) |
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

If `project` is omitted in either env, it remains `None` (no API default change).

### Admin (FLARE API / Admin Console)

The `Session` gains a project context:

```python
sess = new_secure_session(
    username="admin@org_a.com",
    startup_kit_location="./startup",
    project="cancer-research",      # new
)
# All subsequent operations scoped to this project
jobs = sess.list_jobs()             # only caller-visible jobs in cancer-research
sess.submit_job("./my_job")        # tagged to cancer-research
```

Admin console equivalent:

```
> set_project cancer-research
Project set to: cancer-research

> list_jobs
... only shows caller-visible jobs in cancer-research ...
```

A user with roles in multiple projects can switch context:

```
> set_project multiple-sclerosis
Project set to: multiple-sclerosis
```

### Platform Administrator

A new **platform admin** role (distinct from per-project `project_admin`) manages cross-project concerns:

- Assign clients to projects
- Assign project admins
- View system-wide health (without seeing job data)

Project create/archive is deferred for v1 (projects are provisioning-time config in `project.yml`).

---

## Data Model Changes

### Job Metadata

`project` becomes a first-class, immutable field on every job. Set at submission time from the user's active project context. Cannot be changed after creation.

### Job Store Partitioning

New multitenant jobs are stored at `jobs/<project>/<uuid>/` (vs. current `jobs/<uuid>/`). No migration of existing jobs — they remain at `jobs/<uuid>/` and implicitly belong to the `default` project.

Legacy `default` jobs continue to be served by the main server process for compatibility. New server job pods mount only the project-partitioned slice needed for the active job.

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
2. Else if active project is `default` → fall back to cert-embedded role (legacy compatibility)
3. Otherwise → deny (`user not assigned to active project`)

The cert format is unchanged. Existing deployments with `api_version: 3` certs keep working. The cert role field is not removed or made vestigial in this version — it remains the primary source for single-tenant deployments and fallback for the `default` project.

### Admin Role Hierarchy

| Role | Scope | Capabilities |
|------|-------|-------------|
| `platform_admin` | Global | Assign clients/admins to provisioned projects, system shutdown, view all sessions |
| `project_admin` | Per-project | All job ops within project, view project's clients (no client lifecycle control) |
| `org_admin` | Per-project | Manage own-org jobs, view own-org clients within project |
| `lead` | Per-project | Submit/manage own jobs, view own-org clients within project |
| `member` | Per-project | View-only within project |

### Command Authorization Matrix

Every command is scoped to the user's active project. Operations on resources outside the active project are denied.

If the same human has multiple roles (for example `platform_admin` globally and `project_admin` in some projects), no explicit role-switch is required:
- Project-scoped job commands are authorized by the user's role in the active project
- Platform/global commands are authorized by `platform_admin`
- `platform_admin` alone does not imply project job-data permissions

#### Job Operations

| Command | project_admin | org_admin | lead | member |
|---------|:---:|:---:|:---:|:---:|
| `submit_job` | yes | no | yes | no |
| `list_jobs` | all in project | own-org jobs | own jobs | all in project |
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

Shell command behavior needs deeper design discussion because parent-process and job-pod filesystems can diverge (including standard K8s setups). See Unresolved Questions.

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
- **`projects`** — tenant definitions: which sites are enrolled (client-type entries), and (optionally) which admins have which roles. The `admins:` block inside each project is also omitted under SSO.

This separation is intentional: `sites` and `projects.sites` form the **permanent skeleton** of the file. The `admins` sections are an **optional overlay** that exists today but disappears when SSO is introduced — with no restructuring of the rest of the file.

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
    sites: [hospital-a, hospital-b]
    # Omit when using SSO (roles come from IdP claims)
    admins:
      trainer@org_a.com: lead

  multiple-sclerosis:
    sites: [hospital-a, hospital-c]
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
    sites: [hospital-a, hospital-b]
  multiple-sclerosis:
    sites: [hospital-a, hospital-c]
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

1. **Candidate filtering**: Only schedule jobs to sites enrolled in the job's project (client-type sites)
2. **Validation**: `deploy_map` sites must be a subset of the project's enrolled sites
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

Clients participate in all their enrolled projects. **Data isolation is achieved by mounting project-scoped workspace volumes in each job pod.** The Flare client parent process runs in its own pod (or on the node) and does not mount project data volumes — it only orchestrates job pod creation.

| Concern | Mechanism |
|---------|-----------|
| Namespace isolation | Deployment-defined strategy (recommended: one namespace per project; supported: shared namespace or per-job namespace) |
| Storage isolation | Workspace volume resolved by `(project, client, job pod namespace)` (not hostPath) |
| Temp directory isolation | Each pod gets its own `/tmp` via `emptyDir` — no shared host `/tmp` |
| Network isolation | NetworkPolicy scoped by project name |
| Resource limits | ResourceQuota policy per deployment strategy (deferred, see Scheduler) |
| Pod security | PodSecurityPolicy/Standards per namespace |

Workspace volume naming/provisioning must remain project-aware and work with either shared namespaces or per-job namespaces.

### Slurm

- Per-project Slurm accounts/partitions
- Per-project storage paths
- Job submission includes `--account=<project>`

---

## FLARE API Changes

- `Session` gains an optional `project` parameter (defaults to `None`) and `set_project()`/`list_projects()` methods
- `list_jobs` is filtered by active project and caller role (`project_admin`: all in project, `org_admin`: own-org, `lead`: own jobs, `member`: all in project)
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

1. **Phase 1 is ungated**: project plumbing (`project` argument + metadata propagation to launchers) is available independent of `api_version`.
2. **Feature gate for full multitenancy**: project registry, project-scoped RBAC, scheduler constraints, and job-store partitioning are enabled only when `project.yml` has `api_version: 4` with a `projects:` section.
3. **Default project**: all existing jobs, clients, and users are in the `default` project
4. **Cert role fallback**: if no project registry exists, fall back to cert-embedded role; if registry exists but user has no mapping, fallback applies only when active project is `default`
5. **API compatibility**: omitted `project` remains `None` (no default change) across phases
6. **Config version**: `api_version: 4` in `project.yml` signals full multi-project enforcement; version 3 continues to work as single-tenant

### Release Transition Strategy (2.8 -> 2.9)

1. **Upgrade to 2.8 (Phase 1 only)**: optional project tagging/plumbing is available, but no multitenant access-control or scheduler behavior changes are enabled.
2. **Upgrade to 2.9 with existing v3 deployments**: keep current `project.yml` (`api_version: 3`) and startup kits; system remains single-tenant/compatibility mode.
3. **Existing jobs continue to work**: legacy jobs remain at `jobs/<uuid>/` as `default`; no data migration is required.
4. **Activate full multi-project mode when ready**: deploy a v4 `project.yml` (`api_version: 4` + `projects:`) to server startup artifacts and restart server to load registry-backed project scoping.
5. **Provisioning impact**: no full reprovision is required; keep dynamic provisioning behavior by updating server-side artifacts and generating startup kits only for newly added or changed participants.

---

## Design Decisions

| # | Question | Decision |
|---|----------|----------|
| D1 | Can clients participate in multiple projects? | **Yes.** Clients participate in all enrolled projects simultaneously. Data isolation is physical: K8s mounts different PVs per project; Docker mounts different host directories. The Flare parent process does not access project data. |
| D2 | Project lifecycle management? | **Deferred.** Projects are defined at provisioning time in `project.yml`. Runtime project CRUD is not in scope for v1. |
| D3 | Per-project quota management? | **Deferred.** Rely on K8s ResourceQuota per namespace for now. Future: route projects to different K8s scheduling queues via pod labels. |
| D4 | `check_status` information leakage? | **Server has global knowledge, filtering the response is sufficient.** The server parent process knows about all clients and jobs; it filters responses to only include resources in the user's active project. No architectural change needed. |
| D5 | Server-side job store isolation? | **Server job pods must only access their project's data.** The server job process (running in K8s/Docker) must not mount the entire job store — only the project-partitioned slice for new-layout jobs (`jobs/<project>/...`). Legacy jobs remain at `jobs/<uuid>/` under `default`; they are served by the main server process for compatibility and are not mounted into new server job pods. Current `FilesystemStorage` will be replaced by a database or object store in the future, which will enforce project-scoped access natively. |
| D6 | Role storage: certs vs. server-side registry? | **Layered: registry overrides cert.** `project.yml` defines per-project roles; the server loads it at startup via `ProjectRegistry`. Certs continue to authenticate identity (name, org) and carry a role as fallback. No cert format change required. |
| D7 | How do shared clients know which project PV to mount? | **The launcher passes the project name to the client.** Job metadata carries the project; the server includes it when dispatching to clients. The client-side `K8sJobLauncher`/`DockerJobLauncher` uses the project name to select the correct PV/volume mount. |
| D8 | Cross-project isolation in subprocess mode? | **Subprocess mode is single-tenant/trusted only.** Only K8s, Docker, and Slurm launchers provide secure multi-tenant isolation (separate namespaces, volumes, `/tmp`). The default subprocess launcher offers no physical isolation and is only suitable for single-tenant or trusted environments. |
| D9 | Cross-project visibility for `platform_admin` job data? | **No.** `platform_admin` does not get cross-project job metadata/data visibility and there is no `list_jobs --all-projects` behavior in v1. If the same human also has a project-scoped role in the active project, only that project-scoped role grants job access. |
| D10 | Provisioning model at scale? | **Keep dynamic provisioning behavior.** Adding sites/users should not require reprovisioning all existing sites; update server-side config/startup artifacts and generate kits only for newly added or changed participants. |

---

## Unresolved Questions

1. **Shell-command replacement UX**: Parent-process shell commands are backend-dependent and cannot be relied on for job workspace access (notably in K8s, but this can happen in single-project setups too). The right policy and UX replacement need further study (for example log/artifact APIs vs pod-targeted debug workflows).

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

1. Add `project: Optional[str] = None` parameter to `ProdEnv` and `PocEnv`.
2. Pass `project` through to the job metadata at submission time.
3. `K8sJobLauncher` reads `project` from job metadata and selects the corresponding project workspace volume.
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
