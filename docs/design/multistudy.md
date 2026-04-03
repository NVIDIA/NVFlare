# Multi-Study Support

## Introduction

NVFlare can host multiple logical studies inside one deployment. A study defines:

- which client sites participate
- which effective role each admin user has in that study
- which jobs and client-targeted operations are visible from a session bound to that study

The active study is established when the admin session logs in, stored in the authenticated server session, and then
propagated through job submission, job metadata, and scheduling.

This document describes the shipped multi-study design, including session plumbing, job metadata, registry-backed
access control, client filtering, and scheduler enforcement.

---

## Core Principles

1. **Session-scoped study** — study is chosen at session creation/login time, not passed command by command.
2. **Backward-compatible default** — `default` is the built-in fallback study name.
3. **Registry-backed named studies** — named studies come from `project.yml`, are validated at provisioning time, and
   are loaded at server startup from a generated runtime registry.
4. **Two-layer enforcement** — study boundary first, existing RBAC second.
5. **Operational convenience for shared-trust deployments** — multi-study provides study-level authorization and
   scheduling boundaries within a shared deployment, while separate deployments remain the right choice for stronger
   isolation boundaries.

---

## User Experience

### FLARE API

```python
sess = new_secure_session(
    username="admin@org_a.com",
    startup_kit_location="./startup",
    study="cancer-research",
)
```

- If `study` is omitted, the session uses `default`.
- `submit_job`, `list_jobs`, `get_job_meta`, and `clone_job` all operate in the active study context.

### ProdEnv

```python
env = ProdEnv(
    startup_kit_location=args.startup_kit_location,
    study="cancer-research",
)
run = recipe.execute(env)
```

- If `study` is omitted, `ProdEnv` uses `default`.
- Jobs submitted through the recipe stack inherit the environment study.

### Admin Console

```bash
./startup/fl_admin.sh --study cancer-research
```

- If `--study` is omitted, the terminal uses `default`.
- The study is established at login and then inherited by later commands in that terminal session.

---

## Data Model

### Default Study

`DEFAULT_STUDY = "default"` in `nvflare.apis.job_def` is the single source of truth for the built-in fallback study.

`default` is reserved and may not appear under `studies:` in `project.yml`.

### Session State

For `flare_api.Session`, `ProdEnv`, and the admin console:

- the study is sent at login time
- the server stores it as authenticated session state
- the session token carries the active study so recreated sessions keep the same study context

Study names are syntactically validated with the regex:

`^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$`

### Job Metadata

Every submitted job carries a first-class `study` field (`JobMetaKey.STUDY`).

- set from the session's active study at submit time
- immutable after creation
- preserved on clone from the source job

### Legacy Jobs

Jobs created before multi-study may not have a `study` field.

- jobs with no `study` field normalize to `default`
- jobs already persisted on disk with a non-default `study` are treated as belonging to that named study

---

## Provisioning and Configuration

### `project.yml`

Multi-study uses `api_version: 4`.

```yaml
api_version: 4
name: my_project

participants:
  - name: server1.example.com
    type: server
    org: nvidia
  - name: hospital-a
    type: client
    org: org_a
  - name: hospital-b
    type: client
    org: org_b
  - name: admin@nvidia.com
    type: admin
    org: nvidia
    role: project_admin
  - name: trainer@org_a.com
    type: admin
    org: org_a
    role: lead

studies:
  cancer-research:
    sites: [hospital-a, hospital-b]
    admins:
      admin@nvidia.com: project_admin
      trainer@org_a.com: lead

  multiple-sclerosis:
    sites: [hospital-a]
    admins:
      trainer@org_a.com: member
```

Validation rules:

- `studies:` requires `api_version: 4`
- study names must pass `name_check(..., "study")`
- `default` is reserved
- `studies.<name>.sites` must reference existing client participants
- `studies.<name>.admins` must reference existing admin participants
- per-study roles must be one of the existing built-in roles

### Runtime Registry

Provisioning emits `study_registry.json` into the server `local/` directory. The server loads this file at startup as
the authoritative runtime registry for:

- study names
- enrolled sites per study
- per-study admin role mappings

### Update Flow

Study mappings are updated through the normal reprovision and restart flow:

1. edit `project.yml`
2. reprovision
3. deploy the updated server artifacts
4. restart the server

---

## Authorization Model

Multi-study reuses the existing built-in roles:

- `project_admin`
- `org_admin`
- `lead`
- `member`

### Role Resolution

The admin participant `role` in `project.yml` is still baked into the cert. That cert role remains meaningful:

- it is the effective role for the `default` study
- it is also the role used for server-only/global operations

When a session logs in to a named study:

1. the server verifies that a study registry exists
2. the study name exists in the registry
3. the user has an entry in that study's `admins` mapping

If any of those checks fail, login is rejected.

For a valid named study session, the mapped study role becomes the effective role for **study-scoped authorization**.

### What Is Study-Scoped

Study-scoped behavior applies to:

- job listing and job visibility
- direct job access (`get_job_meta`, `clone_job`, `download_job`, etc.)
- job submission and persisted `submitter_role`
- client-targeted and mixed server/client admin operations after target resolution and study filtering

Server-only/global operations keep using the certificate-based role.

### Authorization Layers

Requests are enforced in this order:

1. **Study boundary** — is the target job/client in the session's active study?
2. **Existing RBAC** — evaluate `authorization.json` with the effective role for that request

`authorization.json`, `Person`, `AuthzContext`, `Authorizer`, and policy semantics remain unchanged.

---

## Enforcement Behavior

### Login

- `default` login is always allowed if certificate authentication succeeds
- named-study login is allowed only when the runtime registry exists and the user is mapped into that study
- if `studies:` is absent, named-study login is rejected

### Job Visibility

- `list_jobs` returns only jobs belonging to the active study
- direct job access to a job in another study behaves as "not found"
- cloned jobs preserve the source job's study

### Job Submission

At submit time:

- the job is tagged with the session's active study
- `submitter_role` is persisted as the **effective** role for that session
- `deploy_map` sites must all be enrolled in the active study

### Client Target Filtering

For study-aware client operations:

- `check_status client` shows only enrolled clients in the active study
- mixed `all` operations include the server plus only the enrolled clients for that study

### Scheduler

The scheduler enforces the study boundary again at dispatch time:

- `@ALL` is narrowed to the online clients enrolled in the job's study
- explicit sites outside the study block scheduling
- required sites outside the study block scheduling

This is in addition to submit-time `deploy_map` validation.

---

## Single-Tenant Behavior

If `project.yml` has no `studies:` section:

- the deployment is single-tenant
- only `default` is a valid login study
- submitted jobs are tagged `default`
- legacy jobs with no `study` field normalize to `default`
- jobs persisted with a non-default `study` are hidden from default sessions

This gives one consistent rule: without a provisioned study registry, there are no named study sessions.

---

## Runtime Posture

Multi-study provides authorization and scheduling boundaries within a shared deployment.

In a multi-study deployment, the following are still shared:

- server runtime
- client runtime at a site
- PKI root and cert issuance
- deployment blast radius for `project_admin`

Subprocess-backed execution remains supported as one of the available deployment backends.

---

## When to Use Multi-Study vs. Separate Deployments

Use multi-study when:

- sites participate in multiple studies and duplicate provisioning would be operationally expensive
- studies share the same operational trust boundary
- logical isolation through software enforcement is sufficient

Use separate deployments when:

- you need stronger isolation than a shared deployment can provide
- failure or compromise in one study must not affect another
- studies do not meaningfully share infrastructure or participants

---

## Summary of Shipped Design

| Area | Shipped Design |
|------|----------------|
| Study membership | A client site can participate in multiple named studies. |
| Role model | The feature reuses the built-in roles: `project_admin`, `org_admin`, `lead`, and `member`. |
| Default study | `default` is the built-in fallback study and is reserved in `project.yml`. |
| Runtime source of truth | The server loads generated `study_registry.json` at startup. |
| Named study availability | Named studies are provisioned through `studies:` in `project.yml` and activated by the runtime registry. |
| Authorization config | Existing `authorization.json` RBAC remains in effect, using the effective role for each request. |
| Update model | Study mapping changes are applied through edit, reprovision, redeploy, and restart. |
| Runtime posture | Multi-study provides authorization and scheduling boundaries within a shared deployment. |
