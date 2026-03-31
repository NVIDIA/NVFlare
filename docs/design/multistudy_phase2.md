# Multi-Study Support — Phase 2: Per-Study Access Control

## Revision History

| Version | Notes |
|---------|-------|
| 1 | Initial version |
| 2 | Incorporate feedback and Mayo discussion |
| 3 | Rename `project` → `study`; split Phase 1 / Phase 2 docs |
| 4 | Simplify: reuse existing roles, no new role types, minimal changes |

## Prerequisites

Phase 1 ([multistudy_phase1.md](multistudy_phase1.md)) must be complete: `study` metadata flows from user-facing APIs through to job metadata and launcher plumbing.

## Introduction

Phase 2 adds per-study access control on top of Phase 1's plumbing. The existing role model (`project_admin`, `org_admin`, `lead`, `member`) is unchanged — the only difference is that roles become per-study instead of global. No new roles are introduced.

---

## Core Idea

Today, a user's role is global (baked into the X.509 cert). Phase 2 makes the same role **per-study**: a user can be `lead` in one study and `member` in another.

The existing authorization rules (`authorization.json`) stay exactly as they are. The only new layer is a **study filter**: before evaluating existing RBAC, the server checks whether the resource belongs to the user's active study. If not, the resource is invisible.

---

## How Authorization Works Today

### `authorization.json`

Each deployment ships an `authorization.json` that maps roles to permissions. The default policy:

```json
{
  "format_version": "1.0",
  "permissions": {
    "project_admin": "any",
    "org_admin": {
      "submit_job": "none",
      "clone_job": "none",
      "manage_job": "o:submitter",
      "download_job": "o:submitter",
      "view": "any",
      "operate": "o:site",
      "shell_commands": "o:site",
      "byoc": "none"
    },
    "lead": {
      "submit_job": "any",
      "clone_job": "n:submitter",
      "manage_job": "n:submitter",
      "download_job": "n:submitter",
      "view": "any",
      "operate": "o:site",
      "shell_commands": "o:site",
      "byoc": "any"
    },
    "member": {
      "view": "any"
    }
  }
}
```

Permission conditions: `"any"` = unrestricted, `"none"` = denied, `"n:submitter"` = only if user is the submitter, `"o:submitter"` = only if user is in the same org as the submitter, `"o:site"` = only if user is in the same org as the target site.

### Authorization Flow

1. A `Person(name, org, role)` is constructed from the user's X.509 cert
2. An `AuthzContext(right, user, submitter)` wraps the request
3. The `Authorizer` evaluates the policy: look up `permissions[person.role][right]` and check the condition against the context

**Phase 2 changes only step 1**: the `role` used to construct `Person` can come from the per-study mapping instead of the cert. Steps 2 and 3 are untouched. `authorization.json` is unchanged.

---

## Role Resolution

No new roles. The existing four roles are reused:

| Role | Capabilities (per `authorization.json`, unchanged) |
|------|-----|
| `project_admin` | All operations (`"any"`) |
| `org_admin` | Manage/download own-org jobs, view all, operate own-org sites |
| `lead` | Submit/manage/download own jobs, view all, operate own-org sites |
| `member` | View only |

**Resolution order:**
1. If `project.yml` has a `studies:` section AND the user has a mapping for the active study → use that role
2. Else if active study is `default` → fall back to cert-embedded role (legacy compatibility)
3. Otherwise → deny

### What the participant `role` means

The `role` field on admin participants in `project.yml` serves two purposes:
- It is baked into the X.509 cert at provisioning time (identity + authentication)
- It is the **effective role for the `default` study** and for deployments without a `studies:` section

When a `studies:` section is present, the per-study role overrides the cert role for that study. The cert role still applies to the `default` study as a fallback.

Example: a user with `role: lead` in their participant entry and `member` in the `cancer-research` study mapping is `lead` when operating in the `default` study but `member` when operating in `cancer-research`.

---

## Provisioning: `project.yml`

Minimal addition: a `studies:` section that maps study names to enrolled sites and per-user role overrides. Everything else stays as-is.

```yaml
# Existing sections unchanged
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
    role: project_admin    # cert role; effective role for "default" study
  - name: trainer@org_a.com
    type: admin
    org: org_a
    role: lead             # cert role; effective role for "default" study

# New section — per-study role overrides
studies:
  cancer-research:
    sites: [hospital-a, hospital-b]
    admins:
      trainer@org_a.com: lead        # same as cert role here, but explicit

  multiple-sclerosis:
    sites: [hospital-a]
    admins:
      trainer@org_a.com: member      # overrides cert role for this study
```

- If `studies:` is absent, the system behaves exactly as today (single-tenant, cert roles only).
- Sites listed under a study must reference existing client-type participants.
- Admins listed under a study must reference existing admin-type participants.
- A user not listed under a study has no access to that study (except `default`, which falls back to cert role).

---

## Authorization Enforcement

Two layers, evaluated in order for every command:

1. **Study filter** (new): Does the target resource (job, client) belong to the user's active study? If no → invisible.
2. **RBAC policy** (existing, unchanged): Construct `Person` with the resolved per-study role, evaluate `authorization.json` as today.

The session's active study (set at session start via `--study` or the `study` API parameter) determines which study filter applies.

---

## Job Scheduler

When a `studies:` section is present in `project.yml`:

1. **Site filtering**: Only schedule jobs to sites enrolled in the job's study
2. **Validation**: `deploy_map` sites must be a subset of the study's enrolled sites

No quota or priority changes.

---

## Runtime Isolation

### Kubernetes

The K8s launcher reads `study` from job metadata and resolves study-specific workspace volumes:
- Workspace volume resolved by `(study, client)` tuple
- Each job pod mounts only its study's data volume

### Docker

The Docker launcher reads `study` from job metadata and mounts the corresponding host directory (e.g., `/data/<study>/`) as the workspace volume.

---

## Migration / Backward Compatibility

1. **No `studies:` section** → system behaves as today, single-tenant, cert roles only.
2. **`studies:` section present** → per-study role enforcement enabled; the `default` study falls back to cert roles for compatibility.
3. **Legacy jobs** (no `study` field) → treated as `default` study (Phase 1 behavior, unchanged).
4. **No data migration** — job store layout is unchanged (`jobs/<uuid>/`).

---

## Design Decisions

| # | Question | Decision |
|---|----------|----------|
| D1 | Can clients participate in multiple studies? | **Yes.** Listed under multiple studies in `project.yml`. Data isolation via launcher (K8s PVs / Docker mounts). |
| D2 | New roles needed? | **No.** Existing `project_admin` / `org_admin` / `lead` / `member` reused per-study. |
| D3 | Study lifecycle management? | **Deferred.** Studies defined at provisioning time in `project.yml`. |
| D4 | Per-study quotas? | **Deferred.** Rely on K8s-level resource controls. |
| D5 | How do launchers know which volume to mount? | Job metadata carries the study; the launcher resolves the volume from that. |
| D6 | What does the participant `role` mean when `studies:` exists? | It is baked into the cert and serves as the effective role for the `default` study. Per-study mappings override it for non-default studies. |

