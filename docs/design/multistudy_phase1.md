# Multi-Study Support — Phase 1: Study Plumbing

## Revision History

| Version | Notes |
|---------|-------|
| 1 | Initial version |
| 2 | Incorporate feedback and Mayo discussion |
| 3 | Rename `project` → `study`; split Phase 1 / Phase 2 docs |

## Introduction

Flare currently operates as a single-tenant system. All jobs share a flat store (`jobs/<uuid>/`), and every authorized admin can see and act on every job. There is no data segregation between different collaborations running on the same infrastructure.

Phase 1 introduces a **study** concept as lightweight metadata plumbing. Every job carries a `study` name (defaulting to `"default"`). The study flows from user-facing APIs into job metadata and runtime launchers so that K8s deployments can mount study-specific workspace volumes immediately — without any access-control, provisioning, or job-store changes.

See [multistudy_phase2.md](multistudy_phase2.md) for the full multi-tenancy design (access control, study registry, job-store partitioning, etc.).

### Design Principles

1. **Backward compatible** — a `default` study preserves current single-tenant behavior; legacy jobs missing a `study` field are treated as `default` on read
2. **Phased rollout** — Phase 1 delivers plumbing only; access-control enforcement is deferred to Phase 2
3. **Minimal footprint** — no authorization, no provisioning, no job-store layout changes

---

## Scope

1. `study: str = "default"` parameter on `ProdEnv`, `Session`, `new_session`, `new_secure_session`, `new_insecure_session`.
2. `Session` carries the active study context; `list_jobs` inherits it and returns only jobs in that study.
3. Study is passed through to job metadata at submission time, with syntax validation before persistence.
4. Clone preserves the source job's study (not the session's study).
5. `K8sJobLauncher` reads `study` from job metadata and selects the corresponding study workspace volume (TODO in code).
6. `DockerJobLauncher` unchanged; TODO marker for future study-aware settings resolution.
7. Admin console (`fl_admin.sh`) accepts `--study` at launch time; the study applies to `submit_job` and `list_jobs` commands.
8. No changes to authorization, job store paths, `project.yml` schema, scheduler, or provisioning.

---

## User Experience

### Data Scientist (Recipe API)

The recipe is unchanged. The study is specified via `ProdEnv`:

```python
env = ProdEnv(
    startup_kit_location=args.startup_kit_location,
    study="cancer-research",
)
run = recipe.execute(env)
```

If `study` is omitted, it defaults to `"default"`.

### Admin (FLARE API)

The `Session` gains a study context:

```python
sess = new_secure_session(
    username="admin@org_a.com",
    startup_kit_location="./startup",
    study="cancer-research",
)
jobs = sess.list_jobs()        # only jobs in cancer-research
sess.submit_job("./my_job")   # tagged to cancer-research
```

The study is session-scoped, not a per-command filter.

### Admin Console

```
$ ./startup/fl_admin.sh --study cancer-research

> list_jobs
... only shows jobs in cancer-research ...
```

If `--study` is omitted, the admin terminal uses `default`.

---

## Data Model

### Job Metadata

`study` is a first-class field on every job (`JobMetaKey.STUDY`). Set at submission time from the session's active study. Immutable after creation.

The study value is syntactically validated at the API layer (client-side) and again on the server before persistence. The regex pattern is `^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$` — lowercase alphanumeric with hyphens, 1–63 characters.

### Legacy Jobs

Jobs created before Phase 1 have no `study` field. `get_job_meta_study()` returns `"default"` for these jobs, so they appear in the `default` study transparently.

### Default Study Constant

`DEFAULT_JOB_STUDY = "default"` in `nvflare.apis.job_def` — single source of truth for the default value.

---

## What This Enables

- Data scientists tag jobs with a study and get physical data isolation on K8s immediately (once K8s launcher TODO is implemented).
- Admin/API sessions operate inside one active study context, so Phase 2 authz can validate study access when sessions enter a study.
- Legacy single-tenant deployments are unaffected — everything defaults to `"default"`.

## What This Does NOT Do

- No access control — any user can submit to any valid study name
- No job store partitioning (`jobs/<uuid>/` path unchanged)
- No `project.yml` parsing or `StudyRegistry`
- No Docker launcher behavior change yet
- No `set_study` / `list_studies` admin commands
- Subprocess launcher unchanged (single-tenant/trusted only)
