# Multitenancy Implementation Plan

Companion to [multitenancy.md](multitenancy.md). This document specifies *how* to implement the design with minimal risk.

## Guiding Principles

1. **Feature-gated** — all multitenancy behavior gated on `project.yml` having a `projects:` section (`api_version: 4`). Single-tenant deployments unchanged.
2. **Additive, not migratory** — no job store path migration, no role renames. New code paths only.
3. **Layered role resolution** — registry overrides cert, cert remains fallback. No breaking change to cert format.
4. **Incremental delivery** — three shippable milestones, each independently testable.

---

## Codebase Map

Key files and their roles (discovered via exploration):

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| Job metadata | `nvflare/apis/job_def.py:48-82` | `JobMetaKey` enum |
| Job store | `nvflare/app_common/storages/filesystem_storage.py` | `FilesystemStorage` |
| Job manager | `nvflare/apis/impl/job_def_manager.py` | `SimpleJobDefManager` |
| Job submission | `nvflare/private/fed/server/job_cmds.py:564-618` | `submit_job()` handler |
| Job listing | `nvflare/private/fed/server/job_cmds.py:316-367` | `list_jobs()` handler |
| Job scheduling | `nvflare/app_common/job_schedulers/job_scheduler.py:101-230` | `DefaultJobScheduler._try_job()` |
| Authz policy | `nvflare/fuel/sec/authz.py` | `AuthorizationService`, `Policy`, `Authorizer` |
| Authz filter | `nvflare/fuel/hci/server/authz.py:44-94` | `AuthzFilter.pre_command()` |
| Login/session | `nvflare/fuel/hci/server/login.py:69-119` | `handle_cert_login()` |
| Server session | `nvflare/fuel/hci/server/sess.py:33-86` | `Session` (user_name, user_org, user_role) |
| Conn properties | `nvflare/fuel/hci/server/constants.py:16-45` | `ConnProps` |
| Role from cert | `nvflare/fuel/hci/security.py:74-98` | `get_identity_info()` → `UNSTRUCTURED_NAME` |
| Admin server | `nvflare/private/fed/server/admin.py:95-174` | `FedAdminServer` (filter chain setup) |
| Cmd authz utils | `nvflare/private/fed/server/cmd_utils.py:41-148` | `authorize_job()`, `must_be_project_admin()` |
| FLARE API Session | `nvflare/fuel/flare_api/flare_api.py:65-108` | `Session` (client-side) |
| `new_secure_session` | `nvflare/fuel/flare_api/flare_api.py:944-956` | Session factory |
| ProdEnv | `nvflare/recipe/prod_env.py:45-108` | `ProdEnv` (recipe execution env) |
| SessionManager | `nvflare/recipe/session_mgr.py:40-106` | `SessionManager` |
| Provisioner | `nvflare/lighter/provision.py:132-170` | `prepare_project()`, loads `project.yml` |
| Project entity | `nvflare/lighter/entity.py:370-573` | `Project` class |
| Cert generation | `nvflare/lighter/impl/cert.py:296-347` | `get_pri_key_cert()` |
| x509 role field | `nvflare/lighter/utils.py:129-135` | `x509_name()` → `UNSTRUCTURED_NAME` |
| Admin roles | `nvflare/lighter/constants.py:109-116` | `AdminRole`, `DEFINED_ROLES` |
| Audit | `nvflare/fuel/sec/audit.py:90-125` | `AuditService` singleton |
| Audit filter | `nvflare/fuel/hci/server/audit.py:23-46` | `CommandAudit.pre_command()` |
| Security init | `nvflare/private/fed/utils/fed_utils.py:98-147` | `security_init()` |
| FLAuthorizer | `nvflare/security/security.py:21-74` | `FLAuthorizer`, `COMMAND_CATEGORIES` |

---

## Risk Mitigations

### Problem: Job store path migration
**Mitigation**: no migration. New multitenant jobs go to `jobs/<project>/<uuid>/`. Existing jobs stay at `jobs/<uuid>/` and implicitly belong to `default`. `SimpleJobDefManager` checks both paths when listing `default` project jobs.

### Problem: `project_admin` role rename
**Mitigation**: no rename. `project_admin` already fits as the per-project admin concept. Add `platform_admin` as a new global role. Existing `must_be_project_admin()` checks become "is user `project_admin` in any project OR `platform_admin`" — backward compatible.

### Problem: Cert role becomes vestigial
**Mitigation**: layered resolution. If `ProjectRegistry` has a mapping for this user+project, use it. Otherwise fall back to cert role. Cert format unchanged; no re-provisioning required for existing deployments.

### Problem: Cross-cutting feature requires all-or-nothing
**Mitigation**: three milestones. Milestone 1 adds plumbing with zero behavior change. Milestone 2 adds the registry (gated). Milestone 3 enforces scoping.

---

## Milestone 1: Project-Aware Plumbing

**Goal**: thread `project` through the stack, always `"default"`. All existing tests pass, zero behavior change.

### 1.1 Add `JobMetaKey.PROJECT`

**File**: `nvflare/apis/job_def.py`

```python
class JobMetaKey(str, Enum):
    ...
    PROJECT = "project"
```

### 1.2 Add `ConnProps.PROJECT`

**File**: `nvflare/fuel/hci/server/constants.py`

```python
class ConnProps(object):
    ...
    PROJECT = "_project"
```

### 1.3 Stamp project on job submission

**File**: `nvflare/private/fed/server/job_cmds.py`

In `submit_job()` (~line 604), after setting submitter info:
```python
meta[JobMetaKey.PROJECT.value] = conn.get_prop(ConnProps.PROJECT, "default")
```

Same for `clone_job()` (~line 504).

### 1.4 Add `project` to server-side `Session`

**File**: `nvflare/fuel/hci/server/sess.py`

Add `project="default"` to `Session.__init__()`. Include `"p"` key in token encoding. Set `ConnProps.PROJECT` from session in `LoginModule.pre_command()`.

### 1.5 Add `project` to client-side `Session`

**File**: `nvflare/fuel/flare_api/flare_api.py`

- Add `project="default"` param to `Session.__init__()` and `new_secure_session()`
- Store as `self._project`, pass to server on connect

### 1.6 Add `project` to `ProdEnv`

**File**: `nvflare/recipe/prod_env.py`

Add `project="default"` param, pass through to `SessionManager`.

### 1.7 Add `project` to audit events

**File**: `nvflare/fuel/sec/audit.py`

Add `project` param to `add_event()`. Emit `[P:project]` in log line.

**File**: `nvflare/fuel/hci/server/audit.py`

Pass `conn.get_prop(ConnProps.PROJECT, "default")` to `add_event()`.

### 1.8 Filter `list_jobs` by project

**File**: `nvflare/private/fed/server/job_cmds.py`

Add project predicate to `_job_match()`:
```python
and ((not project) or job_meta.get("project", "default") == project)
```

Extract shared helper `_is_job_in_project(job_meta, project)` for reuse across all job command handlers.

### Milestone 1 Summary

| File | Change |
|------|--------|
| `nvflare/apis/job_def.py` | +1 enum value |
| `nvflare/fuel/hci/server/constants.py` | +1 constant |
| `nvflare/private/fed/server/job_cmds.py` | stamp project on submit/clone, filter list_jobs |
| `nvflare/fuel/hci/server/sess.py` | project field on Session |
| `nvflare/fuel/hci/server/login.py` | set ConnProps.PROJECT from session |
| `nvflare/fuel/flare_api/flare_api.py` | project param on Session + factory |
| `nvflare/recipe/prod_env.py` | project param |
| `nvflare/recipe/session_mgr.py` | pass project through |
| `nvflare/fuel/sec/audit.py` | project field in events |
| `nvflare/fuel/hci/server/audit.py` | pass project from conn |

**~10 files, ~150 lines. Zero behavior change.**

---

## Milestone 2: Project Registry + Role Resolution

**Goal**: add `ProjectRegistry`, per-project role resolution, `platform_admin` role, `set_project`/`list_projects` commands. Gated on `api_version: 4`.

### 2.1 Create `ProjectRegistry`

**New file**: `nvflare/security/project_registry.py` (~150 lines)

```python
class ProjectRegistry:
    """Resolves project membership, client enrollment, and per-project roles.

    Loaded from project.yml at server startup. When absent or api_version < 4,
    operates in single-tenant mode (all users/clients in 'default' project,
    roles from certs).
    """

    def __init__(self):
        self._projects = {}        # name -> {clients: set, users: dict}
        self._multitenant = False

    def load_from_config(self, project_dict: dict):
        """Load from parsed project.yml. Detects api_version >= 4."""
        ...

    def is_multitenant(self) -> bool:
        """True if projects section exists (api_version 4+)."""
        return self._multitenant

    def get_projects(self) -> List[str]:
        """All project names."""
        ...

    def get_project_clients(self, project: str) -> Set[str]:
        """Client names enrolled in project."""
        ...

    def get_user_projects(self, username: str) -> List[str]:
        """Projects this user belongs to."""
        ...

    def get_user_role(self, username: str, project: str) -> Optional[str]:
        """User's role in project, or None if not a member."""
        ...

    def is_platform_admin(self, username: str) -> bool:
        """True if user has global platform_admin role."""
        ...

    def is_user_in_project(self, username: str, project: str) -> bool:
        ...

    def is_client_in_project(self, client_name: str, project: str) -> bool:
        ...
```

Singleton access via `ProjectRegistryService` (follows existing `AuthorizationService` pattern).

### 2.2 Add `platform_admin` role

**File**: `nvflare/lighter/constants.py`

```python
class AdminRole:
    PLATFORM_ADMIN = "platform_admin"  # new
    PROJECT_ADMIN = "project_admin"
    ORG_ADMIN = "org_admin"
    LEAD = "lead"
    MEMBER = "member"

DEFINED_ROLES = [
    AdminRole.PLATFORM_ADMIN,
    AdminRole.PROJECT_ADMIN,
    AdminRole.ORG_ADMIN,
    AdminRole.LEAD,
    AdminRole.MEMBER,
]
```

### 2.3 Load registry at server startup

**File**: `nvflare/private/fed/utils/fed_utils.py` in `security_init()`

After loading authorization policy, load `project.yml` into `ProjectRegistryService`:
```python
project_config = load_yaml(workspace.get_project_config_path())
ProjectRegistryService.initialize(project_config)
```

### 2.4 Layered role resolution in login

**File**: `nvflare/fuel/hci/server/login.py`

In `handle_cert_login()`, after extracting identity from cert:
```python
# Existing: role from cert
role = identity_info.get(IdentityKey.ROLE, "")

# New: override with registry role if multitenant
registry = ProjectRegistryService.get_registry()
if registry and registry.is_multitenant():
    project = ...  # from login request or default
    registry_role = registry.get_user_role(username, project)
    if registry_role:
        role = registry_role
```

### 2.5 Project filter in authz chain

**File**: `nvflare/fuel/hci/server/authz.py`

New `ProjectFilter(CommandFilter)` registered in filter chain between `LoginModule` and `AuthzFilter`:

```python
class ProjectFilter(CommandFilter):
    """Validates user has access to their active project.

    Registered AFTER LoginModule (needs identity) and BEFORE AuthzFilter
    (sets project-scoped role for downstream authz).
    """
    def pre_command(self, conn, args):
        registry = ProjectRegistryService.get_registry()
        if not registry or not registry.is_multitenant():
            return True  # single-tenant, no filtering

        project = conn.get_prop(ConnProps.PROJECT, "default")
        username = conn.get_prop(ConnProps.USER_NAME, "")

        if registry.is_platform_admin(username):
            return True  # platform admin bypasses project check

        if not registry.is_user_in_project(username, project):
            conn.append_error("Access denied: not a member of this project")
            return False

        # Override role to project-specific role
        role = registry.get_user_role(username, project)
        conn.set_prop(ConnProps.USER_ROLE, role)
        return True
```

**File**: `nvflare/private/fed/server/admin.py`

Register `ProjectFilter` after `LoginModule`, before `AuthzFilter`:
```python
login_module = LoginModule(sess_mgr)
cmd_reg.add_filter(login_module)

project_filter = ProjectFilter()          # new
cmd_reg.add_filter(project_filter)        # new

authz_filter = AuthzFilter()
cmd_reg.add_filter(authz_filter)
```

### 2.6 `set_project` and `list_projects` commands

**New command handlers** in `nvflare/private/fed/server/job_cmds.py` (or a new `ProjectCommandModule`):

- `set_project <name>`: validates user membership, updates session's project, re-resolves role
- `list_projects`: returns projects the user belongs to (or all, for `platform_admin`)

**Client-side**: `Session.set_project()` and `Session.list_projects()` in `nvflare/fuel/flare_api/flare_api.py`.

### 2.7 Provisioner: parse `api_version: 4`

**File**: `nvflare/lighter/provision.py`

Extend `prepare_project()` to accept `api_version` 3 or 4. When 4:
- Parse `projects:` section
- Parse per-admin `projects:` mapping
- Validate referenced clients exist in participants

**File**: `nvflare/lighter/entity.py`

Add `projects` property to `Project` class. Add `project_roles` dict to admin `Participant`.

### 2.8 Server startup kit includes `project.yml`

**File**: `nvflare/lighter/impl/static_file.py`

Copy `project.yml` into server startup kit directory.

### Milestone 2 Summary

| File | Change |
|------|--------|
| `nvflare/security/project_registry.py` | **new** — ProjectRegistry + service |
| `nvflare/lighter/constants.py` | +PLATFORM_ADMIN role |
| `nvflare/private/fed/utils/fed_utils.py` | load registry at startup |
| `nvflare/fuel/hci/server/login.py` | layered role resolution |
| `nvflare/fuel/hci/server/authz.py` | +ProjectFilter class |
| `nvflare/private/fed/server/admin.py` | register ProjectFilter |
| `nvflare/private/fed/server/job_cmds.py` | set_project, list_projects handlers |
| `nvflare/fuel/flare_api/flare_api.py` | set_project, list_projects client methods |
| `nvflare/lighter/provision.py` | api_version 4 parsing |
| `nvflare/lighter/entity.py` | projects on Project/Participant |
| `nvflare/lighter/impl/static_file.py` | include project.yml in server kit |

**~11 files (1 new), ~500 lines. Gated on api_version 4.**

---

## Milestone 3: Enforcement + Scheduler

**Goal**: all commands scoped to active project. Scheduler validates project client enrollment.

### 3.1 All job commands: project gate

**File**: `nvflare/private/fed/server/job_cmds.py`

For every job-specific handler (`abort_job`, `delete_job`, `download_job`, `clone_job`, `app_command`, `configure_job_log`), add project validation in `authorize_job_id()`:

```python
def authorize_job_id(self, conn, args):
    ...  # existing: load job, set submitter props

    # New: verify job belongs to active project
    job_project = job.meta.get(JobMetaKey.PROJECT.value, "default")
    active_project = conn.get_prop(ConnProps.PROJECT, "default")
    if job_project != active_project:
        conn.append_error("Job not found in current project")
        return PreAuthzReturnCode.ERROR

    return PreAuthzReturnCode.REQUIRE_AUTHZ
```

This is a single change point since all job commands route through `authorize_job_id()`.

### 3.2 Infrastructure commands: filter to project clients

**File**: `nvflare/private/fed/server/training_cmds.py` (and `cmd_utils.py`)

In `validate_command_targets()`, filter target list to clients enrolled in the active project:

```python
registry = ProjectRegistryService.get_registry()
if registry and registry.is_multitenant():
    project = conn.get_prop(ConnProps.PROJECT, "default")
    for target in targets:
        if not registry.is_client_in_project(target, project):
            conn.append_error(f"Client '{target}' not in project '{project}'")
            return PreAuthzReturnCode.ERROR
```

### 3.3 `check_status`: filter response

**File**: `nvflare/private/fed/server/training_cmds.py`

Filter the client list in `check_status` response to only include clients enrolled in the user's active project.

### 3.4 Scheduler: validate deploy_map against project

**File**: `nvflare/app_common/job_schedulers/job_scheduler.py`

In `_try_job()`, after extracting applicable sites (~line 126):

```python
registry = ProjectRegistryService.get_registry()
if registry and registry.is_multitenant():
    project = job_meta.get(JobMetaKey.PROJECT.value, "default")
    project_clients = registry.get_project_clients(project)
    for site in applicable_sites:
        if site != SERVER_SITE_NAME and site not in project_clients:
            return (SCHEDULE_RESULT_BLOCK, None, f"Site {site} not in project {project}")
```

### 3.5 Job store partitioning (new jobs only)

**File**: `nvflare/apis/impl/job_def_manager.py`

Change `job_uri()` to include project for non-default projects:

```python
def job_uri(self, jid, project=None):
    if project and project != "default":
        return os.path.join(self._uri_root, project, jid)
    return os.path.join(self._uri_root, jid)  # backward compat
```

`get_all_jobs()` scans both `<root>/<uuid>/` and `<root>/<project>/<uuid>/` paths.

### 3.6 Tests

| Test | File | ~Lines |
|------|------|--------|
| ProjectRegistry unit tests | `tests/unit_test/security/project_registry_test.py` | ~200 |
| Project-scoped job filtering | `tests/unit_test/private/fed/server/job_cmds_project_test.py` | ~150 |
| Per-project role resolution | `tests/unit_test/fuel/hci/server/project_filter_test.py` | ~100 |
| Provisioner v4 parsing | `tests/unit_test/lighter/provision_v4_test.py` | ~100 |
| Scheduler project validation | `tests/unit_test/app_common/job_schedulers/scheduler_project_test.py` | ~80 |

### Milestone 3 Summary

| File | Change |
|------|--------|
| `nvflare/private/fed/server/job_cmds.py` | project gate in authorize_job_id |
| `nvflare/private/fed/server/training_cmds.py` | filter infra commands to project clients |
| `nvflare/private/fed/server/cmd_utils.py` | project-aware target validation |
| `nvflare/app_common/job_schedulers/job_scheduler.py` | deploy_map vs project clients |
| `nvflare/apis/impl/job_def_manager.py` | partitioned job_uri for new jobs |
| `tests/unit_test/...` (5 new files) | ~630 lines of tests |

**~5 files + 5 test files, ~550 lines.**

---

## Total Estimates

| Milestone | Files Changed | Files Created | Lines |
|-----------|:---:|:---:|:---:|
| M1: Plumbing | 10 | 0 | ~150 |
| M2: Registry + Authz | 10 | 1 | ~500 |
| M3: Enforcement + Tests | 5 | 5 | ~550 |
| **Total** | **~22** | **~6** | **~1,200** |

---

## Out of Scope (This Plan)

- K8s `K8sJobLauncher` changes (namespace per project, PVC selection)
- Docker `DockerJobLauncher` changes (per-project volume mounts)
- Slurm launcher changes
- Runtime project CRUD (D2 — projects defined at provision time only)
- Per-project quota management (D3 — rely on K8s ResourceQuota)
- External IdP integration (OIDC/SAML)
- Singleton refactoring (document as constraint, defer refactor)
- Shell command restrictions (see Unresolved Questions in design doc)

---

## Unresolved Questions

1. **`set_project` protocol**: does `set_project` require a new server round-trip, or can the client just switch locally and send the new project on the next command? Server round-trip is safer (validates membership) but adds latency.

2. **Token encoding**: should the project be part of the session token, or sent as a command header? Token means re-auth on project switch; header is simpler but requires server-side validation on every command.

3. **`list_jobs --all-projects`**: should `platform_admin` have a flag to see all projects' jobs? Useful for debugging but increases blast radius.

4. **Provisioner backward compat**: when `api_version: 4` project.yml is used, should the provisioner still bake a role into the cert? Options: (a) bake the first project's role as a fallback, (b) leave `UNSTRUCTURED_NAME` empty, (c) bake a sentinel value like `"multitenant"`.
