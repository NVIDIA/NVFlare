# Multi-Study CLI Management

## Table of Contents

- [Introduction](#introduction)
- [Core Principles](#core-principles)
- [Design Principles](#design-principles)
- [Key Design Decisions](#key-design-decisions)
  - [Why the CLI Uses a Simpler Model](#why-the-cli-uses-a-simpler-model)
  - [What the CLI Validates](#what-the-cli-validates)
  - [Machine-readable output](#machine-readable-output)
  - [Structured errors](#structured-errors)
  - [Exit codes](#exit-codes)
  - [No interactive prompts for agents](#no-interactive-prompts-for-agents)
  - [`--schema` on every subcommand](#--schema-on-every-subcommand)
- [Connection Flags](#connection-flags)
- [CLI Commands](#cli-commands)
  - [Study Lifecycle](#study-lifecycle)
  - [User Role Management](#user-role-management)
  - [Dataset Mapping](#dataset-mapping)
- [JSON Output Examples](#json-output-examples)
- [Error Codes](#error-codes)
- [Admin Console Commands](#admin-console-commands)
  - [Study Lifecycle Commands](#study-lifecycle-commands)
  - [User Role Commands](#user-role-commands)
- [Authorization Model](#authorization-model)
  - [Study Lifecycle Authorization](#study-lifecycle-authorization)
  - [User Role Authorization](#user-role-authorization)
  - [Role Values](#role-values)
- [Server-Side Operations](#server-side-operations)
  - [`StudyCommandModule`](#studycommandmodule)
  - [Mutation Flow](#mutation-flow)
  - [`study_registry.json` format (extended)](#study_registryjson-format-extended)
- [Validation Rules](#validation-rules)
  - [Study Name](#study-name)
  - [Sites](#sites)
  - [Users](#users)
  - [Roles](#roles)
  - [Dataset Inputs (`set-dataset` / `unset-dataset`)](#dataset-inputs-set-dataset--unset-dataset)
- [Behavioral Constraints](#behavioral-constraints)
  - [Remove](#remove)
  - [Hot-Reload](#hot-reload)
  - [Admin Self-Removal](#admin-self-removal)
  - [Reprovision Interaction](#reprovision-interaction)
- [Dataset Mapping Companion Design](#dataset-mapping-companion-design)
- [Relationship to Distributed Provisioning](#relationship-to-distributed-provisioning)
  - [End-to-End Workflow](#end-to-end-workflow)
  - [Trust Chain](#trust-chain)
- [Implementation](#implementation)
  - [New Files](#new-files)
  - [Modified Files](#modified-files)
  - [Session API](#session-api)
- [Summary](#summary)

---

## Introduction

The shipped multi-study design provisions studies statically through `project.yml` and requires a reprovision-redeploy-restart cycle to add or remove studies and to change user-role assignments. This document proposes a complementary CLI-driven management surface that allows a running server to accept dynamic study mutations without reprovisioning.

The backend for all mutations is the same `study_registry.json` file that provisioning generates. The server applies mutations through a serialized validate-write-publish flow so the runtime registry and the persisted file remain aligned under normal operation and fail closed on validation/write errors.

---

## Core Principles

1. **Same file, same format** — mutations target `study_registry.json` with `format_version: 1.0`; no new file or format version is introduced.
2. **Authoritative in-memory registry** — after a successful mutation the server hot-reloads the in-memory `StudyRegistry`; a server restart is not required.
3. **Role-based lifecycle** — `remove` requires `project_admin`; `register`, `add-site`, `remove-site`, and user-role management are accessible to `project_admin` and `org_admin`, with `org_admin` visibility and authority scoped to studies where their site is enrolled. `register` is a create-or-merge operation executed atomically under the mutation lock — see Validation Rules for the precise per-role behavior.
4. **Cert role for lifecycle operations** — study lifecycle commands check the certificate-baked role (from distributed provisioning), not the study-mapped role, consistent with the existing `must_be_project_admin` pattern for server-global operations.
5. **Best-effort job-association guard** — `remove` queries the job store before applying the deletion and rejects with `STUDY_HAS_JOBS` if any associated jobs exist at that moment. This is a best-effort guard: the mutation lock prevents two concurrent `remove` calls from racing, but it does not gate job submission. A job submitted concurrently may arrive after the guard passes and before the registry is updated. Jobs are the permanent audit trail regardless of whether the study entry still exists.
6. **Agent-usable by design** — all commands follow the same output, error, exit-code, and flag conventions as the rest of the NVFlare CLI.

---

## Design Principles

`nvflare study` follows the same agent-readiness conventions as all other `nvflare` CLI commands.

## Key Design Decisions

### Why the CLI Uses a Simpler Model

The server can reliably know the caller's identity, role, and org from the presented admin certificate at login time. What it does not have is an independent server-side participant registry. That leads to a deliberate design choice:

- both `project_admin` and `org_admin` must supply an explicit `--sites` list for every `register`, `add-site`, and `remove-site` call
- enrolled sites are accepted declaratively; the server does not verify site org membership
- target users are treated as declarative registry inputs

### What the CLI Validates

This is not "no validation." The server validates:

- caller cert role gates lifecycle and role-mutation authorization
- `org_admin` visibility and mutation authority is scoped to studies where the caller's **site is enrolled** — if the caller's site appears in the study's `sites` list, the study is visible and mutable by that org admin; explicit per-user assignment in the `admins` map is not required for visibility. **Phase 1 caveat:** site membership is accepted declaratively (the server has no independent participant registry to verify that a site name corresponds to a real connected participant). Visibility therefore reflects declared enrollment, not verified connectivity. A consequence is that if a site is removed or renamed, org-admin visibility changes implicitly — see the Phase 1 trust note below.
- study names, site names, and role values are validated syntactically
- both `project_admin` and `org_admin` must supply `--sites` for `register`, `add-site`, and `remove-site`; an empty list is rejected with `INVALID_SITE`
- job-association guard for `remove` is best-effort — see Core Principles for the race with concurrent job submission

The following are intentionally not validated:

- site org membership — site lists are accepted declaratively
- target-user org membership — user strings are stored declaratively

**Phase 1 trust note:** Because site membership is declarative, the trust strength of visibility derived from site enrollment is limited — an org_admin who can write a site name into a study gains visibility based on that claim, not on a verified participant registry. This is a known Phase 1 limitation. A future participant-registration design (`docs/design/participant_registration.md`) would back site membership with strong identity verification. Until then, the declarative model is accepted and must be documented explicitly in operator-facing documentation.

### Machine-readable output

Every command supports `--format json` (global flag registered on the top-level `nvflare` parser; may appear in any position on the command line). The JSON envelope is stable and versioned:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": { "...": "..." }
}
```

In JSON mode, stdout contains exactly one JSON envelope. Human-readable progress, warnings, and prompts go to stderr.

### Structured errors

```json
{
  "schema_version": "1",
  "status": "error",
  "error_code": "STUDY_NOT_FOUND",
  "message": "Study 'cancer-research' not found.",
  "hint": "Verify the study name. If the study exists and you expect access, contact a project_admin to have your site enrolled in the study."
}
```

### Exit codes

```
0   Success
1   Server error (study not found, unauthorized, validation rejected)
2   Connection / authentication failure
3   Timeout
4   Invalid arguments (missing required flag, bad value)
5   Internal error (unexpected exception — do not retry; report a bug)
```

### No interactive prompts for agents

Commands are either accepted or rejected directly with structured errors and exit codes; no interactive confirmation.

### `--schema` on every subcommand

Every `nvflare study` subcommand accepts `--schema` to print a machine-readable JSON description of its arguments and exit.

---

## Connection Flags

All server-backed `nvflare study` commands require a connection to the server. The startup kit location is resolved in the following priority order:

1. `--startup-kit <dir>` — explicit path to the startup kit directory (or its `startup/` subdirectory)
2. `NVFLARE_STARTUP_KIT_DIR` environment variable
3. `--startup-target poc|prod` — looks up `poc.startup_kit` or `prod.startup_kit` from `~/.nvflare/config.conf`

`~/.nvflare/config.conf` is written by `nvflare config`. If `--startup-target` is given and no entry exists for that target, the command fails with exit code 4 and a hint pointing to `nvflare config`.

`--startup-kit` and `--startup-target` are mutually exclusive. All server-backed study commands require the startup kit to be resolved via one of the three sources above; there is no silent default to `poc` because study management is a privileged operation.

---

## CLI Commands

### Study Lifecycle

`--format json` is a global flag and may appear anywhere on the command line.

```
# --sites required for both project_admin and org_admin
nvflare study register    <name> --sites <s1,s2,...> {--startup-kit <dir> | --startup-target poc|prod} [--schema]
nvflare study add-site    <name> --sites <s1,s2,...> {--startup-kit <dir> | --startup-target poc|prod} [--schema]
nvflare study remove-site <name> --sites <s1,s2,...> {--startup-kit <dir> | --startup-target poc|prod} [--schema]

# project_admin only
nvflare study remove      <name> {--startup-kit <dir> | --startup-target poc|prod} [--schema]

# project_admin (all studies) or org_admin (studies where caller's site is enrolled)
nvflare study list        {--startup-kit <dir> | --startup-target poc|prod} [--schema]
nvflare study show        <name> {--startup-kit <dir> | --startup-target poc|prod} [--schema]
```

| Command | Description | Required cert role |
|---------|-------------|-------------------|
| `register` | Create-or-merge a study. `--sites` required for both roles. Caller is recorded in the study's `admins` field with their cert role if not already present. Supplied sites are merged — new sites are added, already-enrolled sites are skipped. For `org_admin`: if the study does not exist it is created; if the study exists and the caller's site is already enrolled, the supplied `--sites` are merged in; if the study exists but the caller's site is not enrolled, returns `STUDY_ALREADY_EXISTS`. | `project_admin` or `org_admin` |
| `add-site` | Add sites to an existing study. `--sites` required for both roles. Idempotent — enrolling an already-enrolled site is a no-op. For `org_admin`: study must be visible (caller's site enrolled in `sites`); otherwise `STUDY_NOT_FOUND`. | `project_admin` or `org_admin` |
| `remove-site` | Remove sites from an existing study. `--sites` required for both roles. For `org_admin`: study must be visible (caller's site enrolled in `sites`); otherwise `STUDY_NOT_FOUND`. | `project_admin` or `org_admin` |
| `remove` | Remove a study and all its entries | `project_admin` only |
| `list` | List studies; `project_admin` sees all, `org_admin` sees only studies where their site is enrolled | `project_admin` or `org_admin` |
| `show` | Show sites and user-role assignments; `org_admin` restricted to studies where their site is enrolled | `project_admin` or `org_admin` |

#### Arguments — `register`

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<name>` | str | Yes | Study name; must match the existing `name_check(..., "study")` contract: `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$` |
| `--sites` | str | Yes | Comma-separated site names to enroll. Required for both `project_admin` and `org_admin`. An empty list is rejected with `INVALID_SITE`. |
| `--startup-kit` | str | No* | Explicit path to the startup kit directory |
| `--startup-target` | `poc\|prod` | No* | Resolves startup kit path from `~/.nvflare/config.conf` |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* `--startup-kit` and `--startup-target` are mutually exclusive. The startup kit must be resolvable — via `--startup-kit`, `--startup-target`, or `NVFLARE_STARTUP_KIT_DIR` — see [Connection Flags](#connection-flags). No silent default: if no source resolves, the command fails.

#### Usage Examples — `register`

**org_admin registers a new study**

```bash
# logged in as org_admin@org_a.com
nvflare study register cancer-research --sites hospital-a --startup-target prod
```

**org_admin re-registers an existing study they belong to (merge — adds supplied sites not yet enrolled)**

```bash
# logged in as org_admin@org_a.com; caller's site is enrolled in the study
nvflare study register cancer-research --sites hospital-b --startup-target prod
```

**project_admin registers a cross-org study**

`project_admin` may enroll sites from any org in a single call.

```bash
# logged in as admin@nvidia.com (project_admin)
nvflare study register cancer-research --sites hospital-a,hospital-b,hospital-c --startup-target prod
```

#### Arguments — `add-site` and `remove-site`

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<name>` | str | Yes | Existing study name |
| `--sites` | str | Yes | Comma-separated site names. Required for both `project_admin` and `org_admin`. An empty list is rejected with `INVALID_SITE`. |
| `--startup-kit` | str | No* | Explicit path to the startup kit directory |
| `--startup-target` | `poc\|prod` | No* | Resolves startup kit path from `~/.nvflare/config.conf` |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* `--startup-kit` and `--startup-target` are mutually exclusive. The startup kit must be resolvable — via `--startup-kit`, `--startup-target`, or `NVFLARE_STARTUP_KIT_DIR` — see [Connection Flags](#connection-flags). No silent default: if no source resolves, the command fails.

Both commands return per-site outcome lists. `add-site`: already-enrolled sites are reported in `already_enrolled` and skipped; newly enrolled sites appear in `added`. `remove-site`: sites not currently enrolled are reported in `not_enrolled` and skipped; removed sites appear in `removed`. Neither command errors on partially overlapping input — the full outcome is always inspectable in the response.

#### Arguments — `remove`

`remove` requires `project_admin` cert role. `org_admin` callers are rejected with `NOT_AUTHORIZED`.

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<name>` | str | Yes | Study name to remove |
| `--startup-kit` | str | No* | Explicit path to the startup kit directory |
| `--startup-target` | `poc\|prod` | No* | Resolves startup kit path from `~/.nvflare/config.conf` |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* `--startup-kit` and `--startup-target` are mutually exclusive. The startup kit must be resolvable — via `--startup-kit`, `--startup-target`, or `NVFLARE_STARTUP_KIT_DIR` — see [Connection Flags](#connection-flags). No silent default: if no source resolves, the command fails.

### User Role Management

```
nvflare study add-user    <study> <user> --role <role> {--startup-kit <dir> | --startup-target poc|prod} [--schema]
nvflare study remove-user <study> <user> {--startup-kit <dir> | --startup-target poc|prod} [--schema]
nvflare study update-user <study> <user> --role <role> {--startup-kit <dir> | --startup-target poc|prod} [--schema]
```

| Command | Description | Required cert role |
|---------|-------------|-------------------|
| `add-user` | Add a user with a role to a registered study | `project_admin` (any study) or `org_admin` (studies where caller's site is enrolled) |
| `remove-user` | Remove a user's role entry from a study | `project_admin` (any study) or `org_admin` (studies where caller's site is enrolled) |
| `update-user` | Change a user's role in a study | `project_admin` (any study) or `org_admin` (studies where caller's site is enrolled) |

#### Arguments — user role commands

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<study>` | str | Yes | Registered study name |
| `<user>` | str | Yes | Username or cert CN string to store in the study mapping |
| `--role` | str | Yes (add, update) | One of: `project_admin`, `org_admin`, `lead`, `member` |
| `--startup-kit` | str | No* | Explicit path to the startup kit directory |
| `--startup-target` | `poc\|prod` | No* | Resolves startup kit path from `~/.nvflare/config.conf` |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* `--startup-kit` and `--startup-target` are mutually exclusive. The startup kit must be resolvable — via `--startup-kit`, `--startup-target`, or `NVFLARE_STARTUP_KIT_DIR` — see [Connection Flags](#connection-flags). No silent default: if no source resolves, the command fails.

#### Usage Examples — user role commands

**org_admin adds a user from their own org to a study**

The org admin adds `trainer@org_a.com` as `lead` to a study where the caller's site is enrolled. The target user string is stored declaratively.

```bash
# logged in as org_admin@org_a.com
nvflare study add-user cancer-research trainer@org_a.com --role lead --startup-target prod
```

**org_admin adds another user-role entry in their study**

```bash
# logged in as org_admin@org_a.com
nvflare study add-user cancer-research analyst@org_a.com --role member --startup-target prod
```

**project_admin adds a user from any org**

```bash
# logged in as admin@nvidia.com (project_admin)
nvflare study add-user cancer-research trainer@org_b.com --role lead --startup-target prod
```

**project_admin adds a cross-org user using an explicit startup kit path**

```bash
nvflare study add-user cancer-research analyst@org_c.com --role member --startup-kit /opt/nvflare/admin_startup
```

**org_admin updates a user's role**

```bash
# logged in as org_admin@org_a.com
nvflare study update-user cancer-research trainer@org_a.com --role member --startup-target prod
```

**org_admin removes a user**

```bash
# logged in as org_admin@org_a.com
nvflare study remove-user cancer-research trainer@org_a.com --startup-target prod
```

### Dataset Mapping

`set-dataset` and `unset-dataset` are local file operations — they do not connect to the server. On-disk format, launcher behavior, validation rules, migration guidance, and companion code changes are defined in `docs/design/study_dataset_mapping.md`.

```
nvflare study set-dataset   <study> <dataset> {--data-path <path> | --pvc <name>} --mode ro|rw [--startup-kit <dir>] [--format json] [--schema]
nvflare study unset-dataset <study> <dataset> [--startup-kit <dir>] [--format json] [--schema]
```

Dataset commands follow the same conventions as all other `nvflare study` commands:

- **Local only** — no server connection; writes directly to `local/study_data.json` in the resolved startup kit directory
- **Same JSON envelope** — `--format json` produces `{"schema_version": "1", "status": "ok", "data": {...}}` or the structured error envelope
- **Same exit codes** — 0 success, 1 error, 4 invalid arguments, 5 internal
- **Same `--schema` flag** — machine-readable argument description, exits immediately
- **Fail-closed on validation** — no file is written until all inputs are validated
- **Idempotent writes** — `set-dataset` is an upsert; calling it twice with the same args is a no-op from the operator's perspective
- **Atomic file writes** — both `set-dataset` and `unset-dataset` write to a temp file in the same directory as `study_data.json`, then rename via `os.replace()`; an interrupted write never leaves a partial or corrupted JSON file

#### Arguments — `set-dataset`

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<study>` | str | Yes | Study name; must match `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$`. The regex excludes `/`, `.`, and special characters to ensure the name is safe as a filesystem path component. |
| `<dataset>` | str | Yes | Dataset name; must match `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$`. Same path-safety guarantee as the study name — used directly as a path component in `/data/<study>/<dataset>`. |
| `--startup-kit` | str | No | Path to the startup-kit root directory; used to locate `local/study_data.json`. If omitted, resolved from `NVFLARE_STARTUP_KIT_DIR` env var. If neither source provides a path, the command fails with `STARTUP_KIT_REQUIRED` (exit 4). |
| `--data-path` | str | Yes* | Docker/subprocess deployment: stored as the `source` field. For Docker, provide the host-absolute path to the dataset directory (e.g. `/host/data/cancer-train`). For subprocess, use `/data/<study>/<dataset>` — the path where the operator will pre-place data on the host. Accepted declaratively; no path-traversal or existence check at CLI time. |
| `--pvc` | str | Yes* | Kubernetes deployment: PVC claim name. Stored as the `source` field. Must satisfy Kubernetes resource name rules: `^[a-z0-9](?:[a-z0-9-]{0,251}[a-z0-9])?$`; rejected with `INVALID_DATASET` if malformed. |
| `--mode` | `ro\|rw` | Yes | Access mode: `ro` for read-only input data; `rw` for read-write staging/output. Any other value is rejected with `INVALID_MODE`. |
| `--format` | `json` | No | Emit structured JSON envelope instead of human-readable output |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* Exactly one of `--data-path` or `--pvc` is required. Both write to the same `source` field in `study_data.json`; the launcher interprets the value based on its own deployment context.

#### Arguments — `unset-dataset`

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<study>` | str | Yes | Study name; must match `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$` — path-safe, same rule as `set-dataset` |
| `<dataset>` | str | Yes | Dataset name; must match `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$` — path-safe, same rule as `set-dataset` |
| `--startup-kit` | str | No | Path to the startup-kit root directory; used to locate `local/study_data.json`. If omitted, resolved from `NVFLARE_STARTUP_KIT_DIR` env var. If neither source provides a path, the command fails with `STARTUP_KIT_REQUIRED` (exit 4). |
| `--format` | `json` | No | Emit structured JSON envelope |
| `--schema` | flag | No | Print command schema as JSON and exit |

`unset-dataset` does not accept `--data-path`, `--pvc`, or `--mode`. It removes the named dataset entry regardless of its current contents. `unset-dataset` is idempotent: if `study_data.json` does not exist, the study key is absent, or the dataset key is absent, the command succeeds with exit 0 and returns `"removed": false` in the response envelope — no error is raised for missing state.

#### Usage Examples

```bash
# Startup kit resolved from NVFLARE_STARTUP_KIT_DIR
nvflare study set-dataset cancer-research training --data-path /host/data/cancer-train --mode ro

# Explicit startup kit path
nvflare study set-dataset cancer-research training --startup-kit /opt/nvflare/hospital-a --data-path /host/data/cancer-train --mode ro

# Kubernetes site — PVC
nvflare study set-dataset cancer-research staging --startup-kit /opt/nvflare/hospital-a --pvc cancer-stage-pvc --mode rw

# Remove a dataset entry
nvflare study unset-dataset cancer-research staging --startup-kit /opt/nvflare/hospital-a
```

#### JSON Output Examples

**`set-dataset` — new entry created:**

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "dataset": "training",
    "created": true,
    "entry": {
      "source": "/host/data/cancer-train",
      "mode": "ro"
    }
  }
}
```

**`set-dataset` — existing entry updated (upsert):**

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "dataset": "training",
    "created": false,
    "entry": {
      "source": "/host/data/cancer-train",
      "mode": "ro"
    }
  }
}
```

**`set-dataset` — Kubernetes site (PVC name as `source`):**

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "dataset": "staging",
    "created": true,
    "entry": {
      "source": "cancer-stage-pvc",
      "mode": "rw"
    }
  }
}
```

**`unset-dataset` — success (entry existed and was removed):**

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "dataset": "staging",
    "removed": true
  }
}
```

**`unset-dataset` — success (entry was already absent; idempotent):**

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "dataset": "staging",
    "removed": false
  }
}
```

**Error — invalid mode:**

```json
{
  "schema_version": "1",
  "status": "error",
  "error_code": "INVALID_MODE",
  "message": "--mode must be 'ro' or 'rw'; got 'readonly'.",
  "hint": "Use --mode ro for read-only input data or --mode rw for read-write staging/output."
}
```

#### Schema Gate

Until the companion launcher change is merged, `set-dataset` and `unset-dataset` handlers return a structured error rather than writing any file. This ensures the CLI surface is available for `--schema` and `--help` discovery without writing a format the launchers cannot yet consume.

```json
{
  "schema_version": "1",
  "status": "error",
  "error_code": "NOT_YET_IMPLEMENTED",
  "message": "Dataset commands are pending the companion launcher change.",
  "hint": "This command will be enabled once the unified study_data.json launcher support is merged."
}
```

---

## JSON Output Examples

### `nvflare study list`

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "studies": ["cancer-research", "multiple-sclerosis"]
  }
}
```

### `nvflare study show <name>`

The `"users"` key in the response corresponds to the `"admins"` map in `study_registry.json`. The CLI response uses `"users"` to avoid exposing the internal file field name.

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "name": "cancer-research",
    "sites": ["hospital-a", "hospital-b"],
    "users": {
      "admin@nvidia.com": "project_admin",
      "trainer@org_a.com": "lead"
    }
  }
}
```

### `nvflare study register <name>`

The response reflects the full resulting state of the study. The `"users"` map is the complete `admins` map after the operation — not just the newly inserted caller.

**New study created** (caller is the only entry):

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "name": "cancer-research",
    "sites": ["hospital-a"],
    "users": {
      "org_admin@org_a.com": "org_admin"
    }
  }
}
```

**Merge call** — `org_admin` on an existing study they already belong to (supplied sites merged in, all pre-existing entries preserved):

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "name": "cancer-research",
    "sites": ["hospital-a", "hospital-b"],
    "users": {
      "org_admin@org_a.com": "org_admin",
      "existing@org_b.com": "lead"
    }
  }
}
```

### `nvflare study add-site <study>`

The response always returns per-site outcome lists so automation can distinguish newly enrolled sites from already-enrolled ones. `sites` is the complete resulting enrollment list for the study.

**Single new enrollment:**
```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "added": ["hospital-b"],
    "already_enrolled": [],
    "sites": ["hospital-a", "hospital-b"]
  }
}
```

**Multi-site call (mix of new and already-enrolled):**
```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "added": ["hospital-b", "hospital-c"],
    "already_enrolled": ["hospital-a"],
    "sites": ["hospital-a", "hospital-b", "hospital-c"]
  }
}
```

**All sites already enrolled (pure no-op):**
```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "added": [],
    "already_enrolled": ["hospital-a"],
    "sites": ["hospital-a"]
  }
}
```

### `nvflare study remove <name>`

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "name": "cancer-research",
    "removed": true
  }
}
```

### `nvflare study add-user <study> <user> --role <role>`

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "user": "trainer@org_a.com",
    "role": "lead"
  }
}
```

### `nvflare study update-user <study> <user> --role <role>`

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "user": "trainer@org_a.com",
    "previous_role": "lead",
    "role": "member"
  }
}
```

### `nvflare study remove-user <study> <user>`

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "user": "trainer@org_a.com",
    "removed": true
  }
}
```

### `nvflare study remove-site <study> --sites <s1,s2,...>`

The response returns per-site outcome lists. `sites` is the complete resulting enrollment list after removal.

**Multi-site call (mix of removed and not enrolled):**
```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "removed": ["hospital-b"],
    "not_enrolled": ["hospital-c"],
    "sites": ["hospital-a"]
  }
}
```

**Single site removed:**
```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "removed": ["hospital-a"],
    "not_enrolled": [],
    "sites": []
  }
}
```

### Error — study has associated jobs

```json
{
  "schema_version": "1",
  "status": "error",
  "error_code": "STUDY_HAS_JOBS",
  "message": "Study 'cancer-research' has 3 associated job(s) and cannot be removed.",
  "hint": "Archive or delete the associated jobs before retrying."
}
```

### Error — not authorized

```json
{
  "schema_version": "1",
  "status": "error",
  "error_code": "NOT_AUTHORIZED",
  "message": "Role 'lead' is not permitted to register a study.",
  "hint": "Log in with a project_admin or org_admin startup kit."
}
```

---

## Error Codes

| Error code | Exit | Meaning |
|------------|------|---------|
| `STUDY_NOT_FOUND` | 1 | Study name does not exist in the registry, or exists but is not visible to the caller (`org_admin`'s site not enrolled in the study's `sites` list) — the same code is used for both cases intentionally to avoid leaking existence. Exception: `register` does NOT use this code (see `STUDY_ALREADY_EXISTS`). |
| `STUDY_ALREADY_EXISTS` | 1 | Study name already registered. Returned by `register` when the study exists but the caller's site is not enrolled. `register` is not a join path — returning `STUDY_NOT_FOUND` on a create would be semantically backwards and confusing. The caller already knows the name; learning "it is taken" reveals nothing actionable. |
| `STUDY_HAS_JOBS` | 1 | Study has associated jobs; `remove` rejected |
| `INVALID_STUDY_NAME` | 4 | Study name fails the name-validation regex or is `"default"` |
| `INVALID_ROLE` | 4 | Role value is not one of the built-in roles |
| `INVALID_SITE` | 4 | A site name in `--sites` is malformed (fails the site-name validator), or `--sites` is an empty list when it is required |
| `SITE_NOT_IN_STUDY` | — | Removed: `remove-site` no longer rejects on non-enrolled sites. Non-enrolled sites are reported in `not_enrolled` in the response and skipped. |
| `ORG_NOT_FOUND` | 4 | Reserved for a future participant-registry-backed validation flow |
| `USER_ALREADY_IN_STUDY` | 1 | `add-user` rejected: user already has a role in this study |
| `USER_NOT_IN_STUDY` | 1 | `remove-user` / `update-user` rejected: user has no role in this study |
| `NOT_AUTHORIZED` | 1 | Caller's cert role is insufficient for this operation |
| `STARTUP_KIT_NOT_CONFIGURED` | 4 | `--startup-target` given but no matching entry in `~/.nvflare/config.conf` |
| `LOCK_TIMEOUT` | 3 | Mutation lock could not be acquired within 30 seconds — another mutation is in progress |
| `CONNECTION_FAILED` | 2 | Cannot connect to or authenticate with the server |

All dataset-command error codes (`INVALID_DATASET`, `INVALID_MODE`, `MISSING_REQUIRED_FLAG`, `STARTUP_KIT_REQUIRED`, `INVALID_STARTUP_KIT`, `DATA_PATH_NOT_FOUND`, `BACKEND_FIELD_MISSING`, `NOT_YET_IMPLEMENTED`, and the dataset-context `STUDY_NOT_FOUND`) are defined in `docs/design/study_dataset_mapping.md`.

---

## Admin Console Commands

The CLI translates each user command into an admin console command sent over the authenticated admin connection. The server-side command handlers live in a new `StudyCommandModule`.

### Study Lifecycle Commands

| Admin command | Arguments | Handler |
|---------------|-----------|---------|
| `register_study` | `<name> --sites s1,s2,...` | `StudyCommandModule.cmd_register_study` — records caller in `admins` with cert role if not already present; merges supplied sites (adds new, skips already-enrolled). `--sites` required for both roles. Three cases: (1) study does not exist → create, record caller in `admins`, add sites; (2) study exists and caller's site is enrolled → merge supplied sites in, succeed; (3) study exists but caller's site is not enrolled → `STUDY_ALREADY_EXISTS`. |
| `add_study_site` | `<name> --sites s1,s2,...` | `StudyCommandModule.cmd_add_study_site`. `--sites` required for both roles. For `org_admin` callers: study must be visible (caller's site must be enrolled); otherwise `STUDY_NOT_FOUND`. |
| `remove_study_site` | `<name> --sites s1,s2,...` | `StudyCommandModule.cmd_remove_study_site`. `--sites` required for both roles. For `org_admin` callers: study must be visible (caller's site must be enrolled); otherwise `STUDY_NOT_FOUND`. |
| `remove_study` | `<name>` | `StudyCommandModule.cmd_remove_study` — rejected if any jobs are associated |
| `list_studies` | _(none)_ | `StudyCommandModule.cmd_list_studies` |
| `show_study` | `<name>` | `StudyCommandModule.cmd_show_study` |

### User Role Commands

| Admin command | Arguments | Handler |
|---------------|-----------|---------|
| `add_study_user` | `<study> <user> <role>` | `StudyCommandModule.cmd_add_study_user` |
| `remove_study_user` | `<study> <user>` | `StudyCommandModule.cmd_remove_study_user` |
| `update_study_user` | `<study> <user> <role>` | `StudyCommandModule.cmd_update_study_user` |

---

## Authorization Model

### Study Lifecycle Authorization

Study lifecycle commands check the certificate-baked role. The study-mapped role is **not** substituted, consistent with other server-global operations.

```
remove_study
    → require cert role == "project_admin" only

register_study, add_study_site, remove_study_site
    → allowed for cert role in {project_admin, org_admin}

list_studies, show_study
    → project_admin: unrestricted (all studies)
    → org_admin: scoped to studies where caller's site is enrolled in the study's `sites` list
```

**Scoping for non-project-admin callers (`org_admin`):**

- `register_study`: `--sites` required for both roles; caller is recorded in the `admins` field with their cert role if not already present; supplied sites merged in (new sites added, already-enrolled skipped). For `org_admin`: if the study does not exist, it is created; if the study exists and the caller's site is already enrolled, supplied sites are merged in and the call succeeds; if the study exists but the caller's site is not enrolled, returns `STUDY_ALREADY_EXISTS`.
- `add_study_site`: `--sites` required for both roles; already-enrolled site is a no-op; sites accepted declaratively. For `org_admin`: the study must be visible (caller's site must appear in the study's `sites` list); if not, return `STUDY_NOT_FOUND` — do not reveal existence.
- `remove_study_site`: `--sites` required for both roles. For `org_admin` callers: the study must be visible (caller's site must appear in the study's `sites` list; otherwise `STUDY_NOT_FOUND`). Sites accepted declaratively.
- `list_studies`: returns only studies where the caller's site is enrolled in the study's `sites` list.
- All other targeted commands (`show_study`, `add_study_site`, `remove_study_site`, `add_study_user`, `remove_study_user`, `update_study_user`): if the caller's site does not appear in the target study's `sites` list, return `STUDY_NOT_FOUND` — do not reveal existence.

`project_admin` sees all studies and may operate on any study without restriction.

Any role not listed above is rejected with `NOT_AUTHORIZED`.

### User Role Authorization

User role commands apply a two-tier check:

1. **`project_admin` (cert)** — may add, update, or remove any user in any study.
2. **`org_admin` (cert)** — may add, update, or remove users only for studies where their site is enrolled in the study's `sites` list. Target user strings are stored declaratively; robust target-user org validation is deferred to participant-registration support.

```
add_study_user, remove_study_user, update_study_user
    → cert role == "project_admin"       → allowed (any org)
    → cert role == "org_admin"           → allowed for studies where caller's site is enrolled
    → other roles                        → NOT_AUTHORIZED
```

### Role Values

Valid role values for user-role commands are the existing built-in roles:

- `project_admin`
- `org_admin`
- `lead`
- `member`

These role values are stored in the study's `admins` field and carry real weight — they are not cosmetic metadata. There are two distinct authorization layers, each consulting a different part of the study record:

- **Named-study login** (an admin connecting to a study to submit or manage jobs) — gated by `admins`: the user must have an entry in that study's `admins` mapping or login is rejected. The stored role value becomes the effective role for all study-scoped authorization within that session. This is defined in `docs/design/multistudy.md` and is not changed by this CLI design.
- **CLI management commands** (`register`, `add-site`, `remove-site`, `list`, `show`, and user-role mutations) — gated by **site enrollment**: an `org_admin` can run these commands for studies where their site appears in the `sites` list, regardless of whether they personally appear in `admins`.

`add-user` and `update-user` are therefore not cosmetic: they determine who can open a named-study session and what effective role they carry inside it.

Role values are distinct from the cert role baked into the admin certificate. Assigning any role value (including `project_admin` or `org_admin`) to a study user grants no server-global privileges; the cert role remains the sole gate for server-global and lifecycle operations. An `org_admin` may assign any of these role values to study users without restriction.

---

## Server-Side Operations

### `StudyCommandModule`

A new `StudyCommandModule` (parallel to `SystemCommandModule` and `JobCommandModule`) registers the study management commands. It embeds `CommandUtil` and holds a reference to the mutable registry service.

### Mutation Flow

All mutating commands follow the same serialized pattern:

1. **Authorize** — check cert role; reject if insufficient.
2. **Validate** — check that the study name, user, role, and sites are syntactically valid before modifying state.
3. **Acquire mutation lock** — a process-local lock serializes all registry mutations so concurrent admin commands cannot lose updates. The acquire call has a 30-second timeout; if the lock is not acquired within that window, the command returns immediately with `LOCK_TIMEOUT` (exit 3). The lock is always released in a `finally` block — a failure during steps 4–9 cannot leave the lock permanently held.
4. **Load current state** — read `study_registry.json` from disk into a working copy while holding the lock.
5. **Guard** — for `remove_study`, query the job store (while holding the lock) for any job tagged with the study name; reject with `STUDY_HAS_JOBS` if any exist. Running inside the lock eliminates the TOCTOU window between two concurrent `remove` calls, but does not prevent a job submission that races in after this check — see the design limitation note below.
6. **Apply mutation** — modify the working copy in memory. Only `register_study` mutates the `admins` map (auto-insert caller with cert role if not already present; existing entries are preserved). `add_study_site` and `remove_study_site` mutate only the `sites` list and never touch `admins`.
7. **Validate resulting config** — construct a new `StudyRegistry` from the updated working copy before touching the live registry pointer.
8. **Write atomically** — write the validated config to a temp file in the same directory, then `os.replace()` to the final path.
9. **Publish hot-reload** — call `StudyRegistryService.initialize(new_registry)` while still holding the lock.
10. **Return result** — reply to the admin connection with the updated study state as a JSON envelope.

If validation of the post-mutation config fails, nothing is written and the live registry remains unchanged. If the disk write fails, nothing is published to the live registry. This still does not make the file update and pointer swap literally atomic across all failure modes, but it avoids the main split-brain case of publishing an invalid registry and makes the rollback behavior explicit.

**Design limitation:** The mutation lock serializes concurrent registry mutations (e.g., two `register` or `remove` calls at the same time). It does not gate job submissions — a job may be submitted to a study being concurrently removed if the submission races the `remove` mutation. The job-association guard (step 5) checks the job store at the time of the guard; any job submitted after that point is not prevented by the lock.

### `study_registry.json` format (extended)

```json
{
  "format_version": "1.0",
  "studies": {
    "cancer-research": {
      "sites": ["hospital-a", "hospital-b"],
      "admins": {
        "admin@nvidia.com": "project_admin",
        "trainer@org_a.com": "lead"
      }
    }
  }
}
```

Mutations only add, modify, or remove entries under `"studies"`. The format is identical to what provisioning generates from `project.yml`. The `"format_version"` field is never changed.

---

## Validation Rules

### Study Name

- Must match the existing runtime validator `name_check(..., "study")`: `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$`
- The regex is chosen to guarantee safe use as a filesystem path component: it excludes `/`, `.`, `..`, spaces, and all shell-special characters, so names can be safely joined into paths such as `/data/<study>/<dataset>` without sanitization
- Dataset names must satisfy the same regex for the same reason — both appear as path components in the data mount path
- `default` is reserved and always rejected with `INVALID_STUDY_NAME`
- `register` is a **create-or-merge** operation, not a pure create. The entire check-then-mutate sequence runs under the mutation lock so two concurrent `register` calls on the same new study name cannot both pass the existence check and produce inconsistent state.

  `register` always applies **merge semantics** on an existing study: supplied sites that are not yet enrolled are added; already-enrolled sites are silently skipped. The caller is recorded in `admins` with their cert role if not already present. `register` never removes sites or `admins` entries — removal requires explicit `remove-site` or `remove-user`.

  **`org_admin` behavior:**
  - Study does not exist → create, record caller in `admins`, add supplied sites. Self-assignment at creation is not a security bypass — the study did not exist before this call, so there is no pre-existing access list to circumvent.
  - Study exists, caller's site already enrolled → merge supplied sites in; succeed.
  - Study exists, caller's site NOT enrolled → `STUDY_ALREADY_EXISTS` (name is taken and caller has no access; `project_admin` must enroll the site first). `register` is not a join path.

  **`project_admin` behavior on existing studies:**
  - Same merge semantics: add supplied sites not yet enrolled; skip already-enrolled sites; record self in `admins` if not already present.

- Unknown names rejected on `remove_study`, `show_study`, and all user-role commands with `STUDY_NOT_FOUND`

### Sites

- both `project_admin` and `org_admin` must supply `--sites` for `register`, `add-site`, and `remove-site`; an empty list or a malformed site name is rejected with `INVALID_SITE`
- Each site name must pass the site-name validator: `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$` (same regex as study names, using `name_check(..., "site")`)
- Site lists are accepted declaratively; the server does not verify site org membership
- Site lists are deduplicated

### Users

- Target user strings are accepted as declarative study membership; the server does not reject unknown users just because they have not connected yet
- For `org_admin` callers: same-org mutation remains the intended policy, but robust target-user org validation requires the participant-registration design described in `docs/design/participant_registration.md`

### Roles

- Must be one of: `project_admin`, `org_admin`, `lead`, `member`; invalid values are rejected with `INVALID_ROLE`
- `add_study_user` rejects a user already present in the study's `admins` map with `USER_ALREADY_IN_STUDY`
- `update_study_user` and `remove_study_user` reject a user not present in the study's `admins` map with `USER_NOT_IN_STUDY`

### Dataset Inputs (`set-dataset` / `unset-dataset`)

Both `<study>` and `<dataset>` are used directly as filesystem path components (e.g. `/data/<study>/<dataset>`). Both must match `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$` — the same path-safe regex as study names. Full validation rules and error codes are defined in `docs/design/study_dataset_mapping.md`.

---

## Behavioral Constraints

### Remove

- **Best-effort job-association guard** — `remove` queries the job store at guard time and returns `STUDY_HAS_JOBS` (exit 1) if any job tagged with the study name exists. This check is not atomic with job submission: a job submitted concurrently may arrive after the guard passes. See Core Principles for the full limitation.
- Existing authenticated sessions continue to run with their session snapshot. New logins to the removed study are rejected immediately after the reload.
- Session eviction is deferred unless a clean `SessionManager` integration is added; `remove` does not disconnect existing sessions.

### Hot-Reload

After any successful mutation the in-memory `StudyRegistry` is replaced through the serialized publish step described above. Sessions that are already authenticated continue using their session-creation snapshot; the new registry takes effect for subsequent logins.

- A user added to a study can log in immediately without a server restart.
- A user removed from a study cannot open a new session for that study, but an existing authenticated session is not forcibly terminated.

### Admin Self-Removal

`remove-user` does not guard against a caller removing themselves from the study's `admins` field. Note that `org_admin` visibility is derived from site enrollment, not from the `admins` field — removing one's own `admins` entry does not lose visibility as long as the site remains enrolled. However, it does remove any within-study role record for that user. Operators should take care when removing user entries that carry meaningful study-level roles.

### Reprovision Interaction

Dynamic mutations through these commands do not modify `project.yml`. A subsequent reprovision will overwrite `study_registry.json` with whatever is in `project.yml`. If dynamic mutations need to survive a reprovision, the operator must backport them to `project.yml`.

---

## Dataset Mapping Companion Design

See `docs/design/study_dataset_mapping.md` for the full companion design: on-disk format, launcher behavior, validation, migration, and required code changes.

---

## Relationship to Distributed Provisioning

The distributed provisioning workflow (`docs/design/distributed_provisioning.md`) decentralizes the **identity layer**: each site generates its own private key, sends only a CSR to the Project Admin, and receives a signed certificate in return. Once the site packages its startup kit and connects to the server, it is a fully authenticated participant — without the Project Admin ever holding its private key, and without reprovisioning any other site.

The multi-study CLI decentralizes the **authorization layer**: once a site is authenticated, its org admin can register studies, enroll sites, and assign user roles — all without reprovisioning and without Project Admin involvement. **Phase 1 caveat:** site enrollment is accepted declaratively; the server does not independently verify that a site name in `--sites` corresponds to an authenticated participant. The org admin's cert proves their identity and role; it does not independently validate the site list they supply.

Together, the two capabilities make a **fully decentralized federation lifecycle** possible:

| Operation | Mechanism | Reprovisioning required? |
|-------|-----------|--------------------------|
| Site joins the federation | `nvflare cert csr` → `nvflare cert sign` → `nvflare package` | No — cert signing is the act of authorization |
| Study created by org admin | `nvflare study register --sites` (org admin supplies explicit site list) | No |
| Users added to the study | `nvflare study add-user` | No |
| Cross-org study created | `nvflare study register --sites` (project admin, sites from multiple orgs) | No |
| Site removed from federation | Certificate revocation (out of scope) | Yes — currently requires reprovisioning |

### End-to-End Workflow

A new site joins and sets up its own study without any central reprovisioning cycle:

```
# Step 1 — Site Admin generates a CSR (once per participant)
nvflare cert csr -n hospital-1 -t client -o ./csr

# Step 2 — Project Admin signs it
nvflare cert sign -r hospital-1.csr -c ./ca -o ./signed --accept-csr-role

# Step 3 — Site Admin packages and starts
nvflare package -e grpc://fl-server:8002 --dir ./csr
cd hospital-1 && ./startup/start.sh

# Step 4 — Org Admin generates their admin CSR and startup kit (same flow)
nvflare cert csr -n admin@org_a.com -t org_admin -o ./admin-csr
# ... sign, package, start admin console ...

# Step 5 — Org Admin registers a study with their site
nvflare study register cancer-research --sites hospital-1 --startup-target prod

# Step 6 — Org Admin adds a lead researcher from their org
nvflare study add-user cancer-research researcher@org_a.com --role lead --startup-target prod

# Step 7 — Lead submits a job to the study
nvflare job submit -j ./my_job --startup-target prod --study cancer-research
```

Steps 1–3 follow the distributed provisioning workflow. Steps 4–7 use the multi-study CLI. The Project Admin is only involved in step 2 (signing the CSR) — the org admin controls the rest independently.

### Trust Chain

The distributed provisioning design already establishes the trust chain for certificate roles:

- The cert type (`project_admin`, `org_admin`, `lead`, `member`) is embedded in `UNSTRUCTURED_NAME` at signing time
- The server reads this at login and uses it as the effective cert role
- The multi-study CLI uses this same cert role to gate study lifecycle and user-role operations

The cert is the gate for **who can call** study lifecycle and user-role commands — it establishes identity and cert role at login. However, the cert alone does not determine **which studies** an `org_admin` can act on. That authority derives from site enrollment in `study_registry.json`, which is accepted declaratively (see Phase 1 trust note in the Key Design Decisions section). An `org_admin` who can write a site name into a study gains management authority over it based on that declared enrollment, not on a cryptographically verified participant registry. No additional trust infrastructure is added in Phase 1; strengthening this is deferred to the participant-registration design.

---

## Implementation

### New Files

| File | Purpose |
|------|---------|
| `nvflare/tool/study/__init__.py` | Package |
| `nvflare/tool/study/study_cli.py` | `nvflare study` subcommand handlers, including local dataset-mapping commands |
| `nvflare/private/fed/server/study_cmds.py` | `StudyCommandModule` server-side handlers |

### Modified Files

| File | Change |
|------|--------|
| `nvflare/cli.py` | Register `nvflare study` top-level subcommand |
| `nvflare/private/fed/server/server_cmd_modules.py` | Register `StudyCommandModule` |
| `nvflare/fuel/flare_api/flare_api.py` | Add `register_study`, `add_study_site`, `remove_study_site`, `remove_study`, `list_studies`, `show_study`, `add_study_user`, `remove_study_user`, `update_study_user` session methods |
| `nvflare/security/study_registry.py` | Add mutation lock support or companion synchronization helper for serialized registry updates |

### Session API

Server-backed handlers (`register`, `add-site`, `remove-site`, `remove`, `list`, `show`, `add-user`, `remove-user`, `update-user`) use `new_secure_session()` for server connectivity, identical to `nvflare job` and `nvflare system`. New session methods are thin wrappers over `AdminAPI.do_command()`.

`set-dataset` and `unset-dataset` are purely local file operations and do **not** use `new_secure_session()` or open any server connection. They resolve the startup-kit directory from `--startup-kit` (explicit path) or `NVFLARE_STARTUP_KIT_DIR` env var, then read and write `local/study_data.json` directly. `--startup-target` is not accepted — poc/prod environment selection is a server-connection concept and does not apply to local file operations.

---

## Summary

| Area | Design |
|------|--------|
| Study lifecycle commands | `register`, `add-site`, `remove-site`, `remove`, `list`, `show` under `nvflare study` |
| User role commands | `add-user`, `remove-user`, `update-user` under `nvflare study` |
| Study lifecycle authorization | `remove`: `project_admin` only. `register`, `add-site`, `remove-site`: `project_admin` or `org_admin` (`--sites` required for both; `org_admin` scoped to studies where their site is enrolled). `list`/`show`: `project_admin` sees all; `org_admin` sees only studies where their site is enrolled |
| User role authorization | `project_admin` (any study); `org_admin` (studies where caller's site is enrolled) |
| Backend file | `study_registry.json` with `format_version: 1.0` |
| Dataset mapping | Included in the CLI surface; on-disk schema, launcher semantics, and CLI commands are specified in `docs/design/study_dataset_mapping.md` |
| Persistence | Serialized mutation flow with lock, temp file, `os.replace()`, then registry publish |
| In-memory reload | Hot-reload after each mutation; no server restart required |
| Job-association guard | Best-effort: `remove` queries the job store at guard time and rejects with `STUDY_HAS_JOBS` if any tagged job exists; not atomic with concurrent job submission |
| Active-session behavior | Existing authenticated sessions unaffected; forced session eviction deferred until a clean session-manager integration is designed |
| Output format | JSON envelope (`schema_version`, `status`, `data`) via `--format json`; human text by default |
| Exit codes | 0 success, 1 server error, 2 connection failure, 3 timeout, 4 invalid args, 5 internal |
| Agent usability | `--schema` on every subcommand; no interactive prompts |
| Reprovision interaction | `project.yml` overwrites `study_registry.json` on reprovision; dynamic mutations must be backported to `project.yml` to survive |
