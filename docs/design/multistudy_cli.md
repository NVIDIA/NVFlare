# Multi-Study CLI Management

## Table of Contents

- [Introduction](#introduction)
- [Core Principles](#core-principles)
- [Design Principles](#design-principles)
- [Key Design Decisions](#key-design-decisions)
  - [Why the CLI Uses a Simpler Model](#why-the-cli-uses-a-simpler-model)
  - [What the System Validates](#what-the-system-validates)
  - [Machine-readable output](#machine-readable-output)
  - [Structured errors](#structured-errors)
  - [Exit codes](#exit-codes)
  - [No interactive prompts for agents](#no-interactive-prompts-for-agents)
  - [`--schema` on every subcommand](#--schema-on-every-subcommand)
- [Connection Flags](#connection-flags)
- [CLI Commands](#cli-commands)
  - [Study Lifecycle](#study-lifecycle)
- [User Membership Management](#user-membership-management)
- [JSON Output Examples](#json-output-examples)
- [Error Codes](#error-codes)
- [Admin Console Commands](#admin-console-commands)
  - [Study Lifecycle Commands](#study-lifecycle-commands)
  - [User Membership Commands](#user-membership-commands)
- [Authorization Model](#authorization-model)
  - [Study Lifecycle Authorization](#study-lifecycle-authorization)
  - [User Membership Authorization](#user-membership-authorization)
- [Server-Side Operations](#server-side-operations)
  - [`StudyCommandModule`](#studycommandmodule)
  - [Mutation Flow](#mutation-flow)
  - [`study_registry.json` format (extended)](#study_registryjson-format-extended)
- [Validation Rules](#validation-rules)
  - [Study Name](#study-name)
  - [Sites](#sites)
  - [Users](#users)
- [Behavioral Constraints](#behavioral-constraints)
  - [Remove](#remove)
  - [Hot-Reload](#hot-reload)
  - [Admin Self-Removal](#admin-self-removal)
  - [Provisioning Boundary](#provisioning-boundary)
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

The shipped multi-study design stores named-study state in `study_registry.json`. In centralized provisioning, `project.yml` may bootstrap the initial contents of that file. This document defines the runtime CLI management surface that lets a running server accept ongoing study mutations without tying them to provisioning.

The backend for all mutations is the same `study_registry.json` file the server loads at startup. The server applies mutations through a serialized validate-write-publish flow so the runtime registry and the persisted file remain aligned under normal operation and fail closed on validation/write errors.

---

## Core Principles

1. **Same file, same format** — mutations target the provisioned `study_registry.json` format used by multi-study, including `site_orgs` and `admins`.
2. **Authoritative in-memory registry** — after a successful mutation the server hot-reloads the in-memory `StudyRegistry`; a server restart is not required.
3. **Role-based lifecycle** — `remove` requires `project_admin`; `register`, `add-site`, `remove-site`, and study-user membership management are accessible to `project_admin` and `org_admin`, with `org_admin` visibility and authority scoped to studies where their org appears in `site_orgs`. `register` is a create-or-merge operation executed atomically under the mutation lock — see Validation Rules for the precise per-role behavior.
4. **Cert role for lifecycle operations** — study lifecycle commands check the certificate-baked role (from distributed provisioning), consistent with the existing `must_be_project_admin` pattern for server-global operations.
5. **Best-effort job-association guard** — `remove` queries the job store before applying the deletion and rejects with `STUDY_HAS_JOBS` if any associated jobs exist at that moment. This is a best-effort guard: the mutation lock prevents two concurrent `remove` calls from racing, but it does not gate job submission. A job submitted concurrently may arrive after the guard passes and before the registry is updated. Jobs are the permanent audit trail regardless of whether the study entry still exists.
6. **Agent-usable by design** — all commands follow the same output, error, exit-code, and flag conventions as the rest of the NVFlare CLI.

---

## Design Principles

`nvflare study` follows the same agent-readiness conventions as all other `nvflare` CLI commands.

## Key Design Decisions

### Why the CLI Uses a Simpler Model

The server can reliably know the caller's identity, role, and org from the presented admin certificate at session creation time. For client sites, the server derives `site -> org` ownership from authenticated client certificates when sites connect. The study registry therefore stores study membership as `site_orgs` instead of a flat `sites` list. That leads to the following model:

- `org_admin` may supply `--sites` only; the server records those sites under the caller's cert org in `site_orgs`
- `project_admin` must supply explicit org grouping for lifecycle mutations because the server must know which org owns each newly enrolled site
- target users are treated as declarative registry inputs

### What the System Validates

This is not "no validation." The system validates:

- caller cert role gates lifecycle and user-membership-mutation authorization
- `org_admin` visibility and mutation authority is scoped to studies where the caller's **org is enrolled** — if the caller's org appears as a key in the study's `site_orgs` mapping, the study is visible and mutable by that org admin, even if that org's current site list is empty; explicit per-user assignment in the `admins` map is not required for visibility
- study names and site names are validated syntactically
- site ownership is validated against the runtime connected-client `site -> org` map derived from authenticated client certificates
- `org_admin` must supply `--sites` for `register`, `add-site`, and `remove-site`; an empty list is rejected with `INVALID_SITE`
- `project_admin` must supply one or more `--site-org <org:s1,s2,...>` groups for `register`, `add-site`, and `remove-site`
- `--sites` and `--site-org` must not appear together on the same lifecycle command; mixed input is rejected with `INVALID_ARGS`
- `org_admin` use of `--site-org` is rejected with `INVALID_ARGS`
- `project_admin` use of `--sites` is rejected with `INVALID_ARGS`
- job-association guard for `remove` is best-effort — see Core Principles for the race with concurrent job submission

**Two-layer enforcement for input-shape checks.** The three `INVALID_ARGS` rules above (mixed input, `org_admin` + `--site-org`, `project_admin` + `--sites`) are enforced at both layers:

- **CLI (fast fail)**: after resolving the startup kit, the CLI reads the caller role from the startup-kit cert and rejects the invalid input before opening a server connection. This gives immediate feedback without a round-trip.
- **Server (authoritative)**: the same checks run again using the authenticated cert role. This ensures the rules hold even if the CLI layer is bypassed.

The following are intentionally not validated:

- target-user org membership — user strings are stored declaratively

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
  "hint": "Verify the study name. If the study exists and you expect access, contact a project_admin to have your org enrolled in the study.",
  "exit_code": 1
}
```

The JSON error envelope includes `exit_code` so machine consumers can rely on the same structured payload for both the error class and the process exit semantics.

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

All server-backed `nvflare study` commands require a connection to the server. Startup kit resolution is identical to all other server-connected `nvflare` commands (`job`, `system`, etc.) and follows this priority order:

1. `--startup-kit <dir>` — explicit path to the startup kit directory (or its `startup/` subdirectory)
2. `NVFLARE_STARTUP_KIT_DIR` environment variable
3. `~/.nvflare/config.conf` — reads `poc.startup_kit` by default, or `prod.startup_kit` when `--startup-target prod` is given

`~/.nvflare/config.conf` is written by `nvflare config`. When no explicit source is provided the config file is consulted automatically and defaults to the `poc` target, so a user who has run `nvflare config` once does not need to pass any flag on every command.

`--startup-kit` and `--startup-target` are mutually exclusive. If all three sources fail to resolve a valid directory, the command exits with code 4 and `STARTUP_KIT_MISSING`.

---

## CLI Commands

### Study Lifecycle

`--format json` is a global flag and may appear anywhere on the command line.

```
# org_admin input
nvflare study register    <name> --sites <s1,s2,...> [--startup-kit <dir> | --startup-target poc|prod] [--schema]
nvflare study add-site    <name> --sites <s1,s2,...> [--startup-kit <dir> | --startup-target poc|prod] [--schema]
nvflare study remove-site <name> --sites <s1,s2,...> [--startup-kit <dir> | --startup-target poc|prod] [--schema]

# project_admin input
nvflare study register    <name> --site-org <org:s1,s2,...> [--site-org <org:s3,...> ...] [--startup-kit <dir> | --startup-target poc|prod] [--schema]
nvflare study add-site    <name> --site-org <org:s1,s2,...> [--site-org <org:s3,...> ...] [--startup-kit <dir> | --startup-target poc|prod] [--schema]
nvflare study remove-site <name> --site-org <org:s1,s2,...> [--site-org <org:s3,...> ...] [--startup-kit <dir> | --startup-target poc|prod] [--schema]

# project_admin only
nvflare study remove      <name> [--startup-kit <dir> | --startup-target poc|prod] [--schema]

# project_admin (all studies) or org_admin (studies where caller's org is enrolled)
nvflare study list        [--startup-kit <dir> | --startup-target poc|prod] [--schema]
nvflare study show        <name> [--startup-kit <dir> | --startup-target poc|prod] [--schema]
```

| Command | Description | Required cert role |
|---------|-------------|-------------------|
| `register` | Create-or-merge a study. `org_admin` must use `--sites`, which are recorded under the caller's cert org in `site_orgs`. `project_admin` must use one or more `--site-org <org:s1,s2,...>` groups. Mixed `--sites` + `--site-org` input is invalid. Caller is recorded in `admins` if not already present. For `org_admin`: if the study does not exist it is created; if the study exists and the caller's org is already enrolled, the supplied `--sites` are merged into that org's site list; if the study exists but the caller's org is not enrolled, returns `STUDY_ALREADY_EXISTS`. | `project_admin` or `org_admin` |
| `add-site` | Add sites to an existing study. `org_admin` must use `--sites`; `project_admin` must use `--site-org`. Mixed `--sites` + `--site-org` input is invalid. Idempotent — enrolling an already-enrolled site is a no-op. For `org_admin`: study must be visible (caller org enrolled in `site_orgs`); otherwise `STUDY_NOT_FOUND`. | `project_admin` or `org_admin` |
| `remove-site` | Remove sites from an existing study. `org_admin` must use `--sites`; `project_admin` must use `--site-org`. Mixed `--sites` + `--site-org` input is invalid. For `org_admin`: study must be visible (caller org enrolled in `site_orgs`); otherwise `STUDY_NOT_FOUND`. | `project_admin` or `org_admin` |
| `remove` | Remove a study and all its entries | `project_admin` only |
| `list` | List studies; `project_admin` sees all, `org_admin` sees only studies where their org is enrolled | `project_admin` or `org_admin` |
| `show` | Show enrolled sites grouped by org and study-user membership; `org_admin` restricted to studies where their org is enrolled | `project_admin` or `org_admin` |

#### Arguments — `register`

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<name>` | str | Yes | Study name; must match the existing `name_check(..., "study")` contract: `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$` |
| `--sites` | str | Yes (`org_admin`) | Comma-separated site names to enroll under the caller's cert org. An empty list is rejected with `INVALID_SITE`. `project_admin` use of `--sites` is rejected with `INVALID_ARGS`. |
| `--site-org` | str | Yes (`project_admin`) | Repeatable `org:s1,s2,...` group. Each group enrolls the listed sites under the named org in `site_orgs`. `org_admin` use of `--site-org` is rejected with `INVALID_ARGS`. |
| `--startup-kit` | str | No* | Explicit path to the startup kit directory |
| `--startup-target` | `poc\|prod` | No* | Resolves startup kit path from `~/.nvflare/config.conf` |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* `--startup-kit` and `--startup-target` are mutually exclusive optional selectors. The startup kit must be resolvable via `--startup-kit`, `NVFLARE_STARTUP_KIT_DIR`, or `~/.nvflare/config.conf` using the explicit `--startup-target` or the default `poc` target — see [Connection Flags](#connection-flags). If no source resolves, the command fails.

`--sites` and `--site-org` are mutually exclusive on the same invocation. If both are provided, the command fails with `INVALID_ARGS`.

#### Usage Examples — `register`

**org_admin registers a new study**

```bash
# logged in as org_admin@org_a.com
nvflare study register cancer-research --sites hospital-a --startup-target prod
```

**org_admin re-registers an existing study they belong to (merge — adds supplied sites not yet enrolled)**

```bash
# logged in as org_admin@org_a.com; caller's org is already enrolled in the study
nvflare study register cancer-research --sites hospital-b --startup-target prod
```

**project_admin registers a cross-org study**

`project_admin` may enroll sites from any org in a single call.

```bash
# logged in as admin@nvidia.com (project_admin)
nvflare study register cancer-research \
  --site-org org_a:hospital-a \
  --site-org org_b:hospital-b,hospital-c \
  --startup-target prod
```

#### Arguments — `add-site` and `remove-site`

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<name>` | str | Yes | Existing study name |
| `--sites` | str | Yes (`org_admin`) | Comma-separated site names in the caller's cert org. `project_admin` use of `--sites` is rejected with `INVALID_ARGS`. |
| `--site-org` | str | Yes (`project_admin`) | Repeatable `org:s1,s2,...` group describing which sites are being mutated under which org. `org_admin` use of `--site-org` is rejected with `INVALID_ARGS`. |
| `--startup-kit` | str | No* | Explicit path to the startup kit directory |
| `--startup-target` | `poc\|prod` | No* | Resolves startup kit path from `~/.nvflare/config.conf` |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* `--startup-kit` and `--startup-target` are mutually exclusive optional selectors. The startup kit must be resolvable via `--startup-kit`, `NVFLARE_STARTUP_KIT_DIR`, or `~/.nvflare/config.conf` using the explicit `--startup-target` or the default `poc` target — see [Connection Flags](#connection-flags). If no source resolves, the command fails.

`--sites` and `--site-org` are mutually exclusive on the same invocation. If both are provided, the command fails with `INVALID_ARGS`.

Both commands return per-site outcome lists. `add-site`: already-enrolled sites are reported in `already_enrolled` and skipped; newly enrolled sites appear in `added`. `remove-site`: sites not currently enrolled are reported in `not_enrolled` and skipped; removed sites appear in `removed`. Neither command errors on partially overlapping input — the full outcome is always inspectable in the response.

#### Arguments — `remove`

`remove` requires `project_admin` cert role. `org_admin` callers are rejected with `NOT_AUTHORIZED`.

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<name>` | str | Yes | Study name to remove |
| `--startup-kit` | str | No* | Explicit path to the startup kit directory |
| `--startup-target` | `poc\|prod` | No* | Resolves startup kit path from `~/.nvflare/config.conf` |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* `--startup-kit` and `--startup-target` are mutually exclusive optional selectors. The startup kit must be resolvable via `--startup-kit`, `NVFLARE_STARTUP_KIT_DIR`, or `~/.nvflare/config.conf` using the explicit `--startup-target` or the default `poc` target — see [Connection Flags](#connection-flags). If no source resolves, the command fails.

### User Membership Management

```
nvflare study add-user    <study> <user> [--startup-kit <dir> | --startup-target poc|prod] [--schema]
nvflare study remove-user <study> <user> [--startup-kit <dir> | --startup-target poc|prod] [--schema]
```

| Command | Description | Required cert role |
|---------|-------------|-------------------|
| `add-user` | Add a user to a registered study's `admins` membership list | `project_admin` (any study) or `org_admin` (studies where caller's org is enrolled) |
| `remove-user` | Remove a user from a study's `admins` membership list | `project_admin` (any study) or `org_admin` (studies where caller's org is enrolled) |

#### Arguments — user membership commands

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `<study>` | str | Yes | Registered study name |
| `<user>` | str | Yes | Username or cert CN string to store in the study membership list |
| `--startup-kit` | str | No* | Explicit path to the startup kit directory |
| `--startup-target` | `poc\|prod` | No* | Resolves startup kit path from `~/.nvflare/config.conf` |
| `--schema` | flag | No | Print command schema as JSON and exit |

\* `--startup-kit` and `--startup-target` are mutually exclusive optional selectors. The startup kit must be resolvable via `--startup-kit`, `NVFLARE_STARTUP_KIT_DIR`, or `~/.nvflare/config.conf` using the explicit `--startup-target` or the default `poc` target — see [Connection Flags](#connection-flags). If no source resolves, the command fails.

#### Usage Examples — user membership commands

**org_admin adds a user from their own org to a study**

The org admin adds `trainer@org_a.com` to a study where the caller's org is enrolled. The target user string is stored declaratively.

```bash
# logged in as org_admin@org_a.com
nvflare study add-user cancer-research trainer@org_a.com --startup-target prod
```

**org_admin adds another user entry in their study**

```bash
# logged in as org_admin@org_a.com
nvflare study add-user cancer-research analyst@org_a.com --startup-target prod
```

**project_admin adds a user from any org**

```bash
# logged in as admin@nvidia.com (project_admin)
nvflare study add-user cancer-research trainer@org_b.com --startup-target prod
```

**project_admin adds a cross-org user using an explicit startup kit path**

```bash
nvflare study add-user cancer-research analyst@org_c.com --startup-kit /opt/nvflare/admin_startup
```

**org_admin removes a user**

```bash
# logged in as org_admin@org_a.com
nvflare study remove-user cancer-research trainer@org_a.com --startup-target prod
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

The `"users"` key in the response corresponds to the `"admins"` list in `study_registry.json`. The CLI response uses `"users"` to avoid exposing the internal file field name.

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "name": "cancer-research",
    "site_orgs": {
      "org_a": ["hospital-a"],
      "org_b": ["hospital-b"]
    },
    "sites": ["hospital-a", "hospital-b"],
    "users": ["admin@nvidia.com", "trainer@org_a.com"]
  }
}
```

### `nvflare study register <name>`

The response reflects the full resulting state of the study. The `"users"` list is the complete `admins` membership after the operation — not just the newly inserted caller.

**New study created** (caller is the only entry):

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "name": "cancer-research",
    "site_orgs": {
      "org_a": ["hospital-a"]
    },
    "sites": ["hospital-a"],
    "users": ["org_admin@org_a.com"]
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
    "site_orgs": {
      "org_a": ["hospital-a", "hospital-b"]
    },
    "sites": ["hospital-a", "hospital-b"],
    "users": ["existing@org_b.com", "org_admin@org_a.com"]
  }
}
```

### `nvflare study add-site <study>`

The response always returns per-site outcome lists so automation can distinguish newly enrolled sites from already-enrolled ones. `site_orgs` is the authoritative grouped structure; `sites` is the complete derived enrollment list for convenience.

**Single new enrollment:**
```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "added": ["hospital-b"],
    "already_enrolled": [],
    "site_orgs": {
      "org_a": ["hospital-a", "hospital-b"]
    },
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
    "site_orgs": {
      "org_a": ["hospital-a", "hospital-b", "hospital-c"]
    },
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
    "site_orgs": {
      "org_a": ["hospital-a"]
    },
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

### `nvflare study add-user <study> <user>`

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "user": "trainer@org_a.com"
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

The response returns per-site outcome lists. `site_orgs` is the authoritative grouped structure; `sites` is the complete derived enrollment list after removal.

**Multi-site call (mix of removed and not enrolled):**
```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "study": "cancer-research",
    "removed": ["hospital-b"],
    "not_enrolled": ["hospital-c"],
    "site_orgs": {
      "org_a": ["hospital-a"]
    },
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
    "site_orgs": {},
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
| `STUDY_NOT_FOUND` | 1 | Study name does not exist in the registry, or exists but is not visible to the caller (`org_admin`'s org not enrolled in the study's `site_orgs` mapping) — the same code is used for both cases intentionally to avoid leaking existence. Exception: `register` does NOT use this code (see `STUDY_ALREADY_EXISTS`). |
| `STUDY_ALREADY_EXISTS` | 1 | Study name already registered. Returned by `register` when the study exists but the caller's org is not enrolled. `register` is not a join path — returning `STUDY_NOT_FOUND` on a create would be semantically backwards and confusing. The caller already knows the name; learning "it is taken" reveals nothing actionable. |
| `STUDY_HAS_JOBS` | 1 | Study has associated jobs; `remove` rejected |
| `INVALID_STUDY_NAME` | 4 | Study name fails the name-validation regex or is `"default"` |
| `INVALID_SITE` | 4 | A site name in `--sites` is malformed (fails the site-name validator), or `--sites` is an empty list when it is required |
| `INVALID_ARGS` | 4 | Mixed `--sites` + `--site-org` input, or a caller role using the wrong lifecycle input shape (`org_admin` with `--site-org`, `project_admin` with `--sites`) |
| `SITE_NOT_IN_STUDY` | — | Removed: `remove-site` no longer rejects on non-enrolled sites. Non-enrolled sites are reported in `not_enrolled` in the response and skipped. |
| `ORG_NOT_FOUND` | 4 | Reserved for a future participant-registry-backed validation flow |
| `USER_ALREADY_IN_STUDY` | 1 | `add-user` rejected: user is already in this study's membership list |
| `USER_NOT_IN_STUDY` | 1 | `remove-user` rejected: user is not in this study's membership list |
| `NOT_AUTHORIZED` | 1 | Caller's cert role is insufficient for this operation |
| `STARTUP_KIT_NOT_CONFIGURED` | 4 | `--startup-target` given but no matching entry in `~/.nvflare/config.conf` |
| `STARTUP_KIT_MISSING` | 4 | No startup kit could be resolved from `--startup-kit`, `--startup-target`, or `NVFLARE_STARTUP_KIT_DIR` |
| `LOCK_TIMEOUT` | 3 | Mutation lock could not be acquired within 30 seconds — another mutation is in progress |
| `CONNECTION_FAILED` | 2 | Cannot connect to or authenticate with the server |

---

## Admin Console Commands

The CLI translates each user command into an admin console command sent over the authenticated admin connection. The server-side command handlers live in a new `StudyCommandModule`.

### Study Lifecycle Commands

| Admin command | Arguments | Handler |
|---------------|-----------|---------|
| `register_study` | `<name> --sites s1,s2,...` for `org_admin`; `<name> --site-org org:s1,s2,... [--site-org ...]` for `project_admin` | `StudyCommandModule.cmd_register_study` — rejects mixed `--sites` + `--site-org` input with `INVALID_ARGS`; rejects `org_admin` use of `--site-org`; rejects `project_admin` use of `--sites`; records caller in `admins` if not already present; merges supplied sites into `site_orgs`. Three cases for `org_admin`: (1) study does not exist → create, record caller in `admins`, add sites under caller org; (2) study exists and caller org is already enrolled → merge supplied sites into that org's site list; (3) study exists but caller org is not enrolled → `STUDY_ALREADY_EXISTS`. |
| `add_study_site` | `<name> --sites s1,s2,...` for `org_admin`; `<name> --site-org org:s1,s2,... [--site-org ...]` for `project_admin` | `StudyCommandModule.cmd_add_study_site` — rejects mixed `--sites` + `--site-org` input with `INVALID_ARGS`; rejects `org_admin` use of `--site-org`; rejects `project_admin` use of `--sites`. For `org_admin` callers: study must be visible (caller org must be enrolled); otherwise `STUDY_NOT_FOUND`. |
| `remove_study_site` | `<name> --sites s1,s2,...` for `org_admin`; `<name> --site-org org:s1,s2,... [--site-org ...]` for `project_admin` | `StudyCommandModule.cmd_remove_study_site` — rejects mixed `--sites` + `--site-org` input with `INVALID_ARGS`; rejects `org_admin` use of `--site-org`; rejects `project_admin` use of `--sites`. For `org_admin` callers: study must be visible (caller org must be enrolled); otherwise `STUDY_NOT_FOUND`. |
| `remove_study` | `<name>` | `StudyCommandModule.cmd_remove_study` — rejected if any jobs are associated |
| `list_studies` | _(none)_ | `StudyCommandModule.cmd_list_studies` |
| `show_study` | `<name>` | `StudyCommandModule.cmd_show_study` |

### User Membership Commands

| Admin command | Arguments | Handler |
|---------------|-----------|---------|
| `add_study_user` | `<study> <user>` | `StudyCommandModule.cmd_add_study_user` |
| `remove_study_user` | `<study> <user>` | `StudyCommandModule.cmd_remove_study_user` |

---

## Authorization Model

### Study Lifecycle Authorization

Study lifecycle commands check the certificate-baked role, consistent with other server-global operations.

```
remove_study
    → require cert role == "project_admin" only

register_study, add_study_site, remove_study_site
    → allowed for cert role in {project_admin, org_admin}

list_studies, show_study
    → project_admin: unrestricted (all studies)
    → org_admin: scoped to studies where caller's org is enrolled in the study's `site_orgs` mapping
```

**Scoping for non-project-admin callers (`org_admin`):**

- `register_study`: `org_admin` uses `--sites`; caller is recorded in `admins` if not already present; supplied sites are merged into `site_orgs[caller_org]`. If the study exists and the caller's org is not already enrolled, return `STUDY_ALREADY_EXISTS`.
- `register_study`: reject mixed `--sites` + `--site-org` input with `INVALID_ARGS`; reject `org_admin` use of `--site-org`; reject `project_admin` use of `--sites`.
- `add_study_site`: already-enrolled site is a no-op. For `org_admin`: the study must be visible (caller org must appear in `site_orgs`); if not, return `STUDY_NOT_FOUND` — do not reveal existence.
- `add_study_site`: reject mixed `--sites` + `--site-org` input with `INVALID_ARGS`; reject `org_admin` use of `--site-org`; reject `project_admin` use of `--sites`.
- `remove_study_site`: for `org_admin` callers, the study must be visible (caller org must appear in `site_orgs`; otherwise `STUDY_NOT_FOUND`).
- `remove_study_site`: reject mixed `--sites` + `--site-org` input with `INVALID_ARGS`; reject `org_admin` use of `--site-org`; reject `project_admin` use of `--sites`.
- `list_studies`: returns only studies where the caller's org is enrolled in `site_orgs`.
- All other targeted commands (`show_study`, `add_study_site`, `remove_study_site`, `add_study_user`, `remove_study_user`): if the caller's org does not appear in the target study's `site_orgs` mapping, return `STUDY_NOT_FOUND` — do not reveal existence.

`project_admin` sees all studies and may operate on any study without restriction.

Any role not listed above is rejected with `NOT_AUTHORIZED`.

### User Membership Authorization

User membership commands apply a two-tier check:

1. **`project_admin` (cert)** — may add or remove any user in any study.
2. **`org_admin` (cert)** — may add or remove users only for studies where their org is enrolled in the study's `site_orgs` mapping. Target user strings are stored declaratively; robust target-user org validation is deferred to participant-registration support.

```
add_study_user, remove_study_user
    → cert role == "project_admin"       → allowed (any org)
    → cert role == "org_admin"           → allowed for studies where caller's org is enrolled
    → other roles                        → NOT_AUTHORIZED
```

Study users are membership only. There is no study-specific role. Two distinct authorization layers remain:

- **Named-study session creation** — gated by `admins`: the user must have an entry in that study's `admins` list or opening that session is rejected. The user's existing certificate-baked role remains the effective role for study-scoped authorization within that session.
- **CLI management commands** (`register`, `add-site`, `remove-site`, `list`, `show`, and user membership mutations) — gated by **org enrollment**: an `org_admin` can run these commands for studies where their cert org appears as a key in `site_orgs`, regardless of whether they personally appear in `admins`.

`add-user` and `remove-user` are therefore not cosmetic: they determine who may open a session for a named study.

---

## Server-Side Operations

### `StudyCommandModule`

A new `StudyCommandModule` (parallel to `SystemCommandModule` and `JobCommandModule`) registers the study management commands. It embeds `CommandUtil` and holds a reference to the mutable registry service.

### Mutation Flow

All mutating commands follow the same serialized pattern:

1. **Authorize** — check cert role; reject if insufficient.
2. **Validate** — check that the study name, user, and sites are syntactically valid before modifying state.
3. **Acquire mutation lock** — a process-local lock serializes all registry mutations so concurrent admin commands cannot lose updates. The acquire call has a 30-second timeout; if the lock is not acquired within that window, the command returns immediately with `LOCK_TIMEOUT` (exit 3). The lock is always released in a `finally` block — a failure during steps 4–9 cannot leave the lock permanently held.
4. **Load current state** — read `study_registry.json` from disk into a working copy while holding the lock.
5. **Guard** — for `remove_study`, query the job store (while holding the lock) for any job tagged with the study name; reject with `STUDY_HAS_JOBS` if any exist. Running inside the lock eliminates the TOCTOU window between two concurrent `remove` calls, but does not prevent a job submission that races in after this check — see the design limitation note below.
6. **Apply mutation** — modify the working copy in memory. `register_study` auto-inserts the caller into `admins` if not already present; `add-user` and `remove-user` mutate `admins` explicitly. Study membership is stored in `site_orgs`: `org_admin` mutations affect only `site_orgs[caller_org]`; `project_admin` mutations may affect multiple org groups in one call.
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
      "site_orgs": {
        "org_a": ["hospital-a"],
        "org_b": ["hospital-b"]
      },
      "admins": ["admin@nvidia.com", "trainer@org_a.com"]
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

  `register` always applies **merge semantics** on an existing study: supplied sites that are not yet enrolled are added under the appropriate org group in `site_orgs`; already-enrolled sites are silently skipped. The caller is recorded in `admins` if not already present. `register` never removes sites or `admins` entries — removal requires explicit `remove-site` or `remove-user`.

  **`org_admin` behavior:**
  - Study does not exist → create, record caller in `admins`, add supplied sites under `site_orgs[caller_org]`. Self-membership at creation is not a security bypass — the study did not exist before this call, so there is no pre-existing access list to circumvent.
  - Study exists, caller's org already enrolled → merge supplied sites into `site_orgs[caller_org]`; succeed.
  - Study exists, caller's org NOT enrolled → `STUDY_ALREADY_EXISTS` (name is taken and caller has no access; `project_admin` must enroll the org first). `register` is not a join path.

  **`project_admin` behavior on existing studies:**
  - Same merge semantics: add supplied sites not yet enrolled under the specified org groups; skip already-enrolled sites; record self in `admins` if not already present.

- Unknown names rejected on `remove_study`, `show_study`, and all user membership commands with `STUDY_NOT_FOUND`

### Sites

- `org_admin` must supply `--sites` for `register`, `add-site`, and `remove-site`; an empty list or a malformed site name is rejected with `INVALID_SITE`
- `project_admin` must supply one or more `--site-org <org:s1,s2,...>` groups for `register`, `add-site`, and `remove-site`
- Mixed `--sites` + `--site-org` input is rejected with `INVALID_ARGS`
- `org_admin` use of `--site-org` is rejected with `INVALID_ARGS`
- `project_admin` use of `--sites` is rejected with `INVALID_ARGS`
- Each site name must pass the site-name validator: `^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$` (same regex as study names, using `name_check(..., "site")`)
- Each org key in `--site-org` must pass `name_check(..., "org")`
- The server validates site ownership from the runtime connected-client map built from authenticated client certificates
- For `org_admin`, each requested site must currently be known to the server and must map to the caller's cert org
- For `project_admin`, each `--site-org <org:s1,s2,...>` entry is accepted only if every listed site is currently known to the server and each site's connected-client cert org matches the specified org
- `project_admin` input must not place the same site in more than one org group within the same command
- `site_orgs` must not contain the same site under multiple orgs after the mutation is applied

### Users

- Target user strings are accepted as declarative study membership; the server does not reject unknown users just because they have not connected yet
- For `org_admin` callers: same-org mutation remains the intended policy, but robust target-user org validation requires the participant-registration design described in `docs/design/participant_registration.md`
- `add_study_user` rejects a user already present in the study's `admins` list with `USER_ALREADY_IN_STUDY`
- `remove_study_user` rejects a user not present in the study's `admins` list with `USER_NOT_IN_STUDY`

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

`remove-user` does not guard against a caller removing themselves from the study's `admins` field. Note that `org_admin` visibility is derived from org enrollment in `site_orgs`, not from the `admins` field — removing one's own `admins` entry does not lose visibility as long as the org remains enrolled. It does, however, prevent that user from opening a new session for the named study until they are added back.

### Provisioning Boundary

These commands mutate runtime study state only.

- centralized provisioning may bootstrap the initial `study_registry.json`
- dynamic provisioning manages participants and certificates only
- dynamic provisioning does not create, update, or remove studies
- ongoing study changes are made through `nvflare study ...`, not through provisioning

---

## Relationship to Distributed Provisioning

The distributed provisioning workflow (`docs/design/distributed_provisioning.md`) decentralizes the **identity layer**: each site generates its own private key, sends only a CSR to the Project Admin, and receives a signed certificate in return. Once the site packages its startup kit and connects to the server, it is a fully authenticated participant — without the Project Admin ever holding its private key.

The multi-study CLI owns the **study authorization layer** at runtime: once sites are authenticated and connected, an org admin can register studies, enroll sites within their org, and manage study-user membership without involving provisioning. Project-admin mutations may enroll sites for multiple orgs in one call by supplying explicit `--site-org` groups.

Together, the two capabilities make a **fully decentralized federation lifecycle** possible:

| Operation | Mechanism | Provisioning step required? |
|-------|-----------|--------------------------|
| Site joins the federation | `nvflare cert request` → `nvflare cert approve` → `nvflare package <signed.zip>` | No — cert approval is the act of authorization |
| Study created by org admin | `nvflare study register --sites` (org admin supplies explicit site list) | No |
| Users added to the study | `nvflare study add-user` | No |
| Cross-org study created | `nvflare study register --site-org ...` (project admin, sites grouped by org) | No |
| Site removed from federation | Certificate revocation / participant removal (out of scope) | Outside study CLI scope |

### End-to-End Workflow

A new site joins and sets up its own study without coupling study management to provisioning:

```
# Step 1 — Site Admin creates a request zip (private key stays local)
nvflare cert request site hospital-1 --org org_a --project example_project

# Step 2 — Project Admin approves the request zip
nvflare cert approve hospital-1.request.zip --ca-dir ./ca

# Step 3 — Site Admin packages and starts
nvflare package hospital-1.signed.zip -e grpc://fl-server:8002 --request-dir ./hospital-1
cd hospital-1 && ./startup/start.sh

# Step 4 — Org Admin creates their admin request and startup kit (same flow)
nvflare cert request user org-admin admin@org_a.com --org org_a --project example_project
# ... approve, package, then activate the startup kit ...

# Step 5 — Org Admin registers a study with a site from their org
nvflare study register cancer-research --sites hospital-1

# Step 6 — Org Admin adds a researcher from their org
nvflare study add-user cancer-research researcher@org_a.com

# Step 7 — Lead submits a job to the study
nvflare job submit -j ./my_job --study cancer-research
```

Steps 1–3 follow the distributed provisioning workflow. Steps 4–7 use the multi-study CLI. The Project Admin is only involved in step 2 (signing the CSR); study creation and study-user membership management happen later through runtime study commands.

### Trust Chain

The distributed provisioning design already establishes the trust chain for certificate roles and participant org identity:

- The cert type (`project_admin`, `org_admin`, `lead`, `member`) is embedded in `UNSTRUCTURED_NAME` at signing time
- The server reads the admin cert at session creation time and uses it as the effective cert role plus caller org
- The server reads the client cert when a site connects and derives that site's org membership from the authenticated connection
- The multi-study CLI uses the admin cert role to gate study lifecycle and study-user membership operations, and uses the connected-client cert org map to validate requested sites

The cert is the gate for **who can call** study lifecycle and study-user membership commands — it establishes identity, cert role, and caller org at session creation time. Connected client certificates establish **which org each site belongs to** at runtime. The registry then determines **which studies** an `org_admin` can act on by checking whether that org appears in the study's `site_orgs` mapping. Centralized provisioning may bootstrap initial study contents, but the running server treats `study_registry.json` as runtime state thereafter.

---

## Implementation

### New Files

| File | Purpose |
|------|---------|
| `nvflare/tool/study/__init__.py` | Package |
| `nvflare/tool/study/study_cli.py` | `nvflare study` subcommand handlers |
| `nvflare/private/fed/server/study_cmds.py` | `StudyCommandModule` server-side handlers |

### Modified Files

| File | Change |
|------|--------|
| `nvflare/cli.py` | Register `nvflare study` top-level subcommand |
| `nvflare/private/fed/server/server_cmd_modules.py` | Register `StudyCommandModule` |
| `nvflare/fuel/flare_api/flare_api.py` | Add `register_study`, `add_study_site`, `remove_study_site`, `remove_study`, `list_studies`, `show_study`, `add_study_user`, `remove_study_user` session methods |
| `nvflare/security/study_registry.py` | Add mutation lock support or companion synchronization helper for serialized registry updates |

### Session API

All handlers (`register`, `add-site`, `remove-site`, `remove`, `list`, `show`, `add-user`, `remove-user`) use `new_secure_session()` for server connectivity, identical to `nvflare job` and `nvflare system`. New session methods are thin wrappers over `AdminAPI.do_command()`.

---

## Summary

| Area | Design |
|------|--------|
| Study lifecycle commands | `register`, `add-site`, `remove-site`, `remove`, `list`, `show` under `nvflare study` |
| User membership commands | `add-user`, `remove-user` under `nvflare study` |
| Study lifecycle authorization | `remove`: `project_admin` only. `register`, `add-site`, `remove-site`: `project_admin` or `org_admin` (`org_admin` uses `--sites`; `project_admin` uses `--site-org`; `org_admin` scoped to studies where their org is enrolled). `list`/`show`: `project_admin` sees all; `org_admin` sees only studies where their org is enrolled |
| User membership authorization | `project_admin` (any study); `org_admin` (studies where caller's org is enrolled) |
| Backend file | `study_registry.json` with `site_orgs` plus `admins` |
| Dataset mapping | Out of scope for this design; deferred to a future iteration |
| Persistence | Serialized mutation flow with lock, temp file, `os.replace()`, then registry publish |
| In-memory reload | Hot-reload after each mutation; no server restart required |
| Job-association guard | Best-effort: `remove` queries the job store at guard time and rejects with `STUDY_HAS_JOBS` if any tagged job exists; not atomic with concurrent job submission |
| Active-session behavior | Existing authenticated sessions unaffected; forced session eviction deferred until a clean session-manager integration is designed |
| Output format | JSON envelope (`schema_version`, `status`, `data`) via `--format json`; human text by default |
| Exit codes | 0 success, 1 server error, 2 connection failure, 3 timeout, 4 invalid args, 5 internal |
| Agent usability | `--schema` on every subcommand; no interactive prompts |
| Provisioning boundary | Centralized provisioning may bootstrap initial study state; dynamic provisioning manages participants only; ongoing study changes are runtime-managed through `nvflare study` |
