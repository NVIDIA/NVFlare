# NVFLARE CLI Design: Enhance NVFLARE CLI Commands

Created Date: 2026-03-26
Updated Date: 2026-04-24


## Problem Statement

NVFlare's admin operations are currently accessible only through an interactive console (`nvflare_admin`) — a stateful REPL that requires a human at the keyboard. This creates three problems:

- Not scriptable: automating FL workflows (submit job, wait, download results) requires fragile stdin piping or the deprecated FLAdminAPI.
- Not CI/CD friendly: pipeline systems (GitHub Actions, Airflow, Kubernetes Jobs) cannot drive an interactive console.
- Incomplete `nvflare job`: submit exists but list, abort, delete, download, clone do not.
- Startup kit selection is too visible in normal workflows: repeated `--startup-target`,
  `--startup-kit`, and environment-only state force users to think about local
  implementation details instead of running commands such as `nvflare job list`,
  `nvflare study list`, and `nvflare system status`.

Goal: make the full NVFlare admin command surface accessible non-interactively — scriptable from CI/CD pipelines, automation tools, and AI agents. This covers:

- `nvflare job` — extended with missing job lifecycle operations
- `nvflare system` — new subcommand for operational control (status, shutdown, restart, version)
- `nvflare recipe` — FL workflow recipe catalog, no server required
- `nvflare network` — cellnet diagnostics for advanced troubleshooting
- `nvflare agent` — bootstrap and context management for agentic workflows
- `nvflare install-skills` — installs skill files into agent framework discovery paths
- `nvflare config kit` — local startup kit registry and active-kit selection for all
  server-connected commands


## Design Principles

From `nvflare_agent_consumer_readiness.md`.

This CLI is Stage 1 of the agent-consumer readiness plan. The design must satisfy agent-usability requirements, not just human usability.

### 1. Machine-readable output on every command

Every command supports a stable, versioned JSON envelope via ``--format json``.
The default output format remains human-readable text (``txt``). When JSON mode is
requested, the output envelope is stable and versioned:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": { "...": "..." }
}
```

`schema_version` allows agents to detect breaking changes. The stdout/stderr contract is:

Contract — stream split in JSON mode:

- For normal command handler execution with ``--format json``, stdout contains exactly one JSON envelope and nothing else.
- Human-readable progress, warnings, prompts, and diagnostics go to stderr.
- Lower-level provisioning diagnostics must be accumulated and attached to the final structured error in JSON mode, rather than printed directly as ad hoc fallback text.

Exceptions: the following are plain text and are explicitly outside the JSON command-output contract:

- `--help` / `-h`: argparse-generated usage text
- `--version`: top-level utility path, not operational command output
- Parser-generated usage errors

Agents must use `--schema` for machine-readable command discovery, not `--help`.

```text
--schema        Print JSON description of this command's arguments and exit
```

### 2. Structured error format

Errors follow a consistent convention so agents know what to do next.

Default (JSON):

```json
{
  "schema_version": "1",
  "status": "error",
  "error_code": "JOB_NOT_FOUND",
  "message": "Job 'abc123' does not exist on this server.",
  "hint": "Use 'nvflare job list' to see available job IDs."
}
```

With ``--format json``, errors are also returned as JSON envelopes on stdout.
Human-facing diagnostics should be written to stderr.

Exit code is non-zero. Error message templates support `str.format_map()` substitution via `**kwargs` in `output_error()`.

### 3. No interactive prompts for agents

All confirmation-required commands (`abort`, `delete`, `shutdown`, `restart`, `remove-client`) accept `--force` to skip confirmation. `--force` must be passed explicitly. Non-interactive contexts (stdin not a tty) without `--force` exit with code 4.

### 4. Exit codes for agent branching

```text
0   Success
1   Server/job error (job not found, unauthorized, etc.)
2   Connection / authentication failure
3   Timeout
4   Invalid arguments
5   Internal error (unexpected exception — likely a bug)
```

Agents branch on exit codes, not output parsing. Exit 4 means "retry with different args". Exit 5 means "report a bug" — never retry.

### 5. CLI contracts are MCP contracts

The JSON output shape and error codes defined here become the MCP tool schemas in Stage 2. The CLI and MCP tool for the same operation must return identical envelopes. Do not define the MCP schema separately — derive it from the CLI contract.

### 6. No new overlapping entrypoints

`nvflare job` is extended, not replaced. `nvflare system` is new but covers a domain with no existing CLI equivalent. FLAdminAPI is not extended — it is deprecated in favor of `Session` (`fuel/flare_api/flare_api.py`) which this CLI wraps via `new_secure_session()`.


## Complete Command Inventory

### `nvflare job` (exists today — requires agent readiness updates)

| Subcommand | Server? | Agent Readiness Status |
| --- | --- | --- |
| `create` | No | Deprecated — retain with stderr warning; use `python job.py --export --export-dir <job_folder>` + `nvflare job submit` instead |
| `submit` | Yes | Needs JSON output, exit codes, structured errors; returns `job_id` immediately |
| `list-templates` | No | Deprecated — retain with stderr warning; use `nvflare recipe list`. Underscore alias `list_templates` kept. |
| `show-variables` | No | Deprecated — retain with stderr warning; use Job Recipe API. Underscore alias `show_variables` kept. |

`submit` already uses `new_secure_session()` (`nvflare/fuel/flare_api/flare_api.py`) for server connectivity. The same session infrastructure is used by `ProdEnv` and `PocEnv`.

### Other existing commands (requires agent readiness updates)

| Command | Subcommands | Agent Readiness Status |
| --- | --- | --- |
| `nvflare simulator` | — | Deprecated — retain with stderr warning; use Job Recipe SimEnv directly (`python job.py`) |
| `nvflare poc` | `prepare`, `start`, `stop`, `clean`, `add user`, `add site` | Add JSON output, exit codes, `--schema`; add `--force` to `prepare` for workspace deletion prompt bypass and to `clean` for stop-before-cleanup; register generated user/admin kits in the shared startup kit registry |
| `nvflare config kit` | `add`, `use`, `show`, `list`, `remove` | User-facing startup kit registry commands; no server connection |
| `nvflare study` | `register`, `show`, `list`, `remove`, `add-site`, `remove-site`, `add-user`, `remove-user` | Add multi-study lifecycle CLI using the active startup kit |
| `nvflare provision` | — | Add JSON output, `--schema`, `--force` for Y/N prompts; restore pre-2.7.0 default: no args = generate `project.yml` |
| `nvflare preflight-check` | — | Add JSON output, `--schema`; exit 0=pass, 1=fail. Underscore alias `preflight_check` accepted for backward compatibility. |
| `nvflare config` | `kit` | Parent command namespace for local CLI settings. The user-facing workflow in this design is `nvflare config kit`. |
| `nvflare dashboard` | — | No changes; excluded from this plan |
| `nvflare authz-preview` | — | Deprecated — retain with stderr warning. Underscore alias `authz_preview` accepted for backward compatibility. |


## Visible Admin Console Commands (No CLI Equivalent Today)

Only commands that serve common end-user or admin tasks are exposed. Diagnostic, internal, and rarely-needed commands stay in the interactive console only.

### Job management (`job_mgmt` scope)

| Command | User | CLI |
| --- | --- | --- |
| `list_jobs` | Both | Yes -> `nvflare job list` |
| `get_job_meta` | Both | Yes -> `nvflare job meta` |
| `submit_job` | End user | Already exists -> `nvflare job submit` |
| `abort_job` | Both | Yes -> `nvflare job abort` |
| `clone_job` | End user | Yes -> `nvflare job clone` |
| `download_job` | End user | Yes -> `nvflare job download` |
| `delete_job` | Admin | Yes -> `nvflare job delete` |
| `list_job` | — | No — internal/debug use |
| `download_job_components` | — | No — specialized, not a common task |
| `configure_job_log` | Both | Yes -> `nvflare job log-config <job_id>` |
| `app_command` | — | No — app-specific, no generic CLI shape |
| `delete_workspace` | — | No — disabled |

### System (`training` + `sys` scopes)

| Command | User | CLI |
| --- | --- | --- |
| `check_status` | Both | Yes -> `nvflare system status` |
| `report_resources` | Admin | Yes -> `nvflare system resources` |
| `shutdown` | Admin | Yes -> `nvflare system shutdown` |
| `restart` | Admin | Yes -> `nvflare system restart` |
| `remove_client` | Admin | Yes -> `nvflare system remove-client` |
| `sys_info` | Both | Yes -> `nvflare system version` |
| `report_env` | — | No — workspace paths; internal/debug |
| `show_scopes` | — | No — internal configuration detail |
| `configure_site_log` | Both | Yes -> `nvflare system log-config <level>` |

### Observability (`info` scope)

| Command | User | CLI |
| --- | --- | --- |
| `show_stats` | Both | Yes -> `nvflare job stats <job_id>` |
| `show_errors` | Both | No — not exposed as a CLI command |
| `reset_errors` | — | No — internal housekeeping |


## Log Operations

### Naming

- `job logs` (plural) = read log content
- `job log-config` = write logging configuration

Current state: users have been using interactive console shell commands (`cat`, `tail`, `grep` via `tail_target_log` / `grep_target` on `Session`) to find errors. These commands are unstructured, security-sensitive, and not agent-usable. The commands below replace this with a proper log API.

When client-side log streaming to the server is enabled, client logs are treated as server-side stored job logs. `nvflare job logs` does not connect to client sites directly and does not execute shell commands on client machines. It asks the server for the job log content that the server has locally for the requested site target.

Client log streaming is enabled by the job, not by `nvflare job logs`. The portable job-level pattern is:

```python
from nvflare.app_common.logging.job_log_receiver import JobLogReceiver
from nvflare.app_common.logging.job_log_streamer import JobLogStreamer

recipe.job.to_clients(JobLogStreamer())
recipe.job.to_server(JobLogReceiver())
```

`JobLogStreamer` runs in each client job process, tails the job `log.txt`, and streams bytes to the server. `JobLogReceiver` runs on the server side for that job, writes incoming chunks into the server job workspace as `<client_name>/log.txt`, and hands the file to the job manager so it is included in the archived job `workspace` artifact. Server-side resource configuration may also provide a global `JobLogReceiver`, but jobs that include both components are self-contained across POC and production deployments.

### `nvflare job logs`

```text
nvflare job logs <job_id> [--site server|<client_name>|all]
```

`--site` defaults to `server` for bounded output and backward-compatible behavior.

Site target behavior:

- `--site server`: return the server-side job log.
- `--site <client_name>`: return that client's job log as streamed to and stored by the server.
- `--site all`: return the server log plus all client logs available in the server-side job log store.

The command never applies local filtering such as `tail` or `grep`; users can pipe or post-process the returned log content if needed.

Human output prints log text directly. For a single `--site server` or
`--site <client_name>` target, no JSON envelope or dict wrapper is printed. For
`--site all`, each site is separated by a short header. `--format json` keeps the
structured response shown below.

Example:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "job_id": "abc123",
    "target": "all",
    "logs": {
      "server": "2026-03-27 10:01:00 INFO ...\n...",
      "site-1": "2026-03-27 10:01:05 ERROR ...\n...",
      "site-2": "2026-03-27 10:01:05 INFO ...\n..."
    }
  }
}
```

If `--site all` is requested and some known job sites do not have stored log content, the command should still return the logs that are available and report missing sites separately. A zero-byte log file is still returned as an available empty string; a missing log file is reported under `unavailable`.

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "job_id": "abc123",
    "target": "all",
    "logs": {
      "server": "2026-03-27 10:01:00 INFO ...\n...",
      "site-1": "2026-03-27 10:01:05 ERROR ...\n..."
    },
    "unavailable": {
      "site-2": "client log stream not available for this job"
    }
  }
}
```

For a specific requested site, missing log content is an error instead of a partial success:

```text
Error: job logs are not available for site 'site-2'
Hint: Verify that client log streaming is enabled and that the site has run this job.
Code: LOG_NOT_FOUND (exit 1)
```

Server-side behavior: `get_job_log <job_id> [server|all|client_name]` returns structured data from server-side stored artifacts. Server logs are read from the live server workspace when available, then from the saved job-store `workspace` component after the run workspace has been archived. Client logs are read from the server's live job workspace at `<job_id>/<client_name>/log.txt` when available, then from the saved job-store `workspace` component member `<client_name>/log.txt`. For compatibility with existing stored receiver outputs, the command can also fall back to client-data components such as `LOG_log.txt_<client_name>`. `tail_target_log` / `grep_target` are insufficient and are not used for this command.

Session API: `Session.get_job_logs(job_id, target, tail_lines=None, grep_pattern=None)` sends the structured server command and returns `logs` plus optional `unavailable`. `tail_lines` and `grep_pattern` are retained as deprecated compatibility arguments for existing Python callers; the CLI no longer exposes these options, and any filtering is applied locally by the session wrapper after retrieving the server-side stored logs.

### `nvflare job log-config`

```text
nvflare job log-config <job_id> [--site server|<client_name>|all] <level_or_mode>
```

Examples:

```bash
nvflare job log-config abc123 DEBUG --site site-1
nvflare job log-config abc123 msg_only --site all
```

Response always uses `sites`:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "job_id": "abc123",
    "sites": ["site-1"],
    "config": "DEBUG"
  }
}
```

### `nvflare system log-config`

```text
nvflare system log-config [--site server|<client_name>|all] <level_or_mode>
```

Same positional form as `nvflare job log-config` but applies at site level. Effective immediately; persists until next restart or reload.

Session API change needed: add `configure_job_log(job_id, config, target)` and `configure_site_log(config, target)` to `Session` in `flare_api.py`.


## Log Retrieval

| Command | Description |
| --- | --- |
| `nvflare job logs <job_id>` | Download job log content from the server-side log store, including streamed client logs when available |

```text
nvflare job logs     <job_id> [--site server|<client_name>|all]
```


## Recipe Catalog

New top-level subcommand. No server connection required.

| Command | Description |
| --- | --- |
| `nvflare recipe list` | List all built-in FL workflow recipes |

```text
nvflare recipe list [--framework <framework>]
```

`--framework` filters by framework (for example `pytorch`, `tensorflow`, `sklearn`). Without filters, returns all recipes whose optional dependencies are installed.

Recipe classes declare metadata as class-level attributes (`recipe_name`, `recipe_description`, `recipe_frameworks`, `recipe_min_clients`). The CLI discovers them at runtime via `importlib` + `inspect`.

Example:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": [
    {
      "name": "fedavg",
      "description": "Federated Averaging for PyTorch",
      "frameworks": ["pytorch"],
      "min_clients": 2,
      "recipe_class": "nvflare.app_opt.pt.recipes.fedavg.FedAvgRecipe"
    }
  ]
}
```


## Monitor / Poll

`nvflare job monitor <job_id>` polls until the job reaches a terminal state via `monitor_job_and_return_job_meta()` callback. Status lines print to stderr; final JSON to stdout.

```bash
JOB=$(nvflare job submit -j ./my_job | jq -r .data.job_id)
nvflare job monitor $JOB && nvflare job download $JOB
```

Exit code 0 on `FINISHED_OK`, exit code 1 on `FAILED` / `ABORTED`.


## Shell Commands

Commands like `pwd`, `ls`, `cat`, `head`, `tail`, and `grep` are not exposed in the CLI. Direct client shell targets were removed per FLARE-2808; `nvflare job logs --site <client_name>` is not a shell target and only reads client logs already streamed to the server-side log store. Server shell targets remain in the interactive console only for controlled debug use.


## Hidden Admin Console Commands

These do not appear in help and should not be exposed in the CLI.

| Command | Module | Purpose | CLI? |
| --- | --- | --- | --- |
| `admin_check_status` | `training_cmds.py` | Internal admin-level status | No |
| `dead` | `sys_cmd.py` | Internal recovery logic | No |
| `_cert_login` | `login.py` | Certificate-based login handshake | No |
| `_logout` | `login.py` | Session logout | No |
| `list_sessions` | `sess.py` | List active admin sessions | No |
| `_check_session` | `sess.py` | Validate session token | No |
| `_commands` | `builtin.py` | Return full command registry | No |
| `pull_binary_file` | `file_transfer.py` | Internal binary file pull | No |
| `push_folder` | `file_transfer.py` | Internal folder push | No |


## Startup Kit Registry and Active Kit

Normal server-connected commands use one active local startup kit. Users should not need
to pass `--startup-target`, `--startup-kit`, or a startup-kit path on every command.

Common POC flow:

```bash
nvflare poc prepare
nvflare poc start
nvflare job list
nvflare job submit -j ./job
```

`poc prepare` registers the generated Project Admin startup kit and makes it active.

Production flow:

```bash
nvflare config kit add alice_example_project /secure/startup_kits/example_project/alice@nvidia.com
nvflare config kit use alice_example_project
nvflare job list
```

Multiple local identities are handled by activating a different registered ID:

```bash
nvflare config kit add cancer_lead /secure/startup_kits/cancer/lead@nvidia.com
nvflare config kit add fraud_org_admin /secure/startup_kits/fraud/org_admin@nvidia.com
nvflare config kit use cancer_lead
nvflare config kit show
```

Activation is local config mutation only. It does not contact the server.

### Startup Kit Registry Storage

`~/.nvflare/config.conf` is the local storage file for the startup kit registry. The
user-facing command is `nvflare config kit`; the stored `startup_kits` data is an
implementation detail, not a separate customer-facing concept.

```hocon
version = 2

startup_kits {
  active = "admin@nvidia.com"

  entries {
    "admin@nvidia.com" = "/tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com"
    "lead@nvidia.com" = "/tmp/nvflare/poc/example_project/prod_00/lead@nvidia.com"
    "org-admin@nvidia.com" = "/tmp/nvflare/poc/example_project/prod_00/org-admin@nvidia.com"
    cancer_lead = "/secure/startup_kits/cancer/lead@nvidia.com"
    fraud_org_admin = "/secure/startup_kits/fraud/org_admin@nvidia.com"
  }
}
```

The registry key is a local ID. It is not a server-side object, not a certificate role,
and not a study role. For POC-generated user startup kits, the default ID is the user
identity, such as `admin@nvidia.com` or `lead@nvidia.com`. Site startup kits are service
identities and are not registered in this CLI identity registry. For production kits, users
choose meaningful local IDs such as `cancer_lead`.

The entry value is only the startup kit path. Metadata such as identity and certificate
role is inspected from the startup kit when needed by commands such as `nvflare config kit show`;
it is not duplicated in `config.conf`. Fields that are not reliably derivable from the
startup kit are omitted from normal output.

Other CLI state may also be stored in `~/.nvflare/config.conf`, but it is outside the
startup kit registry and should not be part of the normal user workflow.

### `nvflare config kit`

`nvflare config kit` manages local startup kit registrations. It is not a server-side resource.
Running `nvflare config kit` without a subcommand should print help and exit without an
`INVALID_ARGS` error.

#### `nvflare config kit add <id> <path>`

Registers a startup kit location under a local ID.

Behavior:

1. Read `~/.nvflare/config.conf`.
2. Validate that `<path>` is a valid admin/user startup kit directory. A valid admin/user
   startup kit must contain all three of:
   - `startup/fed_admin.json`
   - `startup/client.crt`
   - `startup/rootCA.pem`
   Site startup kits are not accepted because they cannot be used for server-connected
   admin CLI sessions.
3. Fail if `<id>` already exists unless `--force` is provided.
4. Create or replace `startup_kits.entries.<id> = "<path>"`.
5. Do not change `startup_kits.active`. Registration and activation are separate.
6. Write config atomically.

Example output:

```text
registered startup kit: alice_example_project
path: /secure/startup_kits/example_project/alice@nvidia.com
next_step: nvflare config kit use alice_example_project
```

#### `nvflare config kit use <id>`

Makes a registered startup kit active.

Behavior:

1. Read `~/.nvflare/config.conf`.
2. Resolve `<id>` to a path from `startup_kits.entries`.
3. Validate that the path still passes the admin/user startup kit check.
4. Set `startup_kits.active = "<id>"`.
5. Write config atomically.
6. Print the active ID and path. Identity and role can be displayed when they can be
   inspected from the startup kit.

No server call is made. The next server-connected command creates a session using the
selected startup kit.

#### `nvflare config kit show`

Shows the configured active startup kit:

```text
active: cancer_lead
identity: lead@nvidia.com
cert_role: lead
path: /secure/startup_kits/cancer/lead@nvidia.com
```

`kit show` reports `startup_kits.active`. It does not replace that value with
`NVFLARE_STARTUP_KIT_DIR`, but it should warn when the environment variable is set:

```text
warning: NVFLARE_STARTUP_KIT_DIR is set (/secure/startup_kits/cancer/lead@nvidia.com)
         normal commands will use this path instead of the active kit above
```

If the active path is missing or invalid, show the stale registration and suggest
`nvflare config kit use <id>` or `nvflare config kit remove <id>`.

#### `nvflare config kit list`

Lists registered startup kits. The command checks each registered path locally and flags
stale entries without contacting the server:

```text
* cancer_lead       ok       lead@nvidia.com        lead          /secure/startup_kits/cancer/lead@nvidia.com
  fraud_org_admin   missing  -                      -             /secure/startup_kits/fraud/org_admin@nvidia.com
  old_lab_admin     invalid  -                      -             /archive/old_lab/admin@nvidia.com
```

The active startup kit is marked with `*`. Missing or invalid paths are shown as
`missing` or `invalid`. Valid site kits are not shown because they are not CLI identities.

#### `nvflare config kit remove <id>`

Removes a local config registration. It does not delete the startup kit directory. If the
removed ID is active, clear `startup_kits.active` and print:

```text
removed startup kit: cancer_lead
warning: no active startup kit is configured
next_step: nvflare config kit use <id>
```

### Resolution Model

All server-connected commands, including `nvflare job`, `nvflare study`, `nvflare system`,
and `nvflare network`, resolve the startup kit using this ordered lookup:

1. If `NVFLARE_STARTUP_KIT_DIR` is set, validate it with the same three-file admin startup
   kit check and use it.
2. Otherwise, use `startup_kits.active` from `~/.nvflare/config.conf`.
3. If neither source resolves to a valid admin/user startup kit, fail before any server
   connection attempt.

The common user documentation should teach the active startup kit model. The environment
variable is an automation override, not the primary user workflow.

### POC Integration

`poc prepare` continues to create the default POC workspace and provision the POC project.
It also registers generated admin/user startup kits in the shared registry.

Behavior:

1. Provision the POC project as today.
2. Find generated admin/user startup kits in the POC prod directory, normally
   `prod_00` after the first `poc prepare`.
3. Keep generated site startup kits in the POC workspace for local service management, but
   do not register them as CLI identities.
4. Read participant identity, org, type, and role from the POC project metadata.
5. Register the Project Admin kit by identity, such as `admin@nvidia.com`.
6. Register each additional admin/user kit by identity, such as `lead@nvidia.com`.
7. Set `startup_kits.active` to the identity of the first `project_admin` participant
   found in the project metadata.
8. If no `project_admin` participant exists, do not set `startup_kits.active`; print a
   hint to create or register an admin startup kit before running admin CLI commands.

`poc prepare --force` keeps its existing meaning: recreate the local POC workspace and
rerun provisioning. It does not mean "override any startup kit registration with the same
ID." During registration, `poc prepare` may replace entries whose paths are under the POC
workspace being recreated. If an ID already points outside the POC workspace, fail with a
hint to remove or replace that kit explicitly.

On the `cli_enhancement2` branch, `dummy_project.yml` includes these local POC user
identities:

```text
admin@nvidia.com     role=project_admin
org-admin@nvidia.com role=org_admin
lead@nvidia.com      role=lead
```

Example result:

```text
startup_kits.active                         -> admin@nvidia.com
startup_kits.entries."admin@nvidia.com"     -> .../prod_00/admin@nvidia.com
startup_kits.entries."org-admin@nvidia.com" -> .../prod_00/org-admin@nvidia.com
startup_kits.entries."lead@nvidia.com"      -> .../prod_00/lead@nvidia.com
```

#### `nvflare poc add user <cert-role> <email> --org <org>`

Creates a new local POC user startup kit and registers it in the shared registry. Valid
`<cert-role>` values are `project_admin`, `org_admin`, `lead`, and `member`. These are
certificate roles, not study-specific roles.

Behavior:

1. Resolve the default POC workspace the same way existing POC commands do.
2. Resolve the active startup kit and require its certificate role to be `project_admin`.
   If `NVFLARE_STARTUP_KIT_DIR` is set, it participates in normal startup-kit resolution
   and must also point to a `project_admin` kit. Otherwise fail with `NOT_AUTHORIZED`
   before mutating project metadata.
3. Read the persisted POC `project.yml` from that workspace. This file is originally
   generated from the default POC `dummy_project.yml` baseline, then becomes the local
   source of truth for later POC additions.
4. If the identity does not exist, append one admin participant with the requested
   identity, organization, and certificate role. With `--force`, update an existing
   participant entry in place instead of appending a duplicate.
5. Locate the current POC output directory, normally `prod_00`, and the existing POC CA
   state/rootCA created by `poc prepare`.
6. Build a reduced dynamic provisioning project from the persisted POC project metadata.
   The reduced project contains the existing server and only the new admin participant.
   Existing clients, admins, and their startup kits are not regenerated.
7. Run the existing POC provisioning/build pipeline on that reduced project, using the
   existing CA state so the root CA is unchanged.
8. Move only the new participant startup kit into the current POC output directory and
   remove the temporary provisioning output. Existing participant directories in `prod_00`
   must remain untouched.
9. Persist the updated project YAML on disk.
10. Register the generated admin/user kit under startup kit ID `<email>`.
11. If there is no active startup kit, make the new kit active.
12. In the normal POC flow, leave the current active kit unchanged and print the activation
   command.

Example output when an active kit already exists:

```text
startup_kit: /tmp/nvflare/poc/example_project/prod_00/bob@nvidia.com
id: bob@nvidia.com
identity: bob@nvidia.com
cert_role: lead
next_step: nvflare config kit use bob@nvidia.com
```

Example output when no active kit exists:

```text
startup_kit: /tmp/nvflare/poc/example_project/prod_00/bob@nvidia.com
id: bob@nvidia.com
identity: bob@nvidia.com
cert_role: lead
active: bob@nvidia.com
```

After the first `poc prepare`, the initial output is normally `prod_00`. A later
`poc add user` is dynamic provisioning: it uses a temporary reduced build to generate only
the new user kit, then places that kit under the existing `prod_00`. The command must not
replace existing participant startup kits or repoint their registry entries.

If the user identity already exists, the command fails unless `--force` is provided. With
`--force`, update the existing project metadata entry in place, create a new startup kit,
and replace only the config registration for that ID. If that ID is active, the active ID
remains the same and now points to the new path.

#### `nvflare poc add site <name> --org <org>`

Creates a new local POC site startup kit and records it in the POC workspace. Site kits
are not registered in the startup kit registry because they are service identities, not
interactive CLI user identities.

Behavior:

1. Resolve the default POC workspace the same way existing POC commands do.
2. Resolve the active startup kit and require its certificate role to be `project_admin`.
   If `NVFLARE_STARTUP_KIT_DIR` is set, it participates in normal startup-kit resolution
   and must also point to a `project_admin` kit. Otherwise fail with `NOT_AUTHORIZED`
   before mutating project metadata.
3. Read the persisted POC `project.yml` from that workspace.
4. Read the service metadata from that POC workspace.
5. If the site does not exist, append one client participant with the requested site name
   and organization. With `--force`, update an existing site participant entry in place
   instead of appending a duplicate.
6. Locate the current POC output directory, normally `prod_00`, and the existing POC CA
   state/rootCA created by `poc prepare`.
7. Build a reduced dynamic provisioning project from the persisted POC project metadata.
   The reduced project contains the existing server and only the new client participant.
   Existing participants are not regenerated.
8. Run the existing POC provisioning/build pipeline on that reduced project, using the
   existing CA state so the root CA is unchanged.
9. Move only the new site startup kit into the current POC output directory and remove the
   temporary provisioning output. Existing participant directories in `prod_00` must remain
   untouched.
10. Persist the updated project YAML on disk.
11. Update the POC service config so commands such as `nvflare poc start -p site-3` know the
   new site exists.
12. Print the generated startup kit path and start instructions.

`poc add site` does not take a separate workspace argument. It is scoped to the active
local POC workspace. Adding a site does not switch the active startup kit because site
kits are service identities, not interactive admin identities.

If the POC system is running, `poc add site` should still be allowed. Existing running
services are not stopped; the new startup kit is used when the site is started or
restarted.

#### `nvflare poc clean`

`poc clean` removes the local POC workspace and clears POC-generated config entries.

Behavior:

1. If the POC system is still running, fail unless `--force` is provided. With `--force`,
   stop the local POC system before cleanup.
2. Remove the POC workspace.
3. Remove startup kit entries whose canonical paths are under the canonical POC workspace
   path.
4. Remove local POC workspace state from `~/.nvflare/config.conf`.
5. If the active startup kit was removed, clear `startup_kits.active`.
6. Leave manually registered production startup kits untouched.

The "under the POC workspace" check must use canonical path comparison, such as
`Path.resolve()` plus `Path.is_relative_to()` or `os.path.commonpath()`. It must not use a
raw string prefix check because `/tmp/nvflare/poc-backup` is not under `/tmp/nvflare/poc`.

### Startup Kit and POC Error Handling

General rules:

- Validate before mutating `~/.nvflare/config.conf`.
- Write config atomically so a failed command does not leave a partially written config.
- Do not contact the server for `nvflare config kit` registration, activation, listing, or
  removal errors.
- Treat identity, role, org, and project inspection as best effort. Failure to inspect
  metadata should not break `kit add` or `kit use` when the startup kit path itself is
  valid.

If `~/.nvflare/config.conf` does not exist:

- `nvflare config kit list` prints an empty list.
- `nvflare config kit show` reports that no active startup kit is configured.
- Normal server-connected commands fail before connecting:

```text
Error: no active startup kit is configured
Hint: Run nvflare poc prepare, or run nvflare config kit add <id> <startup-kit-dir> then nvflare config kit use <id>.
```

If config parsing fails, commands stop without modifying the file:

```text
Error: cannot parse ~/.nvflare/config.conf
Hint: Fix the config file, or move it aside and run nvflare poc prepare.
```

If `startup_kits.active` points to an ID that is not in `startup_kits.entries`:

```text
Error: active startup kit 'cancer_lead' is not registered
Hint: Run nvflare config kit list, then nvflare config kit use <id>.
```

`kit add` error cases:

```text
Error: startup kit path does not exist: /secure/startup_kits/cancer/lead@nvidia.com
```

```text
Error: path is not a valid startup kit: /secure/startup_kits/cancer/lead@nvidia.com
Hint: Use the participant startup directory produced by provisioning.
```

```text
Error: startup kit id 'cancer_lead' already exists
Hint: Use --force to replace this local registration.
```

IDs may contain characters such as `@`, `.`, and `-`. Empty IDs are invalid. If the same
path is registered under more than one ID, allow it; that is local aliasing and does not
affect server behavior.

`kit use` error cases:

```text
Error: startup kit id 'cancer_lead' is not registered
Hint: Run nvflare config kit list.
```

```text
Error: startup kit path for 'cancer_lead' does not exist: /secure/startup_kits/cancer/lead@nvidia.com
Hint: Restore the startup kit, remove the registration, or activate another kit.
```

```text
Error: registered path for 'cancer_lead' is not a valid startup kit
Hint: Run nvflare config kit remove cancer_lead, or replace it with nvflare config kit add cancer_lead <startup-kit-dir> --force.
```

`kit use` must not change `startup_kits.active` until the selected ID resolves to a valid
startup kit path.

`kit list` should not fail just because one registered path is stale. It marks each entry
independently as `ok`, `missing`, or `invalid`. If the active entry is stale, keep the `*`
marker and show the stale status.

`poc prepare` registers POC-generated user kits by identity. `--force` is reserved for
recreating the POC workspace and rerunning provisioning; it is not a general startup kit
registry override. If a target ID already exists:

- If the existing path is under the current POC workspace, replace it.
- If the existing path is outside the current POC workspace, fail and require an explicit
  registry action such as `nvflare config kit remove <id>` or
  `nvflare config kit add <id> <path> --force`.

```text
Error: startup kit id 'lead@nvidia.com' already points outside the POC workspace
Hint: Run nvflare config kit remove lead@nvidia.com, or replace it explicitly with nvflare config kit add lead@nvidia.com <startup-kit-dir> --force.
```

`poc add user` fails before creating or registering a new kit when the requested identity
already exists in the POC project, unless `--force` is provided. With `--force`, update
the existing participant metadata in place and persist it to disk; do not append a
duplicate participant.

`poc add site` follows the same duplicate rule for sites. If services are running, the
command may add the new site kit but must not restart existing services automatically.

Both `poc add user` and `poc add site` require the active startup kit to have the
`project_admin` certificate role. A lead, org admin, member, missing role, or stale kit
must fail before project metadata is changed.

`poc clean` removes only startup kit entries whose canonical paths are under the canonical
POC workspace path. If the active startup kit is removed, clear `startup_kits.active`.
Manual production registrations remain untouched.

### Startup Kit Behavior Guarantees

- `poc prepare` activates the default POC Project Admin startup kit automatically.
- `poc prepare` writes POC admin/user startup kits into `startup_kits.entries`.
- Normal server-connected commands do not expose `--startup-target` or `--startup-kit`.
- Commands resolve the startup kit through `NVFLARE_STARTUP_KIT_DIR` first, then
  `startup_kits.active`.
- `nvflare config kit` is the user-facing startup kit management interface.
- `nvflare config` is the parent command namespace; the normal user workflow in this
  design is `nvflare config kit`.
- `poc add user` registers generated user kits in `startup_kits.entries`.
- `poc add site` generates site kits and updates POC workspace metadata, but it does not
  register site kits in `startup_kits.entries`.
- `poc add user` and `poc add site` do not switch away from the active kit in the normal
  POC flow.
- `poc add site` is allowed while POC services are running and does not stop existing
  services.
- For `nvflare config kit add` and `nvflare poc add user`, `--force` replaces the config
  registration for an existing ID without deleting old startup kit directories.
- For `nvflare poc prepare`, `--force` means recreate the POC workspace; it does not
  override unrelated startup kit registrations outside the POC workspace.
- Persona commands are not needed. The common switching command is `nvflare config kit use <id>`.


## Existing Command Reference

All commands need `--schema` and always emit a JSON envelope unless marked otherwise.

### `nvflare poc`

#### `nvflare poc prepare`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-n`, `--number_of_clients` | int | No | 2 | Number of clients |
| `-c`, `--clients` | str... | No | `[]` | Space-separated client names |
| `-he`, `--he` | flag | No | — | Enable homomorphic encryption |
| `-i`, `--project_input` | str | No | `""` | Path to `project.yaml` |
| `-d`, `--docker_image` | str | No | `None` | Docker image for `docker.sh` generation |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--schema` | flag | No | — | Print command schema and exit |

On success, auto-invoke `install_skills()`; exit 0 on failure because skills are optional.
It also registers generated admin/user startup kits in `startup_kits.entries` and
activates the first Project Admin kit.

#### `nvflare poc start`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--service` | str | No | `"all"` | Participant to start |
| `-ex`, `--exclude` | str | No | `""` | Exclude service directory |
| `-gpu`, `--gpu` | int... | No | `None` | GPU device IDs |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--schema` | flag | No | — | Print command schema and exit |

Must emit `server_url` in the JSON data field when server accepts connections.

#### `nvflare poc stop`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--service` | str | No | `"all"` | Participant to stop |
| `-ex`, `--exclude` | str | No | `""` | Exclude service directory |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--schema` | flag | No | — | Print command schema and exit |

#### `nvflare poc clean`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--force` | flag | No | — | Stop a running POC system before cleanup |
| `--schema` | flag | No | — | Print command schema and exit |

#### `nvflare poc add user`

```text
nvflare poc add user <cert-role> <email> --org <org> [--force] [--schema]
```

Creates or refreshes a local POC admin/user participant from the default POC project YAML,
dynamically generates only that participant's startup kit with the existing POC CA, and
registers that user kit under startup kit ID `<email>`.

#### `nvflare poc add site`

```text
nvflare poc add site <name> --org <org> [--force] [--schema]
```

Creates or refreshes a local POC client participant from the default POC project YAML,
dynamically generates only that participant's startup kit with the existing POC CA, and
updates POC service metadata so `nvflare poc start -p <name>` can manage the new site.

### `nvflare job`

#### `nvflare job create` — deprecated

Retain with stderr deprecation warning. Migrate to `python job.py --export --export-dir <job_folder>` + `nvflare job submit -j <job_folder>`.

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-j`, `--job_folder` | str | No | `"./current_job"` | Job folder path |
| `-w`, `--template` | str | No | `"sag_pt"` | Template name or folder |
| `-sd`, `--script_dir` | str | No | — | Additional script directory |
| `-f`, `--config_file` | str (append) | No | — | Config file with optional `key=value` overrides |
| `-force`, `--force` | flag | No | — | Overwrite existing config |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--schema` | flag | No | — | Print command schema and exit |

#### `nvflare job submit`

Submits a pre-built job config folder to a running server. Returns `job_id` immediately
with no waiting. The job artifact is unchanged at submit time. Server connection uses the
active startup kit described in §Resolution Model. Use `nvflare job monitor` to wait
for results.

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-j`, `--job-folder` | str | No | `"./current_job"` | Pre-built job config folder |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--study` | str | No | `"default"` | Study to submit the job to |
| `--schema` | flag | No | — | Print command schema and exit |

Output:

```json
{"schema_version": "1", "status": "ok", "data": {"job_id": "abc123"}}
```

#### `nvflare job monitor`

Polls a running job until it reaches a terminal state. Prints status updates to stderr; final JSON to stdout.

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `job_id` | str | Yes | — | Job ID to monitor |
| `--timeout` | int | No | 0 | Max seconds to wait |
| `--interval` | int | No | 2 | Poll interval in seconds |
| `--schema` | flag | No | — | Print command schema and exit |

Exit code 0 on `FINISHED_OK`, exit code 1 on `FAILED` / `ABORTED`.

#### `nvflare job list-templates` — deprecated

Retain with stderr warning; use `nvflare recipe list` instead. Underscore alias `list_templates` accepted for backward compatibility.

#### `nvflare job show-variables` — deprecated

Retain with stderr warning; job variables are defined and inspected via the Job Recipe API. Underscore alias `show_variables` accepted for backward compatibility.

### `nvflare provision`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--project-file` | str | No | — | `project.yaml` path |
| `-g`, `--generate` | flag | No | — | Generate sample `project.yaml` |
| `-e`, `--gen-edge` | flag | No | — | Generate sample edge `project.yaml` |
| `-w`, `--workspace` | str | No | `"workspace"` | Workspace directory |
| `-c`, `--custom-folder` | str | No | `"."` | Additional Python code folder |
| `--add-user` | str | No | `""` | YAML file for added user |
| `--add-client` | str | No | `""` | YAML file for added client |
| `-s`, `--gen-scripts` | flag | No | — | Generate startup scripts |
| `--force` | flag | No | — | Skip Y/N confirmation prompts |
| `--schema` | flag | No | — | Print command schema and exit |

### `nvflare simulator` — deprecated

Retain with stderr warning. Use the Job Recipe `SimEnv` directly from Python.

### `nvflare preflight-check`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--package-path` | str | Yes | — | Path to startup kit package |
| `--schema` | flag | No | — | Print command schema and exit |

Exit code must be 0 on all checks pass, 1 on any failure.

### `nvflare config`

`nvflare config` is the parent command namespace for local CLI settings. The startup kit
workflow is exposed through `nvflare config kit`; users should not need to edit or reason
about the underlying storage layout.

Persistence rules:

1. `version = 2` is always the first line in the saved config.
2. Once a config is loaded and re-saved by the CLI, it is normalized to the v2 layout in
   §Startup Kit Registry Storage.
3. Startup kit entries are preserved when unrelated config fields are updated.

### Admin username and `@` in directory names

Admin startup kit directories and startup kit IDs are often email addresses such as
`admin@nvidia.com`. Two issues arise from the `@` character:

1. **Cert name validation** — `_validate_safe_cert_name` currently uses the regex `[A-Za-z0-9][A-Za-z0-9._-]*`, which rejects `@`. The regex must be updated to `[A-Za-z0-9][A-Za-z0-9._@-]*` to allow email-format admin names. Both call sites must be updated: the `--name` argument handler in `cert_commands.py` and the CSR subject CN validator.

2. **Startup kit IDs and paths with `@`** — When an ID or path containing `@` is passed
   on the command line, most shells pass it through unchanged. Some shell configurations
   may attempt expansion on `@`. NVFlare cannot control shell parsing; if a shell expands
   `@`, the user must quote the argument. The config writer must quote IDs when HOCON
   requires quoting.

**Implementation requirement:** update `_SAFE_CERT_NAME_PATTERN` in `nvflare/tool/cert/cert_commands.py` from `[A-Za-z0-9][A-Za-z0-9._-]*` to `[A-Za-z0-9][A-Za-z0-9._@-]*`. No change is needed in argparse or path resolution — `@` is valid at the filesystem level and passes through argparse unchanged.

---

The root `nvflare config` command is a parent namespace in this design. The documented
user-facing workflow is `nvflare config kit`; other local config storage is implementation
state and should not be part of normal user workflows.

### `nvflare dashboard` — deprecated (under review)

No agent-readiness work planned until direction is settled.

### `nvflare authz-preview` — deprecated

Retain with stderr warning. Underscore alias `authz_preview` accepted for backward compatibility.


## Agent-Readiness Change Summary for Existing Commands

| Command | JSON default | `--schema` | Exit Codes | Other |
| --- | --- | --- | --- | --- |
| `poc prepare` | Add | Add | Add | Add `--force`; auto-invoke `install_skills()` on success |
| `poc add user` | Add | Add | Add | Add POC user startup kit and register it in `startup_kits.entries` |
| `poc add site` | Add | Add | Add | Add POC site startup kit and update POC service metadata; site kits are not CLI identities |
| `poc start` | Add | Add | Add | `server_url` in JSON data field when ready |
| `poc stop` | Add | Add | Add | — |
| `poc clean` | Add | Add | Add | Add `--force` to stop a running local POC system before cleanup |
| `config kit add/use/show/list/remove` | Add | Add | Add | Manage local startup kit registry; no server connection |
| `job create` | — | — | — | Deprecated |
| `job submit` | Add | Add | Add | Returns `job_id` immediately |
| `job monitor` | Add | Add | Add | Standalone wait/poll |
| `study register/show/list/remove/add-site/remove-site/add-user/remove-user` | Add | Add | Add | Manage multi-study lifecycle through active startup kit |
| `provision` | Add | Add | Add | Restore pre-2.7.0 default; add `--force` |
| `preflight-check` | Add | Add | Fix | 0=pass, 1=fail; alias `preflight_check` kept |
| `config` | Add | Add | Add | Parent namespace for `config kit`; no normal server connection |
| `job list-templates` | — | — | — | Deprecated; alias `list_templates` kept |
| `job show-variables` | — | — | — | Deprecated; alias `show_variables` kept |
| `simulator` | — | — | — | Deprecated |
| `dashboard` | — | — | — | No changes |
| `authz-preview` | — | — | — | Deprecated; alias `authz_preview` kept |


## Design

### Extend `nvflare job`

Add missing operations to the existing `nvflare job` subcommand:

All server-connected `nvflare job` commands require a startup kit to be resolvable through
the active-kit registry or `NVFLARE_STARTUP_KIT_DIR`.

```text
# Job lifecycle (server must be running)
nvflare job submit    -j <job_folder>
nvflare job monitor   <job_id> [--timeout N] [--interval N]
nvflare job list      [-n prefix] [-i id_prefix] [-r] [-m num] [--study name|all]
nvflare job meta      <job_id>
nvflare job abort     <job_id> [--force]
nvflare job clone     <job_id>
nvflare job download  <job_id> [-o destination]
nvflare job delete    <job_id> [--force]

# Observability (server must be running)
nvflare job stats     <job_id> [--site server|<name>|all]
nvflare job logs      <job_id> [--site server|<name>|all]
nvflare job log-config <job_id> [--site server|<name>|all] <level>
```

### Add `nvflare study`

All `nvflare study` commands connect to the server and require a startup kit to be
resolvable through the active-kit registry or `NVFLARE_STARTUP_KIT_DIR`.

```text
nvflare study register    <name> [--sites <site> [<site> ...] | --site-org <org>:<site>[,<site>...]]
nvflare study show        <name>
nvflare study list
nvflare study remove      <name>
nvflare study add-site    <name> [--sites <site> [<site> ...] | --site-org <org>:<site>[,<site>...]]
nvflare study remove-site <name> [--sites <site> [<site> ...] | --site-org <org>:<site>[,<site>...]]
nvflare study add-user    <name> <user>
nvflare study remove-user <name> <user>
```

Site enrollment is role-sensitive:

- `project_admin` uses `--site-org <org>:<site>[,<site>...]`.
- `org_admin` uses `--sites <site> [<site> ...]`.
- `--sites` accepts either space-delimited or comma-delimited input.
- `--sites` and `--site-org` are mutually exclusive.

### Add `nvflare recipe`

```text
nvflare recipe list [--framework <framework>]
```

### Add `nvflare network`

All `nvflare network` commands query the server's cell network topology over an admin
session and require a startup kit to be resolvable through the active-kit registry or
`NVFLARE_STARTUP_KIT_DIR`.

```text
nvflare network cells
nvflare network peers        <target_cell>
nvflare network conns        <target_cell>
nvflare network route        <to_cell> [--from <from_cell>]
nvflare network msg-stats    <target> [--mode mode]
nvflare network list-pools   <target>
nvflare network show-pool    <target> <pool_name> [--mode mode]
nvflare network comm-config  <target>
nvflare network config-vars  <target>
nvflare network process-info <target>
```

All `nvflare network` commands support `--schema`. User-supplied arguments are validated against strict patterns before interpolation into `do_command()` strings.


## JSON Output Examples

```json
{"schema_version": "1", "status": "ok", "data": [
  {"job_id": "abc123", "name": "cifar10_fedavg", "status": "RUNNING", "submitted_at": "..."}
]}
```

```json
{"schema_version": "1", "status": "ok", "data": {"job_id": "abc123", "name": "..."}}
```

```json
{"schema_version": "1", "status": "error", "error_code": "JOB_NOT_FOUND", "hint": "Use 'nvflare job list' to see available job IDs."}
```

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "server": {"status": "running", "run_number": 3},
    "clients": [{"name": "site-1", "status": "online"}]
  }
}
```


## Add `nvflare system`

All `nvflare system` commands connect to the server and require a startup kit to be
resolvable through the active-kit registry or `NVFLARE_STARTUP_KIT_DIR`.

```text
nvflare system status        [server|client] [client_names...]
nvflare system resources     [server|client] [clients...]
nvflare system shutdown      <server|client|all> [client_names...] [--force]
nvflare system restart       <server|client|all> [client_names...] [--force]
nvflare system remove-client <client_name>
nvflare system log-config    [--site server|<client_name>|all] <level>
nvflare system version       [--site server|<name>|all]
```


## Output

Pass ``--format json`` to any command to receive a structured JSON envelope on stdout. Human-readable progress, warnings, and diagnostics go to stderr.

Exceptions: `--help`, top-level `--version`, and parser-generated usage errors remain plain text.

```text
--schema         Print JSON description of this command's arguments and exit
```

Example:

```json
{
  "schema_version": "1",
  "command": "nvflare job submit",
  "description": "Submit a pre-built job config folder to the FL server. Returns job_id immediately.",
  "args": [
    {"name": "-j/--job-folder", "type": "path", "required": false, "default": "./current_job", "description": "Pre-built job config folder"},
    {"name": "--schema", "type": "bool", "required": false, "default": false, "description": "Print command schema as JSON and exit"}
  ],
  "examples": ["nvflare job submit -j ./my_job"]
}
```


## Exit Codes

```text
0   Success
1   Server/job error (job not found, unauthorized, job FAILED/ABORTED)
2   Connection / authentication failure
3   Timeout
4   Invalid arguments  (includes InvalidTarget — unknown client name passed to shutdown/restart)
5   Internal error (unexpected exception — report a bug; do not retry)
```


## Implementation

### Startup Kit Resolution Implementation

All server-connected commands (`nvflare job`, `nvflare study`, `nvflare system`,
`nvflare network`) resolve the startup kit using the same ordered lookup:

1. `NVFLARE_STARTUP_KIT_DIR` — automation override. If set, validate it with the
   three-file admin startup kit check and use it.
2. `startup_kits.active` — resolve the active ID from `~/.nvflare/config.conf`, then
   validate the registered path.
3. If neither source resolves to a valid admin/user startup kit, fail before any server
   connection attempt.

This resolution order applies uniformly. Command-level descriptions that say "same resolution order as `nvflare job`" refer to this list.

Normal commands do not expose `--startup-target` or `--startup-kit`. Users switch local
identity with `nvflare config kit use <id>`. Automation can use `NVFLARE_STARTUP_KIT_DIR` when
mutating local config is undesirable.

Resolution error examples:

```text
Error: no active startup kit is configured
Hint: Run nvflare config kit use <id>.
```

```text
Error: NVFLARE_STARTUP_KIT_DIR does not point to a valid startup kit for admin use
Path: /secure/startup_kits/cancer/lead@nvidia.com
Hint: Unset NVFLARE_STARTUP_KIT_DIR, or set it to a valid admin startup kit directory.
```

```text
Error: active startup kit 'cancer_lead' points to a missing path
Path: /secure/startup_kits/cancer/lead@nvidia.com
Hint: Run nvflare config kit use <id> or nvflare config kit remove cancer_lead.
```

### Session Reuse

`nvflare job submit` already uses `new_secure_session()` with startup kit discovery. All
server-connected commands follow the same session pattern once the startup kit location is
resolved.

### Session API Coverage

`new_secure_session()` returns a `Session` from `nvflare/fuel/flare_api/flare_api.py`.

Already exposed:

- `submit_job`
- `abort_job`
- `clone_job`
- `download_job`
- `shutdown(target_type, client_names=None)` — `target_type` in `server|client|all`; closes session when server/all
- `restart(target_type, client_names=None)` — `target_type` in `server|client|all`
- `remove_client(client_name)` — removes a single connected client from the federation
- `check_status`
- `report_resources`
- `get_version`
- `configure_site_log`

Needs adding:

- `list_jobs`
- `get_job_meta`
- `delete_job`
- `get_job_logs(job_id, target, tail_lines=None, grep_pattern=None)`
- `configure_job_log(job_id, config, target)`

New methods are thin wrappers over `AdminAPI.do_command()`.

### New Files

| File | Purpose |
| --- | --- |
| `nvflare/tool/system/__init__.py` | Package |
| `nvflare/tool/system/system_cli.py` | All `nvflare system` subcommands |
| `nvflare/tool/recipe/__init__.py` | Package |
| `nvflare/tool/recipe/recipe_cli.py` | `nvflare recipe list` |
| `nvflare/tool/network/__init__.py` | Package |
| `nvflare/tool/network/network_cli.py` | All `nvflare network` handlers |
| `nvflare/tool/kit/__init__.py` | Package |
| `nvflare/tool/kit/kit_cli.py` | `nvflare config kit` command handlers |
| `nvflare/tool/kit/kit_config.py` | Startup kit registry config and metadata helpers |
| `nvflare/tool/agent/__init__.py` | Package |
| `nvflare/tool/agent/agent_cli.py` | `nvflare agent` subcommands |
| `nvflare/agent/__init__.py` | Agent package |
| `nvflare/agent/skills/nvflare-poc-setup.md` | POC setup skill |
| `nvflare/agent/skills/nvflare-job-submit.md` | Job workflow skill |
| `nvflare/agent/skills/nvflare-status-report.md` | Status data collection skill |
| `nvflare/agent/skills/flare_tools.json` | OpenAI / LangChain tool schemas |
| `nvflare/tool/cli_schema.py` | Shared parser serializer |
| `nvflare/tool/cli_output.py` | Shared output helpers |
| `nvflare/tool/cli_errors.py` | Error code registry |
| `nvflare/tool/install_skills.py` | Skill installer |

### Modified Files

| File | Change |
| --- | --- |
| `nvflare/tool/job/job_cli.py` | Add job lifecycle and observability subcommands |
| `nvflare/tool/study/study_cli.py` | Add multi-study lifecycle commands and active startup kit resolution |
| `nvflare/fuel/flare_api/flare_api.py` | Add missing session methods |
| `nvflare/cli.py` | Register new top-level commands |
| `nvflare/tool/cli_session.py` | Resolve `NVFLARE_STARTUP_KIT_DIR` or the active startup kit registry entry |
| `nvflare/tool/poc/poc_commands.py` | Call `install_skills()` on successful `poc prepare`; register generated POC user kits; support `poc add user/site` |
| `nvflare/lighter/provision.py` | Call `install_skills()` on successful `provision` |


## `nvflare install-skills`

```text
nvflare install-skills [--user-dir DIR] [--dry-run]
```

| Flag | Description |
| --- | --- |
| `--user-dir DIR` | Install to a custom directory |
| `--dry-run` | Print what would be installed without writing |

Install targets:

- `claude.skills_dir` -> `~/.claude/skills/nvflare/`
- `openclaw.skills_dir` -> `~/.openclaw/plugins/nvflare/`
- `./.claude/skills/nvflare/` for project-local installs by `poc prepare` / `provision`

On upgrade, existing `nvflare/` skill subdirectory is backed up to `.bak/<timestamp>/` before overwrite.


## Confirmation-Required Commands

`shutdown`, `restart`, `remove-client`, `delete`, and `abort` prompt for confirmation in the interactive console. CLI behavior:

- Default: prompt for confirmation in interactive terminal, or exit 4 in non-interactive mode
- `--force`: execute without confirmation


## Out of Scope

- All shell commands: not exposed in CLI
- Interactive mode: existing `nvflare_admin` console is unchanged
- `app_command`: job-specific and out of scope for initial delivery


## Version Compatibility

### `nvflare system version`

Queries each remote site for the NVFlare version running there.

```text
nvflare system version [--site server|<name>|all] [--schema]
```

Example:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "versions": {
      "server": {"nvflare": "2.8.0"},
      "site-1": {"nvflare": "2.8.0"},
      "site-2": {"nvflare": "2.7.4"}
    },
    "compatible": false,
    "mismatched_sites": ["site-2"]
  }
}
```

`compatible` is true only when all sites report the same NVFlare version as the server version.


## Appendix A: Recipe vs. ExecEnv Separation

### Background — What `job.py` Does Today

Today a user often writes a single `job.py` file that combines two concerns:

- Recipe: what FL algorithm, model, and hyperparameters to use
- ExecEnv: where and how to run

This works for humans but conflates decisions that should vary independently across environments.

### The Problem for CLI and Agent Use

When `ExecEnv` is embedded in `job.py`:

- an agent cannot reuse the same recipe against different environments without editing code
- CI/CD pipelines cannot run sim validation and POC integration from the same artifact

### The Solution — Exported Job Folder as the CLI Contract

For CLI and agent use, the user keeps recipe and execution concerns separate by exporting a job folder before submission:

```python
from nvflare.app_opt.pt.recipes.scaffold import ScaffoldRecipe

recipe = ScaffoldRecipe(
    model=Net(),
    num_rounds=5,
    min_clients=2,
)
```

The CLI-facing artifact is the exported job folder:

```bash
python job.py --export --export-dir ./my_job
nvflare job submit -j ./my_job
```

### ExecEnv Is Not Needed in the CLI Surface

| Surface | ExecEnv needed? | How env is selected |
| --- | --- | --- |
| `python job.py` | Yes | User writes it into `job.py` |
| `python job.py --export` + `nvflare job submit` | No | Exported job folder is env-agnostic |

The existing `job.py` pattern remains fully supported for human use.

### Agent Decision vs. Human Decision

For agentic workflows the agent can still validate in sim or POC through Python APIs and submit exported artifacts through the CLI. Production submission should remain an explicitly approved step.

### Summary

| | `job.py` (Python API) | exported job folder (CLI) |
| --- | --- | --- |
| ExecEnv embedded | Yes | No |
| Env selection | In code | Already resolved before submit |
| Reusable across envs | No | Yes |
| Agent-friendly | No | Yes |
| Human-friendly | Yes | Yes |
