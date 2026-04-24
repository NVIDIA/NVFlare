# NVFLARE CLI Design: Enhance NVFLARE CLI Commands

Created Date: 2026-03-26
Updated Date: 2026-04-14


## Problem Statement

NVFlare's admin operations are currently accessible only through an interactive console (`nvflare_admin`) — a stateful REPL that requires a human at the keyboard. This creates three problems:

- Not scriptable: automating FL workflows (submit job, wait, download results) requires fragile stdin piping or the deprecated FLAdminAPI.
- Not CI/CD friendly: pipeline systems (GitHub Actions, Airflow, Kubernetes Jobs) cannot drive an interactive console.
- Incomplete `nvflare job`: submit exists but list, abort, delete, download, clone do not.

Goal: make the full NVFlare admin command surface accessible non-interactively — scriptable from CI/CD pipelines, automation tools, and AI agents. This covers:

- `nvflare job` — extended with missing job lifecycle operations
- `nvflare system` — new subcommand for operational control (status, shutdown, restart, version)
- `nvflare recipe` — FL workflow recipe catalog, no server required
- `nvflare network` — cellnet diagnostics for advanced troubleshooting
- `nvflare agent` — bootstrap and context management for agentic workflows
- `nvflare install-skills` — installs skill files into agent framework discovery paths


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
| `nvflare poc` | `prepare`, `start`, `stop`, `clean` | Add JSON output, exit codes, `--schema`; add `--force` to `prepare` for workspace deletion prompt bypass |
| `nvflare provision` | — | Add JSON output, `--schema`, `--force` for Y/N prompts; restore pre-2.7.0 default: no args = generate `project.yml` |
| `nvflare preflight-check` | — | Add JSON output, `--schema`; exit 0=pass, 1=fail. Underscore alias `preflight_check` accepted for backward compatibility. |
| `nvflare config` | — | Add JSON output, `--schema` |
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

Current state: there is no proper API to retrieve or parse job logs. Users have been using interactive console shell commands (`cat`, `tail`, `grep` via `tail_target_log` / `grep_target` on `Session`) to find errors. These commands are unstructured, security-sensitive, and not agent-usable. The commands below replace this with a proper log API.

### `nvflare job logs`

```text
nvflare job logs <job_id> [--site server|<client_name>|all] [--tail N] [--grep PATTERN]
```

`--site` defaults to `all`. `--tail N` limits to last N lines per site. `--grep PATTERN` filters lines server-side before returning.

Example:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": {
    "job_id": "abc123",
    "log_source": "workspace",
    "logs": {
      "server": "2026-03-27 10:01:00 INFO ...\n...",
      "site-1": "2026-03-27 10:01:05 ERROR ...\n...",
      "site-2": "2026-03-27 10:01:05 INFO ...\n..."
    }
  }
}
```

Server-side change needed: add a new admin command that returns job log content as structured data. `tail_target_log` / `grep_target` are insufficient.

Session API change needed: add `get_job_logs(job_id, target, tail_lines, grep_pattern)` to `Session` in `flare_api.py`.

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
| `nvflare job logs <job_id>` | Download job log content from server store |

```text
nvflare job logs     <job_id> [--site server|<client_name>|all] [--tail N] [--grep PATTERN]
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

Commands like `pwd`, `ls`, `cat`, `head`, `tail`, and `grep` are not exposed in the CLI. Client targets were removed per FLARE-2808; server targets remain in the interactive console only for controlled debug use.


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
| `--schema` | flag | No | — | Print command schema and exit |

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

Submits a pre-built job config folder to a running server. Returns `job_id` immediately — no waiting. The job artifact is unchanged at submit time; `--startup-target` only selects which startup kit to use for server connection. If neither `--startup-target` nor `--startup-kit` is supplied, submit defaults to the POC startup kit from `~/.nvflare/config.conf`. Use `nvflare job monitor` to wait for results.

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-j`, `--job-folder` | str | No | `"./current_job"` | Pre-built job config folder |
| `--startup-target` | `poc|prod` | No | `poc` | Select startup kit from config; mutually exclusive with `--startup-kit` |
| `--startup-kit` | str | No | — | Explicit admin startup kit directory, or its `startup/` subdirectory; mutually exclusive with `--startup-target` |
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
| `--startup-target` | `poc\|prod` | No | `poc` | Select startup kit from config; mutually exclusive with `--startup-kit` |
| `--startup-kit` | str | No | — | Explicit admin startup kit directory; mutually exclusive with `--startup-target` |
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

Manages `~/.nvflare/config.conf`.

Config file keys use underscores (`startup_kit`, `poll_interval`, `job_timeout`) — this is the HOCON file convention and is distinct from CLI flag naming. CLI flags use dashes (`--startup-kit`, `--startup-target`); the config file is not a CLI surface.

Config file format:

```hocon
version = 2

poc {
    startup_kit = "/tmp/nvflare/poc/workspace/example_project/prod_00/admin@nvidia.com"
    workspace   = "/tmp/nvflare/poc/workspace"
}
prod {
    startup_kit = "/path/to/prod/admin_startup_kit"
}
job_template {
    path = "/path/to/job_templates"
}
agent {
    poll_interval = 10
    job_timeout   = 3600
}
claude {
    skills_dir = "~/.claude/skills/nvflare"
}
openclaw {
    skills_dir = "~/.openclaw/plugins/nvflare"
}
```

Persistence rules:

1. `version = 2` is always the first line in the saved config.
2. Once a config is loaded and re-saved by the CLI, it is normalized to the v2 layout above.
3. Legacy v1 keys such as `startup_kit.path` and `poc_workspace.path` are still read for backward compatibility, but they must not be written back alongside the v2 sections.
4. The compatibility alias `nvflare config --startup_kit_dir <path>` updates `poc.startup_kit` only. It does not populate `prod.startup_kit`.

### Admin username and `@` in directory names

Admin startup kit directories are named after the admin user, typically an email address such as `admin@nvidia.com`. Two issues arise from the `@` character:

1. **Cert name validation** — `_validate_safe_cert_name` currently uses the regex `[A-Za-z0-9][A-Za-z0-9._-]*`, which rejects `@`. The regex must be updated to `[A-Za-z0-9][A-Za-z0-9._@-]*` to allow email-format admin names. Both call sites must be updated: the `--name` argument handler in `cert_commands.py` and the CSR subject CN validator.

2. **Path argument with `@`** — When an admin startup kit path containing `@` is passed on the command line (e.g. `--startup-kit /tmp/nvflare/poc/example_project/prod_00/admin@nvidia.com`), most shells pass it through unchanged. Some shell configurations (notably zsh with certain glob options) may attempt expansion on `@`. NVFlare cannot control shell parsing — if a shell expands `@` in the path, the user must quote the argument: `--startup-kit '/path/to/admin@nvidia.com'`. The NVFlare-side fix is limited to the cert name validator: once `@` is permitted there, the extracted directory basename passes NVFlare's own validation. Shell quoting is the user's responsibility for shells that treat `@` specially.

**Implementation requirement:** update `_SAFE_CERT_NAME_PATTERN` in `nvflare/tool/cert/cert_commands.py` from `[A-Za-z0-9][A-Za-z0-9._-]*` to `[A-Za-z0-9][A-Za-z0-9._@-]*`. No change is needed in argparse or path resolution — `@` is valid at the filesystem level and passes through argparse unchanged.

---

For startup kit resolution order, see §Startup Kit Resolution under Implementation.

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--poc.startup_kit` | str | No | — | POC startup kit path |
| `--poc.workspace` | str | No | — | POC workspace path |
| `--prod.startup_kit` | str | No | — | Production startup kit path |
| `-jt`, `--job_templates_dir` | str | No | — | Job templates path |
| `--agent.poll_interval` | int | No | — | Job monitor poll interval |
| `--agent.job_timeout` | int | No | — | Job monitor timeout |
| `--claude.skills_dir` | str | No | — | Claude Code skill discovery path |
| `--openclaw.skills_dir` | str | No | — | OpenClaw plugin discovery path |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--describe` | flag | No | — | Print config file path, current contents, and format description |
| `--schema` | flag | No | — | Print command schema and exit |

### `nvflare dashboard` — deprecated (under review)

No agent-readiness work planned until direction is settled.

### `nvflare authz-preview` — deprecated

Retain with stderr warning. Underscore alias `authz_preview` accepted for backward compatibility.


## Agent-Readiness Change Summary for Existing Commands

| Command | JSON default | `--schema` | Exit Codes | Other |
| --- | --- | --- | --- | --- |
| `poc prepare` | Add | Add | Add | Add `--force`; auto-invoke `install_skills()` on success |
| `poc start` | Add | Add | Add | `server_url` in JSON data field when ready |
| `poc stop` | Add | Add | Add | — |
| `poc clean` | Add | Add | Add | — |
| `job create` | — | — | — | Deprecated |
| `job submit` | Add | Add | Add | Returns `job_id` immediately |
| `job monitor` | Add | Add | Add | Standalone wait/poll |
| `provision` | Add | Add | Add | Restore pre-2.7.0 default; add `--force` |
| `preflight-check` | Add | Add | Fix | 0=pass, 1=fail; alias `preflight_check` kept |
| `config` | Add | Add | Add | — |
| `job list-templates` | — | — | — | Deprecated; alias `list_templates` kept |
| `job show-variables` | — | — | — | Deprecated; alias `show_variables` kept |
| `simulator` | — | — | — | Deprecated |
| `dashboard` | — | — | — | No changes |
| `authz-preview` | — | — | — | Deprecated; alias `authz_preview` kept |


## Design

### Extend `nvflare job`

Add missing operations to the existing `nvflare job` subcommand:

All server-connected `nvflare job` commands require a startup kit to be resolvable. They accept `--startup-target poc|prod` and `--startup-kit <dir>` as mutually exclusive explicit selectors; see §Startup Kit Resolution for the full lookup order, including env/config fallback.

```text
# Job lifecycle (server must be running)
nvflare job submit    -j <job_folder> [--startup-target poc|prod | --startup-kit <dir>]
nvflare job monitor   <job_id> [--timeout N] [--interval N] [--startup-target poc|prod | --startup-kit <dir>]
nvflare job list      [-n prefix] [-i id_prefix] [-r] [-m num] [--study name|all] [--startup-target poc|prod | --startup-kit <dir>]
nvflare job meta      <job_id> [--startup-target poc|prod | --startup-kit <dir>]
nvflare job abort     <job_id> [--force] [--startup-target poc|prod | --startup-kit <dir>]
nvflare job clone     <job_id> [--startup-target poc|prod | --startup-kit <dir>]
nvflare job download  <job_id> [-o destination] [--startup-target poc|prod | --startup-kit <dir>]
nvflare job delete    <job_id> [--force] [--startup-target poc|prod | --startup-kit <dir>]

# Observability (server must be running)
nvflare job stats     <job_id> [--site server|<name>|all] [--startup-target poc|prod | --startup-kit <dir>]
nvflare job logs      <job_id> [--site server|<name>|all] [--tail N] [--grep PATTERN] [--startup-target poc|prod | --startup-kit <dir>]
nvflare job log-config <job_id> [--site server|<name>|all] <level> [--startup-target poc|prod | --startup-kit <dir>]
```

### Add `nvflare recipe`

```text
nvflare recipe list [--framework <framework>]
```

### Add `nvflare network`

All `nvflare network` commands query the server's cell network topology over an admin session and require a startup kit to be resolvable. They accept `--startup-target poc|prod` and `--startup-kit <dir>` as mutually exclusive explicit selectors; see §Startup Kit Resolution for the full lookup order, including env/config fallback.

```text
nvflare network cells        [--startup-target poc|prod | --startup-kit <dir>]
nvflare network peers        <target_cell> [--startup-target poc|prod | --startup-kit <dir>]
nvflare network conns        <target_cell> [--startup-target poc|prod | --startup-kit <dir>]
nvflare network route        <to_cell> [--from <from_cell>] [--startup-target poc|prod | --startup-kit <dir>]
nvflare network msg-stats    <target> [--mode mode] [--startup-target poc|prod | --startup-kit <dir>]
nvflare network list-pools   <target> [--startup-target poc|prod | --startup-kit <dir>]
nvflare network show-pool    <target> <pool_name> [--mode mode] [--startup-target poc|prod | --startup-kit <dir>]
nvflare network comm-config  <target> [--startup-target poc|prod | --startup-kit <dir>]
nvflare network config-vars  <target> [--startup-target poc|prod | --startup-kit <dir>]
nvflare network process-info <target> [--startup-target poc|prod | --startup-kit <dir>]
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

All `nvflare system` commands connect to the server and require a startup kit to be resolvable. They accept `--startup-target poc|prod` and `--startup-kit <dir>` as mutually exclusive explicit selectors; see §Startup Kit Resolution for the full lookup order, including env/config fallback.

```text
nvflare system status        [server|client] [client_names...] [--startup-target poc|prod | --startup-kit <dir>]
nvflare system resources     [server|client] [clients...] [--startup-target poc|prod | --startup-kit <dir>]
nvflare system shutdown      <server|client|all> [client_names...] [--force] [--startup-target poc|prod | --startup-kit <dir>]
nvflare system restart       <server|client|all> [client_names...] [--force] [--startup-target poc|prod | --startup-kit <dir>]
nvflare system remove-client <client_name> [--startup-target poc|prod | --startup-kit <dir>]
nvflare system log-config    [--site server|<client_name>|all] <level> [--startup-target poc|prod | --startup-kit <dir>]
nvflare system version       [--site server|<name>|all] [--startup-target poc|prod | --startup-kit <dir>]
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

### Startup Kit Resolution

All server-connected commands (`nvflare job`, `nvflare system`, `nvflare network`) resolve the startup kit using the same ordered lookup:

1. `--startup-kit <dir>` — explicit path on the command line
2. `NVFLARE_STARTUP_KIT_DIR` — environment variable
3. `--startup-target poc|prod` — config v2 key lookup (`poc.startup_kit` or `prod.startup_kit`)
4. default target `poc` — uses `poc.startup_kit` from config when `--startup-target` is absent
5. config v1 fallback — legacy keys read for backward compatibility; normalized to v2 on save

This resolution order applies uniformly. Command-level descriptions that say "same resolution order as `nvflare job`" refer to this list.

### Session Reuse

`nvflare job submit` already uses `new_secure_session()` with startup kit discovery. All new server-connected commands follow the same session pattern once the startup kit location is resolved.

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
- `get_job_logs(job_id, target, tail_lines, grep_pattern)`
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
| `nvflare/fuel/flare_api/flare_api.py` | Add missing session methods |
| `nvflare/cli.py` | Register new top-level commands |
| `nvflare/tool/poc/poc_commands.py` | Call `install_skills()` on successful `poc prepare` |
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
