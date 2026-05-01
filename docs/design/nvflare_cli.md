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
- `nvflare system` — new subcommand for operational control (status, shutdown, restart, client access, version)
- `nvflare recipe` — FL workflow recipe catalog, no server required
- `nvflare network` — cellnet diagnostics for advanced troubleshooting
- `nvflare agent` — bootstrap and context management for agentic workflows
- `nvflare install-skills` — installs skill files into agent framework discovery paths
- `nvflare config` — local startup kit registry and active-kit selection for all
  server-connected commands


## Design Principles

From `nvflare_agent_consumer_readiness.md`.

This CLI is Stage 1 of the agent-consumer readiness plan. The design must satisfy agent-usability requirements, not just human usability.

### 1. Machine-readable output on every command

Every command supports a stable, versioned JSON envelope via ``--format json``.
The default output format remains human-readable text (``txt``). Streaming
commands can additionally support newline-delimited JSON via ``--format jsonl``.
When JSON mode is requested, the output envelope is stable and versioned:

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

Contract — stream split in JSONL mode:

- ``--format jsonl`` is supported only by commands that declare ``streaming: true`` in ``--schema``.
- Each stdout line is one complete JSON object.
- Human-readable progress, warnings, prompts, and diagnostics go to stderr.
- The final stream event for wait-like streaming commands must include ``terminal: true``.

Exceptions: the following are plain text and are explicitly outside the JSON command-output contract:

- `--help` / `-h`: argparse-generated usage text
- `--version`: top-level utility path, not operational command output
- Parser-generated usage errors

Agents must use `--schema` for machine-readable command discovery, not `--help`.

Agent-facing commands add explicit command-contract metadata to the existing
argparse-derived schema. The `args` and `examples` fields remain unchanged; the
additional fields describe command behavior that cannot be inferred safely from
argparse:

- `output_modes`: supported machine output modes, for example `["json"]` or
  `["json", "jsonl"]`.
- `streaming`: whether the command can emit multiple machine-readable events.
- `mutating`: whether the command can change local files, local CLI state, or
  server state.
- `idempotent`: whether repeating the same command with the same arguments is
  expected to leave the same final state.
- `retry_token`: optional retry/deduplication token contract. For job submit,
  this is `{"supported": true, "flag": "--submit-token", "scope": "study + submitter + token"}`.

These fields are explicit per-command metadata, not inferred from argparse. The
initial scope is limited to the agent-facing command set: `job submit`, `job
list`, `job meta`, `job wait`, `job monitor`, `job download`, `job logs`,
`config list`, `config inspect`, `config use`, `recipe list`, `recipe show`, and
`study list`.

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

All confirmation-required commands (`abort`, `delete`, `shutdown`, `restart`, `disable-client`, `enable-client`) accept `--force` to skip confirmation. `--force` must be passed explicitly. Non-interactive contexts (stdin not a tty) without `--force` exit with code 4.

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
| `wait` | Yes | Single-envelope automation wait; no progress stream |
| `list-templates` | No | Deprecated — retain with stderr warning; use `nvflare recipe list`. Underscore alias `list_templates` kept. |
| `show-variables` | No | Deprecated — retain with stderr warning; use Job Recipe API. Underscore alias `show_variables` kept. |

`submit` already uses `new_secure_session()` (`nvflare/fuel/flare_api/flare_api.py`) for server connectivity. The same session infrastructure is used by `ProdEnv` and `PocEnv`.

### Other existing commands (requires agent readiness updates)

| Command | Subcommands | Agent Readiness Status |
| --- | --- | --- |
| `nvflare simulator` | — | Deprecated — retain with stderr warning; use Job Recipe SimEnv directly (`python job.py`) |
| `nvflare poc` | `config`, `prepare`, `start`, `stop`, `clean`, `add user`, `add site` | Add JSON output, exit codes, `--schema`; add `--force` to `prepare` for workspace deletion prompt bypass and to `clean` for stop-before-cleanup; register generated user/admin kits in the shared startup kit registry |
| `nvflare config` | `add`, `use`, `inspect`, `list`, `remove` | User-facing startup kit registry commands; keeps 2.7.x root flags `-d/--startup_kit_dir`, deprecated `-pw/--poc_workspace_dir`, and deprecated `-jt/--job_templates_dir`; no server connection |
| `nvflare poc config` | — | POC-specific local config, including the POC workspace path |
| `nvflare study` | `register`, `show`, `list`, `remove`, `add-site`, `remove-site`, `add-user`, `remove-user` | Add multi-study lifecycle CLI using the active startup kit |
| `nvflare provision` | — | Add JSON output, `--schema`, `--force` for Y/N prompts; restore pre-2.7.0 default: no args = generate `project.yml` |
| `nvflare preflight-check` | — | Add JSON output, `--schema`; exit 0=pass, 1=fail. Underscore alias `preflight_check` accepted for backward compatibility. |
| `nvflare config` | — | Parent command namespace for local CLI settings and read-only config inspection. Do not add a nested `kit` subcommand. |
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
| `remove_client` | Admin | No — legacy interactive-console registry cleanup only |
| `disable_client` | Admin | Yes -> `nvflare system disable-client`; persists a server-side disabled flag and rejects reconnect/heartbeat |
| `enable_client` | Admin | Yes -> `nvflare system enable-client`; clears the server-side disabled flag |
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
nvflare job logs <job_id> [--study name] [--site server|<client_name>|all] [--tail N] [--since timestamp] [--max-bytes N]
```

`--study` selects the study that contains the job. If omitted, `job logs`
searches the `default` study. For named-study jobs, callers must pass the same
`--study` used for `job submit` or `job list`; otherwise a valid job ID from a
different study will report `JOB_NOT_FOUND` with the searched study in the
error message.

`--site` defaults to `server` for bounded output and backward-compatible behavior.
If no explicit bound is provided, the CLI returns at most the last 500 lines per
site and reports whether the output was truncated.

Site target behavior:

- `--site server`: return the server-side job log.
- `--site <client_name>`: return that client's job log as streamed to and stored by the server.
- `--site all`: return the server log plus all client logs available in the server-side job log store.

Bound options are applied by the CLI after retrieving the server-side stored log
content:

- `--tail N`: return at most the last N lines per site.
- `--since timestamp`: return timestamped log lines at or after the timestamp
  when line timestamps are parseable. Continuation lines following an included
  timestamped line are included.
- `--max-bytes N`: return at most N UTF-8 bytes per site.

These are CLI output bounds, not server-side retrieval filters. They bound the
human/JSON output and truncation metadata after the server command returns log
content. If the server has already limited a large log to its maximum returned
response size, `--tail` and `--since` are evaluated against that returned
content, not against bytes that were never returned to the CLI.

`grep` is intentionally not a CLI flag; users can pipe or post-process the
returned content when text matching is needed.

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
    },
    "logs_truncated": false,
    "sites": {
      "server": {"available": true, "lines": 500, "bytes": 12000, "logs_truncated": false},
      "site-1": {"available": true, "lines": 200, "bytes": 9000, "logs_truncated": false},
      "site-2": {"available": true, "lines": 150, "bytes": 7000, "logs_truncated": false}
    },
    "filters": {
      "tail": 500,
      "since": null,
      "since_applied": false,
      "max_bytes": null,
      "default_tail_applied": true
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
    "logs_truncated": false,
    "sites": {
      "server": {"available": true, "lines": 500, "bytes": 12000, "logs_truncated": false},
      "site-1": {"available": true, "lines": 200, "bytes": 9000, "logs_truncated": false},
      "site-2": {"available": false, "reason": "client log stream not available for this job"}
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

Session API: `Session.get_job_logs(job_id, target, tail_lines=None, grep_pattern=None)` sends the structured server command and returns `logs` plus optional `unavailable`. `tail_lines` and `grep_pattern` are retained as deprecated compatibility arguments for existing Python callers. The CLI applies `--tail`, `--since`, and `--max-bytes` locally after retrieving server-side stored logs so the JSON response can include truncation metadata. These CLI filters should not be described as reducing server-side read cost or admin API transfer size.

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
| `nvflare recipe show <name>` | Show structured metadata for one built-in FL workflow recipe |

```text
nvflare recipe list [--framework <framework>] [--filter key=value]...
nvflare recipe show <name>
```

`--framework` filters by framework (for example `core`, `numpy`, `pytorch`, `tensorflow`, `sklearn`, `xgboost`) and is a shorthand for `--filter framework=<framework>`.
`--filter` is repeatable and supports these metadata keys:

- `framework`
- `privacy`
- `algorithm`
- `aggregation`
- `state_exchange`

Filter values are normalized so hyphenated and underscored values match the same metadata value, for example `homomorphic-encryption` and `homomorphic_encryption`.
Repeated filters for different keys are combined as an intersection. Repeated filters for the same key are treated as alternatives. Without filters, the command returns the documented built-in recipe catalog plus dynamically discovered recipe entries.
Valid metadata filters that match no available recipes return an empty list.

Recipe classes may declare metadata as class-level attributes (`recipe_algorithm`, `recipe_aggregation`, `recipe_state_exchange`, `recipe_privacy`) or the shorter forms (`algorithm`, `aggregation`, `state_exchange`, `privacy`). When explicit metadata is absent, the CLI infers the common fields from the recipe module, class, and CLI name. The CLI discovers recipes at runtime via `importlib` + `inspect` and supplements that with a documented recipe manifest so recipes remain queryable even when optional framework dependencies are not installed.

`recipe show` returns a single recipe detail document keyed by the same recipe
name returned from `recipe list`. The detail response includes the list-time
metadata plus `client_requirements`, `framework_support`,
`heterogeneity_support`, `privacy_compatible`, constructor `parameters`,
`optional_dependencies`, and `template_references`. Parameter metadata is
derived from the recipe constructor signature when the recipe can be imported,
or from static source parsing when optional dependencies are missing. The CLI
must not instantiate the recipe.

Example:

```json
{
  "schema_version": "1",
  "status": "ok",
  "data": [
    {
      "name": "fedavg",
      "description": "Federated Averaging for PyTorch",
      "framework": "pytorch",
      "module": "nvflare.app_opt.pt.recipes.fedavg",
      "class": "FedAvgRecipe",
      "algorithm": "fedavg",
      "aggregation": "weighted_average",
      "state_exchange": "full_model",
      "privacy": []
    }
  ]
}
```


## Wait / Monitor / Poll

`nvflare job wait <job_id>` is the automation-oriented wait command. It polls until the
job reaches a terminal state and returns one final command result envelope. It must not
stream progress events or status lines as command output. With `--format json`, stdout
contains exactly one JSON envelope after completion, failure, or timeout; diagnostics go
to stderr following the normal CLI output contract.

`nvflare job monitor <job_id>` is the interactive/progress command. It also polls until
the job reaches a terminal state via `monitor_job_and_return_job_meta()` callback. In
human and JSON modes, status lines print to stderr before the final result. In JSONL mode,
status events print to stdout as newline-delimited JSON and the final event has
`terminal: true`. For named-study jobs, pass the same `--study` value used at submit/list
time so wait or monitor opens the correct study session.

```bash
JOB=$(nvflare job submit -j ./my_job | jq -r .data.job_id)
nvflare job wait $JOB --format json && nvflare job download $JOB
```

Both commands accept `--timeout`, `--interval`, and `--study`. `--timeout 0` means no
timeout, `--interval` controls the polling interval in seconds, and omitted `--study`
means the literal default study. Exit code 0 on successful terminal status such as
`FINISHED:COMPLETED` or `FINISHED_OK`, exit code 1 on terminal job failure such as
`FAILED` / `ABORTED`, and exit code 3 on timeout.


## Study Selector Semantics

`--study` always names exactly one concrete study. If `--study` is omitted, commands use
the default study. The string `all` is not a reserved study selector and must not trigger
cross-study behavior in `nvflare job`, `nvflare study`, API sessions, or server-side admin
commands. If a project intentionally creates a study literally named `all`, it is treated
as a normal study name.

Do not add special `study == "all"` handling for job submit/list/monitor, submit-token
lookup, study authorization checks, or session creation. Cross-study enumeration must be
modeled as a separate explicit command or API in a future design, not as an overloaded
study name.


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

Normal server-connected commands use one active local startup kit by default. They also
accept scoped `--kit-id <id>` and `--startup-kit <path>` selectors for scripts, notebooks,
and concurrent workflows that need a per-command identity without changing the active kit.

Common POC flow:

```bash
nvflare poc prepare
nvflare poc start
nvflare job list
nvflare job submit -j ./job
```

`poc prepare` registers the generated Project Admin startup kit and makes it active.
Its JSON payload reports the prior active startup-kit ID, the active POC startup-kit ID,
and whether the active kit changed. Agents use this to restore the user's original
identity after POC workflows.

`poc prepare` also performs a best-effort local preflight for the generated server ports
before `poc start` is run. The preflight reports unavailable ports as warnings in the
success payload; it does not fail preparation because the actual bind happens later in
`poc start`.

Production flow:

```bash
nvflare config add alice_example_project /secure/startup_kits/example_project/alice@nvidia.com
nvflare config use alice_example_project
nvflare job list
```

Multiple local identities are handled by activating a different registered ID:

```bash
nvflare config add cancer_lead /secure/startup_kits/cancer/lead@nvidia.com
nvflare config add fraud_org_admin /secure/startup_kits/fraud/org_admin@nvidia.com
nvflare config use cancer_lead
nvflare config inspect
```

Activation is local config mutation only. It does not contact the server.

### Startup Kit Registry Storage

`~/.nvflare/config.conf` is the local storage file for the startup kit registry. The
user-facing command is `nvflare config`; the stored `startup_kits` data is an
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
role is inspected from the startup kit when needed by commands such as `nvflare config inspect`;
it is not duplicated in `config.conf`. Fields that are not reliably derivable from the
startup kit are omitted from normal output.

Other CLI state may also be stored in `~/.nvflare/config.conf`, but it is outside the
startup kit registry and should not be part of the normal user workflow.

### `nvflare config`

`nvflare config` manages local startup kit registrations. It is not a server-side resource.
Running `nvflare config` without a subcommand prints the current local config in a
read-only JSON/human envelope and exits without an `INVALID_ARGS` error.

Root compatibility flags:

- `-d` / `--startup_kit_dir`: accepted for compatibility with 2.7.x scripts, but
  deprecated. The path is registered under the default startup kit ID and made active.
  New workflows should use `nvflare config add` and `nvflare config use`.
- `-pw` / `--poc_workspace_dir`: accepted for compatibility with 2.7.x scripts, but
  deprecated. New workflows should use `nvflare poc config --pw <poc-workspace-dir>`.
- `-jt` / `--job_templates_dir`: accepted for compatibility with 2.7.x scripts, but job
  template config is deprecated.
- Interim development-only spellings such as `--poc.workspace`,
  `--poc.startup_kit`, and `--prod.startup_kit` are not part of the public
  compatibility contract and must be rejected.

#### `nvflare config add <id> <path>`

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
next_step: nvflare config use alice_example_project
```

#### `nvflare config use <id>`

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

`config use` is a human convenience command and intentionally mutates global CLI
state. Agent and notebook workflows should prefer `--kit-id <id>` or
`--startup-kit <path>` on each server-connected command. In JSON mode, `config
use` returns a warning finding with code `CONFIG_USE_MUTATES_GLOBAL_STATE`.

#### `nvflare config inspect`

Shows the configured active startup kit:

```text
active: cancer_lead
identity: lead@nvidia.com
cert_role: lead
path: /secure/startup_kits/cancer/lead@nvidia.com
```

The JSON payload preserves the existing fields and adds best-effort local
identity metadata:

- `role`: same certificate role as `cert_role`.
- `org`: certificate subject organization when available.
- `project`: certificate issuer common name, normally the project CA name.
- `certificate`: certificate path, `expires_at`, `days_remaining`, and
  `status` (`ok`, `expiring_soon`, `expired`, `unreadable`, or `unknown`).
- `findings`: non-fatal local findings for stale paths, invalid kits, unreadable
  certs, expired certs, or certs expiring soon.

`nvflare config inspect` reports `startup_kits.active`. It does not replace that value with
`NVFLARE_STARTUP_KIT_DIR`, but it should warn when the environment variable is set:

```text
warning: NVFLARE_STARTUP_KIT_DIR is set (/secure/startup_kits/cancer/lead@nvidia.com)
         normal commands will use this path instead of the active kit above
```

If the active path is missing or invalid, show the stale registration and suggest
`nvflare config use <id>` or `nvflare config remove <id>`.

#### `nvflare config list`

Lists registered startup kits. The command checks each registered path locally and flags
stale entries without contacting the server:

```text
* cancer_lead       ok       lead@nvidia.com        lead          /secure/startup_kits/cancer/lead@nvidia.com
  fraud_org_admin   missing  -                      -             /secure/startup_kits/fraud/org_admin@nvidia.com
  old_lab_admin     invalid  -                      -             /archive/old_lab/admin@nvidia.com
```

The active startup kit is marked with `*`. Missing or invalid paths are shown as
`missing` or `invalid`. Valid site kits are not shown because they are not CLI identities.
In JSON mode, each list row includes the same enriched `role`, `org`, `project`,
`certificate`, and `findings` fields as `config inspect`.

#### `nvflare config remove <id>`

Removes a local config registration. It does not delete the startup kit directory. If the
removed ID is active, clear `startup_kits.active` and print:

```text
removed startup kit: cancer_lead
warning: no active startup kit is configured
next_step: nvflare config use <id>
```

### Resolution Model

All server-connected commands, including `nvflare job`, `nvflare study`, `nvflare system`,
and `nvflare network`, resolve the startup kit using this ordered lookup:

1. Optional `--kit-id <id>` — if provided, override the active startup kit for
   this command only by resolving that registered startup-kit ID without changing
   `startup_kits.active`.
2. Optional `--startup-kit <path>` — if provided, override the active startup
   kit for this command only by validating that explicit admin startup-kit path.
3. If `NVFLARE_STARTUP_KIT_DIR` is set, validate it with the same three-file admin
   startup kit check and use it.
4. Otherwise, use `startup_kits.active` from `~/.nvflare/config.conf`.
5. If no source resolves to a valid admin/user startup kit, fail before any server
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

Creates a new local secondary POC user startup kit and registers it in the shared
registry. Valid `<cert-role>` values are `org_admin`, `lead`, and `member`. These are
certificate roles, not study-specific roles. `poc add user` must not add another
`project_admin`; the single POC Project Admin is created by `poc prepare`.

Behavior:

1. Resolve the default POC workspace the same way existing POC commands do.
2. Treat the command as a local POC workspace operation. It uses the local POC project
   metadata and local POC CA created by `poc prepare`; it is not gated by the currently
   active startup kit role.
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
next_step: nvflare config use bob@nvidia.com
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
2. Treat the command as a local POC workspace operation. It uses the local POC project
   metadata and local POC CA created by `poc prepare`; it is not gated by the currently
   active startup kit role.
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
- Do not contact the server for `nvflare config` registration, activation, listing, or
  removal errors.
- Treat identity, role, org, and project inspection as best effort. Failure to inspect
  metadata should not break `config add` or `config use` when the startup kit path itself is
  valid.

If `~/.nvflare/config.conf` does not exist:

- `nvflare config list` prints an empty list.
- `nvflare config inspect` reports that no active startup kit is configured.
- Normal server-connected commands fail before connecting:

```text
Error: no active startup kit is configured
Hint: Run nvflare poc prepare, or run nvflare config add <id> <startup-kit-dir> then nvflare config use <id>.
```

If config parsing fails, commands stop without modifying the file:

```text
Error: cannot parse ~/.nvflare/config.conf
Hint: Fix the config file, or move it aside and run nvflare poc prepare.
```

If `startup_kits.active` points to an ID that is not in `startup_kits.entries`:

```text
Error: active startup kit 'cancer_lead' is not registered
Hint: Run nvflare config list, then nvflare config use <id>.
```

`config add` error cases:

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

`config use` error cases:

```text
Error: startup kit id 'cancer_lead' is not registered
Hint: Run nvflare config list.
```

```text
Error: startup kit path for 'cancer_lead' does not exist: /secure/startup_kits/cancer/lead@nvidia.com
Hint: Restore the startup kit, remove the registration, or activate another kit.
```

```text
Error: registered path for 'cancer_lead' is not a valid startup kit
Hint: Run nvflare config remove cancer_lead, or replace it with nvflare config add cancer_lead <startup-kit-dir> --force.
```

`config use` must not change `startup_kits.active` until the selected ID resolves to a valid
startup kit path.

`config list` should not fail just because one registered path is stale. It marks each entry
independently as `ok`, `missing`, or `invalid`. If the active entry is stale, keep the `*`
marker and show the stale status.

`poc prepare` registers POC-generated user kits by identity. `--force` is reserved for
recreating the POC workspace and rerunning provisioning; it is not a general startup kit
registry override. If a target ID already exists:

- If the existing path is under the current POC workspace, replace it.
- If the existing path is outside the current POC workspace, fail and require an explicit
  registry action such as `nvflare config remove <id>` or
  `nvflare config add <id> <path> --force`.

```text
Error: startup kit id 'lead@nvidia.com' already points outside the POC workspace
Hint: Run nvflare config remove lead@nvidia.com, or replace it explicitly with nvflare config add lead@nvidia.com <startup-kit-dir> --force.
```

`poc add user` fails before creating or registering a new kit when the requested identity
already exists in the POC project, unless `--force` is provided. With `--force`, update
the existing participant metadata in place and persist it to disk; do not append a
duplicate participant.

`poc add site` follows the same duplicate rule for sites. If services are running, the
command may add the new site kit but must not restart existing services automatically.

`poc add user` and `poc add site` do not use the active startup kit as an authorization
boundary. POC is a local deployment owned by the local operator, so the meaningful safety
checks are local integrity checks: valid requested user role, duplicate participant
handling, existing POC project metadata, and generated startup kit shape.

`poc clean` removes only startup kit entries whose canonical paths are under the canonical
POC workspace path. If the active startup kit is removed, clear `startup_kits.active`.
Manual production registrations remain untouched.

### Startup Kit Behavior Guarantees

- `poc prepare` activates the default POC Project Admin startup kit automatically.
- `poc prepare` writes POC admin/user startup kits into `startup_kits.entries`.
- Normal server-connected commands do not expose `--startup-target`.
- Commands resolve the startup kit through `--kit-id` / `--startup-kit` first, then
  `NVFLARE_STARTUP_KIT_DIR`, then `startup_kits.active`.
- `--kit-id` and `--startup-kit` are optional per-command overrides. If
  provided, they take precedence over the active startup kit for that command
  only and must not mutate `startup_kits.active`.
- `nvflare config` is the user-facing startup kit management interface.
- POC-specific local settings belong under `nvflare poc config`; the root
  `nvflare config -pw` flag is compatibility-only.
- `poc add user` registers generated user kits in `startup_kits.entries`.
- `poc add site` generates site kits and updates POC workspace metadata, but it does not
  register site kits in `startup_kits.entries`.
- `poc add user` and `poc add site` do not switch away from the active kit in the normal
  POC flow.
- `poc add site` is allowed while POC services are running and does not stop existing
  services.
- For `nvflare config add` and `nvflare poc add user`, `--force` replaces the config
  registration for an existing ID without deleting old startup kit directories.
- For `nvflare poc prepare`, `--force` means recreate the POC workspace; it does not
  override live unrelated startup kit registrations outside the POC workspace. If an
  outside registration points to a path that no longer exists, it is treated as stale
  local state and may be replaced by the newly generated POC kit.
- Persona commands are not needed. The common switching command is `nvflare config use <id>`.


## Existing Command Reference

All commands need `--schema` and always emit a JSON envelope unless marked otherwise.

### `nvflare poc`

#### `nvflare poc config`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-pw`, `--pw`, `--poc_workspace_dir`, `--poc-workspace-dir` | str | No | — | Set the local POC workspace path |
| `--schema` | flag | No | — | Print command schema and exit |

`nvflare poc config --pw <path>` is the preferred command for setting the local
POC workspace. It writes `poc.workspace` in `~/.nvflare/config.conf`, the same
storage key used by `poc prepare`, `poc start`, `poc stop`, and `poc clean`.
Running `nvflare poc config` without `--pw` reports the effective POC workspace
and any `NVFLARE_POC_WORKSPACE` environment override.

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
activates the first Project Admin kit. The JSON success payload includes:

- `startup_kit.prior_active`: startup-kit ID that was active before prepare, or `null`.
- `startup_kit.active`: startup-kit ID active after prepare, or `null`.
- `startup_kit.changed`: whether prepare changed the active startup-kit ID.
- `port_preflight.checked`: whether server port preflight could be performed.
- `port_preflight.host`: loopback host used for the local availability probe.
- `port_preflight.scope`: `"loopback"` because preflight does not bind wildcard interfaces.
- `port_preflight.ports`: checked local server ports and availability.
- `port_preflight.conflicts`: unavailable ports that may prevent `poc start`.
- `port_preflight.note`: caveat that this is a best-effort loopback check and
  `poc start` can still fail if another local bind address conflicts.

#### `nvflare poc start`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--service` | str | No | `"all"` | Participant to start |
| `-ex`, `--exclude` | str | No | `""` | Exclude service directory |
| `-gpu`, `--gpu` | int... | No | `None` | GPU device IDs |
| `--no-wait` | flag | No | — | Return after process start without readiness wait |
| `--timeout` | int | No | `POC_START_READY_TIMEOUT` | Seconds to wait for readiness |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--schema` | flag | No | — | Print command schema and exit |

Starts server and clients by default and excludes admin console participants unless one is
explicitly selected. In JSON mode, success data includes:

- `status`: `running` after readiness wait, or `starting` with `--no-wait`.
- `ready`: present when readiness was checked or explicitly skipped.
- `ready_timeout`: configured readiness timeout in seconds.
- `server_url`: compatibility URL for the FL server endpoint.
- `server_address`: bound FL server address, for example `localhost:8002`.
- `admin_address`: bound admin endpoint address, for example `localhost:8003`.
- `default_port`, `default_server_port`, `default_admin_port`: POC defaults used for
  comparison and diagnostics.
- `port_conflict`: true when the pre-start local port check found unavailable configured
  server ports.
- `port_preflight`: checked ports and conflict details. It uses the same loopback-scoped
  best-effort contract as `poc prepare`.
- `warnings`: human-readable warning strings derived from `port_preflight.conflicts`.
- `clients`: configured POC client names.

`server_address` and `admin_address` are the source of truth for follow-on automation.
`port_conflict` is advisory: POC does not currently auto-rebind to alternate ports, so a
conflict usually means startup will fail or another local POC system is already using the
configured endpoint.

#### `nvflare poc stop`

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-p`, `--service` | str | No | `"all"` | Participant to stop |
| `-ex`, `--exclude` | str | No | `""` | Exclude service directory |
| `--no-wait` | flag | No | — | Request shutdown and return without waiting for completion |
| `-debug`, `--debug` | flag | No | — | Debug mode |
| `--schema` | flag | No | — | Print command schema and exit |

When stopping the server path, `poc stop` uses coordinated system shutdown logic. By
default it waits until shutdown completes and returns `status: stopped`; with
`--no-wait`, it returns after requesting shutdown with `status: shutdown_initiated`.

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
Allowed `<cert-role>` values are `org_admin`, `lead`, and `member`; this command
does not add another `project_admin`.

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
| `--submit-token` | str | No | — | Caller-generated token for retry-safe submit and later job recovery |
| `--schema` | flag | No | — | Print command schema and exit |

Output:

```json
{"schema_version": "1", "status": "ok", "data": {"job_id": "abc123"}}
```

`--submit-token` is the canonical name for the job submission idempotency/recovery
token. It is optional and, when provided, is a caller-generated opaque value for one
intended job submission. NVFlare does not auto-generate a submit token when the flag is
omitted. It is not an auth token, session token, startup-kit credential, API key, or
certificate secret. It does not grant access; normal startup-kit authentication and
authorization still apply.

The submit token is not part of the submitted job's `meta.json`. Job `meta.json` remains
job-owned metadata for FLARE execution, such as `deploy_map`, `resource_spec`,
`min_clients`, and launcher configuration. Submit-token bookkeeping is server-owned
submission metadata.

A UUID is recommended for automated retries, but any unique non-empty token matching
`^[A-Za-z0-9._:-]{1,128}$` is accepted. Examples:

```bash
nvflare job submit -j ./job --submit-token 550e8400-e29b-41d4-a716-446655440000
nvflare job submit -j ./job --submit-token run-20260429-001
nvflare job submit -j ./job --submit-token agent:session-123:step-2
```

Submit-token scope:

- Lookup key: `(server/project context, study, submitter identity, submit_token)`.
- Validation value: a server-computed job content hash for the submitted job artifact.
- Same scope + same token + same job content: return the existing `job_id`.
- Same scope + same token + different job content: fail with a conflict such as
  `SUBMIT_TOKEN_CONFLICT`.
- Same token in a different study is allowed because studies are separate job namespaces.
- Same submitter and same study without `--submit-token`: keep normal behavior, create
  a new job for each submit, and track it through the normal job store/job history only.
  Do not create a retry-safe submit-token record.
- Same submitter and same study with different `--submit-token` values: keep normal
  behavior and create separate jobs, even if the submitted content is identical.

The job content hash is intentionally not part of the lookup key. The token names the
submission attempt; the content hash proves whether a retry is the same submission. If
the hash were part of the lookup key, accidental token reuse for a different job would
silently create a second job instead of surfacing a conflict.

Submit-token content hashing is defined over the submitted job content root. Directory
input is hashed relative to the provided directory. Zip input may contain one wrapper
directory around the job content; that wrapper is stripped so a normal `zip -r job.zip
job/` archive hashes the same as submitting `job/` directly. Passing the parent
directory that contains `job/` is not equivalent to passing `job/` itself.

Server storage model:

- Persist a separate study-scoped submit record in the server job store. The persistent
  job store is the source of truth; any in-memory lookup map is only an optional cache.
- Do not persist `submit_token` in the uploaded job `meta.json`.
- Do not require runtime clients to receive `submit_token`; deployment/runtime job
  metadata should remain focused on executing the job.
- Store submit records outside the normal job-object namespace so `get_all_jobs()` and
  scheduling continue to see only real jobs.

Recommended persistent namespace:

```text
job_submit_records/<study_hash>/<submitter_hash>/<submit_token_hash>
job_submit_record_index/<job_id_hash>
```

Recommended submit record payload:

```json
{
  "schema_version": 1,
  "submit_token": "run-20260429-001",
  "job_id": "eef2c05b-8b8e-44cf-a6e6-787985ad6a42",
  "state": "creating|created|job_deleted",
  "job_name": "hello-numpy",
  "job_folder_name": "hello-numpy",
  "study": "cancer",
  "submitter_name": "alice",
  "submitter_org": "nvidia",
  "submitter_role": "project_admin",
  "submit_time": "2026-04-29T10:00:00-07:00",
  "job_content_hash": "sha256:...",
  "deleted_time": "2026-04-30T10:00:00-07:00",
  "deleted_by": {"name": "admin@nvidia.com", "org": "nvidia", "role": "project_admin"}
}
```

`deleted_time` and `deleted_by` are present only after the referenced job is deleted.
The reverse index lets `job delete` find submit records by `job_id` without storing the
submit token in job `meta.json`.

Submit handling:

1. If `--submit-token` is absent, do not auto-generate a token. Use the existing submit
   path, create a new job, and rely on normal job-store/job-history tracking only. Do
   not create a retry-safe submit-token record.
2. If `--submit-token` is present, compute the study-scoped submit-record key.
3. If a submit record exists:
   - `state: "job_deleted"`: return `SUBMIT_TOKEN_JOB_DELETED` and do not recreate the
     deleted job;
   - same content hash: return the existing `job_id`;
   - different content hash: return `SUBMIT_TOKEN_CONFLICT` and do not create a job.
4. If no submit record exists:
   - pre-generate a `job_id`;
   - create a submit record with `state: "creating"`, that `job_id`, and the content hash
     using no-overwrite semantics for the submit-record key;
   - create the job using the pre-generated `job_id`;
   - update the submit record to `state: "created"` before returning success.

The submit record should be written before the job object so a retry can recover even if
the server accepts the request but the client times out. If a retry finds a matching
`state: "creating"` record, it should check whether the referenced `job_id` already
exists. If the job exists, update the submit record to `created` and return the existing
`job_id`; if the job does not exist, retry creation with the same pre-generated `job_id`.
Concurrent submissions with the same scope and token must be serialized by the
no-overwrite submit-record create or an equivalent lock.

When `nvflare job delete <job_id>` deletes a job referenced by a submit-token record, it
marks the submit record as `state: "job_deleted"` instead of deleting the record. This
preserves the audit trail and prevents a later retry with the same token from recreating
a job that an operator deliberately deleted. Delete JSON output includes
`submit_records_marked_deleted`.

Recovery after client-side timeout or session loss:

```bash
TOKEN=$(uuidgen)
nvflare job submit -j ./job --study cancer --submit-token "$TOKEN" --format json
nvflare job list --study cancer --submit-token "$TOKEN" --format json
nvflare job meta <job_id> --format json
```

`nvflare job list --submit-token <token>` filters jobs in the selected study. Without
`--study`, it searches the default study, matching normal `job list` behavior. There is
no reserved `all` study selector; callers either omit `--study` for the default study or
specify one concrete study name. The filter should resolve through server submit records,
not by scanning or modifying the job's `meta.json`. Do not add `--submit-token` to
`monitor`, `download`, `abort`, `delete`, or `clone`; recover the `job_id` with
`job list --submit-token` first, then use normal job lifecycle commands.

#### `nvflare job list`

Lists jobs visible to the active startup kit. With no submit-token filter, this keeps the
normal list behavior for the selected study.

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `-n`, `--name` | str | No | — | Filter by job name prefix |
| `-i`, `--id` | str | No | — | Filter by job ID prefix |
| `-r`, `--reverse` | flag | No | — | Reverse sort order |
| `-m`, `--max` | int | No | — | Maximum number of returned jobs |
| `--study` | str | No | `"default"` | Study to list |
| `--submit-token` | str | No | — | Recovery filter for a previous retry-safe submit in the selected study |
| `--schema` | flag | No | — | Print command schema and exit |

`--submit-token` on `job list` is a recovery lookup, not an authorization mechanism. The
server resolves it through study-scoped submit records owned by the current submitter. It
must not scan job `meta.json`, because submit tokens are server-owned submission metadata
and are not part of job execution metadata.

#### `nvflare job wait`

Waits for a running job to reach a terminal state and returns one final command envelope.
This is the preferred command for scripts, CI/CD, and agents because it has no progress
stream to parse.

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `job_id` | str | Yes | — | Job ID to wait for |
| `--timeout` | number | No | 0 | Max seconds to wait; must be >= 0; 0 means no timeout |
| `--interval` | number | No | 2 | Poll interval in seconds; must be > 0 |
| `--study` | str | No | `"default"` | Study containing the job |
| `--schema` | flag | No | — | Print command schema and exit |

Output contract:

- Human mode: print a concise final status summary only after the job is terminal or the
  wait times out.
- JSON mode: stdout contains exactly one JSON envelope after completion, failure, or
  timeout.
- Progress updates are intentionally omitted. Use `nvflare job monitor` when progress
  updates are desired.

Exit code 0 on successful terminal status such as `FINISHED:COMPLETED` or
`FINISHED_OK`, exit code 1 on terminal job failure such as `FAILED`,
`FINISHED_EXCEPTION`, `ABORTED`, or `ABANDONED`, exit code 2 on connection,
authentication, or authorization failure, and exit code 3 on timeout.

#### `nvflare job monitor`

Polls a running job until it reaches a terminal state. This is the progress-oriented
variant: it prints status updates to stderr before the final result. For single-envelope
automation, use `nvflare job wait`.

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `job_id` | str | Yes | — | Job ID to monitor |
| `--timeout` | int | No | 0 | Max seconds to wait; must be >= 0 |
| `--interval` | int | No | 2 | Poll interval in seconds; must be > 0 |
| `--study` | str | No | `"default"` | Study containing the job |
| `--schema` | flag | No | — | Print command schema and exit |

Exit behavior matches `nvflare job wait`.

Output modes:

- Human mode: progress status lines plus a final status summary.
- JSON mode: progress status lines go to stderr; stdout contains exactly one final JSON
  envelope.
- JSONL mode: stdout contains progress events and exactly one final terminal event. Each
  line is a complete JSON object. A successful raw job status such as `FINISHED_OK` is
  normalized to `status: "COMPLETED"` with the original value preserved as `job_status`.
  Timeout emits `status: "TIMEOUT"` and `terminal: true`.

Schema contract:

- `streaming: true`
- `output_modes: ["json", "jsonl"]`

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
workflow is exposed through `nvflare config`; users should not need to edit or reason
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
user-facing workflow is `nvflare config`; other local config storage is implementation
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
| `config add/use/inspect/list/remove` | Add | Add | Add | Manage local startup kit registry; no server connection |
| `job create` | — | — | — | Deprecated |
| `job submit` | Add | Add | Add | Returns `job_id` immediately |
| `job wait` | Add | Add | Add | Single-envelope automation wait/poll |
| `job monitor` | Add | Add | Add | Interactive progress wait/poll |
| `study register/show/list/remove/add-site/remove-site/add-user/remove-user` | Add | Add | Add | Manage multi-study lifecycle through active startup kit |
| `provision` | Add | Add | Add | Restore pre-2.7.0 default; add `--force` |
| `preflight-check` | Add | Add | Fix | 0=pass, 1=fail; alias `preflight_check` kept |
| `config` | Add | Add | Add | Local config parent command and startup kit registry; no normal server connection |
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
nvflare job submit    -j <job_folder> [--study name] [--submit-token token]
nvflare job wait      <job_id> [--study name] [--timeout N] [--interval N]
nvflare job monitor   <job_id> [--study name] [--timeout N] [--interval N]
nvflare job list      [-n prefix] [-i id_prefix] [-r] [-m num] [--study name] [--submit-token token]
nvflare job meta      <job_id>
nvflare job abort     <job_id> [--force]
nvflare job clone     <job_id>
nvflare job download  <job_id> [-o destination]
nvflare job delete    <job_id> [--force]

# Observability (server must be running)
nvflare job stats     <job_id> [--site server|<name>|all]
nvflare job logs      <job_id> [--study name] [--site server|<name>|all] [--tail N] [--since timestamp] [--max-bytes N]
nvflare job log-config <job_id> [--site server|<name>|all] <level>
```

#### `nvflare job download` JSON contract

`nvflare job download <job_id> --format json` keeps the existing server download
protocol unchanged. The CLI downloads the job result through the current transfer API,
then discovers common artifacts under the final local destination on the CLI machine.
It must not expose server workspace, job-store, or transfer temporary paths.

The success envelope's `data` object includes:

- `job_id`: requested job ID.
- `download_path`: final local directory returned by the download API on the CLI
  machine. This may differ from the requested destination if collision handling or the
  transfer layer chooses a different final directory.
- `path`: compatibility alias for `download_path` when present.
- `artifact_discovery`: `"completed"` when local artifact discovery ran, or `"skipped"`
  when the CLI did not have a real local directory to inspect.
- `artifacts`: discovered local artifact paths under `download_path`, such as global
  model, metrics summary, and client log paths. This is `null` when discovery is skipped.
- `missing_artifacts`: expected artifact categories that were not found locally.
  This is `null` when discovery is skipped.

`missing_artifacts` is informational. The command must still succeed when expected
model, metrics, or log artifacts are absent, as long as the download operation itself
succeeded. Agents must use `data.artifacts.*` as the source of truth for consumable
files instead of inferring paths from server locations or assuming a fixed layout under
`download_path`. Agents must treat `artifact_discovery: "skipped"` as "not inspected",
not as proof that expected artifacts are absent.

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

`nvflare study list --format json` is the production submit preflight for the current
phase. The server returns the authenticated identity plus visible studies and per-study
details. The CLI adds startup-kit selection metadata (`source`, `id`, `path`) because
only the CLI knows whether the kit came from `--kit-id`, `--startup-kit`, environment, or
active config. The server remains the source of truth for identity and visibility.

For each returned study, `can_submit_job` and `capabilities.submit_job` are `true` under
the current membership-level model. Hidden or unmapped studies are omitted rather than
returned with denial details. This is not full future authorization-policy introspection;
job submit can still fail for unrelated validation or policy checks.
- `--sites` and `--site-org` are mutually exclusive.

### Add `nvflare recipe`

```text
nvflare recipe list [--framework <framework>] [--filter key=value]...
nvflare recipe show <name>
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
nvflare system shutdown      <server|client|all> [client_names...] [--force] [--no-wait] [--timeout N]
nvflare system restart       <server|client|all> [client_names...] [--force] [--no-wait] [--timeout N]
nvflare system disable-client <client_name> [--force]
nvflare system enable-client <client_name> [--force]
nvflare system log-config    [--site server|<client_name>|all] <level>
nvflare system version       [--site server|<name>|all]
```

For `shutdown` and `restart`, `--timeout N` must be positive. Use `--no-wait`
instead of `--timeout 0` for fire-and-forget operation. `restart all --wait`
waits for the server restart and for previously connected clients to reconnect.

`remove-client` is not exposed as a supported `nvflare system` CLI command. The legacy
interactive-console `remove_client` operation removes only the current active registry
entry; it does not stop the client process, revoke credentials, or prevent reconnect.
`disable-client` is the durable operational control: it persists a disabled flag, removes
any active registry entry, and rejects subsequent registration or heartbeat from the same
client name until `enable-client` clears the flag. This is not certificate revocation.

Disabled client state is stored in the server workspace as `disabled_clients.json`.
In-memory updates and persistence writes must be made under the client manager lock.
The JSON write uses a temporary file plus atomic replace so registration/heartbeat
handling cannot observe a partially written policy file. The file is loaded at server
startup so disabled clients remain disabled across server restarts. If this file exists
but cannot be loaded, the server must fail closed by raising during startup instead of
admitting previously disabled clients.


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
    {"name": "--submit-token", "type": "string", "required": false, "default": null, "description": "Caller-generated token for retry-safe submit and later job recovery"},
    {"name": "--schema", "type": "bool", "required": false, "default": false, "description": "Print command schema as JSON and exit"}
  ],
  "examples": ["nvflare job submit -j ./my_job --submit-token 550e8400-e29b-41d4-a716-446655440000"]
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

1. Optional `--kit-id <id>` — override the active startup kit for this command
   only by resolving a registered startup-kit ID from `~/.nvflare/config.conf`
   without changing `startup_kits.active`.
2. Optional `--startup-kit <path>` — override the active startup kit for this
   command only by validating the explicit path as an admin/user startup kit.
3. `NVFLARE_STARTUP_KIT_DIR` — validate the environment path as an admin/user startup kit.
4. `startup_kits.active` — resolve the active ID from `~/.nvflare/config.conf`, then
   validate the registered path.
5. If no source resolves to a valid admin/user startup kit, fail before any server
   connection attempt.

This resolution order applies uniformly. Command-level descriptions that say "same resolution order as `nvflare job`" refer to this list.

Normal commands do not expose `--startup-target`. Users usually switch local identity with
`nvflare config use <id>`. Automation can optionally use `--kit-id <id>`,
`--startup-kit <path>`, or `NVFLARE_STARTUP_KIT_DIR` when mutating local config is undesirable.

Resolution error examples:

```text
Error: no active startup kit is configured
Hint: Run nvflare config use <id>.
```

```text
Error: NVFLARE_STARTUP_KIT_DIR does not point to a valid startup kit for admin use
Path: /secure/startup_kits/cancer/lead@nvidia.com
Hint: Unset NVFLARE_STARTUP_KIT_DIR, or set it to a valid admin startup kit directory.
```

```text
Error: active startup kit 'cancer_lead' points to a missing path
Path: /secure/startup_kits/cancer/lead@nvidia.com
Hint: Run nvflare config use <id> or nvflare config remove cancer_lead.
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
- `remove_client(client_name)` — removes a single connected client from the active registry
- `disable_client(client_name)` — disables a client from reconnecting until enabled
- `enable_client(client_name)` — enables a disabled client to reconnect
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
| `nvflare/tool/kit/kit_cli.py` | `nvflare config` command handlers |
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

`shutdown`, `restart`, `disable-client`, `enable-client`, `delete`, and `abort` prompt for confirmation in the interactive console. CLI behavior:

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
