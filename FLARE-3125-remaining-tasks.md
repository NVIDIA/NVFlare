# FLARE-3123 / FLARE-3125 — remaining security tasks

Branch: `flare_3123_cellnet_security`. Both tickets are P0/security, fixVersion 2.9.0.
This file tracks what is landed on the branch and what still remains. Full findings
live in the Jira tickets and the workflow outputs; this is the working checklist.

## Landed on this branch

- Internal CellNet listeners bind loopback by default; non-loopback internal
  listeners require mTLS; Docker/K8s/Slurm opt into `listen_host=0.0.0.0`+mTLS.
  (FLARE-3123 fix 1)
- CP admin dispatcher + server-job command receiver validate the parent route
  (origin/destination/peer). (FLARE-3123 fix 4, server side)
- Pre-auth replies do not carry reusable bearer material: `SERVER_MAIN`
  Challenge/Register **and** `CELLNET/Bye`. (FLARE-3123 fix 3)
- `configure_job_log` / `configure_site_log` reject executable `dictConfig`
  (only log levels / built-in modes accepted), server and client paths.
  (FLARE-3123 fix; FLARE-3125 fix 3 — the RCE sink)
- **FLARE-3125 SJ route** enrolled-site -> `server.<job_id>/configure_job_log`
  is rejected: `configure_job_log` is a parent command, gated to `origin==ROOT_SERVER`;
  `ORIGIN` is framework-stamped and unspoofable.
- **FLARE-3125 CJ route** enrolled-site -> `<victim>.<job_id>/CLIENT_COMMAND/*`
  is rejected: `CommandAgent.execute_command` now allows only `origin in {ROOT_SERVER,
  own-parent CP}` addressed to this cell. (commit "authorize CLIENT_COMMAND sender")

## Remaining tasks

### 1. [P0] FLARE-3125 fix 4 — isolate job processes from startup keys
The SJ (and CJ) still run under the server/site UID with read access to
`startup/server.key`. Any residual code-exec primitive can exfiltrate the private
key. Needs a launcher/deployment change: per-job OS identity / container / namespace
that is denied read of the startup kit's private material. This is the main
substantive security item left; it is a deployment change, not a code gate — needs
design. Relevant: `server_engine.py`, the process launcher, provisioning.

### 2. [P0] Integration-validate the CLIENT_COMMAND gate before merge
The `CommandAgent.execute_command` sender check is unit-tested both ways (reject
enrolled peer, still dispatch server/parent origin) but NOT exercised end-to-end.
Verify with a real run:
- production topology: `server -> CP -> CJ` relayed admin command (e.g. abort) still works;
- CP self-management commands (`client_executor` show_stats/etc.) still work;
- simulator (`CJ` connects directly to root, `parent_url=None`) still works;
- an enrolled peer addressing another site's `<victim>.<job_id>` is rejected.

### 3. [P1] Origin <-> authenticated-connection binding (shared dependency)
Both server-side gates ultimately trust the framework-stamped `ORIGIN`. The
origin<->connection identity binding is not hardened: `sfm/conn_manager.require_match`
is mTLS-gated only, and `authenticator.py:~435` fails OPEN when the token resolver
returns `None`. A rogue cell admitted as FQCN `server` would defeat the SJ gate.
Fix: fail closed on unresolved identity; bind endpoint identity on non-mTLS internal
connections (or remove the clear internal path). (FLARE-3123 fix 2)

### 4. [P2] FLARE-3123 fix 4 — other unguarded receivers
Extend receiver authorization (or an incoming auth filter — client cells install
none) to the receivers the branch did not cover:
`server_command_agent.aux_communicate`, `fed_server._listen_command`
(SERVER_PARENT_LISTENER), and assert the CHANNEL explicitly (currently only implied
by callback registration). Note: `AUX_COMMUNICATION` carries legitimate
cross-participant app traffic (swarm/cyclic/broadcast) and must NOT be gated
parent-only — it needs a different model.

### 5. [P2] Stop wildcard command dispatch (FLARE-3123/3125 fix 2)
Both command agents register `topic="*"`. Marginal value now (origin gates reject
unclassified topics before dispatch; unknown topics already return INVALID_REQUEST),
but converting `*` to an explicit topic allow-list is desired defense-in-depth. Must
preserve the `APP_COMMAND` multiplexing path (`AppCommandProcessor` re-dispatches by
inner topic). Deliberate refactor, not a blind change.

### 6. [P3] Doc/behavior cleanup (FLARE-3125 fix 3 follow-up)
`flare_api.configure_job_log` still `json.dumps`es dict configs and
`api_spec.py`/`flare_api.py` docstrings still advertise `dict (dictConfig)` / file
paths, which now fail validation downstream. Reconcile docs + behavior. Optionally
add a sink-level guard in `dynamic_log_config`/`apply_log_config` so fix 3 is not
purely gate-based.
