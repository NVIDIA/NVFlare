# Distributed Provisioning - Scripted Mode (CLI + JSON)

This runbook explains the demo automation in `scripted_mode_demo.sh`. The script
is intentionally written as readable Bash, not as the shortest possible smoke
test, so that it can be adapted into CI, a ticket workflow, or a provisioning
service.

The script uses `--format json` for every NVFLARE CLI call and stores each
command response under `<work_dir>`. It also emits newline-delimited JSON events
on stdout so callers can pipe the output into `jq` or a log collector.

## Inputs
- One **single-site** `site.yml` per participant (server and clients).
- Each `site.yml` must contain: `name`, `org`, `type`.
- `org` must match `^[A-Za-z0-9_]+$` (no hyphens or spaces).

Template: `../site.template.yml`

The script uses the public distributed provisioning command sequence:

```bash
nvflare cert init ...
nvflare cert request ...
nvflare cert approve ...
nvflare package ...
```

The demo runs all commands locally, but the same artifacts can be moved between
roles:

| Phase | Real owner | Command | Handoff |
|-------|------------|---------|---------|
| CA setup | Project Admin | `nvflare cert init` | Keep `ca/` private to Project Admin |
| Request | Requester | `nvflare cert request` | Send only `<name>.request.zip` |
| Approval | Project Admin | `nvflare cert approve` | Return `<name>.signed.zip` and share `rootca_fingerprint_sha256` out-of-band |
| Package | Requester | `nvflare package` | Build startup kit from signed zip and local private key |

## Run
```bash
./scripted_mode_demo.sh <project_name> <server_endpoint> <work_dir> <site_yaml...>
```

Example (server + two clients):
```bash
./scripted_mode_demo.sh my-project grpc://localhost:8002 ./distprov_demo \
  ./server.yml ./site-1.yml ./site-2.yml
```

Security note:
- The script passes the `rootca_fingerprint_sha256` value reported by
  `cert approve` to `package --expected-rootca-fingerprint` for each signed zip.
  In remote deployments, exchange that value with the requester through a
  trusted out-of-band channel.

## What to copy into automation

- Keep `--format json` and parse `.data.*` fields instead of scraping text.
- Treat the request zip and signed zip as the only transfer artifacts.
- Keep the requester's private key in the request folder; never copy `*.key` to
  the Project Admin.
- Use `package --expected-rootca-fingerprint` for non-interactive automation.
- Use `package --confirm-rootca` only for human-driven interactive packaging.
