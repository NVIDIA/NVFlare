# Distributed Provisioning — Scripted Mode (CLI + JSON)

Minimal scripted flow. JSON on stdout via `--format json`.

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
