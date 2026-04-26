# Distributed Provisioning — Scripted Mode

Single scripted flow using **single-site** `site.yml` files and the public
`cert request` -> `cert approve` -> `package` workflow.

## Run
```bash
./scripted_mode_demo.sh <project_name> <server_endpoint> <work_dir> <site_yaml...>
```

Notes:
- Uses `--format json` for all CLI calls; JSON goes to stdout.
- Human-readable progress output (if any) goes to stderr.
- `org` must match `^[A-Za-z0-9_]+$` (no hyphens or spaces).
- `type: client` maps to `cert request site`; `type: server` maps to
  `cert request server`; admin types map to `cert request user`.
