# Secure Local Monitoring Stack

This file provides a separate hardened local Compose path without changing the legacy `docker-compose.yml` and `prometheus.yml` examples in this directory.

## Files

- `docker-compose.secure-local.yml`
- `.env.secure-local.example`

## What Is Different

Compared with the legacy local example, this secure-local variant:

- pins image tags
- binds published ports to `127.0.0.1`
- mounts config and provisioning directories read-only where practical
- requires a non-empty Grafana admin password instead of using the legacy default
- disables Grafana self-sign-up

It intentionally reuses the legacy `prometheus.yml` so the old scrape configuration stays in one place.

## Start the Stack

From this directory:

```bash
cp .env.secure-local.example .env.secure-local
# edit .env.secure-local and set GRAFANA_ADMIN_PASSWORD
set -a
. ./.env.secure-local
set +a
docker compose -f docker-compose.secure-local.yml up -d
```

## Stop the Stack

```bash
docker compose -f docker-compose.secure-local.yml down
```

## Default Local URLs

- Grafana: `http://127.0.0.1:3000`
- Prometheus: `http://127.0.0.1:9090`
- statsd-exporter metrics: `http://127.0.0.1:9102/metrics`

## Notes

- This secure-local stack is additive. It does not replace the legacy local example files in this directory.
- The Grafana password requirement uses Docker Compose required-variable interpolation. If `GRAFANA_ADMIN_PASSWORD` is unset or empty, `docker compose` fails fast.
