# Distributed Provisioning — Agent Only (CLI + JSON)

Minimal agent flow. JSON on stdout via `NVFLARE_CLI_MODE=agent`.

## Inputs
- One **single-site** `site.yml` per participant (server and clients).
- Each `site.yml` must contain: `name`, `org`, `type`.

Template: `../site.template.yml`

## Run
```bash
./agent_demo.sh <project_name> <server_endpoint> <work_dir> <site_yaml...>
```

Example (server + two clients):
```bash
./agent_demo.sh my-project grpc://localhost:8002 ./distprov_demo \
  ./server.yml ./site-1.yml ./site-2.yml
```
