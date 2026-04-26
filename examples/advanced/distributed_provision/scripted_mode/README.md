# Distributed Provisioning - Scripted Mode

This directory demonstrates how to automate the public distributed provisioning
workflow with JSON CLI output:

```bash
nvflare cert init
nvflare cert request
nvflare cert approve
nvflare package
```

The demo script runs every role on one machine so it is easy to try. In a real
deployment, the same command sequence is usually split across machines:

- requester side: `cert request`, then later `package`
- Project Admin side: `cert init`, `cert approve`

The script keeps those role boundaries visible in comments and output events so
it can be used as an automation template.

## Run

```bash
./scripted_mode_demo.sh <project_name> <server_endpoint> <work_dir> <site_yaml...>
```

Example with one server and one client:

```bash
cat > server.yml <<'EOF'
name: server1
org: nvidia
type: server
EOF

cat > site-1.yml <<'EOF'
name: site-1
org: nvidia
type: client
EOF

./scripted_mode_demo.sh example_project grpc://server1:8002 ./distprov_demo \
  ./server.yml ./site-1.yml
```

Notes:

- Each `site.yml` is a single-participant identity file with `name`, `org`,
  and `type`.
- `org` must match `^[A-Za-z0-9_]+$` with no hyphens or spaces.
- `type: client` maps to `cert request site`.
- `type: server` maps to `cert request server`.
- `type: org_admin`, `lead`, or `member` maps to `cert request user`.
- The script saves each CLI JSON response under `<work_dir>` and emits compact
  newline-delimited JSON progress/results on stdout.
- The script reads `rootca_fingerprint_sha256` from `cert approve` output and
  passes it to `package --expected-rootca-fingerprint`.

Output layout:

```text
<work_dir>/
  ca/
  requests/<name>/
  signed/<name>.signed.zip
  workspace/
  request_<name>.json
  approve_<name>.json
  package_<name>.json
```
