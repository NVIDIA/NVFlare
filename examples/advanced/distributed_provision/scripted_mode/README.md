# Distributed Provisioning - Scripted Mode

This demo automates the same federation shown in `../interactive_mode`.

The script is intentionally small and explicit. It automates the distributed
request, approve, and package process with the shared YAML files checked in the
parent `distributed_provision` directory.

## Run

Run the script from the parent `distributed_provision` directory:

```bash
cd examples/advanced/distributed_provision
./scripted_mode/scripted_mode_demo.sh
```

The shared YAML inputs live in that parent directory. By default, the script
writes to `./distprov_demo` relative to the directory where you run it, so the
commands above create `examples/advanced/distributed_provision/distprov_demo`.
To choose a different output directory:

```bash
./scripted_mode/scripted_mode_demo.sh ./my_distprov_run
```

The work directory must not already exist, so each run has clean output.

## What Is Automated

The script uses `project_profile.yaml` and the participant YAML files from the
parent directory. `project_profile.yaml` is the only distributed input that
contains the server endpoint fields: `server.host`, `server.fed_learn_port`,
and `server.admin_port`.

Setup:

1. Project Admin initializes the CA with `nvflare cert init --deploy-version 00`.

Automated distributed provisioning flow:

1. Each requester creates a request zip with `nvflare cert request --participant`.
2. The script copies only `*.request.zip` files to simulate handoff to the Project Admin.
3. Project Admin approves each request with `nvflare cert approve`, which writes
   the server endpoint information into the signed zip.
4. Each requester packages its startup kit with `nvflare package` using only the
   signed zip and local request material.

For automation, the package step uses `--fingerprint <rootca_fingerprint_sha256>`
from the approval JSON output. The package command does not pass an endpoint,
project file, or template argument; those values come from the signed zip.

The signed zip also carries deploy-version metadata, so all packages in this
script land under `prod_00` unless the script is changed to initialize a
different deploy version.

## Dynamic Provisioning

After the initial project has started, use `--add` to automate
request/approve/package for one new participant with the existing CA. This does
not rerun `cert init` and does not repackage existing participants.

Add a new client site:

```bash
./scripted_mode/scripted_mode_demo.sh --add site-3 site-3.yaml ./distprov_demo
```

Add a new admin user:

```bash
./scripted_mode/scripted_mode_demo.sh --add bob@nvidia.com bob.yaml ./distprov_demo
```

The `--add` mode uses `site-3.yaml` or `bob.yaml` from the parent directory,
approves with `./distprov_demo/ca`, and packages the new startup kit under
`./distprov_demo/workspace/fed_project/prod_00/<name>/`.

## Output

```text
distprov_demo/
  ca/
  server.example.com/
  site-1/
  site-2/
  alice@nvidia.com/
  *.request.zip
  *.signed.zip
  workspace/
  01_cert_init.json
  02_request_<name>.json
  03_approve_<name>.json
  04_package_<name>.json
  05_startup_dirs.txt
  06_dynamic_request_<name>.json
  07_dynamic_approve_<name>.json
  08_dynamic_package_<name>.json
  09_dynamic_startup_dirs.txt
```
