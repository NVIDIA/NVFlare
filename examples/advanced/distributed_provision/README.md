# Distributed Provisioning Examples

Distributed provisioning lets each participant create and keep its own private
key. The Project Admin signs certificate requests, returns signed zips, and
never receives participant private keys.

This example directory contains two ways to run the same federation:

| Example | Use When | Entry Point |
|---|---|---|
| Interactive mode | You want to see the role-separated commands a Project Admin, server admin, site admin, and user requester would run. | [`interactive_mode/README.md`](interactive_mode/README.md) |
| Scripted mode | You want one repeatable demo script that automates request, approval, packaging, and dynamic add. | [`scripted_mode/README.md`](scripted_mode/README.md) |

## Responsibilities

| Role | Owns | Sends | Receives |
|---|---|---|---|
| Project Admin | `project_profile.yaml`, CA directory, approval policy | `<participant>.signed.zip`, `rootca_fingerprint_sha256` through a trusted out-of-band channel | `<participant>.request.zip` |
| Server Admin | Server participant definition, server host/ports, local server private key | `server.example.com.request.zip` | `server.example.com.signed.zip` |
| Site Admin | Site participant definition, local site private key | `site-1.request.zip`, `site-2.request.zip`, etc. | matching signed zip |
| User Requester | User participant definition, local user private key | `alice@nvidia.com.request.zip`, etc. | matching signed zip |

Only `*.request.zip` is sent to the Project Admin. The generated `*.key` file
stays with the requester.

## Workflow

The public CLI flow is:

```bash
# Project Admin: initialize the CA. Deploy version defaults to 00.
nvflare cert init --profile project_profile.yaml -o ca --deploy-version 00

# Requester: create local private key, CSR, request metadata, and request zip.
nvflare cert request --participant site-1.yaml

# Project Admin: approve the request zip and return the signed zip.
nvflare cert approve site-1/site-1.request.zip --ca-dir ca --profile project_profile.yaml

# Requester: package the signed zip with the local private key.
nvflare package site-1/site-1.signed.zip --fingerprint <rootca_fingerprint_sha256>
```

`<rootca_fingerprint_sha256>` is the `rootca_fingerprint_sha256` value printed by
`nvflare cert approve`. Share it separately from the signed zip so the requester
can verify the root CA inside the signed zip before packaging.

For a quick local demo where you intentionally skip that out-of-band trust
check, omit `--fingerprint`:

```bash
nvflare package site-1/site-1.signed.zip
```

Normally ignore `--deploy-version`; the default is `00`. All participants
approved with that CA/deploy version package into the same
`workspace/<project>/prod_00/` directory. Use `01`, `02`, etc. only when the
Project Admin intentionally creates a new deployment CA.

## Shared Inputs

The YAML files are shared by both modes and live in this directory:

```text
distributed_provision/
  project_profile.yaml
  project.yaml
  server.yaml
  site-1.yaml
  site-2.yaml
  alice.yaml
  site-3.yaml
  bob.yaml
  interactive_mode/
    README.md
  scripted_mode/
    README.md
    scripted_mode_demo.sh
```

`project_profile.yaml` is the Project Admin profile. It contains the project
name, communication scheme, default connection security, and approved server
endpoint.

Participant definition files use the same top-level shape as a
single-participant `project.yaml`: top-level `name`, optional `description`, and
exactly one item under `participants`. Client and admin user definitions do not
include server endpoint information. The approved endpoint comes from
`project_profile.yaml` and is written into the signed zip during approval.

`site-3.yaml` and `bob.yaml` are used by the dynamic-add examples.

## Run Interactive Mode

Use this when you want clean, role-separated commands and a comparison with
centralized provisioning:

```bash
cd examples/advanced/distributed_provision
```

Then follow [`interactive_mode/README.md`](interactive_mode/README.md).

The interactive walkthrough shows:

- initial CA setup with `--deploy-version 00`;
- request zip creation for server, two client sites, and one admin user;
- Project Admin approval with `project_profile.yaml`;
- requester packaging, with optional `--fingerprint <rootca_fingerprint_sha256>`;
- dynamic add for a new client and a new admin user;
- comparison with centralized `nvflare provision -p project.yaml`.

## Run Scripted Mode

Use this when you want a repeatable automation-style run with JSON command
outputs:

```bash
cd examples/advanced/distributed_provision
./scripted_mode/scripted_mode_demo.sh /tmp/nvflare_distprov_demo
```

The script writes request, approval, and package command outputs to numbered
JSON files under the work directory. It uses `--format json` and passes the
approval fingerprint to `nvflare package --fingerprint`.

Add a new participant to the same `prod_00` deployment:

```bash
./scripted_mode/scripted_mode_demo.sh --add site-3 site-3.yaml /tmp/nvflare_distprov_demo
```

For more details, see [`scripted_mode/README.md`](scripted_mode/README.md).

## Output

Both modes generate startup kits under:

```text
workspace/fed_project/prod_00/<participant>/
```

For example:

```text
workspace/fed_project/prod_00/server.example.com/startup/
workspace/fed_project/prod_00/site-1/startup/
workspace/fed_project/prod_00/site-2/startup/
workspace/fed_project/prod_00/alice@nvidia.com/startup/
```

Dynamic-add participants are added under the same `prod_00` directory when they
use the existing CA and deploy version.
