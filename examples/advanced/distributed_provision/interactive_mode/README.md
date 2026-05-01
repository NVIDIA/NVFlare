# Distributed Provisioning - Interactive Mode

This demo shows the role-separated distributed provisioning flow with direct
`nvflare` CLI commands. There is no wrapper script: each command is what a human
Project Admin or requester runs.

Run all commands in this README from the parent `distributed_provision`
directory, not from `interactive_mode/`. The command examples use paths such as
`server.yaml` and `project_profile.yaml`, and those files are in the parent
directory.

```bash
cd examples/advanced/distributed_provision
```

The shared demo input files are checked in the parent directory:

- `project_profile.yaml`: Project Admin profile used for CA initialization and approvals; includes the approved server endpoint.
- `project.yaml`: Equivalent centralized provisioning project file for comparison.
- `server.yaml`: server requester participant definition.
- `site-1.yaml`: client requester participant definition.
- `site-2.yaml`: client requester participant definition.
- `alice.yaml`: admin user requester participant definition.
- `site-3.yaml`: client requester participant definition for dynamic add.
- `bob.yaml`: admin user requester participant definition for dynamic add.

## Example Scenario

| Role | Name | Type | Org | Server endpoint source |
|---|---|---|---|---|
| Project Admin | - | - | - | - |
| Server | `server.example.com` | `server` | nvidia | `project_profile.yaml` |
| Client 1 | `site-1` | `client` | org1 | signed zip from approval |
| Client 2 | `site-2` | `client` | org2 | signed zip from approval |
| User | `alice@nvidia.com` | `admin` | nvidia | signed zip from approval |

`org` is required and is validated by NVFlare as an organization name.

## Step 1 - Project Admin: Initialize Root CA

```bash
nvflare cert init --profile project_profile.yaml -o ca --version 00
```

This creates `ca/rootCA.pem`, `ca/rootCA.key`, and `ca/ca.json`. The
`ca.json` metadata records `provision_version: "00"` and the root CA
fingerprint used in signed approvals.

## Step 2 - Requesters: Create Request Zips

Each requester runs the command for their own participant definition on the
machine that should keep the private key.

```bash
nvflare cert request --participant server.yaml
nvflare cert request --participant site-1.yaml
nvflare cert request --participant site-2.yaml
nvflare cert request --participant alice.yaml
```

Each request creates a local folder named after the participant. For example,
`site-1/` contains `site-1.key`, `site-1.csr`, `site.yaml`, `request.json`, and
`site-1.request.zip`.

The requester keeps the folder and `.key` private, and sends only the
`.request.zip` file to the Project Admin.

## Step 3 - Project Admin: Approve Request Zips

The Project Admin approves the received zip files directly with the local
`project_profile.yaml`. The profile is the source of the server endpoint:
`server.host`, `server.fed_learn_port`, and `server.admin_port`.

```bash
nvflare cert approve server.example.com/server.example.com.request.zip --ca-dir ca --profile project_profile.yaml
nvflare cert approve site-1/site-1.request.zip --ca-dir ca --profile project_profile.yaml
nvflare cert approve site-2/site-2.request.zip --ca-dir ca --profile project_profile.yaml
nvflare cert approve alice@nvidia.com/alice@nvidia.com.request.zip --ca-dir ca --profile project_profile.yaml
```

Each approval command creates a `<name>.signed.zip` next to the request zip.
The signed zip includes the approved server endpoint information from
`project_profile.yaml`. Return only this signed zip to the matching requester.

The approval command prints `rootca_fingerprint_sha256`. Share this value with
the requester through a trusted out-of-band channel so they can verify it before
packaging.

## Step 4 - Requesters: Package Startup Kits

Each requester runs:

```bash
nvflare package server.example.com/server.example.com.signed.zip --fingerprint <expected_fingerprint>
nvflare package site-1/site-1.signed.zip --fingerprint <expected_fingerprint>
nvflare package site-2/site-2.signed.zip --fingerprint <expected_fingerprint>
nvflare package alice@nvidia.com/alice@nvidia.com.signed.zip --fingerprint <expected_fingerprint>
```

`nvflare package` uses the signed zip endpoint information and the local request
folder containing the private key. No endpoint, project-file, or template
argument is needed. The startup kits are written under
`workspace/<project>/prod_00/<name>/` because this demo initializes the CA with
provision version `00`. All four packages land under the same `prod_00`
directory.

## Dynamic Provisioning - Add Participants Later

After the project has started, do not run `cert init` again and do not
repackage existing participants. A new participant repeats the same
request/approve/package flow with the existing Project Admin `ca/` directory and
`project_profile.yaml`, so the new signed zip receives the same approved server
endpoint.

Example: add a new client site:

```bash
nvflare cert request --participant site-3.yaml
nvflare cert approve site-3/site-3.request.zip --ca-dir ca --profile project_profile.yaml
nvflare package site-3/site-3.signed.zip --fingerprint <expected_fingerprint>
```

Example: add a new admin user:

```bash
nvflare cert request --participant bob.yaml
nvflare cert approve bob@nvidia.com/bob@nvidia.com.request.zip --ca-dir ca --profile project_profile.yaml
nvflare package bob@nvidia.com/bob@nvidia.com.signed.zip --fingerprint <expected_fingerprint>
```

The new participant receives a new startup kit under
`workspace/fed_project/prod_00/<name>/` when using the existing version `00`
CA. Existing startup kits are left as-is.
Provisioning creates the identity and startup kit; study membership and other
runtime policy changes are managed separately.

## Compare With Centralized Provisioning

`project.yaml` describes the same project and participants in the centralized
provisioning format. Run centralized provisioning from the same parent
`distributed_provision` directory:

```bash
nvflare provision -p project.yaml -w centralized_workspace --force
```

Now compare the generated startup-kit layout:

```bash
find workspace/fed_project -path '*/startup' -type d | sort
find centralized_workspace/fed_project -path '*/startup' -type d | sort
```

Expected layout:

- Distributed interactive provisioning packages one signed zip at a time, but
  the provision version comes from the CA. Because this demo uses version `00`,
  all distributed kits are generated under
  `workspace/fed_project/prod_00/<name>/startup`.
- Centralized provisioning generates all participant kits in one run, so the
  kits are together under
  `centralized_workspace/fed_project/prod_00/<name>/startup`.

The startup-kit contents for each participant should be equivalent. To inspect
the generated files:

```bash
find workspace/fed_project -path '*/startup/*' -type f | sort
find centralized_workspace/fed_project -path '*/startup/*' -type f | sort
```

The distributed flow also leaves the role-specific request folders,
`*.request.zip`, and `*.signed.zip` files because those are the artifacts passed
between requesters and the Project Admin. The centralized flow uses the single
`project.yaml` directly and does not model that handoff.

## Optional - Custom Builders

`cert request` and `cert approve` only handle identity and certificate approval.
If a participant needs custom startup-kit generation, add `builders:` to that
participant definition before running `cert request`. `nvflare package` honors
those builders from the local request folder when it builds the signed
participant's kit.
