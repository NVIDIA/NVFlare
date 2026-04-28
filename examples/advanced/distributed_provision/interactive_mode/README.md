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

- `project_profile.yaml`: Project Admin profile used for CA initialization and approvals.
- `project.yaml`: Equivalent centralized provisioning project file for comparison.
- `server.yaml`: server requester participant definition.
- `site-1.yaml`: client requester participant definition.
- `site-2.yaml`: client requester participant definition.
- `alice.yaml`: admin user requester participant definition.
- `site-3.yaml`: client requester participant definition for dynamic add.
- `bob.yaml`: admin user requester participant definition for dynamic add.

## Example Scenario

| Role | Name | Type | Org | Server endpoint fields |
|---|---|---|---|---|
| Project Admin | - | - | - | - |
| Server | `server.example.com` | `server` | nvidia | `server.example.com:8002`, admin `8003` |
| Client 1 | `site-1` | `client` | org1 | connects to `server.example.com:8002` |
| Client 2 | `site-2` | `client` | org2 | connects to `server.example.com:8002` |
| User | `alice@nvidia.com` | `admin` | nvidia | connects to `server.example.com:8003` |

`org` is required and is validated by NVFlare as an organization name.

## Step 1 - Project Admin: Initialize Root CA

```bash
nvflare cert init --profile project_profile.yaml -o ca
```

This creates `ca/rootCA.pem`, `ca/rootCA.key`, and `ca/ca.json`.

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
`project_profile.yaml`.

```bash
nvflare cert approve server.example.com/server.example.com.request.zip --ca-dir ca --profile project_profile.yaml
nvflare cert approve site-1/site-1.request.zip --ca-dir ca --profile project_profile.yaml
nvflare cert approve site-2/site-2.request.zip --ca-dir ca --profile project_profile.yaml
nvflare cert approve alice@nvidia.com/alice@nvidia.com.request.zip --ca-dir ca --profile project_profile.yaml
```

Each approval command creates a `<name>.signed.zip` next to the request zip.
Return only this signed zip to the matching requester.

The approval command prints `rootca_fingerprint_sha256`. Share this value with
the requester through a trusted out-of-band channel so they can verify it before
packaging.

## Step 4 - Requesters: Package Startup Kits

Each requester runs:

```bash
nvflare package server.example.com/server.example.com.signed.zip --confirm-rootca
nvflare package site-1/site-1.signed.zip --confirm-rootca
nvflare package site-2/site-2.signed.zip --confirm-rootca
nvflare package alice@nvidia.com/alice@nvidia.com.signed.zip --confirm-rootca
```

`nvflare package` uses the signed zip and the local request folder containing
the private key. The startup kits are written under
`workspace/<project>/prod_NN/<name>/`.

## Dynamic Provisioning - Add Participants Later

After the project has started, do not run `cert init` again and do not
repackage existing participants. A new participant repeats the same
request/approve/package flow with the existing Project Admin `ca/` directory and
`project_profile.yaml`.

Example: add a new client site:

```bash
nvflare cert request --participant site-3.yaml
nvflare cert approve site-3/site-3.request.zip --ca-dir ca --profile project_profile.yaml
nvflare package site-3/site-3.signed.zip --confirm-rootca
```

Example: add a new admin user:

```bash
nvflare cert request --participant bob.yaml
nvflare cert approve bob@nvidia.com/bob@nvidia.com.request.zip --ca-dir ca --profile project_profile.yaml
nvflare package bob@nvidia.com/bob@nvidia.com.signed.zip --confirm-rootca
```

The new participant receives a new startup kit under
`workspace/fed_project/prod_NN/<name>/`. Existing startup kits are left as-is.
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

Expected difference:

- Distributed interactive provisioning packages one signed zip at a time, so the
  kits are generated under separate `prod_NN` folders, for example
  `workspace/fed_project/prod_00/server.example.com/startup`,
  `workspace/fed_project/prod_01/site-1/startup`, and so on.
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
