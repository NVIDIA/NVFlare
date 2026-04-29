# Distributed Provisioning Examples

These examples use the public distributed provisioning CLI workflow:

1. `nvflare cert init` creates the project CA.
2. The Project Admin keeps a `project_profile.yaml` with `name`, `scheme`,
   `connection_security`, and the server endpoint fields.
3. Each requester creates one participant definition file.
4. `nvflare cert request --participant <participant.yaml>` creates a local private key, CSR, metadata, and request zip.
5. `nvflare cert approve <request.zip> --ca-dir <dir> --profile <project_profile.yaml>` signs the request zip, adds the server endpoint information from the profile to the signed zip, and prints `rootca_fingerprint_sha256`.
6. `nvflare package <signed.zip> --confirm-rootca` combines the signed zip with the local request material to build a startup kit.

Interactive examples use `package --confirm-rootca` so the site admin can
compare the displayed fingerprint with the Project Admin's out-of-band value.
Scripted examples pass the approval output to
`package --expected-rootca-fingerprint`.

## Directory Layout

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

Two example modes are provided:

- `interactive_mode/` for the role-separated human CLI demo and centralized `project.yaml` comparison
- `scripted_mode/` for one simple automation script using the same shared inputs

`site-3.yaml` and `bob.yaml` are used for the dynamic provisioning examples.

Participant definition files use the same top-level shape as a single-participant
`project.yaml`: top-level `name`, optional `description`, and exactly one entry
under `participants`. Client and admin user definitions do not include server
endpoint information. The Project Admin controls the server endpoint in
`project_profile.yaml`, approval writes it into the signed zip, and
`nvflare package` uses the signed zip directly. Server definitions carry their
own ports and may optionally include a local `connection_security` override only
for deployments where the server process is behind a proxy, load balancer, or
ingress that handles the external TLS boundary.

For scripted flows, pass `--format json` so stdout remains machine-readable.
