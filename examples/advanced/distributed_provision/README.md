# Distributed Provisioning Examples

These examples use the public distributed provisioning CLI workflow:

1. `nvflare cert init` creates the project CA.
2. `nvflare cert request` creates a local private key, CSR, metadata, and request zip.
3. `nvflare cert approve` signs the request zip and returns a signed zip.
4. `nvflare package` combines the signed zip with the local request folder to build a startup kit.

Two example modes are provided:

- `interactive_mode/` for the role-separated Bash walkthrough
- `scripted_mode/` for the scripted JSON-output flow
- `site.template.yml` for the single-site identity template

The helper scripts still accept one `site.yml` per participant. They map
`type: client` to `cert request site`, `type: server` to `cert request server`,
and `type: org_admin|lead|member` to `cert request user`.

For scripted flows, pass `--format json` so stdout remains machine-readable.
