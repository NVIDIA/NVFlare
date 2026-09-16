# Provision with CVM Builder

This example configures a TDX server and an AMD SEV-SNP client with an NVIDIA
confidential-computing GPU. Configure a trusted Linux worker with CVM Builder,
approved generic images, and the shared key service before provisioning.

1. Edit `project.yml`: set the reachable server name, `cvm_builder_dir`,
   `cvm_image` folders, Docker archive, and bootstrap egress ports.
2. Edit `cvm_project.yml` with the project's key-service endpoint and existing
   builder credentials. Relative credential paths resolve against this file.
3. Run `nvflare provision -p project.yml -w ./workspace`.

The Docker archive must be a `docker save` archive containing one Linux amd64
image with NVFlare, Bash, and your workload dependencies. NVFlare derives its
image ID; no `image_id` or `release_id` setting is required. `platforms` defaults
to all platforms supplied by each `cvm_image`.

To use a registry image, replace its folder with the immutable reference printed
by `scripts/cvm_publish`, for example:

```yaml
cvm_image: registry.example.org/cvm/tdx-cpu@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

`output_root` is optional. Deliveries default to
`workspace/cvm_project/prod_NN/<participant>/`. Signed startup kits and build
inputs are retained privately under `workspace/cvm_project/.cvm-vault-builds/`.
An explicit `output_root` must be outside the provisioning workspace and uses
fresh per-build subdirectories.

Shared `cvm_project.yml` is discovered beside `project.yml` or in its ancestors.
Set `cvm_vault.project_config` to select another file. Per-participant
`key_service` settings are not accepted.

Builds create application vaults from existing approved CVMs. Provisioning does
not boot or publish them. Use the returned OCI archive paths or JSON result
metadata to distribute deliveries. On failure, retain logs and recovery records
and resolve uncertain key activation before deliberately rebuilding.
