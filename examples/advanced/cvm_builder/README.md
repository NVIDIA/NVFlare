# Provision with CVM Builder

This example configures a TDX server and an AMD SEV-SNP client with an NVIDIA
confidential-computing GPU. Configure a trusted Linux worker with CVM Builder,
approved generic images, and the existing CoCo Trustee before provisioning.

CVM Builder is included at
[`nvflare/lighter/cc/image_builder`](../../../nvflare/lighter/cc/image_builder).
Follow its [build guide](../../../nvflare/lighter/cc/image_builder/BUILD_GUIDE.md)
and [Trustee guide](../../../nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md)
to prepare the worker. The example's `cvm_builder_dir` points to that source tree;
update it if you copy `project.yml` elsewhere.

The [Docker example](docker/README.md) builds the application archive referenced
by `project.yml`. A matching [sample job](jobs/hello-pt_cifar10_fedavg) is also
included for deployment validation.

1. Edit `project.yml`: set the reachable server name,
   `cvm_image` folders, Docker archive, and bootstrap egress ports.
   `cvm_builder_dir` is optional; it defaults to the builder shipped inside the
   installed NVFlare package.
2. Edit `cvm_project.yml` with the project's Trustee endpoint, scoped resource
   token and the acceptance authority's public key that signed the approved CVM
   bundles. Relative credential paths resolve against this file.
3. Run `nvflare provision -p project.yml -w ./workspace`.

The kits keep `/host/bin` mounted because NVFlare's confidential-computing
authorizers look for tools there; set `host_bin: false` per participant to drop
it. `allowed_in_cidrs` and `allowed_out_cidrs` pass through to the vault build.

The Docker archive must be a `docker save` archive containing one Linux amd64
image with NVFlare, Bash, and your workload dependencies. NVFlare derives its
image ID; no `image_id` or `release_id` setting is required. The adapter preserves
the image's `USER`. During privileged vault population, the builder derives its
numeric identity from the authenticated image configuration and applies that
ownership only inside the encrypted vault; unprivileged staging remains owned by
the provisioner. Use a numeric `USER UID[:GID]` for a non-root image; the
empty/default and `root` forms resolve to `0:0`. Optional `workspace_uid` and
`workspace_gid` assertions must be supplied together and match that image
identity. `platforms` defaults to all platforms supplied by each `cvm_image`.

To use a registry image, replace its folder with the immutable reference printed
by `cvmctl publish`, for example:

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
`trustee` settings are not accepted.

Builds create application vaults from existing approved CVMs. Provisioning does
not boot or publish them. Use the returned OCI archive paths or JSON result
metadata to distribute deliveries. On failure, retain logs and recovery records
and resolve uncertain key activation before deliberately rebuilding.
