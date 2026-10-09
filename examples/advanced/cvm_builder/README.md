# Provision bare-metal CVMs

This example uses the unified confidential-computing provisioning interface to
build a TDX server and an AMD SEV-SNP client with an NVIDIA confidential GPU.

The project-wide [`cc_project.yml`](cc_project.yml) contains the Trustee,
approval, and trusted CVM Builder settings. Each protected participant selects
`cc_deployment_mode: bare_metal_cvm` through its own `cc_config` file:

- [`cc_server.yml`](cc_server.yml) selects a TDX CPU-only CVM.
- [`cc_site-1.yml`](cc_site-1.yml) selects an SNP plus NVIDIA CC CVM.

Build the Docker-save archive described in [docker/README.md](docker/README.md),
replace the placeholder image and credential paths, then run:

```bash
nvflare provision -p project.yml -w ./workspace
```

The common packager verifies and removes the plaintext startup kits before it
invokes CVM Builder. OCI artifacts are delivered under the normal production
directory unless `build_tools.bare_metal_cvm.output_root` selects an external
private build location. Common delivery records are written to
`cc_manifests/<participant>.json`.

See the [unified deployment guide](../../../docs/user_guide/confidential_computing/deployment.rst),
the [CVM build guide](../../../nvflare/lighter/cc/image_builder/BUILD_GUIDE.md),
and the [Trustee guide](../../../nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md).
