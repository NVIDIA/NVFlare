# Provision bare-metal CVMs

This example uses the unified confidential-computing provisioning interface to
build a TDX server and an AMD SEV-SNP client with an NVIDIA confidential GPU.
For the documented starting point, use the
[single-platform CPU-only example](cpu_only/README.md); return here when the
federation requires a mixed CPU/GPU topology.

The project-wide [`cc_project.yml`](cc_project.yml) contains the Trustee,
approval, and trusted CVM Builder settings. Each protected participant selects
`cc_deployment_mode: bare_metal_cvm` through its own `cc_config` file:

- [`cc_server.yml`](cc_server.yml) selects a TDX CPU-only CVM.
- [`cc_site-1.yml`](cc_site-1.yml) selects an SNP plus NVIDIA CC CVM.

Build the Docker-save archive described in [docker/README.md](docker/README.md),
replace the placeholder image and credential paths, and supply approved generic
CVM artifacts at the two `cvm_image` paths. Production approval is a separate,
one-time operation for each exact generic CPU/GPU bundle. The builder's official
[`acceptance_runner.py`](../../../tests/integration_test/lighter/cc/image_builder/acceptance_runner.py)
runs the complete Trustee plus hardware matrix and writes the report used by
`cvmctl admin approve`; follow the
[exact-manifest approval procedure](../../../nvflare/lighter/cc/image_builder/BUILD_GUIDE.md#21-exact-manifest-acceptance-and-approval)
before using a new generic image here.

Then run:

```bash
nvflare provision -p project.yml -w ./workspace
```

The common packager verifies and removes the plaintext startup kits before it
invokes CVM Builder. OCI artifacts are delivered under the normal production
directory unless `build_tools.bare_metal_cvm.output_root` selects an external
private build location. Common delivery records are written to
`cc_manifests/<participant>.json`.

The bundled client job uses `ClientAPIExecutor` to launch
`/local/custom/client.py` from the measured application image. Because the job
does not ship custom code, `cc_site-1.yml` explicitly grants that exact class in
the participant's `class_allow_list`. This is a reviewed per-site opt-in; do not
replace it with a package prefix or add it to the global default allowlist.

See the [unified deployment guide](../../../docs/user_guide/confidential_computing/deployment.rst),
the [CVM build guide](../../../nvflare/lighter/cc/image_builder/BUILD_GUIDE.md),
and the [Trustee guide](../../../nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md).
