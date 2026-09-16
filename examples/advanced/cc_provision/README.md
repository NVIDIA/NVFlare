# Confidential-computing provisioning

Use the [CVM Builder example](cvm_builder/README.md) to provision an Intel TDX
server and an AMD SEV-SNP client with an NVIDIA confidential-computing GPU.
CVM Builder is included in this repository at
[`nvflare/lighter/cc/image_builder`](../../../nvflare/lighter/cc/image_builder).

1. Prepare the build worker, Trustee and approved generic images using the
   [build guide](../../../nvflare/lighter/cc/image_builder/BUILD_GUIDE.md) and
   [Trustee guide](../../../nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md).
2. Build and save a Linux amd64 application image containing NVFlare and the job
   dependencies. The [Docker example](docker/README.md) provides an application
   image and code for the [sample job](jobs/hello-pt_cifar10_fedavg).
3. Edit [cvm_builder/project.yml](cvm_builder/project.yml) and the shared
   [cvm_project.yml](cvm_builder/cvm_project.yml), then run:

   ```sh
   nvflare provision -p cvm_builder/project.yml -w ./workspace
   ```

4. Distribute the resulting OCI deliveries and follow the
   [user guide](../../../nvflare/lighter/cc/image_builder/USER_GUIDE.md) to verify,
   launch and stop them on their target hosts.

The `cvm_vault` configuration builds encrypted application vaults from approved
CVM images. It uses the normal workspace, static-file, certificate and signature
builders shown in the example. No additional `CCBuilder` or `build_image_cmd`
entry is needed for this workflow.
