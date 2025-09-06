# How to use CC provision

This guide explains how to use **CC (Confidential Computing) Provision** in NVFLARE, including setting up site configurations, enabling the CC builder, and using Docker images for CC workloads.


## 1. Define CC Configuration per Site (`cc_config`)

Each site participating in a CC job must provide a **CC configuration file**. This file describes the trusted execution environment (e.g., AMD SEV-SNP on-prem CVM), drive allocations, and attestation policies.

Here is an example (`cc_server.yml`):


```yaml
compute_env: onprem_cvm
cc_cpu_mechanism: amd_sev_snp
role: server

# All drive sizes are in GB
root_drive_size: 10
applog_drive_size: 1
user_data_drive_size: 1
secure_drive_size: 10
data_source: /tmp/data
docker_archive: base_images/app.tar.gz

allowed_ports:
- 8002

cc_issuers:
  - id: snp_authorizer
    path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
    token_expiration: 150 # needs to be less than check_frequency
cc_attestation:
  check_frequency: 300
  failure_action: stop_job
```

## 2. Reference `cc_config` in `project.yml`

In your `project.yml`, reference the CC configuration file for each site using the `cc_config` key:

```yaml
participants:
  - name: server1
    type: server
    org: nvidia
    fed_learn_port: 8002
    cc_config: cc_server1.yml
```

## 3. Add the CCBuilder

At the end of the `builders` section in your `project.yml`, add the `CCBuilder`:

```yaml
builders:
  - path: nvflare.lighter.cc_provision.impl.cc.CCBuilder
```

This builder sets up all CC-related configurations and assets.

## 4. Add the DockerImageBuilder

A Docker image is required for all CC jobs. The DockerImageBuilder uses your base_dockerfile and appends all necessary NVFLARE components into it.

Add the following to the `builders` section:

```yaml
builders:
  ...
  - path: nvflare.lighter.impl.docker_image_builder.DockerImageBuilder
    args:
      base_dockerfile: Dockerfile.base
      requirement: git+https://github.com/NVIDIA/NVFlare.git@main
```

Note:
    1. The `requirement` is an optional argument
    2. No custom runtime code is allowed, everything must be pre-installed in the image
    3. The final image will be encrypted for execution in a Confidential VM

## 5. Add the OnPremPackager

To generate startup kits for on-premises deployment, add the `OnPremPackager`:

```yaml
packager:
  path: nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager
  args:
    build_image_cmd: build_cvm_image.sh
```

Note:
    1. `build_image_cmd`: Path to the script used to build the CVM disk image.
    2. You can obtain `build_cvm_image.sh` from the NVFLARE team

## 6. Generate the Startup Kits

Once you add all the required sections into your `project.yml`, run the provision command:

```bash
nvflare provision -p project.yml
```

This will do:
    1. Build and encrypt the Docker image
    2. Package the confidential computing environment
    3. Output `.tgz` that contains CVM image for each site

## 7. Distribute and deploy

Each site's result will be located in 

```bash
workspace/example_project/prod_xx/[site_name]/cvm_xxx.tgz
```

You can distribute these tgz file to each site.

To deploy on each site, do:

```bash
tar -zxvf cvm_xxx.tgz
cd cvm_xxx
./launch_vm.sh
```

The confidential VM will start, and the NVFLARE server and clients will automatically connect and begin operation.
You can now use the NVFlare admin console to communicate with the NVFlare system.
