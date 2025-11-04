# How to use CC provision

This guide explains how to use **CC (Confidential Computing) Provision** in NVFLARE, including setting up site configurations, enabling the CC builder, and using Docker images for CC workloads.


## 0. Prepare application docker image workload

In CC, we don't allow custom code, all the codes and required libs must be built-in in the docker image.
In this example, we show you how to build NVFlare docker images in [docker/](docker/README.md)


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
user_config_drive_size: 1
user_data_drive_size: 1

# Docker image archive saved using:
# docker save <image_name> | gzip > app.tar.gz
docker_archive: /tmp/base_images/app.tar.gz

allowed_ports:
- 8002

cc_issuers:
  - id: snp_authorizer
    path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
    token_expiration: 100 # seconds, needs to be less than check_frequency
    args:
      snpguest_binary: "/host/bin/snpguest"
cc_attestation:
  check_frequency: 120 # seconds
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

Note that this CCBuilder needs to be placed **after** the "StaticFileBuilder" and
**before** the "SignatureBuilder".
This builder sets up all CC-related configurations and assets.

## 4. Add the OnPremPackager

To generate startup kits for on-premises deployment, add the `OnPremPackager`:

```yaml
packager:
  path: nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager
  args:
    build_image_cmd: build_cvm_image.sh
```

Note:
    1. `build_image_cmd`: Path to the script used to build the CVM disk image.
    2. For 2.7.0 Technical Preview release, please contact `federatedlearning@nvidia.com` to receive the `build_cvm_image.sh`

## 5. Generate the Startup Kits

Once you add all the required sections into your `project.yml`, run the provision command:

```bash
nvflare provision -p project.yml
```

## 6. Distribute and deploy

Each site's result will be located in 

```bash
workspace/example_project/prod_xx/[site_name]/[site_name].tgz
```

You can distribute these tgz file to each site.

To deploy on each site, do:

```bash
tar -zxvf [site_name].tgz
cd cvm_xxx
./launch_vm.sh
```

The confidential VM will start, and the NVFLARE server and clients will automatically connect and begin operation.
You can now use the NVFlare admin console to communicate with the NVFlare system.
