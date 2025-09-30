.. _cc_deployment_guide:

################################################
FLARE Confidential Federated AI Deployment Guide
################################################

On Premise: AMD CVM and NVIDIA GPU
-----------------------------------

Prerequisites
=============

Hardware Prerequisites
^^^^^^^^^^^^^^^^^^^^^^

- The machine that needs to run CC needs to have AMD SNP-SEV enabled CPU.
- To run GPU, the machine needs to have a “production” board of NVIDIA GPU that has CC-enabled.
- The AMD firmware is updated to at least 1.55. Here is how to update the firmware:

.. code-block:: bash

   wget https://download.amd.com/developer/eula/sev/amd_sev_fam19h_model0xh_1.55.29.zip
   unzip amd_sev_fam19h_model0xh_1.55.29.zip
   sudo mkdir -p /lib/firmware/amd
   sudo cp amd_sev_fam19h_model0xh_1.55.29.sbin /lib/firmware/amd/amd_sev_fam19h_model0xh.sbin
   sudo reboot

After reboot, verify the firmware is updated:

.. code-block:: bash

   $ sudo dmesg | grep -i sev
   [    0.000000] SEV-SNP: RMP table physical range [0x0000000035600000 - 0x0000000075bfffff]
   [    5.030228] ccp 0000:45:00.1: sev enabled
   [    5.128124] ccp 0000:45:00.1: SEV firmware update successful
   [    8.265240] ccp 0000:45:00.1: SEV API:1.55 build:29
   [    8.265248] ccp 0000:45:00.1: SEV-SNP API:1.55 build:29
   [    8.273638] kvm_amd: SEV enabled (ASIDs 100 - 509)
   [    8.273640] kvm_amd: SEV-ES enabled (ASIDs 1 - 99)
   [    8.273642] kvm_amd: SEV-SNP enabled (ASIDs 1 - 99)

Software Prerequisites
^^^^^^^^^^^^^^^^^^^^^^

- Ubuntu 25.04 ( Host must be 25.04, Guest could be 24.04)
- Has qemu on the machine.
- Get the codes from NVFlare main GitHub repo: `git clone https://github.com/NVIDIA/NVFlare.git`
- Get image builder codes:
  - Get the image builder codes from NVFlare team (email federatedlearning@nvidia.com)
- Get base images from NVFlare team.
- Copy them inside `~/nvflare-github/nvflare/lighter/cc/image_builder/base_images`.
- Get or build your own kbs client that needs to match the kbs server, we are using commit: `a2570329cc33daf9ca16370a1948b5379bb17fbe`.
- Copy it inside `~/nvflare-github/nvflare/lighter/cc/image_builder/kbs`.
-- note::
    The current KBS doesn’t support updating individual rules.
    So we have to update the whole rule file every time a new CVM is added.
    We use that file as the master copy of the policy file.
    You just need to create this folder `/shared/policy` and place the following files
    there: `policy.rego`, `set-policy.sh`, `private.key`, please get these files from the NVFlare team.

Project Admin Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Learn `Trustee Service:<https://www.redhat.com/en/blog/introducing-confidential-containers-trustee-attestation-services-solution-overview-and-use-cases>`_
- Following the Instruction of `Trustee:<https://github.com/confidential-containers/trustee?tab=readme-ov-file>`
- Deploy the Trustee KBS server: Deployment Guide of Trustee KBS and HashiCorp Vault :ref:`hashicorp_vault_trustee_deployment`



Usage
=====

Docker Image Build Step
-----------------------

The CC image builder is for any generic workload not restricted to NVFlare, it will load the docker_archive specified in the configs and launch it inside the CVM.

To use with NVFlare, we need to first build a docker image, for example, we can use this Dockerfile to build:

.. code-block:: docker

   ARG BASE_IMAGE=python:3.12

   FROM ${BASE_IMAGE}

   ENV PYTHONDONTWRITEBYTECODE=1
   ENV PIP_NO_CACHE_DIR=1

   RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install zip
   RUN pip install -U pip
   RUN pip install git+https://github.com/NVIDIA/NVFlare.git@main
   COPY application_code.zip application_code.zip
   RUN nvflare pre-install install -a application_code.zip

   ENTRYPOINT ["/user_config/nvflare/startup/sub_start.sh"]

Note that for CC jobs, we don't allow custom codes, so we must pre-install those codes inside each CVM. We utilize our nvflare pre-install command to do that.

First, we need to prepare the application workload as docker image

<ADD TODO >
<ADD TODO >
<ADD TODO >


Provision Step
--------------

Switch directory to NVFlare example: `NVFlare/examples/advanced/cc_provision`.

Edit the `project.yml` and change the following fields:
- `Build_image_cmd` under the `OnPremPackager`: change it to the absolute path of the image builder code, for example:

.. code-block:: yaml

   packager:
     path: nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager
     args:
       # this needs to be replaced with the real path of the image build scripts
       build_image_cmd: /localhome/local-yuantingh/nvflare-github/nvflare/lighter/cc/image_builder/cvm_build.sh

(Optional) Customize the CC configuration, this is optional for users, but mandatory now for QA testing, please refer to the next section for each field’s meaning:

Edit the `cc_server1.yml`:
- To pre-install the docker workload
    <todo>

Edit the `cc_site-1.yml`:
- To pre-install the custom code for the job, we need to package it to a NVFlare code package, please refer to the sections below called “NVFlare code package”, then we add this:
  - `nvflare_package: application_code.zip`
- For the NVFlare version we want, since 2.7 is not released, we change to main branch:
  - `nvflare_version: git+https://github.com/NVIDIA/NVFlare.git@main`
- Add the IP of the server (if known beforehand for testing), for example:
  - `host_entries:`
    - `server1: 10.176.4.244`
- Remove the GPU authorizer and `cc_gpu_mechanism` line since we DO NOT have production board in this machine.

You can refer to the How to use CC section of the document for the real YAMLs that I was using.

Run `$ nvflare provision -p project.yml` (takes around 1000 seconds to build each CVM).

The startup packages will be generated inside each site’s folder:

.. code-block:: text

   ./workspace/example_project/prod_00/
      /server1/server1.tgz
      /site-1/site-1.tgz

We can then distribute these folders to each site.

Run step
--------

Once each folder is distributed to each site, we can un-tar it:

.. code-block:: bash

   $ tar -zxvf server1.tgz

Then we just start it using `launch_vm.sh`:

.. code-block:: bash

   $ cd cvm_xxx
   $ ./launch_vm.sh

Similarly, do the same for client site-1:

.. code-block:: bash

   $ tar -zxvf site-1.tgz
   $ cd cvm_yyy
   $ ./launch_vm.sh

The server and client will be started automatically inside each CVM. We can then use the admin client to interact with the system.

Switch directory to NVFlare example:

.. code-block:: bash

   $ cd NVFlare/examples/advanced/cc_provision

Copy job inside admin client:

.. code-block:: bash

   $ cp -r jobs/* ./workspace/example_project/prod_00/admin@nvidia.com/transfer/

(Optional) if the server name is NOT a public Domain Name, please add an entry in your `/etc/hosts` for the admin client machine.

Start the admin:

.. code-block:: bash

   $ ./workspace/example_project/prod_00/admin@nvidia.com/startup/fl_admin.sh

Inside the admin console, we can submit the job:

.. code-block:: bash

   submit_job hello-pt_cifar10_fedavg

CC Configuration
================

.. list-table::
   :header-rows: 1

   * - Field name
     - Field value
     - Meaning
   * - compute_env
     - onprem_cvm
     - Computation environment
   * - cc_cpu_mechanism
     - amd_sev_snp
     - CC CPU mechanism
   * - role
     - Server / client
     - Role in NVFlare system
   * - root_drive_size
     - An integer
     - GBs for root drive
   * - applog_drive_size
     - An integer
     - GBs for applog drive
   * - user_config_drive_size
     - An integer
     - GBs for user_config drive
   * - user_data_drive_size
     - An integer
     - GBs for user_data drive
   * - docker_archive
     - /localhome/local-yuantingh/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz
     - Absolute path to the docker image saved using: `docker save <image_name> | gzip > app.tar.gz`
   * - user_config
     - A list of key-value pairs,
     - This “value” path will be mounted in the docker container inside “/user_config/[key]”
   * - cc_issuers
     - 
     - Contains lists of issuers that are implemented in NVFlare
   * - id
     - snp_authorizer
     - ID of the issuer
   * - path
     - "nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer"
     - Path to the issuer class
   * - token_expiration
     - 100
     - Token expiration in seconds, needs to be less than “check_frequency”
   * - cc_attestation
     - 
     - 
   * - check_frequency
     - 120
     - In seconds, how frequent should we do attestation check


Reference YAMLs for testing on 10.176.200.152 machine
=====================================================

.. code-block:: yaml

   $ cat project_local.yml
   api_version: 3
   name: example_project
   description: NVIDIA FLARE sample project yaml file

   participants:
     # Change the name of the server (server1) to the Fully Qualified Domain Name
     # (FQDN) of the server, for example: server1.example.com.
     # Ensure that the FQDN is correctly mapped in the /etc/hosts file.
     - name: server1
       type: server
       org: nvidia
       fed_learn_port: 8002
       cc_config: cc_server1_local.yml
     - name: site-1
       type: client
       org: nvidia
       cc_config: cc_site-1_local.yml
       # Specifying listening_host will enable the creation of one pair of
       # certificate/private key for this client, allowing the client to function
       # as a server for 3rd-party integration.
       # The value must be a hostname that the external trainer can reach via the network.
       # listening_host: site-1-lh
     - name: admin@nvidia.com
       type: admin
       org: nvidia
       role: project_admin

   # The same methods in all builders are called in their order defined in builders section
   builders:
     - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
     - path: nvflare.lighter.impl.static_file.StaticFileBuilder
       args:
         # config_folder can be set to inform NVIDIA FLARE where to get configuration
         config_folder: config

         # scheme for communication driver (currently supporting the default, grpc, only).
         # scheme: grpc

         # app_validator is used to verify if uploaded app has proper structures
         # if not set, no app_validator is included in fed_server.json
         # app_validator: PATH_TO_YOUR_OWN_APP_VALIDATOR

         # download_job_url is set to http://download.server.com/ as default in fed_server.json.  You can override this
         # to different url.
         # download_job_url: http://download.server.com/

         overseer_agent:
           path: nvflare.ha.dummy_overseer_agent.DummyOverseerAgent
           # if overseer_exists is true, args here are ignored.  Provisioning
           #   tool will fill role, name and other local parameters automatically.
           # if overseer_exists is false, args in this section will be used and the sp_end_point
           # must match the server defined above in the format of SERVER_NAME:FL_PORT:ADMIN_PORT
           #
           overseer_exists: false
           args:
             sp_end_point: server1:8002:8002

     - path: nvflare.lighter.impl.cert.CertBuilder
     - path: nvflare.lighter.impl.signature.SignatureBuilder
     - path: nvflare.lighter.cc_provision.impl.cc.CCBuilder
   packager:
     path: nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager
     args:
       # this needs to be replace with the real path of the image build scripts
       build_image_cmd: /localhome/local-yuantingh/nvflare-github/nvflare/lighter/cc/image_builder/cvm_build.sh

.. code-block:: yaml

   $ cat cc_server1_local.yml
   compute_env: onprem_cvm
   cc_cpu_mechanism: amd_sev_snp
   role: server

   # All drive sizes are in GB
   root_drive_size: 30
   applog_drive_size: 1
   user_config_drive_size: 1
   user_data_drive_size: 1
   # Docker image archive saved using:
   # docker save <image_name> | gzip > app.tar.gz
   docker_archive: /localhome/local-yuantingh/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz
   # will be mount inside docker "/user_config/nvflare"
   user_config:
     nvflare: /tmp/startup_kits

   allowed_ports:
   - 8002

   cc_issuers:
     - id: snp_authorizer
       path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
       token_expiration: 100 # seconds, needs to be less than check_frequency

   cc_attestation:
     check_frequency: 120 # seconds
     failure_action: stop_job

.. code-block:: yaml

   $ cat cc_site-1_local.yml
   compute_env: onprem_cvm
   cc_cpu_mechanism: amd_sev_snp
   role: client

   # All drive sizes are in GB
   root_drive_size: 30
   applog_drive_size: 1
   user_config_drive_size: 1
   user_data_drive_size: 1
   # Docker image archive saved using:
   # docker save <image_name> | gzip > app.tar.gz
   docker_archive: /localhome/local-yuantingh/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz

   # for debugging purpose
   hosts_entries:
      server1: 10.176.200.152

   # will be mount inside docker "/user_config/nvflare"
   user_config:
     nvflare: /tmp/startup_kits

   cc_issuers:
     - id: snp_authorizer
       path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
       token_expiration: 100 # seconds, needs to be less than check_frequency

   cc_attestation:
     check_frequency: 120 # seconds
     failure_action: stop_job

