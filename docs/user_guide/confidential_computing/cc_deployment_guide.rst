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
- Host OS must be Ubuntu 25.04.
- To run GPU, the machine needs to have an NVIDIA GPU that is CC-enabled.
- The AMD firmware that supports SEV.

Here is how to get the firmware:

.. code-block:: bash
    echo 'deb http://archive.ubuntu.com/ubuntu plucky-proposed main restricted universe multiverse' | \
      sudo tee /etc/apt/sources.list.d/plucky-proposed.list

    sudo tee /etc/apt/preferences.d/99-plucky-proposed <<'EOF'
    Package: *
    Pin: release a=plucky-proposed
    Pin-Priority: 100
    EOF

    sudo apt update
    sudo apt install -t plucky-proposed ovmf

The firmware will installed as ``/usr/share/ovmf/OVMF.amdsev.fd``.

Software Prerequisites
^^^^^^^^^^^^^^^^^^^^^^

- Ubuntu 25.04
- Has qemu on the machine.
- Get the codes from NVFlare main GitHub repo: `git clone https://github.com/NVIDIA/NVFlare.git`
- Get image builder codes:
  - Get the image builder codes from NVFlare team (email federatedlearning@nvidia.com)
- Get base images from NVFlare team.
- Let's assume the image builder code are in `~/cc/image_builder`.
- Copy the base images to `~/cc/image_builder/base_images`.
- Get or build your own kbs client that needs to match the kbs server, we are using commit: `a2570329cc33daf9ca16370a1948b5379bb17fbe`.
- Copy the kbs-client and credentials to `~/cc/image_builder/binaries`.
- Build snpguest with version v0.9.2.
- Copy snpguest and credentials to `~/cc/image_builder/binaries`

.. note::
    The current KBS does'’t support updating individual rules.
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

The CC image builder is for any generic workload not restricted to NVFlare,
it will load the docker_archive specified in the configs and launch it inside
the CVM.

To use with NVFlare, we need to first build a docker image, for example, we can
use the following Dockerfile to build:

.. code-block:: dockerfile

  ARG BASE_IMAGE=python:3.12

  FROM ${BASE_IMAGE}

  ENV PYTHONDONTWRITEBYTECODE=1
  ENV PIP_NO_CACHE_DIR=1

  RUN pip install -U pip &&\
      pip install nvflare~=2.7.0rc
  COPY code/ /local/custom
  COPY requirements.txt .
  RUN pip install -r requirements.txt

  ENTRYPOINT ["/user_config/nvflare/startup/sub_start.sh", "--verify"]

Note that for CC jobs, we don't allow custom codes, so we must include those
codes inside the docker image.



Provision Step
--------------

1. Switch directory to NVFlare example: `NVFlare/examples/advanced/cc_provision`.

2. Edit the `project.yml` and change the following fields:
  
  - `build_image_cmd` under the `OnPremPackager`: change it to the absolute path of the image builder code, for example:

    .. code-block:: yaml

       packager:
         path: nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager
         args:
           # this needs to be replaced with the real path of the image build scripts
           build_image_cmd: ~/nvflare-github/nvflare/lighter/cc/image_builder/cvm_build.sh


3. Customize the CC configuration for each site, please refer to the next section for each field’s meaning:

  - Edit the `cc_server1.yml`:

    - Edit the `docker_archive` field:

      .. code-block:: yaml

          docker_archive: ~/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz

  - Edit the `cc_site-1.yml`:

    - Edit the `docker_archive` field:

      .. code-block:: yaml

          docker_archive: ~/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz

    - If the server name is NOT a public domain name, please add the following section:

      .. code-block:: yaml

        host_entries:
          server1: 10.176.4.244

    - If you don't have GPU, please remove the GPU authorizer and `cc_gpu_mechanism` line

4. Run ``$ nvflare provision -p project.yml`` (takes around 1000 seconds to build each CVM).

5. The startup packages will be generated inside each site's folder:

.. code-block:: text

   ./workspace/example_project/prod_00/
      /server1/server1.tgz
      /site-1/site-1.tgz

We can then distribute these folders to each site.


Content of the CC startup kit
-----------------------------
For the admin, they can see it the same as the non-cc provision.
For the server and sites, we now generate the NEW startup kit which only contains "server1.tgz",
Once you untar it you will see the following:

.. code-block:: bash

   $ ls server1/cvm_885fe8f608b3/
applog.qcow2  crypt_root.qcow2  initrd.img  launch_vm.sh  OVMF.amdsev.fd  README.txt  user_config.qcow2  user_data.qcow2  vmlinuz

Each file is explained as below:

  - Applog.qcow2: the disk file to store the application logs, NVFlare app logs will be written inside, we will have a section below explaining how to mount a qcow2 to inspect its content. Drive image for /applog. This is an unencrypted drive and can be mounted on any VM.
  - Crypt_root.qcow2: Encrypted root drive. A key is required to mount it.
  - Initrd.img: initramfs with init-app
  - Launch_vm.sh: The launch script for the CVM
  - OVMF.amdsev.fd: The firmware with support for kernel-hashes=on
  - README.txt: To explain the content of this folder
  - User_config.qcow2: the disk file to store the application configuration file, NVFlare use this to store the startup kits, users can mount and modify the content
  - User_data.qcow2: This is just a placeholder for user_data drive. It's very small. Users may need to extend it.


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
     - ~/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz
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

How to inspect content of a qcow2 file
======================================

You can use the following command to inspect the content of a qcow2 file:

  - Load the nbd kernel module: 
  
  .. code-block:: bash

    sudo modprobe nbd max_part=8

  - Connect the QCOW2 image: sudo qemu-nbd --connect=/dev/nbd0 user_config.qcow2

  .. code-block:: bash

    sudo qemu-nbd --connect=/dev/nbd0 user_config.qcow2

  - Mount the image to local file system: sudo mount /dev/nbd0 /mnt/user_config 

  .. code-block:: bash

    sudo mount /dev/nbd0 /mnt/user_config 

  - Check the content inside: 

  .. code-block:: bash

    ls /mnt/user_config


  - In NVFlare, we will put the startup kits inside user_config, so we can 
    check inside it has the startup kit content:

  .. code-block:: bash

    ls /mnt/user_config/nvflare/

  - Now we can safely unmount:

  .. code-block:: bash

    sudo umount /mnt/user_config

  - And disconnect:

  .. code-block:: bash

    sudo qemu-nbd --disconnect /dev/nbd0


Reference YAMLs for machine with AMD SNP-SEV enabled CPU
========================================================

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

    - path: nvflare.lighter.impl.cert.CertBuilder
     - path: nvflare.lighter.impl.signature.SignatureBuilder
     - path: nvflare.lighter.cc_provision.impl.cc.CCBuilder
   packager:
     path: nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager
     args:
       # this needs to be replace with the real path of the image build scripts
       build_image_cmd: ~/nvflare-github/nvflare/lighter/cc/image_builder/cvm_build.sh

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
   docker_archive: ~/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz
   # will be mount inside docker "/user_config/nvflare"
   user_config:
     nvflare: /tmp/startup_kits

   allowed_ports:
    - 8002

   allowed_out_ports:
   - 443
   - 8002
   - 8999

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
   docker_archive: ~/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz

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

