.. _cc_deployment_guide:

################################################
FLARE Confidential Federated AI Deployment Guide
################################################

Overview
========

This guide provides step-by-step instructions for deploying NVIDIA FLARE with Confidential Computing (CC) capabilities on-premises using AMD SEV-SNP CPUs and NVIDIA GPUs.

The deployment involves building a Confidential VM (CVM) image that contains the FLARE application, provisioning the system for participants, and launching the CVMs on each site.

Deployment Environment
======================

This guide covers the following deployment configuration:

- **Platform**: On-Premise AMD CVM with NVIDIA GPU
- **CPU**: AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging)
- **GPU**: NVIDIA GPU with Confidential Computing support (optional)
- **Host OS**: Ubuntu 25.04

Prerequisites
=============

Hardware Requirements
---------------------

**CPU Requirements**

- AMD CPU with SEV-SNP enabled
- AMD firmware supporting SEV-SNP

**GPU Requirements (Optional)**

- NVIDIA GPU with Confidential Computing support (H100, Blackwell)

**Host System**

- Host OS: Ubuntu 25.04

Software Requirements
---------------------

**Required Software**

- Ubuntu 25.04
- QEMU (for virtualization)
- Docker

**NVFlare Components**

1. **NVFlare Source Code**

   Clone from GitHub:

   .. code-block:: bash

      git clone https://github.com/NVIDIA/NVFlare.git

2. **Image Builder**

   - Obtain the image builder code from the NVFlare team
   - Contact: federatedlearning@nvidia.com
   - Install location: ``~/cc/image_builder``

3. **Base Images**

   - Obtain base images from the NVFlare team
   - Copy to: ``~/cc/image_builder/base_images``

4. **KBS Client**

   - Build or obtain KBS client matching your KBS server
   - Recommended commit: ``a2570329cc33daf9ca16370a1948b5379bb17fbe``
   - Copy the kbs-client and credentials to: ``~/cc/image_builder/binaries``

5. **SNPGuest Tool**

   - Build SNPGuest version v0.9.2
   - Copy snpguest and credentials to: ``~/cc/image_builder/binaries``

**AMD Firmware Installation**

To install the AMD SEV-SNP firmware:

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

The firmware will be installed at ``/usr/share/ovmf/OVMF.amdsev.fd``.

**Policy Files Setup**

.. note::

   The current KBS doesn't support updating individual rules. You must update the entire rule file when adding a new CVM.

Create the policy directory and obtain the required files:

.. code-block:: bash

   mkdir -p /shared/policy

Place the following files in ``/shared/policy`` (obtain from the NVFlare team):

- ``policy.rego`` - Master policy file
- ``set-policy.sh`` - Policy update script
- ``private.key`` - Authentication key

Project Admin Requirements
---------------------------

As the project admin, you need to:

1. **Understand Trustee Service**

   - Learn about `Trustee Service <https://www.redhat.com/en/blog/introducing-confidential-containers-trustee-attestation-services-solution-overview-and-use-cases>`_
   - Review the `Trustee documentation <https://github.com/confidential-containers/trustee?tab=readme-ov-file>`_

2. **Deploy Trustee KBS Server**

   Follow the :ref:`hashicorp_vault_trustee_deployment` guide to deploy the Trustee Key Broker Service with HashiCorp Vault.

Deployment Workflow
===================

The deployment consists of four main steps:

1. **Build Docker Image** - Create the application container
2. **Provision** - Generate CVM images and startup kits
3. **Distribute** - Send startup kits to each site
4. **Launch** - Start CVMs at each site

Step 1: Build Docker Image
---------------------------

The CC image builder supports any generic workload. For NVFlare, create a Docker image with the application pre-installed.

**Example Dockerfile:**

.. code-block:: dockerfile

   ARG BASE_IMAGE=python:3.12

   FROM ${BASE_IMAGE}

   ENV PYTHONDONTWRITEBYTECODE=1
   ENV PIP_NO_CACHE_DIR=1

   RUN pip install -U pip && \
       pip install nvflare~=2.7.0rc

   COPY code/ /local/custom
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   ENTRYPOINT ["/user_config/nvflare/startup/sub_start.sh", "--verify"]

.. note::

   For CC jobs, custom code at runtime is not allowed. All application code must be included in the Docker image.

**Build and save the image:**

.. code-block:: bash

   docker build -t nvflare-site:latest .
   docker save nvflare-site:latest | gzip > nvflare-site.tar.gz

Step 2: Provision
-----------------

Navigate to the example directory:

.. code-block:: bash

   cd NVFlare/examples/advanced/cc_provision

**2.1 Configure Project**

Edit ``project.yml`` and update the ``build_image_cmd`` path:

.. code-block:: yaml

   packager:
     path: nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager
     args:
       # Update this path to your image builder location
       build_image_cmd: ~/nvflare-github/nvflare/lighter/cc/image_builder/cvm_build.sh

**2.2 Configure Server**

Edit ``cc_server1.yml`` and set the ``docker_archive`` path:

.. code-block:: yaml

   docker_archive: ~/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz

**2.3 Configure Client**

Edit ``cc_site-1.yml``:

1. Set the ``docker_archive`` path:

   .. code-block:: yaml

      docker_archive: ~/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz

2. If the server name is not a public domain, add host entries:

   .. code-block:: yaml

      host_entries:
        server1: 10.176.4.244

3. If no GPU is available, remove the GPU authorizer and ``cc_gpu_mechanism`` configuration.

**2.4 Run Provision**

.. code-block:: bash

   nvflare provision -p project.yml

.. note::

   Provisioning takes approximately 1000 seconds to build each CVM image.

**2.5 Output**

Startup packages are generated in:

.. code-block:: text

   ./workspace/example_project/prod_00/
      server1/server1.tgz
      site-1/site-1.tgz
      admin@nvidia.com/

Step 3: Distribute
-------------------

Distribute the generated startup kits to each participant:

- Send ``server1.tgz`` to the server site
- Send ``site-1.tgz`` to client site-1
- Admin keeps the admin package locally

CVM Startup Kit Contents
^^^^^^^^^^^^^^^^^^^^^^^^^

Each startup kit (e.g., ``server1.tgz``) contains:

.. code-block:: bash

   $ tar -zxvf server1.tgz
   $ ls server1/cvm_885fe8f608b3/

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``applog.qcow2``
     - Application log storage (unencrypted, can be mounted and inspected)
   * - ``crypt_root.qcow2``
     - Encrypted root filesystem (requires decryption key)
   * - ``initrd.img``
     - Initramfs with InitApp for attestation
   * - ``launch_vm.sh``
     - CVM launch script
   * - ``OVMF.amdsev.fd``
     - AMD SEV-SNP firmware with kernel-hashes support
   * - ``README.txt``
     - Documentation
   * - ``user_config.qcow2``
     - User configuration storage containing NVFlare startup kits
   * - ``user_data.qcow2``
     - User data storage (placeholder, can be extended)
   * - ``vmlinuz``
     - Linux kernel

Step 4: Launch CVMs
--------------------

**4.1 Launch Server**

On the server machine:

.. code-block:: bash

   tar -zxvf server1.tgz
   cd server1/cvm_*
   ./launch_vm.sh

**4.2 Launch Client**

On each client machine:

.. code-block:: bash

   tar -zxvf site-1.tgz
   cd site-1/cvm_*
   ./launch_vm.sh

The server and clients will automatically start the NVFlare system inside their respective CVMs.

**4.3 Start Admin Console**

On the admin machine:

.. code-block:: bash

   cd NVFlare/examples/advanced/cc_provision

   # Copy jobs to admin transfer folder
   cp -r jobs/* ./workspace/example_project/prod_00/admin@nvidia.com/transfer/

.. note::

   If the server name is not a public domain, add an entry in ``/etc/hosts`` on the admin machine.

Start the admin console:

.. code-block:: bash

   ./workspace/example_project/prod_00/admin@nvidia.com/startup/fl_admin.sh

**4.4 Submit Job**

In the admin console:

.. code-block:: bash

   submit_job hello-pt_cifar10_fedavg

Configuration Reference
=======================

CC Configuration Parameters
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Example Value
     - Description
   * - ``compute_env``
     - ``onprem_cvm``
     - Computation environment type
   * - ``cc_cpu_mechanism``
     - ``amd_sev_snp``
     - CPU confidential computing mechanism
   * - ``role``
     - ``server`` / ``client``
     - Role in the NVFlare system
   * - ``root_drive_size``
     - ``30`` (GB)
     - Size of the root filesystem drive
   * - ``applog_drive_size``
     - ``1`` (GB)
     - Size of the application log drive
   * - ``user_config_drive_size``
     - ``1`` (GB)
     - Size of the user configuration drive
   * - ``user_data_drive_size``
     - ``1`` (GB)
     - Size of the user data drive
   * - ``docker_archive``
     - ``~/path/to/app.tar.gz``
     - Path to Docker image archive (created with ``docker save``)
   * - ``user_config``
     - Key-value pairs
     - Paths mounted in container at ``/user_config/[key]``
   * - ``allowed_ports``
     - List of ports
     - Inbound ports to whitelist
   * - ``allowed_out_ports``
     - List of ports
     - Outbound ports to whitelist
   * - ``cc_issuers``
     - List of authorizers
     - CC attestation token issuers
   * - ``token_expiration``
     - ``100`` (seconds)
     - Token validity duration (must be < ``check_frequency``)
   * - ``check_frequency``
     - ``120`` (seconds)
     - Attestation check interval
   * - ``failure_action``
     - ``stop_job``
     - Action on attestation failure

Complete Configuration Examples
--------------------------------

**Project Configuration (project.yml)**

.. code-block:: yaml

   api_version: 3
   name: example_project
   description: NVIDIA FLARE sample project yaml file

   participants:
     - name: server1
       type: server
       org: nvidia
       fed_learn_port: 8002
       cc_config: cc_server1.yml
     - name: site-1
       type: client
       org: nvidia
       cc_config: cc_site-1.yml
     - name: admin@nvidia.com
       type: admin
       org: nvidia
       role: project_admin

   builders:
     - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
     - path: nvflare.lighter.impl.static_file.StaticFileBuilder
       args:
         config_folder: config
     - path: nvflare.lighter.impl.cert.CertBuilder
     - path: nvflare.lighter.impl.signature.SignatureBuilder
     - path: nvflare.lighter.cc_provision.impl.cc.CCBuilder

   packager:
     path: nvflare.lighter.cc_provision.impl.onprem_packager.OnPremPackager
     args:
       build_image_cmd: ~/nvflare-github/nvflare/lighter/cc/image_builder/cvm_build.sh

**Server Configuration (cc_server1.yml)**

.. code-block:: yaml

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

   # Will be mounted inside docker at "/user_config/nvflare"
   user_config:
     nvflare: /tmp/startup_kits

   # Inbound ports whitelist
   allowed_ports:
     - 8002

   # Outbound ports whitelist
   allowed_out_ports:
     - 443    # HTTPS
     - 8002   # NVFlare
     - 8999   # Trustee KBS

   cc_issuers:
     - id: snp_authorizer
       path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
       token_expiration: 100  # seconds, must be < check_frequency

   cc_attestation:
     check_frequency: 120  # seconds
     failure_action: stop_job

**Client Configuration (cc_site-1.yml)**

.. code-block:: yaml

   compute_env: onprem_cvm
   cc_cpu_mechanism: amd_sev_snp
   role: client

   # All drive sizes are in GB
   root_drive_size: 30
   applog_drive_size: 1
   user_config_drive_size: 1
   user_data_drive_size: 1

   # Docker image archive
   docker_archive: ~/NVFlare/examples/advanced/cc_provision/docker/nvflare-site.tar.gz

   # For non-public domain server names
   hosts_entries:
     server1: 10.176.200.152

   # Will be mounted inside docker at "/user_config/nvflare"
   user_config:
     nvflare: /tmp/startup_kits

   cc_issuers:
     - id: snp_authorizer
       path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
       token_expiration: 100  # seconds, must be < check_frequency

   cc_attestation:
     check_frequency: 120  # seconds
     failure_action: stop_job

Troubleshooting
===============

Inspecting QCOW2 Disk Images
-----------------------------

To inspect the contents of a QCOW2 disk image (e.g., ``user_config.qcow2``):

**1. Load the NBD kernel module:**

.. code-block:: bash

   sudo modprobe nbd max_part=8

**2. Connect the QCOW2 image:**

.. code-block:: bash

   sudo qemu-nbd --connect=/dev/nbd0 user_config.qcow2

**3. Mount the image:**

.. code-block:: bash

   sudo mount /dev/nbd0 /mnt/user_config

**4. Inspect the contents:**

.. code-block:: bash

   ls /mnt/user_config

For NVFlare startup kits:

.. code-block:: bash

   ls /mnt/user_config/nvflare/

**5. Unmount:**

.. code-block:: bash

   sudo umount /mnt/user_config

**6. Disconnect:**

.. code-block:: bash

   sudo qemu-nbd --disconnect /dev/nbd0

Common Issues
-------------

**Issue: CVM fails to boot**

- Check the ``applog.qcow2`` for boot logs
- Verify firmware is correctly installed
- Ensure kernel-hashes is enabled in the firmware

**Issue: Attestation failure**

- Verify Trustee KBS server is accessible
- Check network connectivity to attestation service
- Ensure correct ports are whitelisted in ``allowed_out_ports``

**Issue: Server/Client connection fails**

- Verify ``/etc/hosts`` entries if not using public domain
- Check firewall rules
- Ensure correct ports are configured in both server and client

Next Steps
==========

After successfully deploying the system:

- Review the :ref:`NVFlare CC Architecture <cc_architecture>` for understanding the security model
- Consult :ref:`confidential_computing_attestation` for attestation details
- Explore advanced configuration options for your specific use case
