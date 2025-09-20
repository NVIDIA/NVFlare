.. _cc_on_prem_cvm_architecture:

##############################################################
FLARE Confidential Computing Based IP Protection Architecture
##############################################################

.. contents::
   :local:
   :depth: 2

Introduction
============

In an era where artificial intelligence drives critical decisions across industries, protecting the intellectual property (IP) of machine learning models has become paramount—especially during inference and federated learning. These models often represent years of research, proprietary algorithms, and strategic data investments, making them highly valuable assets. Inference, typically conducted on edge or client devices, and federated learning, which distributes model training across decentralized nodes, both expose models to untrusted environments where IP theft or reverse engineering is a significant risk. Without robust IP protection, organizations face not only financial losses but also threats to competitive advantage and compliance. Ensuring model confidentiality during both training and inference is therefore essential for secure deployment, responsible innovation, and sustained trust in AI systems.

The risks to model IP stem from multiple critical phases in the deployment time and runtime lifecycle.

Deployment-Time Risks
---------------------

At deployment time, the model IP is particularly vulnerable if introduced into an untrusted or unverified environment. An untrusted host or malicious host owner can intercept the model by modifying the application code, tampering with the execution environment, or delaying the activation of security mechanisms such as attestation and encryption. Without strict controls over when and how the model is decrypted or loaded, attackers can gain early access before protections are in place. This makes the deployment phase a critical point of exposure, especially in environments where hosts are not fully controlled or are operated by third parties.

Runtime Risks
-------------

Even after deployment, model IP remains exposed to runtime threats. A host system—whether trusted or compromised—can still leak the model if sufficient safeguards are not maintained. Attackers may exploit vulnerabilities to gain remote access, copy the model from memory, intercept it over the network, or extract it from disk-based checkpoints. Insider threats or physical access to a machine can also lead to data exfiltration. While VM-based Trusted Execution Environments (TEEs) provided by Confidential Computing offer strong isolation guarantees, these mechanisms are not infallible. If the attacker can directly access the CVM TEE or modify the application inside the TEE, then the TEE protection doesn’t help the IP protection: here are a few possible ways that model IP can be leaked out at runtime:
- Compromised participant machines
- Unauthorized access to the remote training machine (via direct access or network access)
- Remote access or a leak from the network
- Leak from storage (such as a model checkpoint)

Design Proposal: Securing AI Workloads with Confidential Computing
==================================================================

Challenge
---------

Simply deploying applications in a Confidential VM (CVM) is insufficient to protect model IP. A comprehensive security architecture is required.

Proposed Solution
-----------------

A secure deployment architecture combining:

- Specialized CVM Image
- Hardware-backed chain of trust from hardware to application
- Enhanced security controls for network, storage, and access
- Measured boot and runtime attestation
- Pre-packaged Workload Container
- FLARE training applications or inference services
- Model weights and proprietary code

Security Guarantee
------------------

Our Minimum Viable Product (MVP) product design ensures model IP remains protected throughout the entire lifecycle, from deployment through execution, even in potentially compromised environments.

IP Protection Security Architecture
===================================

The Approach
------------

The high-level approach for generating a Confidential VM (CVM) image involves embedding the application workload within a secure virtual machine that leverages VM-based Trusted Execution Environment (TEE) architecture. To ensure strong security guarantees, the CVM is fully locked down—no shell access, no open ports except for explicitly whitelisted ones, and all data access restricted to encrypted disk partitions.

To protect against tampering during deployment, the boot process is anchored in Confidential Computing’s chain of trust, extending from hardware up to the application layer. Critical disk partitions are encrypted, and decryption keys are withheld until remote attestations are successfully completed. This attestation verifies both the base system and the application against expected measurements at a remote trustee service. Only after passing this check does the trustee’s key broker service release the decryption key, allowing the CVM to proceed securely.

The attestations will be completed with two-stages, once the kernel is booted normally, the attestation service will perform 2nd stage attestation (both CPU + GPU attestation), if the attestation is verified then the normal workload will be started.

Assumptions
-----------

- We fully trust the individual who builds the CVM image, as well as the host machine used during the image creation process. This ensures that the CVM is constructed in a secure and controlled environment.
- We trust the remote trustee service, including its integrated key broker service, to be secure and reliable. The internal protection mechanisms of the trustee service are considered out of scope for this design.
- To verify the integrity and confidentiality of the CVM application's boot process, we assume that CPU-based attestation at boot time is sufficient. Specifically, we rely on a one-time, hardware-backed attestation during CVM startup to establish trust, without requiring ongoing or continuous runtime verification.
- On-going continuous attestation will be handled at Application level (with both GPU and CPU attestation, such as NVFLARE).

Security Architecture
=====================

Key Challenges in Securing Application-Level Integrity
------------------------------------------------------

By default, Chain of Trust Stops at the Kernel:
Confidential Computing's hardware-backed chain of trust typically ends at the kernel. User-level application code is not included in the default measurement and attestation process.

Application Integrity Risk:
Without extending the chain of trust to cover the application, malicious modifications can occur at boot time. This risks compromising both the application’s integrity and the overall confidentiality of the system, even if kernel-level attestation is successful.

Necessity of Application Measurement:
To ensure end-to-end trust, application-level measurements must be automatically calculated by the kernel and cryptographically signed by CC-enabled hardware. Relying on external or manual hash values creates potential attack vectors.

Use Case Consideration – Disk Content Not Measured:
Confidential Computing attestation is designed to measure memory-loaded components during boot. Application binaries and data stored on disk are not covered. This is not a flaw in the architecture but a challenge that must be addressed for use cases requiring full application trust.

Security Implication for Application Deployment:
If the application and its associated data are not part of the attested set, the CVM cannot ensure their integrity or confidentiality—posing a significant risk for secure deployment in sensitive scenarios.

Design Approach
---------------

This design addresses the above challenges with the following approaches:

- **Encrypted Storage**: The CVM encrypts critical storage partitions to protect sensitive code and data from unauthorized access.

- **Customer-Specific Key**: A unique decryption key is associated with each customer and stored securely in the remote key broker service, along with the expected attestation reference values.

- **Attestation-Bound Key Release**: The decryption key is released only upon successful CPU-based attestation, ensuring it is provided exclusively to trusted environments that match both CVM and application measurements and possess valid cryptographic signatures.

- **Two-stage attestation & two-stage-key release**:
  - CPU verification ⇒ GPU verification (extending the chain of the trust from CPU to GPU)
  - Two-stage key releases with partition dm-verity.

Additional Security Hardenings
------------------------------

- **Disk Security**: Leverage both dm-crypt for encryption and dm-verity for integrity verification of disk partitions. Disable auto-mount.
- **Access Control**: Disable login mechanisms, including SSH and console access, to prevent unauthorized entry into the CVM.
- **Network Hardening**: Configure strict firewall rules and disable all unnecessary services and ports, allowing only explicitly whitelisted network access.

Reference Value and Key Storage
===============================

There are different approaches to store the reference values, leveraging:

- Trustee service with remote key broker services
- Trusted Platform Module (TPM)
- Virtual TPM (vTPM)

For our most common deployment scenarios, we will build a CVM image in one trusted host (Host A), then distribute and deploy it to another untrusted host (Host B). In this design, we choose to use the remote trustee service.

Design of the CVM Boot Up Process
=================================

The sequence diagram of the boot up process:

Here, we are leveraging the Initapp in a TEE context to enable application-level attestation, using the kernel as an indirect attesting environment.

Kernel as an Attesting Environment – via InitApp in TEE
-------------------------------------------------------

Concept Overview
----------------

In a Confidential Computing environment (e.g., AMD SEV-SNP, Intel TDX), the kernel is already measured at boot time by the hardware-backed chain of trust. Rather than modifying the kernel or injecting measurement logic earlier in the boot flow, we delegate application-level attestation to a lightweight agent called InitApp, which runs in early user space—right after the kernel, but before any application workload or sensitive data is accessed.

Key Design Principles
---------------------

- **Trusted Kernel Base**: The kernel serves as the base of trust. It is measured by the TEE platform during boot, forming part of the trusted launch.
- **InitApp as Attesting Agent**: InitApp is responsible for:
  - Performing application-level attestation.
  - Interacting with the trustee service and key broker.

Placement and Measurement
-------------------------

- The measurement must include initramfs, kernel, and kernel arguments (command line). With AMD, this is achieved by kernel-hashes=on flag.
- InitApp must be included in the initramfs, ensuring it is loaded into kernel memory and automatically measured as part of the attested launch context.
- Avoid placing InitApp outside initramfs (e.g., in /oem/initapp), as this bypasses automatic measurement and increases attack surface via replay attacks.

Measurement Integrity
---------------------

Embedding InitApp within initramfs ensures:
- It is measured with initramfs via attestation SDK.
- Replay or tampering is prevented.
- No need for custom measurement mechanisms.

Bootup Sequence Overview
-------------------------

- BIOS/UEFI
  ↓
- Bootloader (GRUB) loads:
  - vmlinuz
  - initramfs (includes InitApp and minimal network tools)
  ↓
- initramfs executes /bin/init-app
  ↓
- InitApp:
  - Brings up network interface (e.g., eth0)
  - Performs attestation using CPU TEE
  - Contacts trustee and key broker service
  - Decrypts and mounts secure root filesystem
  ↓
- InitApp executes: switch_root /new_root /sbin/init

Secure Filesystem Layout for initramfs
--------------------------------------

.. code-block:: text

   initramfs/
   ├── bin/
   │   └── init-app         # Attesting agent
   ├── init                # Stub to call /bin/init-app
   ├── dev/
   ├── etc/
   ├── lib64/
   ├── mnt/
   ├── proc/
   ├── sys/
   ├── tmp/

QEMU Launch Example
-------------------

.. code-block:: bash

   qemu-system-x86_64 \
    -kernel vmlinuz \
    -initrd initramfs.img

In this setup, initramfs.img is loaded into kernel memory and included in the TEE measurement, securing both InitApp and its logic. Placing InitApp elsewhere (e.g., mounted later from disk) breaks the measurement chain and introduces the risk of replay or tampering.

What Needs to Be Measured and Signed?
=====================================

When preparing a Confidential VM (CVM) image, it's crucial to ensure that key components are measured and cryptographically verified to maintain a trusted boot process.

With TEE platforms like AMD SEV-SNP or Intel TDX, the firmware measures and includes the hashes of the following in the attestation report:
- Kernel binary
- Initramfs (which includes InitApp)
- Kernel command-line parameters
- Firmware (UEFI/BIOS)
- EFI boot configuration (depending on platform and setup)

These measurements are rooted in hardware and cannot be forged by the host. Any tampering with measured components—such as modifying InitApp—will result in a different TEE measurement hash. Consequently, the Trustee will detect the mismatch and deny key release, preventing decryption of sensitive data.

.. note::

   You do not need to sign or measure the entire CVM disk image. Focusing on these critical boot-time components is sufficient to establish a robust and verifiable chain of trust.

Process of Build and Boot up CVM Image
======================================

The sequence diagram:

Build Process: Creating a Confidential VM (CVM) Image
-----------------------------------------------------

Goal: Produce a secure CVM image with all trusted measurements registered in the Trustee service in a trusted host A.

1. **Build Base CVM Image**
   - Follow your standard CVM creation guide or automation pipeline.
   - Choose a supported OS (e.g., Ubuntu 22.04 LTS).

2. **System Requirements**
   - Install guest OS patches for AMD SEV-SNP or Intel TDX.
   - Install Confidential Computing drivers:
     - AMD: kvm_amd, sev, snp kernel modules
     - TDX: TDX guest drivers (tdx_guest)
   - (Optional) Install GPU drivers (e.g., NVIDIA vGPU with CC support).

3. **Install Required Packages**
   - Install the attestation SDK CLI tools or libraries.
   - Install tooling to generate initramfs.

4. **Prepare InitApp + Initramfs**
   - Build the InitApp binary (early boot attestation code).
   - Generate initramfs:
     - Include InitApp, attestation tools, and measurement logic.
     - Call InitApp in the default init (via /bin/init).
   - Generate a unique CVM_ID for this VM.
   - Add to kernel boot arguments:
     - initrd=/boot/initramfs.img
     - append(vm_id=”$CVM_ID")

5. **Partition Disks & Apply Security Hardening**
   - Partition the disk and prepare encrypted volumes.
   - See the disk partitioning section for more details.

6. **Install the Workload**
   - Deploy the pre-approved workload (e.g., a Docker image).
   - Install the workload on the CVM’s encrypted disk.

7. **Apply Additional Security Enhancements**
   - Harden access:
     - Disable password logins
     - Restrict or disable SSH and console access
     - Configure firewall rules and disable unneeded services and ports.

8. **Finalize & Encrypt**
   - Power off the CVM.
   - Generate an encryption key.
   - Encrypt the root FS using LUKS.

9. **Get the CVM measurement**
   - Boot up the CVM, the CVM kernel panics because it can't retrieve the key due to the measurement not registered yet. The InitApp prints measurements in the log.
   - Todo: It's much faster to calculate the measurement with this tool but it generates invalid result: https://github.com/virtee/sev-snp-measure

10. **Update the resource policy or reference values to the Trustee**
    - Update the policy in Trustee with measurement.
    - Store the encryption key with Trustee with the namespace /keys/root/$CVM_ID.

11. **Package the CVM Bundle**
    - All the files generated by CVM builder is packaged a gzipped tar.

Runtime Boot-Up Process (CVM)
=============================

Boot up Sequence in host B
--------------------------

- Boot up CVM image on host.
- Launch CVM instance.
- UEFI loads kernel and initramfs (via initrd=/boot/initramfs.img).
- Initramfs starts network.
- Initramfs starts initApp.
- initApp requests CPU attestation report.
- initApp send key requests to trustee with its attestation report.
- Receives encryption key if system is not tampered with.
- Decrypt encrypted filesystem using received key.
- Pivot root to the decrypted rootfs mapper (switch_root).
- systemd takes over and continues normal runtime.
- Attestation agent service perform 2nd stage attestation for CPU and GPU report.
- Workload is started.
- Monitor /bootlog to verify CVM boot health.
- Monitor /applog logs for application issue.
- User may optionally mount external NFS volume data (such as training data in FL case).

Design Secure CVM Image
=======================

Image Size and Partitions
-------------------------

CVM Image storage size estimation:

Minimum Disk Size with CUDA & GPU Confidential Computing Drivers
----------------------------------------------------------------

Since you need CUDA and GPU Confidential Computing drivers (e.g., AMD SEV-SNP or Intel TDX with GPU passthrough), the disk size requirements increase.

.. list-table::
   :header-rows: 1

   * - Component
     - Approximate Size
   * - Ubuntu Minimal (CLI-only)
     - ~2GB
   * - CUDA Toolkit & Drivers
     - ~5GB–10GB
   * - NVIDIA cuDNN & Other Libraries
     - ~2GB
   * - Confidential GPU Driver (e.g., NVIDIA Confidential Compute)
     - ~1GB
   * - Confidential Computing Stack (SEV-SNP, TDX, etc.)
     - ~500MB–1GB

Updated Recommended Minimum Disk Sizes
--------------------------------------

.. list-table::
   :header-rows: 1

   * - Use Case
     - Recommended Disk Size
   * - Minimal GPU Setup (No PyTorch, No Large Apps)
     - 16GB
   * - With CUDA & Confidential GPU Drivers
     - 32GB

Expanding Disk Size Later
-------------------------

If needed, we can expand the disk dynamically using:
- Virtual Disk (QCOW2, RAW, VHDX) – Use qemu-img resize + growpart + resize2fs.

Disk Security
=============

Disk layout and Partitions
--------------------------

.. list-table::
   :header-rows: 1

   * - Partition
     - Mount Point
     - Contents
     - Encryption
     - Notes
   * - Protective MBR / GPT Header
     - n/a
     - GPT structures
     - ❌
     - Standard GPT disk format. Tampered partition will cause boot failure.
   * - EFI System Partition (ESP)
     - /boot/efi
     - Bootloader binaries (GRUB, systemd-boot, etc.)
     - ❌
     - Required by UEFI. This is protected by the normal secure boot procedures.
   * - Kernel + Initramfs
     - /boot
     - Kernel image, basic initramfs
     - ❌
     - This is not protected. The tampered image will cause measurement change so encryption key can't be retrieved.
   * - Boot Logging Partition
     - /bootlog
     - Early logs from initramfs and InitApp
     - (Write-once, then RO)
     - Write-once early, mount read-only after transition. Visible to host.
   * - Root Filesystem
     - /
     - Full Ubuntu/OS install
     - dm-crypt
     - root OS.
   * - Encrypted vault
     - /vault
     - Logs, scratch
     - ✅ (LUKS or dm-crypt)
     - Writable at runtime.
   * - Encrypted workspace
     - /vault/workspace
     - workloads,
     - LUKS
     - Writable
   * - App Logging Partition
     - /applog
     - Record app logs, especially client server communication failure etc.
     - 
     - We may not need to expose the training log. Writable at runtime. App Log is designed as a separate image, so that when CVM shutdown the log can be still read.
   * - (Optional) User Data Volume
     - /data
     - User-mounted data
     - Optional
     - Separate image attached. Can be S3/NFS/external encrypted volume.
   * - Temporary Filesystem
     - /tmp,
     - Runtime files (RAM only)
     - ❌ (RAM only)
     - This is RAM disk and protected by TEE.

Disk Encryption and Integrity Protection
----------------------------------------

Encryption is performed during the image build stage. The decryption key is securely stored in a remote key broker service. The disk image includes multiple partitions with encryption and integrity protection:
- Root partition (/): Encrypted using dm-crypt
- /boot partition: Protected with root FS.
- /workspace partition: A writable partition encrypted with dm-crypt, providing both confidentiality and integrity. The NVFLARE workspace is stored here.
- /tmp as tmpfs: This maps to RAM. In a Confidential VM, the TEE ensures this memory is encrypted.
- Swap is disabled to prevent the operating system from unintentionally writing sensitive data to disk.

Mount Security
--------------

Auto-mounting is disabled to prevent unauthorized or accidental mounting of external devices.

Access Security
===============

Admin Access
------------

The system is configured to be admin-less by removing all users from the sudoers file.

OS Login
--------

OS-level login is disabled entirely.

Remote Access
-------------

SSH (sshd) is disabled. The serial console is disabled (see Appendix D for details).

Network Security
================

All network connections are authenticated and encrypted. We use TLS for secure communication and to authenticate attestation services.

Firewall Configuration
----------------------

All ports are blocked by default using ipTables, except for explicitly whitelisted ports.

Whitelisted ports include:
- Application communication ports
- Attestation service ports
- Experiment tracking ports (e.g., MLflow for FL training)

Other Disk Partitions
=====================

Logging
-------

/bootlog:
This log records the boot process and is essential during setup and debugging, especially when diagnosing boot failures. Initially, the boot log is writable. After the system completes its transition, it becomes read-only to preserve integrity.

/applog:
This log captures application-level output (e.g., FLARE logs). It is writable to aid debugging—for instance, when investigating connectivity issues between clients and servers. The log is visible to the host and implemented as a separate image file. This allows log analysis to continue even after the CVM is shut down.

Input Data
==========

The user provides a predefined command to mount input data (e.g., training data), which is mounted to the data partition on the FL client. No dynamically attached disks are allowed.

For this design:
- Input data is assumed to be unencrypted.
- Only NFS-based mounts are supported.

Input Data Volume NFS Mount Design
==================================

Requirements:
This block device must have one partition and its label must be DATAPART. This partition must be formatted as ext4 file system. Its group and user id must be set to 1000 (the default user).

The CVM instance will automatically mount this DATAPART to /data folder.

Automatic NFS mount with NFS client
-----------------------------------

Currently, only NFS mount is supported if CVM instances need to access files outside the DATAPART (i.e., the additional .qcow2 file).

CVM instance will locate ext_mount.conf file in the /data directory. If found, it will run nfs client to mount the exported nfs server directory to /data/mnt folder.

The format of ext_mount.conf for NFS mount is:

.. code-block:: text

   NFS_EXPORT=$NFS_SERVER_NAME_or_IP:$EXPORT_DIR

example:

.. code-block:: text

   NFS_EXPORT=172.31.53.113:/var/tmp/nfs_export

The CVM instance will run:

.. code-block:: bash

   sudo mount -t nfs $NFS_EXPORT /data/mnt

We must create the /data/mnt folder for mount point before we run the command.

CVM Image Components
====================

Based on the current design, the special CVM Image design will essentially consist of the following:

- Base Confidential VM image with hardened security measures and CC drivers (GPU, CPU)
- Initramfs.img which contains initApp
- initApp contains:
  - trustee client, attestation agent
  - AMD SEV-SNP or Intel TDX attestation SDKs
  - Workload docker
  - Application code and dependencies
  - Attestation service agent
  - No need for FLARE
  - Needed for all other non-CC aware applications.
  - Systemd service that will start the workload docker

CVM Image Services and Workload Interface
=========================================

In this section, we will discuss the contract and interaction between services (systemd) and workload, two types of service we have in mind:

- Attestation service agent performs initial self-attestation and periodically self-attestation.
  - If succeeded, trigger workload start.
  - docker run
  - If failed,
    - Initial self-attestation failed, won’t call start.
    - Periodical check failed ⇒
    - Kill docker
  - How to deal with the network interruptions, do we tolerate occasional attestation failure due to network interruptions?

CVM Image Build Configuration
=============================

For each CVM (NVFLARE will build many CVM images, one for each client/site), we will have configuration:

- cc_params.yml
- cc_build.yml

Here is an example of cc_params.yml:

.. code-block:: yaml

   computer_env: onprem_cvm
   cc_cpu_mechanism: amd_sev_snp
   cc_gpu_mechanism: nvidia_cc
   role: client
   root_drive_size: 256
   secure_drive_size: 128
   nvflare_version: 2.6.0
   data_source: /tmp/data
   nfs_mount: nfs-server.local:/data
   custom_code: /tmp/custom

   site_required_python_packages:
    - "numpy==1.21.5"
    - "pandas==1.3.5"
    - "pyarrow==11.0.0"
    - "pydantic==2.3.1"
    - "pyyaml==6.0.1"
    - "requests==2.26.0"

cc_build.yml
------------

.. code-block:: yaml

   vault_file: ~/vault.img
   nvflare_folder: /vault/nvflare
   workspace_folder: /vault/workspace
   venv_folder: /vault/venv
   service_folder: /vault/service
   logging_folder: /applogs

   required_system_packages:
      - "cryptsetup:2.2"
      - "lvm2:2.03"
      - "parted:3.3"
      - "iptables:1.8"
      - "systemd:245"
      - "dmsetup:1.02"
      # Additional security packages
      - "apparmor:3.0"
      - "selinux-utils:3.1"
      - "auditd:3.0"
      - "fail2ban:0.11"
      - "rkhunter:1.4"
      # Monitoring packages
      - "sysstat:12.0"
      - "prometheus-node-exporter:1.0"
      # Backup tools
      - "rsync:3.1"
      - "duplicity:0.8"

   required_python_packages:
      - "ansible"
      - "libvirt-python==11.3.0"
      - "pyyaml"

Trustee Service and Attestations
================================

To protect the model IP, confidential computing hardware alone is not sufficient. Additional infrastructure and services are required—most critically, the Trustee Service, which includes the following components:
- Attestation Service
- Key Broker Service

The Trustee Service must support CPU-level attestation across AMD, Intel, and ARM architectures during the boot process. For this design, we adopt the CNCF Confidential Containers (CoCo) Project Trustee Service and Guest components:
🔗 https://github.com/confidential-containers/trustee
But any other open-source or proprietary trustee service should be able to do the job. This infrastructure is swappable.

Design Rationale
----------------

This design is chosen based on the following key factors:
- Our main focus is on protecting the integrity and confidentiality of initApp during boot up.
- The initApp is a small Rust program that runs independently of the GPU, so GPU attestation is not required at this stage.
- We need an open-source trustee service that has both key broker service and attestation, and basic configuration support. CoCo Trustee Service is the only option we can find at the moment.

CoCo Trustee Architecture
=========================

Components
----------

- **Key Broker Service**: The KBS is a server that facilitates remote attestation and secret delivery. Its role is similar to that of the Relying Party in the RATS model.
- **Attestation Service**: The AS verifies TEE evidence. In the RATS model, this is a Verifier.
- **Reference Value Provider Service**: The RVPS manages reference values used to verify TEE evidence. This is related to the discussion in section 7.5 of the RATS document.
- **KBS Client Tool**: This is a simple tool which can be used to test or configure the KBS and AS.

Note: We are not using the RVPS component. There are no supported APIs to use. We are not using the CDH (Confidential Data Hub) component.

CoCo Trustee Services
---------------------

- Create reference values
- Login credentials
- Role-based Access Control (RBAC) ⇒ missing
- Identity namespace ⇒ use “path” for now ⇒ missing proper identity namespace
- Retrieval reference value
- Identity namespace ⇒ use “path” for now ⇒ missing proper identity namespace
- Access control ⇒ missing
- TLS communication (PR merged) ⇒ fixed

Trustee Policies
================

The "trustee policy" refers to the rules and configurations governing how secrets are released and how the trustworthiness of a confidential workload is verified before granting access to sensitive data. It involves two main types of policies: resource policies and attestation policies.

- **Resource Policies**: These policies determine which secrets are released to a specific workload, typically scoped to the container. They control what secrets are available to the workload, ensuring that only necessary information is provided.
- **Attestation Policies**: These policies define how the claims about the Trusted Computing Base (TCB) are compared to reference values to determine the trustworthiness of the workload. They specify how the attestation process verifies that the workload is running in a trusted environment.

What we do: Currently, we only need to use resource policy, we will use the default attestation policy.

One can set the policy to the needed measurement (hash values) or referring to the reference values. We choose to use the resource policy for now.

Set Policy
----------

Here is a policy example. The resource policy we set to ensure only CVM with the measurement matching the value can get the resource (the key for LUKS).

.. code-block:: text

   package policy
   default allow = false
   allow {
       input["submods"]["cpu0"]["ear.veraison.annotated-evidence"]["snp"]["measurement"] == "Cwa8qBJimP2freTTrrpvAZVbEQEyAhPY4fZGgSn9z4qtt0CAGmcS+Otz96qQZ92k"
   }

And the command to set this policy into the Trustee service.

.. code-block:: bash

   #!/usr/bin/env bash
   TRUSTEE_ADDRESS=<your organization trustee service addresss>
   PORT=8999

   ROOTCA=keys/rootCA.crt

   sudo kbs-client --url https://$TRUSTEE_ADDRESS:$PORT --cert-file $ROOTCA config --auth-private-key private.key  set-resource-policy --policy-file resource_policy.rego



Set & Get Resource
------------------

Here is the command for KBS client to set and get resources:

.. code-block:: bash

   kbs-client --url https://$TRUSTEE_ADDRESS:$PORT --cert-file $ROOTCA config --auth-private-key $PRIVATE_KEY set-resource --resource-file $SECRET_FILE --path $URL_PATH
   kbs-client --url https://$TRUSTEE_ADDRESS:$PORT --cert-file $ROOTCA get-resource --path $URL_PATH

NOTE:
- --path $URL_PATH
  This is used for identity namespace isolation for now.

CVM Image Measurement
=====================

Measurement Tool:
For AMD, here is the tool to perform the measurements, the value (hashes) can be used for resource policy or reference values.

🔗 https://github.com/virtee/sev-snp-measure

What does it measure:

.. list-table::
   :header-rows: 1

   * - Component
     - Measured by Default
     - Measured with kernel-hashes=on
   * - OVMF
     - ✅ Yes
     - ✅ Yes
   * - Kernel (vmlinuz)
     - ❌ No
     - ✅ Yes
   * - initrd/initramfs
     - ❌ No
     - ✅ Yes
   * - Kernel args
     - ❌ No
     - ✅ Yes

The SEV-SNP measurement is a SHA-384 hash of:
- OVMF + firmware state
- Kernel
- Initrd
- Kernel command line
- Platform launch policy
- Guest-supplied report_data
- etc.

As long as:
- Provide the same inputs to both sev-snp-measure and the runtime SEV-SNP launch process (i.e., QEMU/KVM with SEV-SNP enabled),
- Don’t introduce randomness between build and runtime (e.g., dynamic kernel arguments, timestamps, UUIDs),

The measurement will match exactly.

Attestation Stages
==================

1. **Boot-Time Attestation**
   - Scope: CPU only
   - Ensures the integrity of the CVM and the early boot process, including initApp.
   - Performed using the Trustee Service at startup.

2. **Runtime Attestation**
   - Scope: CPU + GPU
   - Required to protect the application workload during runtime execution.
   - Likely involves an application-level attestation agent.
   - FLARE integrates a Confidential Computing (CC) Manager that performs attestation at multiple stages, including runtime, to maintain trust across the system lifecycle.

Application Level Security
==========================

In addition to the basic CVM Security, we also need additional security at application level. This might be different for different type of applications.

General Security Measure
------------------------

For all applications, we need the following additional security measures:
- **Attestation service agent**:
  - Perform the self-attestation using both CPU and GPU attestation service at start.
  - Boot level attestation is only for CPU, we need to attest GPU as well.
  - Perform periodical self-tests to make sure the system is not compromised.
- **Code Level security**:
  - No dynamic code changes.

FLARE: Federated Learning Application & Security
================================================

Federated Learning Provision Process
------------------------------------

Federated learning provision is a process to prepare the software packages (FLARE’s startup kits) for each participating organization. Clients and the server will obtain different startup kits. The package is prepared by the system owned by the project admin and then distributed to each participant. Then, FL Server needs to start first, FL Client site will start the startup kit, connect to FL server.

There are three distinguished phases:

- **Provision processes** – prepare the software artifacts (the startup kits).
- **Distribution process** – software packages are distributed to participants.
- **Run-time processes** – At each participant’s host machine, the participant deploys the package, starts the FL system, and establishes the communication between the FL server and the participant.

Terminology
-----------

To simplify discussions, we define the following roles:

- **Project Admin**: The individual responsible for initiating and managing the overall project. This includes approving participants, provisioning resources, and triggering the Confidential VM (CVM) build process.

- **Model Owner**: The entity (person or organization) that owns both the pre-trained model and the final trained model. They are primarily concerned with protecting the intellectual property of the model.

- **Data Owner**: The entity that owns the private data used in training. Data privacy and security are their primary concerns.

- **Org Admin**: An IT administrator from a participating organization. This person is responsible for setting up the local environment and launching the site-specific Federated Learning (FL) system instance (e.g., the FL client).

The Process
-----------

- **Provision Process**: The generated CVM image will be a lockdown with no access. This is done via additional hardened security measures described above.
- **Distribution process**: For CLI based provision, we will let customers decide the best way to distribute the CVM image file. For FLARE Dashboard, user should be able to download CVM image.
- **Deploy/start**: The participant, deployed the CVM image to a CC-enabled Host, add NFS data volume need for the training, run start scripts to start the system.

FLARE Attestation Verification
------------------------------

FLARE’s CC manager performs three different attestations:

- **Self-attestation**
- **Cross-verification among client and server**
- **Periodical cross-verification**

FLARE Workload Execution and Access Control Policies
----------------------------------------------------

- All training and inference code must be pre-reviewed and approved before inclusion in the workload.
- The application and its dependencies are pre-installed in the workload docker.
- Job execution is triggered by submitting a predefined job configuration—no dynamic or custom or user-supplied code is allowed at runtime.

For IP Protection Use Cases
---------------------------

- Only the Project Admin is authorized to download results, including the global model and logs.
- Download permissions are disabled for all other users and cannot be overridden at the individual site level.

Threat Model and Mitigations
============================

This section describes the threat models that the current design helps to mitigate, and the new risks with this process.

The following attacks are outside of the scope of this document:
- Software supply chain attacks that apply to guest Unified Extensible Firmware Interface (UEFI) firmware, the bootloader and kernel, and third-party dependencies for the workload.
- Attacks on Trustee Service.

Possible Attacks
----------------

The current CVM architecture is designed to defend against the following possible attacks by an untrustworthy host workload operator:
- Modify disk contents, intercept network connections, and attempt to compromise the TEE at runtime.
- Tamper CVM image file at deployment time, before launch in the remote host:
  - Modify boot process in the image to retrieve encryption key.
  - Modify workload code to write checkpoint path, save model to unencrypted disk.
  - Modify network port rules to allow model to send over the network to unauthorized location.
  - Modify access rules to enable access at runtime.

CVM at Runtime
--------------

- Add login console to directly login to CVM.
- SSH to CVM.
- Network attack for the open port.
- Copy the model checkpoint from the disk.

Attack Surfaces
---------------

The following table describes the attack surfaces that are available to attackers.

.. list-table::
   :header-rows: 1

   * - Attacker
     - Target
     - Attack surface
     - Risks
   * - Host owner or workload operator
     - TEE, Workload
     - Disk reads
     - Anything read from the disk is within the attacker's control. Dynamic disk attachments mean that an attacker can modify disk contents dynamically and at will.
   * - Workload Operator
     - TEE, Workload
     - Disk writes
     - Anything written to disk is visible to an attacker.
   * - Host owner or workload operator
     - TEE, Workload
     - Network
     - External network connection to or Attestation can be intercepted. For FLARE FL Server, two ports open for FL Client communication (Inbounds). FLARE FL Server also open ports for outbound communication: Experimental tracking and statsd system monitoring (if allowed).
   * - Host Owner or Workload operator
     - Attestation Service communication
     - Attestation messages
     - Intercept the message to perform man-in-the-middle attack.
   * - Host owner
     - CVM image file
     - initApp
     - Tampered initApp to trick trustee service to release the decryption key.
   * - Input Data
     - TEE, Workload
     - User Input dataset
     - User input dataset could be exposed to possible poison attacks. But this is not scope of protection in this document.
   * - Output Data
     - TEE, Workload
     - Output result
     - User output dataset could be exposed to possible IP theft.

Threats and Mitigations
=======================

Confidential Computing is used to defend against various attack vectors on Confidential Virtual Machines (CVMs), including tampering, disk access, and network intrusion. Below is a breakdown of the threat surfaces and corresponding mitigations.

- **CVM tampering risk**: The confidential computing attestation protocol helps protect the boot sequence CVM boot as well application initApp. The workload will be encrypted to avoid modification at rest. Any tampering attempt will cause attestation failure, which will not be able to decrypt the CVM root-fs.
- **Disk risk**: A CVM Image encrypted disk with integrity protection is designed to mitigate risks from disk attacks. After initApp is read from disk, it's measured and that data is never re-read from disk again. The description is only retrieved after the verification and then the root fs is decrypted.
- **Network attack Risks**: Attacks are mitigated by having authenticated, end-to-end encrypted channels. External network access, such as SSH, Login, adding a serial console are disabled in the image. Strict firewall input/output rules for the CVM, ports are blocked except for whitelisted ports.

The following tables describe the threats and mitigations:

Attacks on the Measured Boot Process
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Threat
     - Mitigation
     - Mitigation Implementation
   * - Attacker disables measured boot using old firmware
     - Trustee-based attestation detects failure
     - Confidential Computing enforces attestation check before trust is granted.
   * - Attacker disables measured boot and tampers InitApp
     - No key released without successful attestation
     - Remote Key Broker only releases keys after valid CC + InitApp attestation.
   * - Attacker tampers InitApp to steal keys after replaying measurements
     - Attestation fails due to changed InitApp & nonce check
     - Measurements include initramfs; nonces ensure freshness; replay attacks are rejected.
   * - Memory corruption in early boot (e.g., BootHole, Sequoia)
     - Early boot components are measured before processing
     - Attestation fails if grub.cfg or file system config is modified; no auto-mounts.
   * - Disk TOCTOU: tamper boot binaries after measurement but before execution
     - Read-measure-execute pattern; encrypted partitions
     - Boot disk is read-once into memory; dm-verity and dm-crypt enforce disk integrity.
   * - Modify device drivers or user services after kernel loaded
     - dm-crypt root
     - Modify of root file system will cause IO error.
   * - GPU with GPU Hypervisor is compromised
     - The attacker attempted to steal the decryption key once released to the TEE memory after the CPU attestation succeeded.
     - Since the bootup InitApp attestation only attests CPU measurement, the compromised GPU hypervisor is within the TEE trust boundary once the GPU driver is loaded.
     - The GPU hypervisor will try a DMA attack on the TEE memory to steal the decryption key.
     - Unless there is joint CPU + GPU attestation, this is an identified theoretical security hole.
     - The final security fix may require a new industrial solution.
     - Currently, with careful design of the CVM and attestation flow, the risk is really small.
     - 1) GPU driver is trustworthy
     - The GPU driver is part of the root-fs system, which is encrypted. If a tampered GPU driver (without encryption key) will cause the GPU failure to load.
     - If the GPU is successfully loaded, it is trustworthy.
     - 2) CPU driver, hypervisor, and kernel are trustworthy
     - Otherwise, we would be able to pass the attestation at bootup time.
     - 3) 2nd phase GPU attestation will be started before any workload starts
     - If GPU attestation fails, the system will shut down.
     - The compromised GPU will need to steal the decryption key only via the bounced buffer (PCI passthrough) (H100 GPU). Since there is no secret placed in the bounced buffer, there is nothing to steal.
     - For TDISP enabled GPUs, the logic still applies.
     - CVM design mostly mitigates the risk.

Attacks on Trustee Attestation
------------------------------

This table describes potential threats and mitigation strategies to Trustee Attestation.

.. list-table::
   :header-rows: 1

   * - Threat
     - Mitigation
     - Mitigation Implementation
   * - An attacker intercepts the network connection between the CVM attestation client and Trustee to steal the secret token.
     - Use of authenticated, encrypted TLS connection prevents passive eavesdropping.
     - Attacker cannot impersonate the service (lacks TLS key).
     - Attacker cannot impersonate the client (identity verified by attestation protocol).

Attacks on Workloads
--------------------

This table describes potential threats and mitigation strategies related to workloads.

.. list-table::
   :header-rows: 1

   * - Threat
     - Mitigation
     - Mitigation Implementation
     - Location
   * - An attacker tries to SSH or log in and connect to the running instance.
     - SSH is disabled, and the login password is randomized.
     - No SSHD running; randomized login password ensures no external access.
     - Confidential VM image
   * - An Attacker tries to copy the model check-point from the disk accessible from Host where CVM is running
     - The disk partition where model is saved is encrypted
   * - An attacker downloads the final training model from the admin console or API.
     - FLARE permissions restrict access.
     - Fine-grained permissions enforced within FLARE prevent unauthorized model access.
     - Workload application
   * - An attacker steals the model from a host with a GPU that does not support Confidential Computing (CC) or where CC is disabled.
     - Runtime attestation verifies both CPU and GPU at multiple stages.
     - InitApp attests CPU integrity only during boot.
     - Application attestation service performs:
     - Start stage: self-verification for CPU & GPU.
     - Periodic cross-verification.
     - Workload attestation
   * - An attacker passes a malformed and encrypted dataset to the workload.
     - Out of scope in current design
     - Defensive parsing code in the workload.
     - Input data is strictly validated and parsed securely.
     - Workload
   * - An attacker passes a skewed or poisoned dataset to the workload to learn from others’ data.
     - Out of scope in current design; differential privacy can mitigate.
     - Google Confidential Space mentions using differential privacy for this threat.
     - Workload

References
==========

- RATS architecture: https://www.rfc-editor.org/rfc/rfc9334.html
- Google Confidential Space Security Overview: https://cloud.google.com/docs/security/confidential-space
- Confidential containers trustee attestation service solution overview and use cases https://www.redhat.com/en/blog/introducing-confidential-containers-trustee-attestation-services-solution-overview-and-use-cases
- Confidential Container Trustee: https://github.com/confidential-containers/trustee
- Azure confidential computing: harden the linux image to remove sudo users: https://learn.microsoft.com/en-us/azure/confidential-computing/harden-the-linux-image-to-remove-sudo-users
- Microsoft Secure the Windows boot process. https://learn.microsoft.com/en-us/windows/security/operating-system-security/system-security/secure-the-windows-10-boot-process
- Microsoft Secure Boot. Note these links to the above article.
  - https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/oem-secure-boot
- Real-world Linux boot process (https://0pointer.net/blog/brave-new-trusted-boot-world.html)
- Authenticating each boot stages (https://0pointer.net/blog/authenticated-boot-and-disk-encryption-on-linux.html)
- https://github.com/virtee/sev-snp-measure




