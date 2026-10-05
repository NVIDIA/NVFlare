.. _cc_deployment:

Unified Confidential Computing Deployment
==========================================

NVIDIA FLARE provisions all confidential participants through one interface.
Each protected server or client names a ``cc_config`` file, and the project
names one ``cc_project_config`` file. The required ``cc_deployment_mode`` is
one of ``bare_metal_cvm``, ``coco``, or ``azure_cc``.

The interface standardizes configuration, signing, peer verification, private
staging, and result manifests. The deliverables remain different: a complete
CVM OCI artifact, a CoCo Pod plus encrypted image identity, or a signed Azure
startup kit.

Common project configuration
----------------------------

Add the common builder and packager to ``project.yml``. A participant's role
comes from its ``type`` and is never repeated in its CC file.

.. code-block:: yaml

   api_version: 3
   name: cc_project
   cc_project_config: cc_project.yml

   participants:
     - name: server.example.com
       type: server
       org: example
       fed_learn_port: 8002
       admin_port: 8003
       cc_config: cc_server.yml
     - name: site-1
       type: client
       org: example
       cc_config: cc_site-1.yml
     - name: admin@example.com
       type: admin
       org: example
       role: project_admin

   builders:
     - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
     - path: nvflare.lighter.impl.static_file.StaticFileBuilder
     - path: nvflare.lighter.impl.cert.CertBuilder
     - path: nvflare.lighter.cc_provision.impl.cc.CCBuilder
     - path: nvflare.lighter.impl.signature.SignatureBuilder

   packager:
     path: nvflare.lighter.cc_provision.impl.cc_packager.CCPackager

``cc_project.yml`` owns services, credentials, approval keys, registries, and
trusted build commands. Paths in this file resolve relative to this file.

.. code-block:: yaml

   schema_version: 1
   attestation_services:
     trustee:
       type: trustee
       kbs_endpoint: https://trustee.example.org:8443
       ca_cert_file: ./credentials/kbs-ca.pem
       admin_token_file: ./credentials/kbs-resource-token.jwt
       attestation_token_endpoint: http://127.0.0.1:8006/aa/token
       attestation_signing_public_key_file: ./credentials/trustee-as-public.pem
       token_expiration_seconds: 300
       check_frequency_seconds: 120
       registration_token_timeout_seconds: 300
       refresh_token_timeout_seconds: 30
       get_token_request_timeout_seconds: 45
       # Optional verifier policy shared by bare-metal CVM and CoCo.
       proof_iat_leeway_seconds: 180
       workload_constraints:
         server:
           cpu_tee: tdx
           tdx_mr_td: <approved-96-character-lowercase-hex-value>
         site-1:
           cpu_tee: snp
           init_data: <approved-64-character-lowercase-hex-value>
       retry:
         max_attempts: 10
         initial_delay_seconds: 1.0
         max_delay_seconds: 15.0
         backoff_multiplier: 2.0
         jitter_ratio: 0.5
     azure_maa:
       type: azure_maa
       endpoint: https://sharedeus2.eus2.attest.azure.net
       token_expiration_seconds: 100
       check_frequency_seconds: 60

   container_registries:
     coco_workloads:
       endpoint: registry.example.org:5000
       ca_cert_file: ./credentials/registry-ca.pem
       publisher_username_file: ./credentials/registry-username
       publisher_password_file: ./credentials/registry-password

   approval:
     public_key_files:
       - ./credentials/acceptance-signing.pub

   build_tools:
     bare_metal_cvm:
       cvm_builder_dir: ../../../nvflare/lighter/cc/image_builder
       # output_root is optional.
     coco:
       build_command: ./admin/build_coco_image.sh
       build_timeout_seconds: 3600

The loopback attestation endpoint is inside a CoCo confidential guest. It is
not the provisioning host or remote Trustee address. Bare-metal CVM and CoCo
both select ``attestation.service: trustee`` and therefore consume this same
object. Trustee settings must not be repeated in participant files.
If ``workload_constraints`` is present, it must contain exactly every protected
participant that selects this service. Use logical name ``server`` for the root
server and participant names for clients. ``proof_iat_leeway_seconds`` accepts
0 through 180 and defaults to 180. Constraint values pin signed Trustee EAR
claims; obtain them from authenticated reference measurements. Do not set
``gpu_required`` in this mapping. Provisioning derives that internal signed-claim
constraint from each participant's ``gpu_tee``: ``nvidia_cc`` requires CPU plus
GPU evidence and ``none`` requires CPU-only evidence.

Common participant fields
-------------------------

Every ``cc_config`` starts with these fields:

.. code-block:: yaml

   schema_version: 1
   cc_deployment_mode: coco
   cpu_tee: amd_sev_snp
   gpu_tee: nvidia_cc
   attestation:
     service: trustee
   class_allow_list: []

``cpu_tee`` is ``intel_tdx`` or ``amd_sev_snp``. ``gpu_tee`` is ``none`` or
``nvidia_cc`` and also controls the peer attestation GPU requirement. An empty
``class_allow_list`` adds no custom classes; it never
means unrestricted. Entries must be complete reviewed Python class paths.

Workload sources form a closed enum:

* ``docker_archive`` supplies ``path`` and is valid only for
  ``bare_metal_cvm``.
* ``docker_build`` supplies ``context`` and a context-relative ``dockerfile``
  and is valid only for ``coco``.
* ``external`` has no other fields and is valid only for ``azure_cc``. Azure
  tooling owns image and resource creation while NVFlare signs the kit and
  installs attestation components.

Bare-metal CVM
--------------

.. code-block:: yaml

   schema_version: 1
   cc_deployment_mode: bare_metal_cvm
   cpu_tee: intel_tdx
   gpu_tee: none
   attestation:
     service: trustee
   workload:
     source:
       type: docker_archive
       path: /srv/nvflare/images/application.tar
   bare_metal_cvm:
     cvm_image: oci://registry.example.org/cvm/tdx@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
     storage:
       vault_size_gib: 16
       applog_size_gib: 1
       user_config_size_gib: 1
       user_data_size_gib: 1
     network:
       allowed_in_ports: []
       allowed_out_ports: [8443]
       allowed_in_cidrs: []
       allowed_out_cidrs: []

``cvm_image`` is an immutable registry reference or a pulled CVM directory.
Its approved contract must match ``cpu_tee`` and ``gpu_tee``. The application
image identity is read from the Docker archive. Federation, admin, relay, and
listener ports are derived from the signed kit and merged with the requested
network additions.

The application container receives the measured ``kbs-client`` read-only and
uses it to generate Trustee-backed, identity-bound NVFlare peer proofs. Vault
key release remains a separate Trustee authorization decision.

CoCo
----

.. code-block:: yaml

   schema_version: 1
   cc_deployment_mode: coco
   cpu_tee: amd_sev_snp
   gpu_tee: nvidia_cc
   attestation:
     service: trustee
   workload:
     source:
       type: docker_build
       context: ./site-1
       dockerfile: Dockerfile
   coco:
     release_name: site-1-v1
     registry: coco_workloads
     registry_repository: workloads/site-1
     platform_config_file: ../admin/platform.env

``registry`` selects the project-level registry; ``registry_repository`` is
only its repository path. ``platform_config_file`` pins the approved Kata
runtime profile and must not own registry credentials. The trusted project
``build_command`` consumes a generated request containing the participant's
staged source. It cannot replace or override that source.

The CPU/GPU pair selects one of ``kata-qemu-snp``,
``kata-qemu-nvidia-gpu-snp``, ``kata-qemu-tdx``, or
``kata-qemu-nvidia-gpu-tdx``. Startup stdout and stderr are redirected to
``/dev/null`` before signing so the host cannot collect application console
output; guest-local NVFlare logs remain available.

Azure CC
--------

.. code-block:: yaml

   schema_version: 1
   cc_deployment_mode: azure_cc
   cpu_tee: amd_sev_snp
   gpu_tee: none
   attestation:
     service: azure_maa
   workload:
     source:
       type: external
   azure_cc:
     deployment_target: confidential_vm

``deployment_target`` is ``confidential_vm`` or
``confidential_container``. The NVFlare 2.9 integration accepts only AMD
SEV-SNP with ``gpu_tee: none``. Intel TDX and NVIDIA confidential GPU targets
fail validation until the authorizer, claim policy, deployment workflow, and
end-to-end tests support them together.

Results and migration
---------------------

``CCPackager`` verifies every signed kit, moves all confidential plaintext
kits to private state, and publishes ``cc_manifests/<participant>.json``.
Each manifest uses schema ``nvflare-cc-delivery/v1`` and records the mode, TEE
selection, attestation service, artifact checksum, and mode-specific immutable
identities. Private paths, credentials, and receipts never enter a manifest.

This interface is intentionally incompatible with the earlier CC provisioning
syntax. Top-level ``cvm_vault``, ``compute_env``, ``cc_cpu_mechanism``,
``cc_gpu``, participant ``role``, raw authorizer class paths, and the separate
``CoCoPackager`` entry point are rejected. Convert the whole project and its
examples together; there is no runtime compatibility mode.
