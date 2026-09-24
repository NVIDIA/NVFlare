.. _cc_architecture:

##################################
CVM Builder Security Architecture
##################################

.. contents::
   :local:
   :depth: 2

This page describes the trust model, trust boundaries and chain of trust behind the
on-premises IP protection deployment. Read it before provisioning with
:ref:`cvm_builder`, because the guarantees below depend on who operates each
component, not only on the hardware.

The authoritative engineering documents live with the builder in the NVFlare source
tree; see `Reference guides`_ for links to each one.

Trust model
===========

Confidential Computing hardware alone does not protect model IP. The architecture
assumes the following division of trust.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Actor
     - Trusted
     - Notes
   * - CPU/TEE firmware (AMD SP, Intel TDX module, TDVF/OVMF)
     - Yes
     - Root of trust; vendor-signed.
   * - KBS / Attestation Service operator (Trustee)
     - Yes
     - Holds vault keys and reference values. Must be run by the data owner or a
       mutually trusted party.
   * - Builder host
     - At build time only
     - Sees plaintext vault content and the vault key while building. Must scrub
       its scratch space.
   * - CVM host / hypervisor operator
     - **No**
     - Can read or replace any disk image, edit the QEMU command line, and spoof
       the network.
   * - Workload (Docker image)
     - By the party that built the vault
     - Runs inside the TEE and remains part of the trusted computing base.

The application and the admitted guest services stay inside the trusted computing
base. Guest services run as root unless configured otherwise, and executable-path
validation is not a privilege sandbox. By default, a container has a read-only root
filesystem and receives the authenticated application payload read-only, with
explicit writable mounts for the application ``runtime/`` and ``data/`` directories
and the clear ``/applog`` output disk. Authenticated configuration may add bind
mounts from those writable locations to other container paths, including paths
named ``/vault/services``, ``/vault/config`` or ``/vault/docker``, and may disable
the read-only container root. Those choices can mask authenticated files inside
the container's mount namespace, although they do not modify the underlying vault
content. Guest service definitions, launch configuration, application executables
and Docker metadata remain protected by the guest's systemd and vault mount
controls. Operators must review these authenticated application settings rather
than assume the defaults are still in force. A compromised workload can still
disclose any secret it is authorized to use.

Attacks in scope
----------------

- Booting a modified root filesystem.
- Obtaining keys from debug-enabled or otherwise unapproved CPU configurations.
- Substituting a vault header, or presenting a different binding in evidence.
- Altering vault ciphertext, authentication tags or initialization vectors.
- Replaying stale attestation evidence.
- Reading vault secrets from persistent storage.
- Preventing the attestation services from responding.

The host can read and replace the clear sidecar disks. Applications therefore treat
``user_config`` and ``user_data`` as untrusted inputs and never place secrets in
them, and ``/applog`` carries only output that is intentionally disclosed to the
host.

Out of scope
------------

Side channels, host denial of service, physical attacks, and bugs in the workload
itself. Cloning an authorized root-and-vault combination is also not prevented, and
replay of previously valid mutable disk sectors or snapshots is not detected:
attestation freshness and disk-state freshness are distinct properties. A deployment
that needs disk rollback protection requires an external trusted state service.

Deployment requirements
-----------------------

These follow from the trust model and are requirements, not recommendations.

- **One Trustee instance per administration domain.** Use a dedicated Trustee
  instance, resource storage and publisher state for each independently
  administered project or tenant and security profile. The KBS endpoint and trust
  roots are measured build inputs inherited by every vault. Bundle-scoped resource
  roles restrict paths inside a trusted administration domain; they do not isolate
  mutually untrusted tenants that share a generic bundle. Do not issue these
  administrative credentials to independent tenants on a shared instance.
- **Every participant trusts the Trustee operator** with its vault keys. A
  federation whose participants do not share that trust cannot use a single generic
  profile and a single Trustee for all participants.
- **Protection is asymmetric.** A data owner running an opaque encrypted workload
  must trust the workload publisher with all data given to that workload. The CVM
  protects application and model secrets from the host; it does not prove that the
  workload preserves the data owner's confidentiality, and it does not prevent an
  authorized workload from exporting data. CIDR and port controls reduce
  destinations but are not a data-use policy. Review the workload, its output
  policy and the organizational agreement separately.

Chain of trust
==============

The hardware chain of trust normally stops at the kernel, so application code and
data on disk are not covered by the launch measurement. The architecture extends it
in four steps.

1. **Measured launch.** Firmware measures the kernel, initramfs and kernel command
   line into the TEE launch measurement. AMD SEV-SNP uses ``kernel-hashes=on``;
   Intel TDX uses TDVF measured direct boot and reports MRTD and RTMR values.
2. **Verified root.** The root filesystem is read-only dm-verity, and its root hash
   is covered by the launch measurement. A tmpfs upper layer keeps anything written
   at runtime from persisting in the clear. The initramfs only opens the dm-verity
   root and stacks the overlay; it performs no attestation and holds no KBS
   endpoint, CA certificate or network configuration.
3. **Attested key release.** Attestation and vault unlock run as a systemd service
   inside the already-verified root. The key is released only against a signed
   appraisal that matches the registered measurements (see
   `Attestation-bound key release`_).
4. **Authenticated vault.** Application code, confidential configuration and state
   live on authenticated LUKS2 storage, so payload authentication is enforced on
   every read, including reads of guest services, scripts and Docker state.

Any startup verification failure -- verity, attestation, key release, vault identity
or payload authentication -- prevents workload startup and halts the guest. An
integrity or attestation failure detected after startup stops the workload and
powers off the guest.

Attestation layers
==================

CVM Builder and NVFlare perform separate attestation operations with different
purposes. Keeping the layers separate is deliberate.

CVM vault authorization
-----------------------

The verified guest attests to Trustee before the workload starts. Initial appraisal
and key release require CPU evidence for a CPU-only profile and composite CPU/GPU
evidence for a GPU profile. This establishes that the measured CVM, its attached
vault and, when configured, its GPUs are authorized to run together.

The CVM supervisor repeats the same hardware appraisal and verifies current key
authorization every five minutes. A periodic failure stops the workload, closes the
vault so its key leaves the kernel, and retries fresh appraisal and key retrieval
within a bounded quarantine window. A successful retry reopens the same vault and
restarts the workload; expiry of the window or a quarantine failure powers off the
guest. This layer is owned by CVM Builder and the Trustee deployment and is
configured through the builder profile rather than through NVFlare job
configuration.

Runtime attestation
-------------------

NVFlare's ``CCManager`` and its ``CCAuthorizer`` components provide a separate
application-level layer. They generate and cross-verify participant tokens for the
lifetime of the system, and a site that fails validation is removed from the
federation. See :ref:`confidential_computing_attestation` for that workflow.

The split matters because CVM-level reauthorization protects local vault access,
while NVFlare participant attestation decides whether remote peers remain admitted
to the federation. Neither layer substitutes for the other.

Attestation-bound key release
=============================

Each sealed vault has one KBS resource at ``keys/<build_id>/<binding_id>``, where
``build_id`` identifies the approved generic CVM bundle and ``binding_id`` is the
digest of the LUKS header the guest actually uses to open that vault. The guest
presents ``binding_id`` inside signed TEE evidence, and the verified guest checks
the digest against the header it opens.

Releasing a key therefore requires all of the following at once: an approved CPU
appraisal, the registered measurements of the bundle named in the key path, a path
binding equal to the header digest in the evidence, and an existing secret at that
exact path. An instance running vault A cannot obtain vault B's key merely because
both use the same root.

Resource policy
---------------

CVM Builder generates the KBS resource policy from the approved bundle manifests and
installs it as ``kbs/resource-policy.rego``. The policy denies by default and admits
only freshly issued, affirming appraisals:

.. code-block:: text

   package policy
   import rego.v1
   default allow := false
   cpu := input["submods"]["cpu0"]
   ev := cpu["ear.veraison.annotated-evidence"]
   tv := cpu["ear.trustworthiness-vector"]
   approved_cpu(policy_id) if {
       fresh_token
       cpu["ear.appraisal-policy-id"] == policy_id
       cpu["ear.status"] == "affirming"
       tv["executables"] == 3
       tv["hardware"] == 2
       tv["configuration"] == 2
   }

``fresh_token`` bounds the issued-at, not-before and expiry claims so an appraisal
older than five minutes is rejected, which is what makes replay of stale evidence
ineffective.

One rule is emitted per approved bundle. An Intel TDX rule pins the measurements and
requires the evidence binding to equal the requested resource path:

.. code-block:: text

   allow if {
       approved_cpu("<attestation_policy_id>")
       is_string(ev["init_data"])
       ev["tdx"]["quote"]["body"]["mr_td"] == "<mr_td>"
       ev["tdx"]["quote"]["body"]["rtmr_0"] == "<rtmr_0>"
       ev["tdx"]["quote"]["body"]["rtmr_1"] == "<rtmr_1>"
       ev["tdx"]["quote"]["body"]["rtmr_2"] == "<rtmr_2>"
       ev["tdx"]["td_attributes"]["debug"] == false
       regex.match("^[0-9a-f]{64}0{32}$", ev["init_data"])
       binding_id := ev["init_data"]
       data.plugin == "resource"
       data["resource-path"] == ["keys", "<build_id>", binding_id]
   }

The AMD SEV-SNP rule has the same shape, pinning ``snp.measurement`` and requiring
``policy_debug_allowed`` and ``policy_migrate_ma`` to be false. A GPU profile adds
conditions requiring exactly ``gpu_count`` distinct NVIDIA GPU submodules in the same
signed appraisal before the key is released; CPU-only rules are unaffected.

Because the policy is derived from the generic bundle rather than from individual
applications, adding or rebuilding a vault leaves reference values and the resource
policy unchanged. There is no per-vault authorization ledger to maintain.

Reference guides
================

The builder's engineering documentation ships in the NVFlare source tree at
``nvflare/lighter/cc/image_builder``. These are operator and developer references
rather than user documentation, and they are versioned with the code:

- :github_nvflare_link:`DESIGN.md <nvflare/lighter/cc/image_builder/DESIGN.md>` -
  goals and non-goals, full threat model, architecture, boot flow, and the
  measurement and key-binding rules summarized above
- :github_nvflare_link:`TRUSTEE_GUIDE.md <nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md>` -
  deploying and administering the Trustee instance, installing reference values and
  policies, and uploading or revoking vault keys
- :github_nvflare_link:`BUILD_GUIDE.md <nvflare/lighter/cc/image_builder/BUILD_GUIDE.md>` -
  preparing the build host and building generic CVMs and vaults
- :github_nvflare_link:`USER_GUIDE.md <nvflare/lighter/cc/image_builder/USER_GUIDE.md>` -
  launching and operating a delivered CVM on a runtime host
- :github_nvflare_link:`CONFORMANCE.md <nvflare/lighter/cc/image_builder/CONFORMANCE.md>` -
  the conformance criteria each profile must meet, and their current status
- :github_nvflare_link:`VALIDATION.md <nvflare/lighter/cc/image_builder/VALIDATION.md>` -
  the recorded hardware validation log and the remaining production gates
- :github_nvflare_link:`GPU_BUILD.md <nvflare/lighter/cc/image_builder/GPU_BUILD.md>` and
  :github_nvflare_link:`TDX_TROUBLESHOOTING.md <nvflare/lighter/cc/image_builder/TDX_TROUBLESHOOTING.md>` -
  platform-specific build and troubleshooting notes
