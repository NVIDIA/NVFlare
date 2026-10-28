.. _cc_operations:

Confidential Computing Operations
=================================

This page is the start-to-finish checklist and first-response guide for all
three deployment modes. Complete the common steps once, then follow the selected
mode. Keep hardware qualification, workload approval, and a successful
functional run as separate records.

Common start checklist
----------------------

#. Choose ``bare_metal_cvm``, ``coco``, or ``azure_cc`` for every protected
   participant and record its CPU/GPU TEE combination.
#. Confirm hardware and firmware support before reserving or buying machines.
   Enable the TEE and IOMMU in firmware, reboot, and verify the operating-system
   interfaces. For NVIDIA CC, qualify the exact GPU, VBIOS, driver, and attester
   combination; a model name alone is insufficient.
#. Plan external dependencies and egress: Trustee/KBS, DNS, NTS, AMD KDS or
   Intel collateral, NVIDIA attestation endpoints, registry, and federation
   ports. Start any Intel Tiber Trust account and platform-registration work
   during procurement because it can add lead time.
#. Build the workload into the approved image. BYOC and external-process
   executors are denied in CC mode. Add only exact, reviewed class names to
   ``class_allow_list``.
#. Put the shared Trustee or Azure MAA object in ``cc_project.yml`` and the mode
   selection in each participant ``cc_config``. Run ``nvflare provision`` and
   retain the private build receipts and public ``cc_manifests`` separately.
#. Perform positive and negative hardware acceptance, then record the exact
   artifact digest, references, policies, firmware, and expiry dates that were
   approved.

Bare-metal CVM checklist
------------------------

#. Use the candidate warning and validation record at the start of the
   :github_nvflare_link:`CVM Builder README
   <nvflare/lighter/cc/image_builder/README.md>`. For learning without a TEE,
   use its dev-mode walkthrough.
#. On TDX, configure ``qgsd`` port 4050, the collateral service, platform
   registration, and a signed-quote smoke test. On SNP, run ``cvmctl preflight
   host --platform amd_sev_snp --firmware PATH``; add ``--gpu`` for an installed
   GPU target.
#. Prepare the generic bundle, collect references on trusted target hardware,
   complete the acceptance matrix, and approve a newly versioned production
   profile. The checked-in candidate cannot be relabeled as production.
#. Build an uncompressed ``application.tar``, size the vault, provision, and
   seal the application-specific vault. Launch only the immutable delivered
   artifact.
#. After a failed launch, inspect the fixed ``phase=`` identifier and the newest
   retained ``applog-history`` image before retrying.

CoCo checklist
--------------

#. Use Ubuntu 24.04 for the secure-services host in this pinned workflow and
   the documented 26.04 baseline for the other role hosts. Give the registry
   publisher its exact egress IPv4 ``/32``; it must not be the site egress.
#. Install Trustee, registry, and reference policies through the role-separated
   service stages. Keep the KBS default-deny until a reviewed release handoff is
   merged.
#. Rehearse and approve the exact Kata runtime profile, then install Kubernetes,
   Kata, and (for GPU targets) GPU Operator. Set
   ``GPU_OPERATOR_DRIVER_ENABLED=0`` only with a working host driver.
#. Run ``coco/45-check-gpu-operator.sh`` after every GPU Operator install or
   upgrade. The launch script repeats this check and reports operand drift.
#. Transfer the digest-authenticated Pod handoff to CoCo IT. The launch script
   treats a ``FailedCreatePodSandBox`` event as proof that the effective Kata
   handler failed and prints the event instead of waiting silently.
#. Verify attestation, key release, the real FL job, renewal, and required denial
   cases. ``kubectl exec``, attach, copy, port-forward, and application log
   collection from the host are intentionally unavailable.

Azure checklist
---------------

#. Use Azure tooling to prepare the confidential VM or container resource and
   image, then configure ``workload.source.type: external``.
#. Configure the project-level Azure MAA service and participant
   ``cc_deployment_mode: azure_cc``. NVFlare 2.9 supports AMD SEV-SNP with
   ``gpu_tee: none`` in this mode.
#. Provision and install the signed startup kit using the selected Azure guide,
   then validate MAA evidence and the NVFlare job on the deployed resource.

Shared terminology and configuration names
------------------------------------------

.. list-table:: Names used for the same boundary
   :header-rows: 1
   :widths: 22 25 25 28

   * - Meaning
     - Unified provisioning
     - Bare-metal profile
     - CoCo role scripts
   * - Trustee HTTPS service
     - ``kbs_endpoint``
     - ``kbs_url``
     - ``https://${SERVICE_FQDN}:8443``
   * - Trustee host name
     - ``trustee.example.org``
     - ``trustee.example.org``
     - ``SERVICE_FQDN``
   * - Participant CPU TEE
     - ``intel_tdx`` / ``amd_sev_snp``
     - platform map key
     - selected Kata runtime
   * - Signed EAR CPU evidence
     - ``tdx`` / ``snp`` inside ``workload_constraints``
     - appraisal evidence
     - appraisal evidence
   * - NVIDIA confidential GPU
     - ``gpu_tee: nvidia_cc``
     - ``gpu: nvidia_cc``
     - NVIDIA GPU Kata runtime

Both on-premises modes resolve their endpoint names to the same upstream
Trustee service. ``SERVICE_FQDN`` must resolve from inside the confidential
guest, not only on the host. The different 2.9 keys are retained for
compatibility; do not invent additional aliases.

Troubleshooting by symptom
--------------------------

.. list-table:: Symptom, check, and diagnostic location
   :header-rows: 1
   :widths: 29 43 28

   * - Symptom
     - First check and next action
     - Diagnostic source
   * - ``phase=attestation`` or no Trustee token
     - Check DNS for ``SERVICE_FQDN``, KBS/AS TLS, NTS reachability and clock
       convergence, then platform collateral egress.
     - Retained CVM applog; Trustee AS logs
   * - ``phase=key_release``
     - Check the immutable resource path, release policy, admin/JWT expiry, and
       RVPS reference expiry. A valid quote alone does not authorize a key.
     - Retained applog; KBS policy audit
   * - ``phase=vault_identity``
     - Check that the exact vault and generic bundle match and that the header,
       UUID, platform, and hardware binding were not changed.
     - Retained applog; ``vault_manifest.json``
   * - ``phase=payload_authentication``
     - Check dm-integrity, manifest identity, application archive identity, and
       measured service files. Do not bypass the failing check.
     - Retained applog; acceptance result
   * - TDX quote stops after about 168 hours
     - Match the cache-expiry and collateral error sequence, coordinate a
       ``qgsd`` restart, then run a fresh challenge-bound quote test.
     - ``journalctl -u qgsd`` and the :github_nvflare_link:`TDX runbook
       <nvflare/lighter/cc/image_builder/TDX_TROUBLESHOOTING.md>`
   * - TDX appraisal is ``OutOfDate``
     - Confirm that Trustee ``tcb_update_type`` matches the approved
       ``standard`` or ``early`` collateral channel; update firmware for the
       selected baseline rather than weakening policy.
     - Trustee verifier logs and deployed KBS config
   * - SNP certificate fetch returns HTTP 429
     - Stop tight retries, use bounded exponential backoff, and cache the
       authenticated VCEK for its chip/TCB identity according to site policy.
     - Trustee verifier logs; KDS response status
   * - A shared-KBS release name already exists
     - Release names are immutable within the shared policy namespace. The
       provisioner checks the authenticated active policy before invoking the
       publisher; choose a unique project/release name. The secure-services
       install lock repeats the check to close concurrent races.
     - Active KBS resource policy and release handoff
   * - GPU Operator never becomes ready
     - Run ``45-check-gpu-operator.sh``. Restore each reported ClusterPolicy
       operand. With ``driver.enabled=false``, verify the host driver first.
     - ClusterPolicy; node labels and allocatable resources
   * - ``FailedCreatePodSandBox``
     - Use the event text. For a missing Kata handler, rerun stages 30 and 35
       after verifying containerd imports the selected runtime fragment.
     - Pod Events; containerd and kubelet journals
   * - GPU quarantine after periodic appraisal failure
     - Treat quarantine as terminal for that guest. Preserve diagnostics and
       restart only after the attestation or readiness cause is corrected.
     - Trusted guest audit output and service appraisal logs

Credential and reference expiry belongs in the deployment calendar. Trustee
administration JWTs are capped at 30 days, and RVPS records use the explicit
``--expires`` value. Set ``expiry_warning_seconds`` to the site's maintenance
lead time (seven days by default). Rotate before that window; do not wait for a
production key-release denial.

Day-2 rebuild matrix
--------------------

.. list-table:: What changed and what must be rebuilt
   :header-rows: 1
   :widths: 31 33 36

   * - Change
     - Bare-metal CVM
     - CoCo
   * - Application code, dependency, command, or class grant
     - Rebuild and reseal the application vault; keep the approved generic CVM
       only when its contract is unchanged.
     - Create a new immutable release: rebuild, encrypt, sign, publish, generate
       policy/Pod handoffs, and merge the release policy.
   * - Guest kernel, root package, runtime, firmware, attester, or measured policy
     - New profile version, generic build, references, full acceptance and
       approval, then rebuild every dependent vault.
     - New runtime profile, trusted rehearsal/reference approval, redeployment,
       and new workload release.
   * - CPU/GPU/VBIOS/driver baseline
     - Update reviewed references and profile inputs; repeat affected hardware
       acceptance and rebuild/approve as required by the changed contract.
     - Update platform/GPU references through service stages, verify them, and
       launch a newly approved release when its policy binding changes.
   * - Trustee AS signing key or endpoint identity
     - Re-provision signed kits and rebuild artifacts that pin the key or CA.
     - Re-provision kits and redistribute the authenticated key/CA handoff.
   * - Expiring JWT or RVPS expiry only
     - Rotate the credential or renew reviewed reference records under the same
       approved values; do not rebuild unrelated application bytes.
     - Rotate the credential or renew reviewed reference records; verify the
       active policy and release afterward.

Deferred tooling
----------------

An aggregated ``doctor`` command, automatic SNP TCB discovery, a productized
lab Trustee bootstrap, a self-contained measured TDX signed-quote probe, and
unattended approval for CI require separate security design. Existing
120-second approvals remain interactive and fail closed. Until that probe is
qualified, ``cvmctl preflight host --quote-probe`` requires a site-approved
executable that boots a bounded TD and verifies current collateral.
