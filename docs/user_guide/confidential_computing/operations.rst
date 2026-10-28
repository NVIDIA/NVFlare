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
   GPU target and ``--snp-vcek-store PATH`` to verify the exact offline
   collateral entry used by Trustee.
#. Run the combined ``cvmctl build`` path with the trusted acceptance runner,
   approval key, and, when the same role owns them, Trustee install and release
   signing options. Use the split commands only for role separation or
   cross-host measurement. The checked-in candidate cannot be relabeled as
   production.
#. Build an uncompressed ``application.tar``, size the vault, and provision.
   Provisioning invokes ``cvmctl vault`` for each protected participant; a
   direct builder workflow can call it with publication options. Launch the
   immutable delivery with ``cvmctl run``.
#. After a failed launch, inspect the fixed ``phase=`` identifier and the newest
   retained ``applog-history`` image before retrying.

CoCo checklist
--------------

#. Start with an existing CoCo-ready Kubernetes cluster, image registry, and
   configured Trustee. Use the supported Kata confidential runtime and follow
   the upstream NVIDIA CoCo/Kata instructions for GPU targets. NVFlare
   provisioning does not install or reconfigure this infrastructure.
#. Capture the installed runtime measurements using the CoCo measurement guide,
   then have the Trustee administrator install and read back the exact CPU and,
   when required, GPU reference values.
#. For a GPU target, repeat the read-only ClusterPolicy, operand, worker-label,
   ``vfio-pci``, and allocatable ``nvidia.com/pgpu`` checks documented in the
   CoCo README after each GPU Operator upgrade or configuration change.
#. Run ``nvflare provision`` to create encrypted images, private Trustee release
   handoffs, and Pod YAML. The Trustee administrator registers each unique
   release using the retained complete resource-policy export before launch.
#. Transfer the Pod handoff to the cluster operator and use the documented
   launcher. It reports a ``FailedCreatePodSandBox`` event promptly while
   retaining the Pod for diagnosis.
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
     - CoCo workflow
   * - Trustee HTTPS service
     - ``kbs_endpoint``
     - ``kbs_url``
     - ``KBS_URL``
   * - Trustee host name
     - ``trustee.example.org``
     - ``trustee.example.org``
     - host in ``KBS_URL``
   * - Participant CPU TEE
     - ``intel_tdx`` / ``amd_sev_snp``
     - platform map key
     - selected Kata runtime
   * - CPU TEE configuration
     - ``intel_tdx`` / ``amd_sev_snp``
     - ``intel_tdx`` / ``amd_sev_snp``
     - ``intel_tdx`` / ``amd_sev_snp``
   * - NVIDIA confidential GPU
     - ``gpu_tee: nvidia_cc``
     - ``gpu: nvidia_cc``
     - NVIDIA GPU Kata runtime

Both on-premises modes resolve their endpoint names to the same upstream
Trustee service. The host in ``KBS_URL`` must resolve from inside the confidential
guest, not only on the host. The different 2.9 keys are retained for
compatibility. Trustee still signs the upstream ``tdx`` or ``snp`` evidence
object key on the wire; NVFlare translates that internal protocol detail and
does not accept it as a configuration value.

Troubleshooting by symptom
--------------------------

.. list-table:: Symptom, check, and diagnostic location
   :header-rows: 1
   :widths: 29 43 28

   * - Symptom
     - First check and next action
     - Diagnostic source
   * - ``phase=configuration``
     - Check the measured runtime configuration and root-only runtime-state
       directory before investigating Trustee.
     - Retained CVM applog; ``runtime.json``
   * - ``phase=network``
     - Check the measured nftables table, learned DNS resolvers, and configured
       bootstrap egress.
     - Retained CVM applog; host network configuration
   * - ``phase=time_sync``
     - Check NTS reachability and clock convergence before attestation.
     - Retained CVM applog; chrony sources
   * - ``phase=attestation`` or no Trustee token
     - Check DNS for the Trustee host, KBS/AS TLS, and platform collateral egress.
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
   * - ``phase=supervision``
     - Check proof renewal, workload lifecycle, vault closure, and containment
       diagnostics after successful startup.
     - Retained applog; CVM audit records
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
     - Release names are immutable within the shared policy namespace. Choose a
       unique project/release name and reconcile the Trustee administrator's
       retained complete resource-policy export before registering another
       release; a stale export can overwrite resources or remove unrelated rules.
     - Retained complete resource-policy export and release handoff
   * - GPU Operator never becomes ready
     - Repeat the CoCo README's ClusterPolicy, operand, worker-label,
       ``vfio-pci``, and allocatable ``nvidia.com/pgpu`` checks. Repair the
       existing platform according to its GPU Operator management model.
     - ClusterPolicy; node labels and allocatable resources
   * - ``FailedCreatePodSandBox``
     - Use the event text. For a missing Kata handler, verify the selected
       RuntimeClass and repair the existing containerd/Kata installation.
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
     - Capture the changed platform, update and read back Trustee CPU/GPU
       references, and launch a newly approved release when its policy binding
       changes.
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
