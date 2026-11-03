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
#. Build the workload into the approved image. CC mode blocks BYOC in every
   deployment mode. The packaged CoCo runtime does not support external-process
   execution; the bare-metal examples use a reviewed in-process executor fixed
   in the measured image. Add only exact, reviewed class names to
   ``class_allow_list`` and validate the executor for the selected mode.
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
#. Follow the build guide's split exact-manifest flow: construct/finalize,
   qualify the isolated Trustee's administration, install the candidate, seal
   its test vault, run acceptance, approve, and install/publish. The shipped
   ``acceptance_runner`` needs that candidate vault and Trustee first. Combined
   ``cvmctl build --acceptance-runner`` automation requires a site-supplied
   adapter; ``site_acceptance_runner`` is not shipped. The checked-in candidate
   cannot be relabeled as production.
#. Build an uncompressed ``application.tar``, size the vault, and provision.
   The outer Docker-save tar must be uncompressed; compressed image layers
   inside it are supported. Their expanded size, not just the saved tar size,
   contributes to the vault's capacity requirements. The CPU-only example
   supplies a separate CPU-wheel Dockerfile to avoid unused GPU packages.
   Provisioning invokes ``cvmctl vault`` for each protected participant; a
   direct builder workflow can call it with publication options. Launch the
   immutable delivery with ``cvmctl run`` for its first launch. Restart the same
   delivery with the absolute ``launch_cvm.sh`` path inside the materialized
   directory; rematerializing restores a pristine vault and loses prior guest
   writes.
#. Start clients first so guest prescan, attestation, vault unlock and Docker
   import finish before starting the server. Clients wait for the server's FL
   endpoint. Then start the server and confirm every required client registers
   within CCManager's **600-second** startup window. This window begins at its
   initial membership check, is fixed in this release, and is independent of
   ``verify_frequency``. A large archive can take longer than 600 seconds to
   prepare; measure cold starts and preserve this launch order rather than
   weakening the required-participant check.
#. After a failed launch, inspect the fixed ``phase=`` identifier in the
   delivery's ``console-<timestamp>.log`` or attached serial output. Serial
   capture starts before guest boot. The applog disk is available only after
   the vault mounts, and the failed run is copied to ``applog-history`` at the
   next launch.

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

.. _cc_log_retrieval:

Retrieve job logs and early-boot diagnostics
---------------------------------------------

Use an authenticated NVFlare administrator session to retrieve application
logs for both bare-metal CVMs and CoCo. CC provisioning configures protected
clients to stream ``log.txt`` as well as ``error_log.txt`` to the server's
``JobLogReceiver``. The latter also retains its post-run error-log upload for
failures before job startup. No guest shell, Kubernetes log stream, or mount of
the confidential filesystem is required.

For a running or completed job, use these FLARE Console commands:

.. code-block:: text

   get_job_log JOB_ID all
   list_job_components JOB_ID
   download_job_components JOB_ID

The equivalent methods on an authenticated FLARE API ``Session`` are:

.. code-block:: python

   logs = sess.get_job_logs(job_id, target="all")
   print(logs.get("logs", {}))
   print(logs.get("unavailable", {}))
   print(sess.list_job_components(job_id))
   log_directory = sess.download_job_components(job_id)

``get_job_logs`` returns the available tail, bounded to 5 MiB per file. For
complete streams after they finish, download the additional components:
``LOG_log.txt_<site>`` and ``ERRORLOG_<site>``. Use the component names returned
by ``list_job_components``; do not assume they appear inside the ordinary
``download_job`` / ``download_job_result`` workspace archive. That archive
contains the server workspace, not every client's private workspace.
These are NVFlare/Python logging files; raw ``print()`` output, subprocess
stderr, and application-specific files are not automatically included. Use the
configured Python logger for application diagnostics that must be retrieved.

Logs are disclosed to the trusted federation server and authorized FL
administrator. Before provisioning, a site can set
``allow_log_streaming: false`` on its client participant in ``project.yml``
to disable both streams. The opt-out is included in the signed startup kit.
If disabled, or if the guest dies before sending data, the admin API reports
unavailable logs; it cannot recover unsent bytes. Retrieve logs before removing
the server's job storage. Reprovision existing CC deliveries to install the new
streamer configuration, and use the updated NVFlare package in the workload
image so both file streamers can coexist.

Confidential NVFlare logs stay inside the CVM vault or CoCo container filesystem
and travel over the authenticated federation connection. They are not mirrored
to the clear ``/applog`` disk or Kubernetes stdout. A workload may explicitly
write non-confidential output to ``/applog``, but that is a separate disclosure
decision. The clear disk is reformatted at every CVM boot; the launcher retains
three preceding disk images under ``applog-history`` as a best-effort aid.

If the application never starts, use the controlled diagnostics for its mode:

* **Bare-metal CVM:** the launcher prints the path of
  ``console-<timestamp>.log`` beside the materialized delivery. It records the
  same output-only guest serial bytes seen on the terminal, including
  initramfs failures and sanitized ``phase=`` startup errors before ``/applog``
  exists. The three newest files are retained with mode ``0600``; each new
  capture keeps at most the first 16 MiB. At the byte limit or on a write
  failure, the launcher warns and stops retaining bytes while continuing to
  drain the logging channel and display terminal output. Copy needed files
  before another launch or rematerialization; if capture cannot be created,
  the launcher warns and keeps terminal output. QEMU's own host-side
  errors remain on launcher stderr. These logs are advisory host files, not
  signed attestation or acceptance evidence.
* **CoCo:** before a workload registers, use Pod status and Events, the assigned
  worker's kubelet/containerd journal, and the trusted KBS/AS/RVPS logs described
  in the :github_nvflare_link:`CoCo launch troubleshooting guide
  <examples/devops/coco/LAUNCH.md>`. These identify scheduling, sandbox,
  image-decryption, and attestation failures. They are not guest syslog.
  The sealed agent continues to deny exec, attach, and application log access
  from the cluster host.

Raw guest ``/var/log`` and journals are not exported. They can contain secrets
and reside in private guest storage. Do not add a clear syslog disk, enable
guest debug shells, or relax the measured CoCo agent policy to obtain them.

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
   * - CPU TEE and platform selection
     - ``intel_tdx`` / ``amd_sev_snp``
     - matching platform map key
     - configuration value plus selected Kata runtime
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

Short glossary
--------------

TEE
   Trusted Execution Environment. Hardware and firmware isolation that protects
   an approved guest's execution from the host within the documented threat
   model.
KBS
   Key Broker Service. The Trustee API that applies resource policy and returns
   a specific key or resource only after the required authorization succeeds.
AS
   Attestation Service. The Trustee component that verifies evidence, evaluates
   appraisal policy and reference values, and signs an attestation result.
RVPS
   Reference Value Provider Service. Trustee storage and APIs for approved
   measurements and TCB values used during appraisal.
EAR
   EAT Attestation Result. The signed appraisal result consumed by release and
   NVFlare participant-authorization policy.
InitData
   CoCo launch data whose digest is bound into TEE evidence, including inputs
   used to select the admitted guest-agent policy.
genpolicy
   The CoCo/Kata tool that derives a guest-agent policy from the intended Pod
   workload. Its output is a security input that must be reviewed and bound.

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
     - Attached ``cvmctl run``/``launch_cvm.sh`` serial output; ``runtime.json``
   * - ``phase=network``
     - Check the measured nftables table, learned DNS resolvers, and configured
       bootstrap egress.
     - Attached launcher serial output; host network configuration
   * - ``phase=time_sync``
     - Check NTS reachability and clock convergence before attestation.
     - Attached launcher serial output; chrony sources
   * - ``CVM_STARTUP_FAILURE=verity``
     - Treat the generic root as corrupt or mismatched. Re-authenticate and
       rematerialize the delivery; do not disable dm-verity.
     - Initramfs output on the attached serial console
   * - ``dm-verity device corrupted`` / kernel panic after startup
     - Treat the generic root as corrupt at runtime. A kernel panic need not
       emit a startup marker or Python phase. Preserve serial output and
       re-authenticate/rematerialize the approved image; keep verity enabled.
     - Kernel output on the attached serial console
   * - ``phase=attestation`` or no Trustee token
     - Check DNS/TLS and collateral first, then the selected policy, AS signing
       key, approved TCB/measurements, and RVPS expiry. A non-affirming EAR or
       wrong AS key fails here before resource retrieval.
     - Attached launcher serial output; Trustee AS/verifier logs
   * - ``phase=key_release``
     - After an affirmative EAR, check the immutable resource path, release
       policy, resource presence, and bundle/platform binding.
     - Attached launcher serial output; KBS policy audit
   * - ``phase=vault_identity``
     - Check that the exact vault and generic bundle match and that the header,
       UUID, platform, and hardware binding were not changed.
     - Attached launcher serial output; ``vault_manifest.json``
   * - ``phase=payload_authentication``
     - Check dm-integrity, manifest identity, application archive identity, and
       measured service files. Do not bypass the failing check.
     - Attached launcher serial output; mounted applog; acceptance result
   * - ``phase=supervision``
     - Check proof renewal, workload lifecycle, vault closure, and containment
       diagnostics after successful startup.
     - Mounted applog; CVM audit records
   * - Required CC participant is missing or unreachable
     - Confirm every protected participant registered within 600 seconds of the
       server's initial membership check. Start slow clients before the server;
       check federation ports and provisioned names.
     - Authenticated admin client status while the server is alive; serial
       boot milestones and Trustee audit timestamps. Parent-process logs stay
       encrypted in a sealed CVM; see the diagnostic limits below.
   * - Registration or peer proof is rejected
     - Compare the participant name, issuer, signed EAR claims, proof key, and
       configured workload constraints. Do not resubmit the job until the peer
       identity mismatch is understood.
     - Trustee appraisal/policy audit and authenticated admin status; reviewed
       streamed job logs when a job has started. Parent-process logs remain
       encrypted in a sealed CVM.
   * - CCManager shuts down a server or client
     - Find the first validation failure before the shutdown line, correct the
       failed participant or policy, then restart the whole required federation.
     - Retained admin status, serial output and Trustee audits. The detailed
       ``Shutting down site ... due to`` message is in encrypted service logs
       unless an approved guest diagnostic channel was configured.
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

After a successful launch, the launcher remains attached, the server and every
required client appear online in the admin console, and each NVFlare log records
``Initialized CCManager`` followed by ``Cross-site validation passed``. A job is
then accepted without ``CC participant validation failed``. Early startup phases
run before the applog disk is mounted, so their ``phase=`` lines are in the
serial output and its ``console-<timestamp>.log`` capture.
``applog-history`` is created at the next launch and
contains the preceding run, not the run currently starting.

Credential and reference expiry belongs in the deployment calendar. Trustee
administration JWTs are capped at 30 days, and RVPS records use the explicit
``--expires`` value. Set ``expiry_warning_seconds`` to the site's maintenance
lead time (seven days by default). Rotate JWTs before that window. An RVPS
deadline cannot be extended for an existing profile: create a new profile
version, repeat acceptance and approval, and install its new record before the
old one expires.

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
     - Create a new profile version; rebuild, run full acceptance and approval,
       then rebuild every dependent vault because the URL, CA, and AS key are
       measured inputs.
     - Re-provision kits and redistribute the authenticated key/CA handoff.
   * - Expiring administration JWT only
     - Rotate the credential; no CVM or vault rebuild is required.
     - Rotate the credential and verify the active policy afterward.
   * - Expiring RVPS approval
     - Create a new profile version, references, acceptance, and approval, then
       rebuild dependent vaults. Re-importing the same profile does not renew it.
     - Install a newly reviewed reference record according to the CoCo rollout
       policy; verify the active policy afterward.

Deferred tooling
----------------

An aggregated ``doctor`` command, automatic SNP TCB discovery, a productized
lab Trustee bootstrap, a self-contained measured TDX signed-quote probe, and
unattended approval for CI require separate security design. Approval remains an
explicit offline-authority action and fails closed. Until that probe is
qualified, ``cvmctl preflight host --quote-probe`` requires a site-approved
executable that boots a bounded TD and verifies current collateral.

Live progress logs from confidential participants
--------------------------------------------------

In a sealed bare-metal CVM, parent/service NVFlare logs stay inside the encrypted
vault. Clear applog contains attestation/audit records, not a copy of those logs.
The job-log API below cannot retrieve parent-process CCManager messages before
a job starts. While the server is alive, retain authenticated client/job status;
correlate it with serial boot milestones, network reachability and Trustee
appraisal/release audits. If shutdown happens before administration is possible,
the exact service-level reason is unavailable through the default external
interfaces. Use an approved guest diagnostic facility if your deployment
provides one, or reproduce the configuration with non-secret test inputs in an
isolated diagnostic deployment. Do not expose production vault contents or
enable guest shells to recover a log.

:ref:`live_log_streaming` is supported over the authenticated FLARE connection
from inside a CVM or CoCo workload. It discloses the selected log content to the
trusted federation server and authorized administrator. Review the application's
log policy first: never log keys, raw evidence, proprietary payloads or private
training data merely to obtain progress. The site can disable streaming with
``allow_log_streaming: false`` in its participant configuration before signing.
Keep guest consoles and Kubernetes exec/attach/log access restricted.

For a job that needs INFO-level progress on 2.9, add the documented
``nvflare.app_common.logging.job_log_streamer.JobLogStreamer`` components to
``config_fed_client.json`` for ``log.txt`` and ``error_log.txt``. If those streams
are already supplied by the signed site configuration, use that configuration
instead of declaring duplicates. Review and include the exact class path in
the participant's ``class_allow_list`` before provisioning; do not allow a whole
package. The bundled CVM CIFAR10 job shows the explicit job-level configuration.

Write progress through the configured Python logger at an appropriate level.
Raw ``print()``, subprocess output and arbitrary application files are not
automatically part of ``log.txt``. The CIFAR10 example logs the round and
periodic batch progress without logging model parameters or input samples.
Retrieve the current tail while a round is still running:

.. code-block:: bash

   nvflare job logs JOB_ID --site site-1 --tail 100

``client log stream not available`` means no stream has arrived; it is not a
progress indication. Check that the job started, the reviewed streamer is
enabled, ``allow_log_streaming`` permits it, the server receiver is present,
and the code writes to the configured Python logger. If the guest dies before
the first stream, use the deployment's startup diagnostics. See
:ref:`live_log_streaming` for receiver configuration and heartbeat behavior.
