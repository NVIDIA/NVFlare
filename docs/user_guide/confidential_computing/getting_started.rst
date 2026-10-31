.. _cc_getting_started:

Getting Started with Confidential Computing
===========================================

Use this guide to choose and prepare a confidential federation before writing
provisioning configuration. Federated Confidential Computing with NVIDIA FLARE
is a **technical preview**: configuration support is not hardware acceptance
for every platform, software revision, or workload.

Role and handoff map
--------------------

Production normally separates the credentials held by these roles. One person
or organization may perform more than one role only when the deployment's trust
policy permits it.

.. code-block:: text

   reviewed sources and workload
              |
              v
   [trusted build/provisioning worker]
        | candidate bundle + test vault
        v
   [acceptance operator/authority] -- signed approval --> [Trustee administrator]
        | acceptance report                                  | approved references,
        +---------------------> build worker                  | policy, and scoped key
                                                              v
   [build worker] -- signed sealed delivery ------------> [runtime host]
        | scoped vault key upload                              | launches untrusted host/VMM
        +-----------------------------> [Trustee] <--- evidence/resource request ---+
                                                                           |
                                                                   [confidential guest]

The build worker hands off an exact candidate and later a signed sealed
delivery. The acceptance authority signs the report for that exact candidate.
The Trustee administrator installs approved references and policy and issues
only scoped resource credentials. The runtime host receives public artifacts
and launch instructions, never approval or Trustee-administration keys.

1. Decide what needs protection
-------------------------------

Identify which servers and clients need confidential execution. A protected
server can keep individual updates inside its trusted execution boundary during
aggregation; protected clients can run reviewed code and model weights against
local data. Ordinary, verifier-only participants can verify protected peers
without claiming that their own environments are confidential.

Agree on who approves the application, authorizes local data access, operates
the trust service, and receives models, results, or logs. The application can
read its inputs, so confidential execution does not replace an output policy.
See :ref:`cc_architecture_overview` for the responsibilities and trust boundaries.

.. _cc_choose_deployment:

2. Choose a deployment approach
-------------------------------

.. list-table:: Deployment choices in NVFlare 2.9
   :header-rows: 1
   :widths: 26 44 30

   * - Deployment form
     - Choose this when
     - Preparation guide
   * - Standalone Confidential VM (CVM)
     - You need a standalone VM with a verified generic root and a separately
       sealed application vault, without Kubernetes.
     - :ref:`cvm_builder`
   * - Confidential Containers (CoCo)
     - You deploy Kubernetes workloads inside Kata confidential VMs, with
       encrypted workload images and InitData-bound guest policy.
     - :github_nvflare_link:`CoCo-ready cluster workflow
       <examples/devops/coco/README.md>`
   * - Azure confidential computing
     - You use Azure confidential VMs or confidential container instances.
       External Azure tooling creates the images and resources.
     - :doc:`azure/index`

A federation may mix these deployment forms. Select the form for each
participant according to its environment and the agreed trust model, not only
its ability to start a VM or container.

3. Check platform and operational prerequisites
-----------------------------------------------

The 2.9 configuration supports AMD SEV-SNP and Intel TDX targets for standalone
CVM and CoCo deployments, with CPU-only or NVIDIA confidential-GPU profiles.
For CoCo, **full security qualification is incomplete**; the historical mixed
run recorded below does not qualify **TDX+GPU or a protected-server topology**.
The Azure integration currently accepts AMD SEV-SNP with ``gpu_tee: none``.

.. list-table:: Documented 2.9 component baselines
   :header-rows: 1
   :widths: 25 35 40

   * - Deployment component
     - Documented baseline
     - Qualification note
   * - Standalone CVM build worker and generic guest
     - Ubuntu 26.04; the checked-in candidate records kernel 7.0.0-31 and
       QEMU 10.2.1.
     - The exact profile remains a candidate until its complete hardware and
       negative-test matrix is approved.
   * - CoCo/Kata worker
     - Ubuntu 25.10 with the example's pinned Kata 3.29.0 profile, following
       the NVIDIA CoCo v1.0 platform matrix.
     - Do not infer that the newer Ubuntu 26.04/Kata 4.0 matrix qualifies the
       pinned 3.29.0 example.
   * - Shared Trustee backend
     - Upstream Trustee v0.22.0, paired with CoCo v0.23.0 when CoCo shares the
       service.
     - Revalidate policies, references, APIs, and appraisal on every upgrade.
   * - Azure
     - The versions required by the selected Azure confidential VM or ACI
       deployment guide.
     - NVFlare does not impose one common guest-image baseline across those
       externally created resources.

* For standalone CVMs, review the :github_nvflare_link:`revision-specific
  validation record <nvflare/lighter/cc/image_builder/VALIDATION.md>` and
  :github_nvflare_link:`conformance requirements
  <nvflare/lighter/cc/image_builder/CONFORMANCE.md>`. Prepare the trusted build
  worker, approved generic CVM, and Trustee service described in
  :ref:`cvm_builder`.
* For CoCo, the :ref:`implementation scope and hardware evidence limits
  <coco_security_architecture>` describe one historical mixed TDX CPU-only and
  SNP+NVIDIA GPU functional run on **2026-10-04**, with an ordinary trusted server.
  The workload performed CPU arithmetic, not GPU training. **Before the four-step workflow,
  deploy and verify both Trustee and the CoCo-ready Kubernetes cluster.** Follow
  the `official Trustee installation guide
  <https://confidentialcontainers.org/docs/attestation/installation/>`_ and, for
  GPU workloads, the `NVIDIA CoCo/Kata GPU deployment guide (v1.0)
  <https://docs.nvidia.com/confidential-computing/containers/v1.0.0/confidential-containers-deploy>`_
  to prepare the Kata confidential GPU runtime and GPU Operator. CPU-only
  profiles do not require GPU components. Use the supported component versions,
  then proceed with the inputs in the
  :github_nvflare_link:`four-step deployment workflow
  <examples/devops/coco/README.md>`.

  The optional :github_nvflare_link:`beginner prerequisite appendix
  <examples/devops/coco/PREREQUISITES.md>` helps beginners bring up the
  prerequisite systems for the NVFlare CoCo workflow. Consult the official
  documentation for each component and the version you deploy. If there is any
  discrepancy, the component's official documentation takes precedence. Its
  two-machine example keeps Trustee separate from Kubernetes; review its
  Ubuntu 26.04 and pinned-runtime qualification limits before following it.
* For Azure, follow the selected :doc:`Azure deployment guide <azure/index>`
  for the cloud resources and attestation setup.

Review the selected deployment's architecture and approval requirements before
releasing secrets or data to it. A successful local documentation or
configuration check is not hardware acceptance.

4. Prepare the workload and data policy
---------------------------------------

* Package reviewed application code and dependencies in the approved workload
  image. CC mode blocks BYOC jobs, including launcher-defined executable content.
  Use exact, reviewed ``class_allow_list`` entries for additional components.
* Validate the job's executor against the selected deployment. The packaged
  CoCo runtime does not support external-process execution; bare-metal examples
  use a reviewed in-process executor fixed in the measured image. Follow the documented
  :github_nvflare_link:`in-process executor guidance
  <examples/devops/coco/provision/README.md#prepare-the-workload-and-project>`.
  Changing code or policy requires a new approved release.
* Keep confidential checkpoints, credentials, and logs within protected storage
  or an explicitly approved secure output path. CVM Builder's ``user_config``,
  ``user_data``, and ``applog`` sidecars are clear storage, not secret stores.
* Define who may receive datasets, individual updates, final models, and logs.
  An encrypted image does not implement dataset authorization or output review.
* Treat GPU driver and VBIOS versions as approved policy inputs. See the
  :github_nvflare_link:`CoCo GPU reference-value procedure
  <examples/devops/coco/TRUSTEE.md#set-gpu-references-when-required>` before deploying or
  changing a GPU baseline.

5. Provision, launch, and validate
----------------------------------

Once the deployment, prerequisites, and workload approvals are established,
follow :ref:`cc_deployment` for the shared provisioning configuration. Select
``bare_metal_cvm``, ``coco``, or ``azure_cc`` in each protected participant's
``cc_config``, and define shared services in ``cc_project_config``.

Create confidential-computing projects with CLI provisioning. The Dashboard's
legacy project record does not generate ``cc_config``, Trustee policy, or a
protected deployment artifact and therefore does not expose a CC control.

Use the chosen mode's guide for building or creating the protected artifacts,
approval, distribution, and launch. Validate the selected platform and workload,
including the required attestation and denial behavior, before approving the
deployment for use. Keep configuration support, successful functional runs, and
security acceptance as separate claims.

Follow :ref:`cc_operations` for the linear checklist, shared terminology,
failure table, and Day-2 rebuild decisions across all three modes.

Beginner FAQ
------------

**What does confidential computing add to federated learning and TLS?**
TLS protects traffic between authenticated endpoints. Confidential computing
adds evidence about the endpoint's TEE and approved software before peers admit
it or a service releases a secret. See :ref:`cc_architecture_overview`.

**Which parties and software must I still trust?**
Trust the workload and approved guest software, platform roots of trust,
approval authority, Trustee or Azure attestation authority, and the policy for
data and outputs. The host can still deny service. See the roles and boundaries
in :ref:`cc_architecture_overview`.

**Do I need a confidential GPU?**
No. CPU-only profiles use ``gpu_tee: none``. Select ``nvidia_cc`` only when the
reviewed workload needs a qualified confidential GPU and both CPU and GPU
appraisal are part of its release policy.

**Can I evaluate the workflow without TEE hardware?**
The CVM Builder :github_nvflare_link:`dev-mode walkthrough
<nvflare/lighter/cc/image_builder/README.md#30-minute-dev-mode-walkthrough>`
exercises construction, sealing, and launch without live attestation or key
release. It does not qualify hardware, denial behavior, Trustee policy, or a
production profile.

**How do I choose standalone CVM, CoCo, or Azure, and check qualification?**
Use :ref:`cc_choose_deployment` and the baseline table above, then read the
selected mode's validation record. Configuration support and a functional run
do not establish production qualification.

**How do attestation, workload approval, and key authorization differ?**
Attestation appraises evidence from the current environment. Approval records
which exact workload/profile an authority accepts. Resource policy separately
decides whether that appraised environment may receive a particular key. See
:ref:`cc_architecture_overview`.

**Which files are protected?**
Protected guest memory and authenticated encrypted storage depend on the mode.
For standalone CVMs, ``user_config``, ``user_data``, and ``applog`` are clear
host-visible sidecars; keep secrets in the vault. CoCo host logs and external
storage require their own reviewed protection. See :ref:`cc_operations`.

**What happens after a failure, update, or revocation?**
The deployment fails closed for required appraisal or release-policy failures,
but secrets already released cannot be recalled from an authorized running
workload. Updates may require new measurements, approval, references, vaults,
or provisioning. Use the failure and Day-2 matrices in :ref:`cc_operations`.
For detailed CoCo-specific answers, see :ref:`coco_security_faq`.

.. _cc_official_references:

Official component references
-----------------------------

Use these official references to review the platform, services, and tools used
by the workflow. Select documentation matching the installed component versions
and keep the NVFlare deployment's version, policy, and hardware-acceptance
requirements. For CPU/GPU hardware and attestation specifications, see
:ref:`cc_architecture_references`.

.. list-table:: Upstream documentation
   :header-rows: 1
   :widths: 22 40 38

   * - Component
     - Official documentation
     - Review
   * - Confidential Containers
     - `CoCo getting started <https://confidentialcontainers.org/docs/getting-started/>`_
     - Cluster prerequisites and confidential workloads.
   * - Trustee
     - `Trustee installation <https://confidentialcontainers.org/docs/attestation/installation/>`_
       and `Attestation with Trustee <https://confidentialcontainers.org/docs/attestation/>`_
     - Deploy the prerequisite service; review policies, reference values, and resource release.
   * - Kata Containers
     - `Kata 3.29.0 documentation <https://github.com/kata-containers/kata-containers/tree/3.29.0/docs>`_
     - Runtime and guest behavior for the supported Kata version.
   * - NVIDIA GPU Operator
     - `NVIDIA CoCo/Kata GPU deployment (v1.0) <https://docs.nvidia.com/confidential-computing/containers/v1.0.0/confidential-containers-deploy>`_
     - Prepare the prerequisite Kata GPU runtime and GPU Operator for the supported profile.
   * - Kubernetes
     - `RuntimeClass <https://kubernetes.io/docs/concepts/containers/runtime-class/>`_
       and `crictl <https://kubernetes.io/docs/tasks/debug/debug-cluster/crictl/>`_
     - Runtime selection and CRI diagnostics.
   * - Open Policy Agent / Rego
     - `Upgrading Rego to v1 <https://www.openpolicyagent.org/docs/v0-upgrade>`_
     - Rego v1 syntax and migration of existing policies.
   * - Docker
     - `Docker Build <https://docs.docker.com/build/>`_
     - Application-image builds.
   * - Skopeo
     - `Skopeo command documentation <https://github.com/podman-container-tools/skopeo/tree/main/docs>`_
     - Registry login, image copying/inspection, and TLS options.
   * - Cosign
     - `Container signing <https://docs.sigstore.dev/cosign/signing/signing_with_containers/>`_
       and `signature verification <https://docs.sigstore.dev/cosign/verifying/verify/>`_
     - Image-signing keys and signature checks.
