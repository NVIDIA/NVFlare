.. _cc_getting_started:

Getting Started with Confidential Computing
===========================================

Use this guide to choose and prepare a confidential federation before writing
provisioning configuration. Federated Confidential Computing with NVIDIA FLARE
is a **technical preview**: configuration support is not hardware acceptance
for every platform, software revision, or workload.

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
     - :github_nvflare_link:`CoCo collaboration and deployment walkthrough
       <examples/devops/coco/FL-DEPLOYMENT.md>`
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
For CoCo, **full security qualification is incomplete**; the recorded mixed
run does not qualify **TDX+GPU or a protected-server topology**.
The Azure integration currently accepts AMD SEV-SNP with ``gpu_tee: none``.

* For standalone CVMs, review the :github_nvflare_link:`revision-specific
  validation record <nvflare/lighter/cc/image_builder/VALIDATION.md>` and
  :github_nvflare_link:`conformance requirements
  <nvflare/lighter/cc/image_builder/CONFORMANCE.md>`. Prepare the trusted build
  worker, approved generic CVM, and Trustee service described in
  :ref:`cvm_builder`.
* For CoCo, the :github_nvflare_link:`hardware validation record
  <examples/devops/coco/RUNTIME-VARIANTS.md#support-status-and-hardware-validation>`
  reports a mixed TDX CPU-only and SNP+NVIDIA GPU functional run on
  **2026-10-04**, with an ordinary trusted server. The workload performed CPU
  arithmetic, not GPU training. Review the machine prerequisites and
  role-separated handoffs in the
  :github_nvflare_link:`deployment walkthrough
  <examples/devops/coco/FL-DEPLOYMENT.md>`.
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
* Validate the job's executor against the selected deployment. External-process
  execution has failed in sealed CoCo deployments; follow the documented
  :github_nvflare_link:`in-process executor workaround
  <examples/devops/coco/FL-DEPLOYMENT.md#executor-limitation-and-workaround>`.
  Changing code or policy requires a new approved release.
* Keep confidential checkpoints, credentials, and logs within protected storage
  or an explicitly approved secure output path. CVM Builder's ``user_config``,
  ``user_data``, and ``applog`` sidecars are clear storage, not secret stores.
* Define who may receive datasets, individual updates, final models, and logs.
  An encrypted image does not implement dataset authorization or output review.
* Treat GPU driver and VBIOS versions as approved policy inputs. See the
  :github_nvflare_link:`CoCo GPU reference-value procedure
  <examples/devops/coco/service/GPU-REFERENCE-VALUES.md>` before deploying or
  changing a GPU baseline.

5. Provision, launch, and validate
----------------------------------

Once the deployment, prerequisites, and workload approvals are established,
follow :ref:`cc_deployment` for the shared provisioning configuration. Select
``bare_metal_cvm``, ``coco``, or ``azure_cc`` in each protected participant's
``cc_config``, and define shared services in ``cc_project_config``.

Use the chosen mode's guide for building or creating the protected artifacts,
approval, distribution, and launch. Validate the selected platform and workload,
including the required attestation and denial behavior, before approving the
deployment for use. Keep configuration support, successful functional runs, and
security acceptance as separate claims.
