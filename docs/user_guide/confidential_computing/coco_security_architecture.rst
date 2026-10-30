.. _coco_security_architecture:

#####################################
CoCo + NVFlare Security Architecture
#####################################

This is the authoritative security model for the Confidential Containers
(CoCo) integration in ``examples/devops/coco`` and its NVFlare provisioning and
attestation components. It is for workload owners, data owners, federation
administrators, infrastructure operators, and security reviewers. Read it
before deciding which participants or infrastructure providers to trust.

It describes the supported Kata/genpolicy 3.29.0 contract and Trustee appraisal
and resource-authorization policies, with the implementation scope below. It is not a security proof,
certification, or assurance that every supported hardware configuration has
passed end-to-end testing. Reassess these assumptions when changing versions,
policies, applications, or deployment topology. The separate
:ref:`CVM Builder architecture <cc_architecture>` uses a different guest and
storage design; its guarantees must not be imported into this CoCo deployment.

.. important::

   **Implementation scope:** ``2.9`` revision
   `0b2d5dd39 <https://github.com/NVIDIA/NVFlare/commit/0b2d5dd39>`_ includes
   `PR #5344 <https://github.com/NVIDIA/NVFlare/pull/5344>`_, extending the
   SNP-plus-GPU framework to all four explicit SNP/TDX CPU-only/GPU targets.
   This architecture change documents that implementation; it adds no runtime
   support itself.
   The September 29, 2026 real-quote test was rejected because both launch and
   current TCB were ``OutOfDate``; that failure remains part of the record.
   On October 4, 2026, a separate mixed functional deployment passed with an
   ordinary trusted server, a CPU-only TDX client, and an SNP plus NVIDIA GPU
   client. Authorized Kata guests launched encrypted images, obtained release
   resources, registered with required peer proofs, and completed a finite
   job with both clients. The job used CPU arithmetic; GPU appraisal was
   required for the SNP client. That result does not qualify TDX plus GPU,
   sustained renewal, confidentiality denial tests, or the complete negative
   test matrix. See :ref:`acceptance and evidence limits <coco_security_acceptance>`.
   Implemented targets and passing offline tests are not hardware acceptance.

.. contents:: Find a security answer
   :local:
   :depth: 2

.. _coco_security_guarantees:

Security promise and deployment choices
=======================================

The intended guarantee is that an adversarial cluster operator can schedule an
authorized encrypted workload without obtaining its plaintext image or changing
security-relevant launch inputs while retaining access to its protected keys.
That guarantee depends on the authorized hardware, guest software, application,
independent secure services, and correctly enforced policies described below.
It is not a claim that every YAML field is immutable or that the current
implementation has complete field-level coverage. In particular, the launch
sequence below identifies an unresolved exact-image-digest authorization
limitation. The operator can always refuse to run the workload or stop it.

There are **two separate authorization decisions**:

* **Workload authorization:** Trustee's Key Broker Service (KBS) releases
  resources only when attestation and release-specific policy permit it. The
  guest verifies the image and enforces its agent policy before running it.
* **Federation participation:** NVFlare's CCManager and CoCoAuthorizer verify
  proofs from the configured protected participants. This does not replace KBS
  authorization, image protection, application review, or normal FL identity
  authentication.

Encryption protects image layers; it does not hide all image metadata.
Attestation verifies specified claims about a trusted execution environment
(TEE); it does not prove that an application is harmless, that training is
correct, or that released results cannot leak data. An authorized application
can disclose any secret it is allowed to access.

Target profiles and implementation scope
----------------------------------------

.. list-table:: Select each target explicitly
   :header-rows: 1
   :widths: 17 27 22 17 17

   * - Target
     - Kata RuntimeClass
     - Participant CC configuration
     - Required resource-release appraisal
     - Implementation scope
   * - SNP, CPU-only
     - ``kata-qemu-snp``
     - ``cpu_tee: amd_sev_snp``; ``gpu_tee: none``
     - CPU only
     - Implemented in ``2.9``
   * - SNP with NVIDIA GPU
     - ``kata-qemu-nvidia-gpu-snp``
     - ``cpu_tee: amd_sev_snp``; ``gpu_tee: nvidia_cc``
     - CPU and GPU
     - Implemented in ``2.9``
   * - TDX, CPU-only
     - ``kata-qemu-tdx``
     - ``cpu_tee: intel_tdx``; ``gpu_tee: none``
     - CPU only
     - Implemented in ``2.9``
   * - TDX with NVIDIA GPU
     - ``kata-qemu-nvidia-gpu-tdx``
     - ``cpu_tee: intel_tdx``; ``gpu_tee: nvidia_cc``
     - CPU and GPU
     - Implemented in ``2.9``

CPU-only is an intentional deployment mode,
not a fallback after GPU attestation fails. CPU-only profiles need no NVIDIA GPU
or GPU Operator.
Current launch contracts support one application container, no host namespaces
or volumes, omitted CPU/memory resource requests and limits, and zero or one
``nvidia.com/pgpu``. Arbitrary VM sizing, sidecars, persistent volumes, and
multiple GPUs require additional implementation and security review.

Protected clients and protected servers are supported. An ordinary server can
verify CoCo client proofs without running in CoCo; that **does not protect the
server's own memory, aggregation, credentials, or data from its administrator**.
Protecting the server requires its own CC configuration, encrypted release,
authorized TEE launch, and inclusion in the required attested participant set.
Ordinary participants outside that set have no CC attestation requirement.

.. _coco_security_trust_boundaries:

Parties, assets, and trust boundaries
=====================================

The four deployment roles
-------------------------

.. list-table:: Role separation is part of the security model
   :header-rows: 1
   :widths: 24 38 38

   * - Party
     - Responsibilities and authority
     - Trust requirement
   * - Workload owner / NVFlare provisioning node
     - Reviews application code, builds images and signed startup kits,
       encrypts/signs releases, prepares guest policies and handoffs.
     - Trusted with plaintext code, identity credentials, signing keys, and
       release keys. Compromise here can produce an authorized malicious image.
   * - Trusted platform authority / trusted system
     - Reviews the platform baseline and launch artifacts; collects and
       verifies fresh hardware evidence; exports reference profiles.
     - Trusted to choose acceptable software and TCB, not merely copy values
       supplied by the future cluster operator.
   * - Secure-services administrator
     - Operates registry write authorization, Trustee/KBS, Attestation Service
       (AS), Reference Value Provider Service (RVPS), and their policies/keys.
     - Trusted to protect keys, validate handoffs, enforce release rules, and
       retain the platform authority's configuration records.
   * - CoCo IT / cluster operator
     - Schedules delivered Pod manifests on the existing runtime.
     - Untrusted. Controls the host OS, Kubernetes, containerd, hypervisor
       arguments, disks, CNI, DNS, routing, scheduling, and cluster APIs.

Roles may share a machine or organization only when that does not collapse a
required trust boundary. In particular, giving the adversarial cluster owner
control of AS/RVPS/KBS policy or its administrative credentials defeats the
independent key-release decision. A checksum calculated by that owner does not change Trustee's configured
reference or workload authorization.

The data owner and workload publisher also have distinct interests. Protecting
the publisher's code from the data owner does not prove the publisher's code
will preserve the data owner's privacy. Both parties control what code
may consume sensitive inputs and what outputs it may emit.

Architecture
------------

.. figure:: ../../resources/cc-federation-architecture.svg
   :alt: Trusted preparation supplies authorized artifacts and policies. Trustee authorizes resources for the confidential guest inside an untrusted host, while NVFlare peers separately verify identity-bound proofs over TLS.
   :width: 640px
   :align: center

   Four deployment roles and two separate security decisions: workload resource
   release by secure services, and participant verification by NVFlare peers.

For the complete role and service interactions, open the
:download:`detailed CoCo architecture diagram <../../resources/coco-nvflare-security-architecture.svg>`.
In this CoCo deployment, the protected guest is a Kata confidential VM.

Registry storage and network transport are not where image plaintext should
appear on the hostile side. Decryption occurs inside the confidential guest.
Public registry certificates, ciphertext, signatures, and workload policy may
be copied without granting decryption authority.

Trusted computing base
----------------------

The trusted computing base includes the CPU security implementation and
firmware; vendor endorsement roots and verifier implementations; the authorized
guest boot chain, kernel and services; Kata agent policy enforcement; the
attestation agent (AA), confidential data hub (CDH), image verification and
decryption code; NVFlare and the admitted application/dependencies; trusted
build/provisioning systems; and the independently administered secure services.
GPU targets add the confidential GPU implementation and its authorized guest
stack and verification path.

The host OS, Kubernetes admission policies, CNI, host-side seccomp, RuntimeClass
name, installer, and launch script are **not** trusted enforcement mechanisms
against the owner of that host. Their checks prevent mistakes by cooperative
operators. Remote acceptance must rely on evidence and independent policies,
not on the fact that the installer reported success.

Assets and visibility
---------------------

.. list-table:: Secrets are not interchangeable
   :header-rows: 1
   :widths: 25 40 35

   * - Asset
     - Authorized holders / use
     - Cluster operator visibility
   * - Plaintext image and participant startup kit
     - Trusted provisioning storage and the authorized guest. Kits contain
       participant identity material; packaging retains private recovery files.
     - Not delivered. Local image encryption does not erase the builder's
       plaintext, caches, or backups.
   * - Per-release image key
     - Workload owner, authorized secure-services storage, authorized guest.
     - Never put in Pod YAML, Kubernetes Secrets, public images, or logs.
   * - Image-signing private key
     - Workload owner; signs the immutable encrypted-image manifest digest.
     - Public verification key/signature may be visible; private key is not.
   * - AS token-signing private key
     - Secure services; signs attestation results.
     - Only the configured public key goes to peer verifiers.
   * - KBS administrative credential
     - Secure-services administrators; can change protected policy/resources.
     - Never delivered to CoCo IT or workload users merely submitting releases.
   * - TLS private keys and registry publisher credential
     - Endpoint owners; publisher credential goes only to the trusted publisher.
     - Public CA/certificates may be distributed. TLS trust is not the AS
       token-signing public key.
   * - Guest attested private key
     - AA and authorized guest code; used for resource decryption and the
       CoCoAuthorizer possession proof.
     - The raw AA token response must never be exported. This is not a claim
       that application-accessible key material is hardware-nonexportable.
   * - Pod, InitData, policy, digests, manifests, encrypted blobs, reference values
     - Public or nonsecret coordination material; format and consistency are checked.
     - Image/configuration metadata, sizes, timing and resource usage remain
       observable. Do not put secrets in command lines or environment metadata.

.. _coco_security_two_site_example:

Worked example: one model owner and two data owners
===================================================

This example maps the deployment roles to a federation: **model owner M** runs
the FL server, **data owner A** participates as ``site-1``, and **data owner B**
participates as ``site-2``. Each client runs in its own adversarially operated
CoCo cluster. The concrete setup uses the **SNP plus NVIDIA GPU** profile;
consult the implementation-scope qualification above and the
:github_nvflare_link:`runtime-aware FL deployment guide <examples/devops/coco/README.md>`
for other target choices.
It is an architectural deployment recipe, not a supplied end-to-end training
application or a claim that this exact two-site deployment has been tested.

.. figure:: ../../resources/coco-two-site-summary.svg
   :alt: Model owner M supplies reviewed client code to a mutually trusted provisioner. Separate encrypted releases run in hostile clusters for site-1 and site-2, with independent secure services. Each data owner authorizes a user-supplied TLS dataset connector into its guest. Authorized model updates go to M's ordinary trusted server.
   :width: 640px
   :align: center

   Solid paths use the CoCo/NVFlare deployment model. Dashed dataset paths
   require application-specific implementation and authorization; they are
   not supplied by the CoCo packaging scripts.

Open the :download:`detailed two-site architecture diagram
<../../resources/coco-two-site-fl-example.svg>` for the full handoffs and trust
boundaries.

Agree on authority before preparing machines
--------------------------------------------

.. list-table:: FL stakeholders and deployment authority
   :header-rows: 1
   :widths: 24 40 36

   * - Stakeholder
     - Operates or supplies
     - Responsibilities / trust
   * - Model owner M
     - Model, reviewed client/server application, ordinary FL server, and
       trusted FL project-administration console.
     - A and B select the admitted client code, permitted job/model inputs,
       and outputs to M. M's server administrators can see those outputs.
   * - Data owner A / B
     - Its dataset service and storage, separately administered from its
       untrusted compute operator.
     - Each owner selects code that may access its data, dataset scope,
       purpose, recipients, retention, and output policy. An explicitly
       delegated reviewer may act for that owner; CoCo IT does not administer it.
   * - Trusted provisioning operator
     - ``admin/`` image helpers, build/sign/encrypt pipeline, project provisioning,
       and separate client releases.
     - M, A, and B must trust this operator with application plaintext,
       participant credentials, build inputs, signing keys, and image keys.
       M may perform this role only with that agreement.
   * - Trusted platform authority
     - ``trusted_system/`` installed-runtime capture helpers.
     - Reviews guest artifacts, effective launch inputs and firmware/TCB
       baseline; supplies references and launch contracts.
       The future hostile cluster operator is not this authority.
   * - Independent secure-services administrator
     - ``service/`` reference/resource helpers and existing Trustee/KBS/AS/RVPS policy.
     - Acts as a trusted delegate for the affected owners; checks their
       configured requirements before installing platform or workload policy.
       Trustee does not implement an automatic multi-owner vote or quorum.
   * - CoCo IT A / B
     - ``coco/`` launch helper and the existing host/Kubernetes/runtime; launches its
       site's delivered Pod.
     - Receives no plaintext startup kit, image key, registry publishing
       password, or secure-services administration credentials.

Code confidentiality from compute IT must not prevent the data owners' trusted
reviewers from assessing what will consume their data. Image signing proves
publisher authorization, not privacy-preserving behavior. Record the accepted
code/build recipe, dependencies, release identifiers, policies, and allowed
outputs; organization-level consent is not produced by ``nvflare provision``.

The ``admin/`` deployment kit is the **trusted provisioning role**, not the
``admin@example.com`` FL console startup kit. The latter authenticates the FL
project administrator. Neither belongs on an adversarial cluster host.
Trusted roles may be co-located only if all affected owners accept their
combined administrative authority and secret exposure. Separate VMs under one
hostile administrator do not create independent trust.

Machines and inputs to prepare
-------------------------------

.. list-table:: Prerequisites for this SNP plus GPU example
   :header-rows: 1
   :widths: 25 40 35

   * - Machine / operator
     - Required environment
     - Intended final state
   * - Provisioning node / trusted provisioning operator
     - Supported Linux environment with matching NVFlare, existing image/policy
       tools and engine privileges; reviewed application contexts, supplied TLS
       trust material, and registry publishing credentials.
     - Private build workspace, signed kits, per-site encrypted releases and
       handoffs. No cluster administration is needed to package a release.
   * - Trusted platform system / platform authority
     - Trusted control of the selected existing CoCo-ready SNP/GPU worker
       during capture; runtime/QEMU access and configured firmware.
     - Verified fresh CPU evidence and an authorized reference/launch handoff;
       no cluster or runtime installation is part of this workflow.
   * - Secure-services host / independent administrator
     - Existing configured Trustee/KBS/AS/RVPS and registry endpoints, TLS roots,
       administrative credentials, and the canonical current resource-policy
       export used to preserve existing rules during workload registration.
     - Authorized platform references and separate workload release authorizations.
   * - Two site compute hosts / CoCo IT A and B
     - CoCo-ready Kubernetes workers with the supported installed SNP runtime,
       matching captured launch profile, and one allocatable confidential GPU
       per running protected GPU Pod.
     - Existing Kubernetes, authorized Kata runtime and GPU support; one
       confidential NVFlare client at each site. Host software remains untrusted.
   * - FL server and administration host / M
     - Trusted OS and matching NVFlare environment; private signed server and
       FL admin kits; reachable server DNS and authenticated FL transport.
     - Ordinary server verifies both client proofs locally with the pinned AS
       public key; trusted console verifies registration and job results.
   * - Dataset endpoints / A and B independently
     - Owner-controlled storage and authenticated data API, plus a reviewed
       guest connector and recipient authorization mechanism supplied by users.
     - Data goes only to the authorized site's guest; persistence and result
       delivery obey the owner's separate storage/output policy.

Two simultaneous clients in this example consume two allocatable GPUs. A
protected server is optional and requires its own encrypted release and the
logical CC identity ``server`` in the required attested set. Selecting the same
GPU profile for the server requires a third concurrent GPU; a separately
authorized CPU-only server profile needs none. An ordinary server needs no CoCo
runtime to verify client proofs, but its operator remains trusted with
aggregation and permitted updates.

Ensure the independently configured registry and KBS endpoints resolve
**inside each guest** and are reachable through authenticated HTTPS.
Host files are insufficient. The existing registry must permit credential-free
guest image/signature reads while rejecting anonymous writes; publisher
credentials remain on trusted provisioning infrastructure.

The **prebuilt Kata guest** and the **application container** are different
artifacts. The authorized existing Kata 3.29.0 installation supplies the runtime and
guest boot artifacts, including kernel, rootfs and guest services; operators
do not build a new guest OS for each client. M instead supplies reviewed
application Dockerfiles containing matching NVFlare, Python 3.11 or later,
bash/coreutils, required userspace libraries, and authorized training/connector
code. The example Dockerfiles contain deliberately non-runnable placeholder
bases: replace them with reviewed, digest-pinned bases before provisioning.
Provisioning adds the individual client's signed startup kit to its image.

Measurement, packaging and handoff order
----------------------------------------

Start with CoCo-ready Kubernetes workers, the supported Kata confidential
runtime and GPU Operator for GPU targets, and existing configured Trustee
KBS/AS/RVPS plus a TLS registry. Follow
:github_nvflare_link:`examples/devops/coco/README.md` from an NVFlare source
checkout. The workflow has four operational steps and no manual approval phase.

1. **Capture references from the installed runtime.** The platform operator runs
   ``capture-platform.py CONFIG OUTPUT`` on the selected worker under its control.
   The collector is bound to that worker. Capture verifies fresh challenge-bound
   hardware evidence, TCB constraints, and local input consistency, then directly
   writes ``platform-reference-values.json`` and
   ``approved-workload-launch-profile.json``. The existing verifier executables
   come from the operator's trusted tool environment; capture records their
   hashes for local change detection, not initial upstream provenance.
   Prepare a collector on trusted build infrastructure only if one is needed,
   keeping publisher credentials off cluster workers. See
   :github_nvflare_link:`examples/devops/coco/MEASUREMENTS.md`.

2. **Set values in the configured Trustee.** The administrator installs the
   supplied references with the reference helper and receives authenticated API
   readback. CPU/GPU appraisal policies already exist and must implement the
   expected reference and signed-claim contracts. Project participants reference
   one named Trustee service in ``cc_project.yml``; administrative credentials
   belong to the separate service-operation configuration. See
   :github_nvflare_link:`examples/devops/coco/TRUSTEE.md`.

3. **Provision NVFlare images and Pod manifests.** Provisioning signs startup
   kits, builds and encrypts participant images, signs and publishes their
   immutable digests, and generates guest policies/InitData, Pod YAMLs, and
   private Trustee release handoffs. The image runner is noninteractive.
   It does not configure Trustee automatically. See
   :github_nvflare_link:`examples/devops/coco/provision/README.md`.

4. **Register releases and launch Pods.** The administrator passes each received
   private six-file handoff directly to the installer. It installs resources
   first and the merged KBS policy last, preserving other releases. CoCo IT
   receives only its final Pod and invokes the launch helper, which validates,
   performs a server dry run, applies, and waits. The trusted FL administrator
   checks registration and required peer proofs/CC validation. Pod readiness
   alone does not establish NVFlare application readiness. See
   :github_nvflare_link:`examples/devops/coco/LAUNCH.md`.

Provision two separate client releases
--------------------------------------

Start from the complete ``provision/project.yaml``. Retain its server and FL
admin participants, builder order and common ``CCPackager``; replace the placeholder
server/admin names with the actual identities and configure reachable server
DNS/port before provisioning. Replace its one-client participant entry
with these two entries within ``participants`` (this fragment is not a complete
project file):

.. code-block:: yaml

   - name: site-1
     type: client
     org: data-owner-a
     cc_config: cc_site-1.yml
   - name: site-2
     type: client
     org: data-owner-b
     cc_config: cc_site-2.yml

Copy/adapt ``cc_site-1.yml`` to ``cc_site-2.yml`` and supply the reviewed
application contexts. Keep ``cc_deployment_mode: coco``, select the required
``cpu_tee`` and ``gpu_tee``, and let the participant ``type`` supply the role.
Use these distinct packaging values:

.. list-table:: Per-client packaging inputs
   :header-rows: 1
   :widths: 34 33 33

   * - Field
     - ``cc_site-1.yml``
     - ``cc_site-2.yml``
   * - ``workload.source.context``
     - ``./site-1``
     - ``./site-2``
   * - ``workload.source.dockerfile``
     - ``Dockerfile``
     - ``Dockerfile``
   * - ``coco.release_name``
     - ``site-1-v1``
     - ``site-2-v1``
   * - ``coco.registry_repository``
     - ``workloads/site-1``
     - ``workloads/site-2``

Both participants select the same named Trustee service from ``cc_project.yml``.
Each ``coco.platform_config_file`` must reference a
prepared configuration for its destination in the source example's admin
helper directory. Use the matching captured launch profile and cluster service
settings for each target. Use reviewed, fully
qualified ``class_allow_list`` entries for admitted
application components. Do not upload executable code through a job to avoid
rebuilding: CCManager disables bring-your-own-code jobs.

From the source example's ``provision/`` directory in the matching NVFlare
environment, run::

   nvflare provision -p project.yaml

The packager invokes the image builder for each protected participant,
and publishes it without an interactive review or approval gate.
It produces separate participant kits, encrypted/signed images, InitData and
policies, secure-services handoffs and Pod manifests. The same authorized base
code may be reused, but the sites must not share a packaged identity or image
key. Code, dependency, startup-kit or security-policy changes require a fresh
release; measured launch-input changes also need fresh platform capture. Protect the ordinary server and FL admin kits separately.

Dataset access, training outputs and persistence
------------------------------------------------

**The CoCo package does not implement the dataset path.** A suitable design for
this example is a guest-initiated, end-to-end TLS/mTLS connection from each
client to its own owner's data service, with the connector baked into the
reviewed image. A supplies only A's authorized records to site-1; B supplies
only B's authorized records to site-2. Keep the data services, authorization
secrets and storage outside the hostile compute administrators' control.

Before sending data, the owner-controlled service must authenticate the
recipient and authorize its site, authorized workload, dataset scope and expiry.
A participant certificate proves an identity, not that authorized code is
executing. CoCoAuthorizer's peer proof is also **not an exact-image workload
authorization or a dataset ACL**. Implement and review the service's proof and
credential-delivery protocol; do not export the raw AA response or its private
key, and do not put dataset credentials in public Pod YAML or host-managed
Kubernetes Secrets.

.. important::

   Strict release of data **only to the authorized application** remains a
   deployment gate: establish the actual-image authorization and guest
   boot/rootfs integrity coverage discussed in
   :ref:`the Pod launch analysis <coco_security_pod_launch>`, and test the
   resulting recipient authorization. The current exact-image-digest gap
   must not be hidden by the diagram or by a successful peer TEE proof. Until
   these requirements are met, this example is not a demonstrated confidential
   dataset-delivery solution for that threat model.

Do not use ``hostPath`` or PVC mounts for the dataset: the current packager
rejects volumes. Consume plaintext in private guest memory, or in storage with
separately demonstrated confidentiality and integrity; being named a guest
filesystem does not establish that a host-backed disk is protected. The
provided writable workload state is ephemeral, not a durable encrypted data
or checkpoint service.

For persistence, users must implement a reviewed result/checkpoint protocol to
owner-controlled network storage, including authentication, key custody,
retention and rollback/replay handling. If storage administrators are also
untrusted, encrypt and authenticate data inside the guest before upload and
keep decryption authority outside their control. Neither durable confidential
storage nor freshness protection is supplied automatically by CoCo.

During training M sends the authorized model/configuration; clients compute over
their authorized inputs and return permitted model updates through NVFlare's
authenticated transport. Keeping raw data site-local is an **application and
output-policy requirement**, not a consequence of attestation alone. Updates
can disclose training information. This setup does not automatically provide
differential privacy, secure aggregation, or protection from malicious model
inputs. A and B control what M may receive; M's ordinary server sees those
updates in plaintext. Review and validate these application-level controls
before enabling sensitive training.

.. _coco_security_attack_surface:

Attack-surface inventory
========================

The **attack surface** is the set of interfaces and data-processing paths
through which an actor can influence trusted execution or reach sensitive
assets. It includes more than listening network ports: boot inputs, agent
requests, image and evidence parsers, administrative handoffs, local APIs,
application inputs, and output/storage channels all cross relevant boundaries.

This inventory maps each surface to its possible caller, protected assets,
enforcement point, and remaining risk. It is an architecture-level inventory,
not a complete list of kernel interfaces, device operations, parser bugs, or
currently open listeners. Verify the actual deployment's reachability and
versions. A stated deployment obligation is not a claim that the scripts have
implemented or tested every necessary control. The release-scope qualification
above also applies here.

Authentication, attestation, and default-deny authorization do not eliminate
the code that parses a request before deciding whether to accept it. Malformed
input and resource exhaustion remain concerns even when no secret is released.
For individual attacks and their outcomes, see
:ref:`Threats, enforcement, and residual risks <coco_security_threats>`.

Host-controlled and guest-local interfaces
------------------------------------------

.. list-table:: Interfaces reaching the confidential guest
   :header-rows: 1
   :widths: 30 35 35

   * - Surface and possible caller
     - Assets and enforcement boundary
     - Residual risk / deployment obligation
   * - **Pod, runtime, and launch configuration.** CoCo IT controls Kubernetes,
       CRI requests, runtime selection, boot inputs, InitData delivery, and
       effective container configuration.
     - **Assets:** authorized guest and workload execution, release keys.
       Authorized measurements, independent expected InitData, and effective
       guest-policy checks constrain selected inputs; host validation is not
       the adversarial boundary.
     - Not every YAML field is bound. Equivalent or unconstrained changes may
       pass, and the host can prevent execution. Use the field-change tables
       and demonstrate integrity coverage for external guest rootfs inputs.
   * - **Host-to-guest agent requests and virtual I/O.** The host/VMM supplies
       agent requests, virtual-device responses, and shared-I/O inputs to the
       guest, regardless of Kubernetes admission controls.
     - **Assets:** guest kernel/services and private CPU/GPU state. The authorized
       guest stack and TEE enforce the boundary; agent rules constrain
       effective requests and deny interactive exec/stream and policy-replacement
       paths. Permitted lifecycle operations still exist.
     - Agent parsers, guest drivers, and allowed request handling remain trusted
       attack surface. Attesting their identity does not prove absence of
       vulnerabilities. Review enabled devices/interfaces and shared data;
       do not claim guest seccomp, universal side-channel protection, or host
       denial-of-service resistance.
   * - **Registry responses and image artifacts.** The guest processes
       manifests, configuration, signatures, and encrypted layers supplied
       through the registry; publishers and compromised infrastructure can
       affect available content, while the host controls transport.
     - **Assets:** image confidentiality and guest execution integrity.
       Configured TLS trust, guest signature/integrity/decryption checks, and
       independent KBS authorization are the enforcement points. Anonymous
       ciphertext pulls do not grant publishing or decryption authority.
     - Image parsers/unpackers remain trusted code. Another accepted-key,
       same-repository signed image is subject to the documented unresolved
       exact-digest authorization limitation. Review image content and parser
       dependencies; encryption alone neither hides all metadata nor prevents
       substitution attempts or exhaustion.
   * - **Guest-local AA/CDH interfaces.** Admitted guest processes deliberately
       use token/resource services. The reviewed REST configuration enables
       guest-component APIs, including AA's ``/aa/token``; CDH also has a
       guest-local socket interface.
     - **Assets:** raw AA response and its guest private key, EARs, and released
       resources. CoCoAuthorizer restricts its AA call to guest loopback,
       disables proxies/redirects, and bounds the response. KBS separately
       authorizes protected resource requests.
     - These caller checks are not isolation from a compromised application
       inside the guest. Review all enabled REST/socket endpoints and their
       reachability, not only the URL used by CoCoAuthorizer. Do not export
       them through a Service, host port, proxy, or diagnostic output; do not
       assume a host-controlled network policy protects them.

Secure services and trusted-side interfaces
-------------------------------------------

.. list-table:: Interfaces capable of changing trust or releasing resources
   :header-rows: 1
   :widths: 30 35 35

   * - Surface and possible caller
     - Assets and enforcement boundary
     - Residual risk / deployment obligation
   * - **Build/provisioning inputs and handoff import.** Workload owners and
       platform authorities supply source, dependencies, project configuration,
       startup kits, evidence, references, and release files; supply-chain
       producers influence those inputs.
     - **Assets:** plaintext images, FL credentials, signing/publishing keys,
       image keys, and configuration records. The importer consumes received handoff
       files directly, validates their structure, and reconstructs policy from a service-owned
       template instead of accepting arbitrary supplied Rego.
     - Trusted code execution and build caches remain sensitive. An attacker
       controlling a trusted build or configuration authority can authorize malicious
       content. Review dependencies and imported files, and protect retained
       plaintext/recovery data.
   * - **KBS attestation and resource HTTPS requests.** Any client that can
       reach the endpoint can attempt protocol requests; it need not already
       be an authorized confidential workload.
     - **Assets:** per-release keys and verification policy/material. AS
       appraisal, session/guest-key binding, and KBS's exact resource-path and
       workload authorization gate release. Authorized responses are encrypted
       to the attested guest key, not to an identity asserted by the host.
     - TLS endpoint authentication is not resource authorization. HTTP,
       evidence, and policy processing still require security review and
       operational capacity limits. A successful appraisal grants no store-wide
       access; a permissive resource policy can defeat release isolation.
   * - **AS/RVPS service calls and reference ingestion.** KBS forwards evidence
       for appraisal; trusted administrators install references and policies.
       Hostile evidence therefore reaches verifier code even when backends are
       not directly reachable from the cluster.
     - **Assets:** appraisal integrity, AS signing authority, reference values,
       and TCB floors. Evidence verification, configured baselines,
       service-owned appraisal policy, and controlled administration form the
       trusted boundary.
     - Keep backend interfaces restricted and verify actual listeners/routes;
       do not infer isolation from component names. Verifier flaws, compromised
       reference ingestion, or administrator mistakes can admit an
       unacceptable guest. Verify persisted references and policy after changes.
   * - **Administrative APIs, publishing, and host management.** Secure-services
       administrators and authorized registry publishers hold distinct
       credentials; remote clients may still reach administrative request
       parsers on the public HTTPS front end.
     - **Assets:** KBS resources/policies, AS/TLS keys, registry content, and
       backups. The existing Trustee configuration enforces administrative
       authentication and authorization; this example consumes opaque credentials
       and does not validate their audience. Registry publishing uses separate
       authorization. OS/container administrators remain trusted.
     - The TLS proxy forwards KBS routes; administrative routes are not
       guaranteed a separate private listener. Protect credentials and
       management access independently of CoCo IT. A valid privileged identity
       can change trust decisions; attestation cannot veto that administrator.
   * - **Vendor evidence, endorsement, and collateral inputs.** Verifiers use
       hardware evidence, vendor-issued certificates/reference material,
       appraisal responses, and possibly locally cached copies.
     - **Assets:** platform identity, freshness, and accepted security baseline.
       Vendor trust roots and verifier checks establish authenticity;
       configured AS/RVPS policy decides acceptability. The
       CPU/GPU-specific verification sections describe the supported paths.
     - Vendor services, verifier dependencies, cache contents, freshness, and
       availability remain part of the trust/operations review. Valid signatures
       alone do not satisfy a configured TCB. Do not relax checks or accept stale material
       solely to restore connectivity.

Federation, application, and output interfaces
----------------------------------------------

.. list-table:: Interfaces still relevant after successful attestation
   :header-rows: 1
   :widths: 30 35 35

   * - Surface and possible caller
     - Assets and enforcement boundary
     - Residual risk / deployment obligation
   * - **FL transport and peer-proof verification.** Network actors can disrupt
       or redirect connections; authenticated peers supply EARs and signed
       proofs that the receiving participant must parse and verify.
     - **Assets:** participant identity and federation admission. End-to-end FL
       mTLS, pinned AS trust, guest-key proof verification, expected
       site/audience, and time/replay checks enforce separate transport and
       attestation boundaries. Use site-aware verification, not merely
       compatibility ``verify(token)``.
     - Authentication does not make a peer's inputs harmless. Cached EARs,
       process-local replay state, clocks, verifier restarts, and replicas limit
       freshness/revocation guarantees. Host CNI, source IP, and a successful
       local installer are not independent identity authorities.
   * - **Application/job inputs and FL administration.** Admitted participants,
       data sources, and authorized FL administrators supply application
       requests, datasets, model updates, and permitted job configuration.
     - **Assets:** training data, model/code confidentiality, computation, and
       guest-accessible credentials. Application authorization, reviewed code
       and dependencies, and CCManager's job-code restrictions complement
       attestation. Administrator connections are not attested as worker
       participants by this CCManager path.
     - Allowed component classes are trusted code, not a sandbox. A vulnerable
       or malicious admitted application can expose its keys/data, offer an
       interactive endpoint, or misuse valid proofs. Review actual listeners,
       input handling, administrator privileges, and permitted outputs; TEE
       identity does not establish application benevolence or training privacy.
   * - **Logs, results, writable storage, and checkpoints.** Authorized code emits
       outputs; recipients and any host-backed sink may observe them. CoCo IT
       controls host storage, traffic metadata, scheduling, and restart/replay
       of public launch inputs.
     - **Assets:** guest files, secrets, derived data, and persistent state.
       Guest-private state, denied agent streams, silent generated startup,
       and authenticated application channels protect specific paths.
       Persistent storage needs its own confidentiality/integrity and key
       management design.
     - Console redirection does not stop alternate output channels or disable
       Kubernetes logs. Image encryption does not encrypt arbitrary writable
       volumes or prevent rollback, duplicate execution, or inference from
       released results. Configure each output/storage destination.

Using the inventory during deployment review
--------------------------------------------

For each row, record the actual endpoint or local interface, listener/bind
scope, reachable callers, required credentials, owning trusted authority,
software/policy revision, and positive and negative test evidence. Include
indirect reachability: a private AS can still parse evidence forwarded by KBS,
and a guest-local API can become exposed by an application proxy. Record the
result of each check rather than treating this table as a completed audit.

Remove unnecessary listeners, debug services, and API forwarding. Review any
new device, mount, sidecar, application service, dependency, or administrative
route as a change to the attack surface. Where a measured launch input or
workload authorization changes, obtain new authorized references/releases as
required. These are review obligations, not additional protections introduced
by this documentation.

.. _coco_security_platform_approval:

Capturing verified platform references
============================================

A **measurement** is a digest of defined launch inputs or measured events, not
a generic hash of everything on a machine. **Evidence** is the hardware-backed
report/quote and supporting data used to verify those claims. A **reference**
is an configured expected value. A valid signature establishes
origin/integrity under a trust root; it does not decide whether a measured
program or firmware version is acceptable.

The platform operator runs ``capture-platform.py CONFIG OUTPUT`` on the selected
existing worker. A short-lived collector obtains a fresh challenge-bound
report/quote inside the confidential VM. Capture records the installed
configuration and actual launch, verifies hardware evidence using the supplied
verifier, checks the configured TCB/security constraints and local input hashes,
and writes ``capture-summary.json`` plus both handoff files directly.
There is no separate approval command, offline SNP model, upstream Kata image
download, or second collection phase.

Preinstalled verifier executables are supplied by the operator's trusted tool
environment. Capture records their hashes to detect later local changes; it does
not establish the initial upstream origin of those executables. Cryptographic
verification still requires the AMD chain/report signature or Intel quote/
collateral, fresh challenge binding, and the selected security baseline.

The provisioning node coordinates two distinct direct file handoffs:

* ``platform-reference-values.json`` goes to secure services for RVPS
  reference updates under the existing AS policies.
* ``approved-workload-launch-profile.json`` goes to the workload owner as
  a target-aware authorized contract. Its SNP/TDX and CPU/GPU selection must match
  the configured profile; editing its fields cannot select another target.
  The contract constrains trusted-side generation and records captured runtime
  settings, launch shape, application security context, and guest token-API
  capability. It is not itself an attestation claim.

CoCo IT uses the existing authorized runtime and receives only final Pod YAML.
It has no authority to update reference values. Handoff files are accepted
as supplied; format checks and locally recorded checksums do not automatically
establish a sender's identity or the transferred file's origin. Trusted recipients
are responsible for supplying the intended files. Replacement of an entire
handoff and its checksum manifest can authorize a different workload; the direct
handoff workflow does not detect that substitution. Optional independently known
digests can check supplied bytes, but are not required by this workflow.

SNP verification
----------------

For the usual VCEK-based SNP path, AMD firmware produces the signed guest
report. The verifier validates the endorsement chain and report binding; AMD's
Key Distribution Service provides chip/TCB-specific endorsement certificates.
An Intel-style host QGS is not needed for this path. Certificate retrieval may
be online or use a separately managed cache; successful retrieval alone does not establish acceptable TCB.

The service policy checks membership in ``snp_launch_measurement`` and four
configured minimum reported-TCB values: bootloader, TEE, SNP
firmware, and microcode. Missing/nonnumeric floor data cannot pass. Debug and
migration-agent permission must be disabled. Multiple authorized measurements
share these four floors; the current SNP format does not associate different
TCB minima with each measurement. Never choose permissive floors solely from
the platform under test.

TDX verification
----------------

The collection/export and RVPS/AS policy implementation is included in
``2.9``. Evaluate acceptance for the selected target, runtime revision, and
deployment using the recorded :ref:`acceptance evidence and limits
<coco_security_acceptance>`.

A TDX guest obtains a TD report; a Quote Generation Service on the same physical
platform uses Intel's quoting infrastructure to produce a remotely verifiable
quote. PCK certification and verification collateral support this trust chain.
Platform provisioning/certificate caching and quote generation are separate
from application authorization. The NVFlare participant does not need a PCS API key
to call the guest AA API. The host's chosen provisioning workflow may need one.

The trusted collector and service verifier/policy require a valid quote and
collateral, challenge/InitData binding, non-debug mode, acceptable ``UpToDate``
TCB, and replay of the measured-boot log. The current-TCB result, when supplied,
must also be acceptable; expired collateral is rejected. An open QGS socket,
booted TDX VM, decoded quote, or valid signature alone is insufficient.

RVPS stores authorized profiles under ``coco_tdx_profiles_v2``. Each profile is
one complete tuple:

.. code-block:: text

   mr_td, rtmr_0, rtmr_1, rtmr_2, rtmr_3, xfam,
   tdvfkernel, tdvfkernelparams

MRTD covers the initial TD measurement. RTMRs accumulate measurements through
extend operations; the event log explains the events whose replay must agree
with the quoted registers. Their contents depend on the authorized boot and
runtime measurement sequence. All four RTMRs must match the same authorized
profile, together with XFAM and the kernel/kernel-parameter event digests.
Combining individually authorized fields from different profiles is forbidden.

Collection and workload appraisal need a stable authorized
attestation phase. Runtime extensions to RTMR3 cannot be handled by ignoring
it, assuming it is always zero, or accepting arbitrary values. Review profiles
for intended states or redesign the measurement sequence.

GPU appraisal and downgrade prevention
--------------------------------------

GPU profiles require a supported GPU in production confidential-computing mode
and NVIDIA appraisal through the deployed Trustee verification path. GPU
allocation, a successful CUDA command, or a host-reported CC flag is not remote
attestation. NVIDIA Remote Attestation Service (NRAS) appraises GPU evidence
using Reference Integrity Manifest (RIM) and certificate-status services.
Vendor evidence appraisal checks the GPU's reported state against
the applicable signed reference material; it does not make arbitrary driver
versions or future releases automatically acceptable to an organization.

CPU-only release rules require exactly
``cpu0``. GPU release rules require exactly ``cpu0`` and ``gpu0`` and
successful appraisals of both. A failed or
missing GPU cannot silently downgrade a GPU-required release. Conversely,
installing a GPU verifier does not make GPU evidence mandatory for every
workload. Unified provisioning derives the peer verifier's internal GPU
requirement from each participant's ``gpu_tee``. ``nvidia_cc`` requires signed
CPU and GPU appraisals; ``none`` requires CPU-only evidence. Operators do not
repeat an internal ``gpu_required`` boolean in ``workload_constraints``. Peer
verification remains separate from the release's KBS CPU/GPU requirements.

Measurement coverage must be demonstrated
-----------------------------------------

Identical CPU model, runtime name, container image, or Pod YAML does not by
itself guarantee identical launch measurements. The effective firmware,
kernel, boot arguments, VM configuration, and measured event sequence matter.
Changes require review and a new capture as appropriate rather than weakening
reference matching.

A trusted-host file hash proves which file was reviewed there; it does not
remotely prove what an adversarial host supplied later. In particular, do not
claim that an external block-device root filesystem is protected merely
because capture recorded its hash. The authorized boot path must demonstrate
that security-critical guest content is covered by hardware measurements or
cryptographic integrity verification rooted in measured code. Measured
firmware/kernel/initrd and externally supplied disk content are not
interchangeable. Treat missing proof of that binding as an unresolved
deployment acceptance requirement, not an assumed guarantee.

.. _coco_security_workload_release:

From provisioning to authorized execution
=========================================

Preparing an immutable release
------------------------------

The trusted workload owner runs ``nvflare provision -p project.yaml`` with
explicit CC settings for each protected participant. The CoCo builder generates
CCManager/CoCoAuthorizer configuration and prepares the startup kit before
signature generation. The packager incorporates that signed kit into a reviewed
container build and creates a separate protected release for each participant.

The publication pipeline encrypts every workload image layer, resolves the
immutable encrypted-image manifest digest, signs that digest, and generates
the image signature policy and strict Kata agent policy. InitData carries the
guest trust/configuration and agent policy. The pipeline generates release-specific workload authorization and a
confidential six-file handoff for secure services. It contains the image key, public image-signing key, image
policy, release authorization, resource-policy fragment, and checksum manifest.

The secure-services administrator reads the received handoff directly and
reconstructs and checks the release policy using the service-owned template, rather
than trusting arbitrary workload-supplied Rego. The administrator installs the
resources and reviewed rule into a default-deny global resource policy. Only
then does CoCo IT receive the Pod YAML for direct launch-helper consumption.

Plaintext startup kits, build contexts, Docker caches, signing material, and
recovery outputs on the provisioning node remain sensitive. Packaging is not
secure erasure or a transactional publication protocol. A failed build/push
can leave private local artifacts or partially published ciphertext; investigate
before retrying and retain the private recovery state.

What InitData binds
-------------------

The Pod's ``io.katacontainers.config.hypervisor.cc_init_data`` annotation carries
encoded/compressed InitData. Its confidentiality is not required. The security
property is that the exact InitData digest is bound into CPU evidence and
compared against independent release authorization:

* SNP uses a 32-byte SHA-256 digest bound through ``HOST_DATA``.
* TDX uses that SHA-256 digest followed by 16 zero bytes in the 48-byte
  ``MRCONFIGID``. Nonzero padding or arbitrary truncation is not accepted.

This is **attestation-bound InitData**, not a separately signed InitData file.
It commits to the guest agent policy and guest configuration, including KBS
endpoint/trust and image-policy location. Image digest and process arguments
are present in the authorized policy. The CPU does not directly hash the later
decrypted application image and magically certify its behavior; trusted guest
verification and enforcement connect that application to the authorized policy.

Changing the decoded embedded policy or trust configuration changes the
expected binding. Editing YAML is possible: changes outside the permitted
effective guest configuration are intended to fail at the relevant boundary.
The field-change tables below distinguish implemented checks, allowed changes
and unresolved coverage. Unmeasured metadata changes do not necessarily change
CPU measurements, and re-encoding identical InitData bytes need not change its
digest.

The key-release decision
------------------------

During a workload launch:

1. In the intended launch, the guest fetches the encrypted image by immutable
   digest and establishes the KBS attestation exchange using its configured
   TLS trust. The actual-image authorization limitation is detailed below.
2. CPU evidence, and GPU evidence when required, binds the attestation session
   and guest public key. AS verifies evidence and evaluates its appraisal
   policy against authorized RVPS references; it signs the attestation result.
3. For **each** resource request, KBS evaluates the actual plugin/resource path,
   exact required submodules/vectors, authorized CPU type, expected InitData,
   image identity and process arguments extracted from the attested policy.
   KBS does not inspect the host's live image-pull or process-creation request.
4. Only an authorized request receives the resource encrypted to the attested
   guest key. The authorized guest image-verification/decryption path and agent
   policy must then permit execution. KBS acceptance alone is not proof that
   the application started successfully.

The three allowed resource paths are release-specific:

.. code-block:: text

   default/image-key/<release>
   default/sig-public-key/<release>
   default/security-policy/<release>

Knowing or guessing another path does not authorize it. A successful platform
appraisal is not permission to read the whole KBS store. Although signing
public keys and image policy are not secrets, controlling their integrity is
necessary; the service applies the release authorization to all three resources.

The exact accepted EAR trust vectors for this policy are:

.. list-table:: Policy contract, not a universal confidence scale
   :header-rows: 1
   :widths: 28 24 24 24

   * - Appraisal
     - ``executables``
     - ``hardware``
     - ``configuration``
   * - SNP CPU
     - 3
     - 2
     - 2
   * - TDX CPU
     - 3
     - 2
     - 2
   * - NVIDIA GPU
     - 3
     - 2
     - 2

These vectors contain exactly the three listed keys. Unappraised AR4SI
categories are omitted rather than asserted with a zero value. The checks
require exact vectors, not "at least" a numeric score. In Rego,
``configuration := 2 if { ... }`` assigns that claim when its conditions hold;
it neither counts conditions nor authorizes a resource by itself.

.. _coco_security_pod_launch:

Pod launch sequence and four outcomes
-------------------------------------

The following diagram starts at ``kubectl apply -f pod.yaml``. Platform
references, the confidential release resources, and service-owned policies
must already be installed. RuntimeClass selects a container-runtime handler;
it is not a separate service and its name is not a hardware-attested identity.
The host-side API, admission, launch-script and checksum checks are bypassable
by CoCo IT and are not the security boundary against that operator.

.. figure:: ../../resources/coco-launch-summary.svg
   :alt: Five conceptual checkpoints: guest launch, policy and encrypted artifacts, hardware appraisal, resource authorization, and guest checks before workload startup. Failed checkpoints stop that launch path; exact image-digest binding remains a documented limitation.
   :width: 640px
   :align: center

   Read the checkpoints from top to bottom. These are conceptual dependencies,
   not a literal packet trace. Guest checks, image handling, and resource
   requests can interleave or repeat. The table below distinguishes startup
   failure, key-release denial, guest-request denial, and an allowed launch.

Open the :download:`full-size vector diagram <../../resources/coco-pod-launch-sequence.svg>`
for the detailed actor-by-actor interactions. The
:download:`editable Mermaid source <../../resources/coco-pod-launch-sequence.mmd>`
retains the full logical trace of the same actors, checkpoints, outcomes and
coverage limitation; the SVG groups it into narrower phases for readability.

.. list-table:: Four outcomes, not one universal attestation failure
   :header-rows: 1
   :widths: 23 42 35

   * - Outcome
     - Meaning
     - Security interpretation
   * - 1. Startup failure
     - No usable handler, unschedulable Pod, failed VM boot, unavailable registry/service, or another operational failure.
     - No successful workload launch. A host-reported failure is not evidence that a security policy enforced the boundary.
   * - 2. Key-release denial
     - Evidence verification/appraisal or KBS authorization fails: wrong platform, required GPU, InitData, resource path or attested policy claims.
     - The protected resource is not released. The confidential VM may already have booted.
   * - 3. Guest-request denial
     - Guest image signature/integrity/decryption checks or agent checks of the effective container/process request fail.
     - The prohibited request cannot complete. KBS might already have released resources inside the authorized guest; this does not expose them to IT.
   * - 4. Allowed change / successful launch
     - The effective launch satisfies the implemented checks despite a changed, unconstrained or equivalent YAML value. The unchanged authorized launch follows this path too.
     - The workload can run. This does not certify every YAML field, prove exactly-once execution, or close the image-digest coverage limitation below.

These are expected outcomes from source inspection, not a completed live
mutation-test suite. Test the enforcing guest/service boundary, not only the
cooperative launch script. ``Running`` status on the hostile cluster is not
trusted proof that the intended workload is executing.

RuntimeClass and guest-boot changes
-----------------------------------

.. list-table:: Runtime selection is not runtime-name attestation
   :header-rows: 1
   :widths: 31 40 29

   * - YAML/runtime change
     - Checkpoint
     - Expected outcome
   * - Nonexistent RuntimeClass or unavailable handler
     - Kubernetes/container runtime cannot create the sandbox using that configuration.
     - 1: startup failure before attestation, under normal cluster behavior.
   * - Ordinary, nonconfidential runtime
     - It cannot supply the required confidential CPU evidence for the protected release.
     - 2: no protected image key. An unrelated unencrypted container can still run.
   * - Switch an SNP-authorized release to TDX, or vice versa
     - KBS requires the release's CPU evidence type and exact authorized trust vectors.
     - 2: denial, even if the other platform passes its own appraisal.
   * - Switch a GPU-required workload to CPU-only
     - KBS requires the exact CPU/GPU submodule set and GPU appraisal; guest device constraints also apply.
     - 2: missing required GPU evidence denies the key; an earlier guest/runtime failure is also possible.
   * - Different class name/handler producing the same authorized guest
     - KBS checks evidence, InitData and workload authorization, not ``runtimeClassName``.
     - 4: can pass after IT bypasses exact-name preflight. A different name alone is not a violation of the attested guest boundary.
   * - Kernel, firmware or other guest-boot overrides
     - AS compares covered measured inputs with authorized references; the guest boot path must authenticate other security-critical content.
     - 2 for unauthorized measured inputs, or 1 for boot failure. Ignored/unmeasured changes have no automatic remote denial; establish external disk/rootfs integrity separately.

Image changes and the current authorization limitation
-------------------------------------------------------

The intended image controls combine signatures, digest/integrity checks,
encrypted layers, release-scoped keys and guest policy. They must not be
described as a universal comparison of the requested image with the authorized
digest in the current implementation:

* KBS compares the image in the attested InitData policy, not the host's actual
  pull request. AS's validated image identifier is derived from that policy.
* The pinned guest Rego allows ``io.kubernetes.cri.*`` annotation keys without
  a generic equality check on their values. Its ``image_guest_pull`` storage
  rule checks the rootfs destination but leaves ``source`` and
  ``driver_options`` validation incomplete.
* The generated image signature policy uses ``sigstoreSigned`` with
  ``signedIdentity: matchRepository``. That is not an independent allowlist of
  exactly one authorized manifest digest.

The provisioner creates a separate encryption key and resource path for each
immutable release. For independently provisioned images A and B with different
keys, A's key cannot decrypt B's encryption material. Requesting B's key while
retaining A's InitData must fail B's exact resource authorization rule. Changing
InitData invokes the other release's independently installed authorization;
it does not inherit A's release authorization. Repository-scoped signature acceptance alone
does not bypass these controls. Registry write access also does not supply a
signing key or a decryption key.

.. warning::

   Exact-image-digest authorization is an unresolved coverage limitation.
   A direct comparison of the effective image digest with the authorized digest
   has not been demonstrated. Any proposed replacement must still have an
   accepted signature, be decryptable using authorized resources, and satisfy
   every other guest check. The missing comparison is a source-review finding,
   not a demonstrated bypass of per-release encryption or a successful
   substitution/key-extraction attack. Add an explicit effective-image binding
   and a negative test before
   claiming that every image-field substitution is blocked.

.. list-table:: Image substitutions have different failure paths
   :header-rows: 1
   :widths: 34 38 28

   * - Image change
     - Checkpoint
     - Expected outcome
   * - Unsigned image or unauthorized signing key
     - Guest image signature policy.
     - 3: verification should reject the image.
   * - Corrupted signed manifest or encrypted layer
     - Guest signature, digest and authenticated-decryption checks.
     - 3: verification/decryption should fail; missing blobs can instead cause 1.
   * - Image requiring an unauthorized key path
     - KBS checks the actual requested release-specific resource path.
     - 2: resource denied.
   * - Another image signed by the accepted key in the permitted repository
     - Current signature policy is repository-scoped; KBS's image claim is policy-derived, not a direct observation of that pull.
     - Unresolved: do not promise 3. It could reach 4 if the other checks pass; exact-digest authorization and a negative test are required.

Other Pod fields and effective guest checks
-------------------------------------------

The provisioning validators allow a much narrower manifest shape than the
complete Kubernetes Pod API. They reject unsupported additions before handoff,
but IT can modify the final file or bypass the launcher. The table therefore
describes what the independent guest/service checks cover, not just what
``validate_workload_pod`` accepts. Checks operate on effective guest requests,
which are not identical to literal YAML fields.

.. list-table:: Field changes after host-side preflight is bypassed
   :header-rows: 1
   :widths: 27 45 28

   * - Field or change
     - Actual checkpoint / limitation
     - Possible outcome
   * - ``cc_init_data`` policy, endpoint, CA or image-policy location
     - Guest validates the InitData hardware binding; KBS expects the configured digest of the decoded data.
     - 2 if decoded data changes; guest validation can also stop startup. Re-encoding identical bytes can reach 4.
   * - ``command`` / effective arguments
     - Guest agent compares argument count and values against the authorized rules, including permitted substitutions.
     - 3 for an unauthorized process; 4 if the same or explicitly permitted effective process results.
   * - ``env``
     - Each supplied value must match guest rules. The pin does not require every authorized variable to remain present; ordering and some substitutions are permitted.
     - 3 for disallowed supplied values; removals/reordering or allowed substitutions can reach 4.
   * - ``runAsUser`` / ``runAsGroup``
     - Guest checks effective UID/GID. Supplementary groups may be a subset of authorized groups, not arbitrary additions.
     - 3 for an unauthorized identity; 4 for equivalent/permitted identity settings.
   * - Capabilities, ``allowPrivilegeEscalation``, ``readOnlyRootFilesystem``
     - Guest checks normalized capability sets, ``NoNewPrivileges`` and rootfs readonly mode.
     - 3 when those effective constraints are violated.
   * - ``runAsNonRoot`` / ``privileged`` flags themselves
     - No independent attested comparison of the literal flags; effective UID, capabilities, mounts and other guest properties enforce the boundary.
     - 3 if the resulting request violates guest rules; 4 is possible if checked properties do not change.
   * - ``seccompProfile``
     - Trusted validation requires ``RuntimeDefault``, but the pin requires null OCI Seccomp in the guest. This is not guest seccomp-filter enforcement.
     - A different YAML value resulting in the same null guest field can reach 4. Do not depict this as a universal seccomp denial.
   * - Volumes / mounts
     - The trusted manifest schema forbids them. Guest rules check effective mounts/storage but allow specified patterns, exceptions and some omissions.
     - 3 for requests outside those rules; permitted effective requests can reach 4. Arbitrary host-backed storage is not confidential by default.
   * - Additional sidecar, init or debug containers
     - Each guest create request must match an authorized policy entry. The pin does not globally consume each entry once or enforce exactly-once execution.
     - 3 for an unauthorized request; do not promise every duplicate request matching an authorized entry is denied.
   * - ``tty`` / ``stdin``
     - Effective terminal mode is checked. Exec and read/write streams are separately denied; ``stdin`` is not independently attested as a YAML flag.
     - 3 for prohibited guest requests; a flag change with no disallowed effect can reach 4.
   * - Names, namespace, labels and other annotations
     - Selected guest annotations have equality/pattern checks; there is no blanket YAML metadata check or proof of host-side metadata truth.
     - 3 for a checked mismatch; unconstrained/equivalent values can reach 4.
   * - CPU/memory limits, overhead, node selection and scheduling
     - Source-profile checks reject unsupported shapes. Remote checks do not universally compare these YAML values; some changes affect measured VM inputs or guest requests.
     - 1 for unschedulable/failed launches; 2 or 3 for a covered mismatch; 4 for changes outside those constraints.
   * - ``restartPolicy``
     - Host can restart or recreate the same authorized workload. KBS has no restart-policy or general anti-duplication claim.
     - 4 is possible; no exactly-once guarantee. Host can also deny service.
   * - DNS, networking and ports
     - Host controls routing. End-to-end TLS/mTLS authenticates services/peers; attestation does not certify the whole network configuration.
     - 1 for connectivity failures, 3 for a checked guest mismatch, or 4 if the workload remains usable. Traffic metadata remains exposed.
   * - Service account, token automount and automatic service links
     - Injected mounts/environment may violate guest rules. The raw Kubernetes settings are not KBS authorization identities.
     - 3 for prohibited injected requests; otherwise 4 can be possible. Do not treat Kubernetes service-account identity as the workload's attested identity.
   * - ``imagePullPolicy``
     - Alters fetch/cache behavior, not cryptographic authorization of an image. The guest image checks still apply, with the exact-digest limitation above.
     - 1 if the image is unavailable, 3 if image checks fail, or 4 when usable permitted image content is obtained.
   * - ``apiVersion``, ``kind``, invalid types or newly added fields
     - Normal Kubernetes parsing/schema checks can reject malformed/unsupported input; trusted provisioning has its own allowlist. Neither is hostile-host enforcement.
     - 1 under normal API rejection. Otherwise classify the resulting boot inputs and guest requests; there is no universal remote check for every Kubernetes field.

The diagram's allowed-change branch is deliberate. Successful execution after
a permitted metadata or equivalent-runtime change is not itself a plaintext
disclosure. Conversely, a claim that all security-relevant substitutions fail
requires closure of the stated coverage limitation and live negative tests.

Guest enforcement, network, and storage
---------------------------------------

The authorized agent policy restricts container creation, process arguments,
environment, mounts, identity, capabilities, and other reviewed OCI properties.
Exec/stream requests and policy replacement are denied; global and
per-container exec allowlists remain empty. Validation checks the pinned rule
implementation as well as generated data, not just a few default-deny lines.
It is not a verifier for arbitrary user-written Rego.

For NVFlare, the reviewed profile uses non-root UID/GID 65532, dropped Linux
capabilities, no privilege escalation, and a **writable guest-local** root
filesystem for logs/job state. No hostPath or PVC is implied. The authorized
``RuntimeDefault`` seccomp YAML field is not proof of guest seccomp filtering:
the pinned Kata policy requires null guest OCI seccomp. Do not advertise host
Kubernetes security settings as equivalent guest-enforced controls.

The runtime enables ``agent.guest_components_rest_api=all`` so the application
can access the guest-local AA token API. It is part of the reviewed measured
launch profile. Do not expose this API through a Kubernetes Service, host port,
proxy, or diagnostic dump: the token response includes a private key. Guest
application access is deliberate and extends the trusted computing base.

Networking remains hostile. CNI, DNS, network policies, service selectors and
source IPs are not independent trust anchors. Use owner-controlled FL mTLS,
certificate/peer authentication, and application authorization end to end.
Forward protected-server connections without terminating TLS outside the
authorized guest. Policy validation of CNI-supplied values does not make the
cluster's routing, peer identity, or connectivity trustworthy.

Generated NVFlare startup redirects inherited stdout/stderr to ``/dev/null``
before launching the participant. Guest file logs may exist in ephemeral
storage. This does not disable Kubernetes' log API or stop admitted code from
opening another output channel. No sensitive data may be intentionally emitted
to host-visible logs, command metadata, debug interfaces, or untrusted peers.
Persistent data/checkpoints need a separate authenticated encryption, key
management, and rollback design; encrypted image layers do not provide it.

.. _coco_security_peer_attestation:

NVFlare peer attestation and its limits
=======================================

Generation and verification protocol
------------------------------------

Inside a protected client or server, CoCoAuthorizer calls the guest-local AA
token API. For a new attestation token, AA interacts with Trustee; it may also
return a cached token. The authorizer disables environment proxies and
redirects for this sensitive local request and bounds its response size.

The response contains an AS-signed Entity Attestation Result (EAR) and a guest
private key. The authorizer verifies the EAR with its independently pinned
P-256 AS public key, checks the expected profile, timestamps and exact trust
vectors, and confirms the private key matches the public key authenticated in
the EAR. It produces an outer signed proof containing:

.. code-block:: text

   ear   AS-signed appraisal
   sub   configured logical FL site name
   aud   project-specific audience
   iat   issuance time
   exp   expiration time
   jti   random proof identifier

Only this proof is transmitted. The raw AA response and private key must not
be logged, persisted, or sent to another participant. The configured site name
is signed by the guest; it is not independently assigned by CPU hardware or AS.

The recipient validates the inner EAR with the pinned AS key, extracts the
attested public key, and validates the outer proof with that key. It checks the
project audience and, through ``verify_for_site()``, the signed subject against
the **independently authenticated FL peer**. Incoming JWT key URLs/certificates
are not accepted as new trust roots. Proof possession prevents merely forwarding
an EAR without its associated key from satisfying these checks.

Verification is local: it does not contact Trustee, query RVPS, need a GPU,
or require a Kata guest. The ordinary server can therefore verify a CoCo
client. ``verify(token)`` alone validates a proof but does not bind it to an
expected transport peer; use the site-aware operation at that boundary and
keep normal FL authentication enabled.

Required participants and server identity
-----------------------------------------

Provisioning fixes ``cc_enabled_sites`` and each site's exact required
attestation namespaces through ``required_site_verifier_ids``. Peer-provided
discovery lists cannot silently reduce those requirements. Missing, duplicate,
extra, or invalid required proofs fail closed. Ordinary participants outside
the required set are not thereby confidential; administrator connections are
not attested as worker participants by this CCManager path.

The protected server's logical attestation subject is ``server``, not its
certificate DNS name. TLS still authenticates the endpoint. Adding the server's
CC configuration includes that identity in generated requirements and gives
ordinary clients verifier-only components where needed. Without that server
configuration, client-only deployment does not attest the server.

All CoCo participants in the project must share the required AS trust,
effective verifier policy, and attestation timing. Site-specific workload pins
are represented in the shared constraint map. Changes to membership or policy
require trusted reprovisioning; an unreachable required participant is not
automatically removed from the trust requirement.

What the default authorizer accepts
-----------------------------------

The authorizer accepts exactly
``cpu0`` or exactly ``cpu0`` and ``gpu0``. It
selects SNP/TDX from AS-signed evidence and verifies every present appraisal
against its exact vector. Unknown CPU types, unknown submodules, malformed
claims, and failed GPU appraisal are rejected. Stripping GPU claims from an
EAR breaks the AS signature.

Peer verification requires GPU evidence exactly when that protected
participant selects ``gpu_tee: nvidia_cc`` and requires CPU-only evidence for
``gpu_tee: none``. It does not directly compare the participant's image digest
or command. Workload-specific key authorization remains KBS's job.

``workload_constraints`` can pin each protected site's InitData digest,
SNP measurement, CPU TEE and TDX MRTD/RTMR values. With constraints enabled,
unlisted sites or missing/mismatched required claims fail. The legacy
``measurement`` key means SNP, never TDX. The canonical InitData constraint is
SHA-256, with strict TDX padding normalization. These restrictions narrow
accepted peers but do not replace AS appraisal or KBS resource authorization.

The outer audience is project-specific (generated as ``nvflare-trustee:<project>``).
It is different from the optional inner ``ear_audience`` constraint. By default
there is no independent expected EAR audience or AS policy identifier. A
successful proof is not by itself proof that one particular application is
authorized for that project; protect FL credentials, service policy and, when
required, site-specific workload constraints accordingly.

Freshness, clocks, and replay
-----------------------------

.. list-table:: Current authorizer defaults
   :header-rows: 1
   :widths: 32 25 43

   * - Check
     - Default
     - Meaning
   * - Maximum EAR age
     - 300 seconds; configurable 1--300
     - Measured from EAR ``iat``; clock tolerance does not extend this age cap.
   * - EAR timestamp leeway
     - 180 seconds; configurable 0--180
     - Applied to ``iat``, ``nbf``, and ``exp`` checks; does not repair clocks.
   * - Outer proof lifetime / maximum age
     - 300 seconds
     - Declared lifetime and observed age are bounded independently of the EAR.
   * - Outer future-``iat`` leeway
     - 180 seconds; configurable 0--180
     - Allows bounded clock skew. Outer ``exp`` and optional ``nbf`` stay strict.
   * - Replay tracking
     - Process-local, up to 10,000 entries
     - Rejects a seen subject/identifier until strict proof expiration; resets
       on restart and is not shared with other verifiers.

A newly signed outer proof can contain a cached EAR. Neither a new proof nor
a periodic verification proves that fresh hardware attestation happened at
that instant. A proof may be accepted once by each independent verifier, and
replay state does not survive restart. There is no verifier-issued challenge
protocol or durable federation-wide replay ledger here.

Keep clocks synchronized on guests and secure services. Bounded leeway mitigates
some skew, including a guest clock lagging the AS, but must not be described as
accepting arbitrary time errors or expired proofs. Large skew, stale EARs and
expired outer proofs still fail.

The direct constructor exposes these timing options. The supplied provisioning
CC YAML exposes ``token_expiration`` (mapped to maximum EAR age), retry settings,
``proof_iat_leeway_seconds``, and ``workload_constraints``; it does not expose
every constructor option, including ``ear_audience``, ``ear_leeway_seconds``,
or ``proof_lifetime_seconds``. Use the configuration runbook rather than inserting
unsupported arguments into a project file.

Validation schedule, retries, and failure behavior
--------------------------------------------------

CCManager validates protected-client registration and, when required, the
server's registration proof. A scheduling-related check runs if cross-site
validation has not yet run; it is **not an unconditional fresh check before
every job**. Periodic cross-validation continues afterward. Generated CoCo
configuration defaults to a 120-second interval; the generic CCManager
constructor's default is 600 seconds. The first periodic round waits the
configured interval plus up to 20 percent jitter to accommodate startup.

Token generation retries are bounded by attempt count, elapsed-time budgets
and cancellation. Defaults are 10 attempts, initial delay 1 second, maximum
delay 15 seconds, multiplier 2, and jitter ratio 0.5. Only transient connection
or timeout failures and guest-API HTTP 429/502/503/504 are retryable by this
authorizer. Invalid evidence, failed appraisal, malformed responses and generic
authorization failures do not become successful through a retry bypass.
``generate()`` itself remains single-attempt; CCManager uses the bounded wrapper.

An invalid protected-client registration is rejected. Failure to authenticate
a required protected server stops that client. Cross-validation failures,
including unavailable required participants, follow the fail-closed shutdown
path: the server initiates federation-wide shutdown; a client exits. This is
not automatic quarantine of just one failing peer. Retries improve resilience
to some transient outages without changing that policy or making availability
a guarantee.

.. _coco_security_threats:

Threats, enforcement, and residual risks
========================================

The following matrix assumes the independent services and authorized guest and
application are not compromised. "Denied" means the specified enforcement
must reject the request; the acceptance checklist below requires testing the
real boundary, not merely trusting an installer's preflight.

.. list-table:: Review the remaining risk, not just the mitigation
   :header-rows: 1
   :widths: 24 39 37

   * - Attack or concern
     - Enforcement
     - Remaining risk / requirement
   * - Read plaintext image from registry, node cache, or host disk
     - All workload layers are encrypted; release keys go only to an authorized guest.
     - Manifests/configuration, sizes and timing are not generally secret. Protect builder caches separately.
   * - Replace image, manifest, signature, or ciphertext layer
     - Digest/signature/integrity checks protect fetched content; KBS binds the attested policy's image claim, not the actual pull request.
     - Registry writes do not forge signatures or keys. Another accepted-key/same-repository signed image requires explicit exact-digest authorization and a negative test; do not promise universal substitution rejection.
   * - Change command, environment, user, capabilities, mounts, or guest policy
     - Reviewed guest rules constrain effective OCI requests; KBS checks exact InitData and its image/command claims.
     - Host checks are bypassable. Some guest rules allow patterns, subsets or omissions. Confirm effective enforcement and do not infer unimplemented seccomp protection.
   * - Replace KBS endpoint, CA, image policy, or InitData
     - Independent expected InitData binding, authenticated service endpoint and exact resource authorization.
     - Compromise of trusted policy or trust-root distribution is outside the hostile-host guarantee.
   * - Replace guest/runtime launch inputs
     - Authorized measurements, verified boot events and integrity checks in the authorized boot path.
     - Host binary pins are not remote proof. Demonstrate integrity coverage of external rootfs/disk inputs.
   * - Forge evidence, enable debug, or downgrade platform TCB
     - Report/quote signatures, challenge binding, verifier checks, authorized references and security baseline.
     - Vendor/firmware/verifier vulnerabilities remain assumptions. A valid certificate alone is insufficient.
   * - Mix fields from several authorized TDX profiles
     - AS requires one complete authorized tuple, including all four RTMRs.
     - Each accepted state must match the configured reference tuple; event-log consistency alone is insufficient.
   * - Omit GPU or submit failed GPU evidence
     - GPU release policy requires exactly CPU+GPU and both exact vectors; present failed GPU proofs are rejected.
     - Select ``gpu_tee: nvidia_cc`` for the participant. Provisioning derives the bound peer-verifier requirement; ``gpu_tee: none`` requires CPU-only evidence.
   * - Request another workload's decryption key
     - KBS checks the actual exact path plus that release's platform/workload constraints on every request.
     - Knowing a path or passing CPU appraisal grants no store-wide access. A permissive global policy defeats isolation.
   * - Use exec, attach, copy, ephemeral debug, or SSH
     - Guest agent request policy denies interactive paths; application review must exclude SSH/debug services and credentials.
     - Custom images are not automatically scanned for daemons. A vulnerable or intentionally interactive authorized application can provide another entry point.
   * - Inspect or modify private guest CPU/GPU memory
     - Authorized CPU TEE and, when selected, confidential GPU protections.
     - Not a universal physical/side-channel guarantee; shared I/O and intentionally disclosed outputs remain outside private memory.
   * - Spoof DNS, CNI, peer address, or server routing
     - End-to-end TLS/mTLS, provisioned trust, authenticated FL identity and site-bound proofs.
     - Host retains traffic metadata, routing and availability control. Network policy alone is not a trust boundary.
   * - Forward another site's token or tamper with its claims
     - AS signature, guest-key possession proof, expected subject/audience and FL peer authentication.
     - Authorized guest code can access its key. Compatibility verification without expected-site binding is insufficient.
   * - Replay evidence/proofs or use stale authorization
     - Attestation-session freshness, token time checks and local proof replay rejection.
     - Cached EARs, verifier restarts/replicas and issued-key lifetime matter; this is not immediate revocation or exactly-once execution.
   * - Clone workload, repeat a job, or roll back persistent data
     - No general prevention in the supplied design.
     - Add an independently trusted state/lease protocol and authenticated persistent-storage design if required.
   * - Extract secrets from logs, environment metadata, or outputs
     - Silent generated startup, denied agent streams, reviewed application output and authenticated channels.
     - Admitted code can emit secrets. Kubernetes log availability and traffic-analysis resistance are not disabled by attestation.
   * - Compromise the authorized application or an admitted dependency
     - Build review, restricted job-code admission and application security controls reduce exposure.
     - Attestation preserves identity, not benevolence. Application compromise can leak keys/data and sign proofs.
   * - Infer training data, poison updates, or falsify aggregation
     - Requires separate application review, privacy controls and protocol/algorithm defenses.
     - CoCo does not by itself provide differential privacy, Byzantine robustness, verifiable computation, or cryptographic secure aggregation.
   * - Compromise provisioning, AS signing, KBS administration, or reference configuration
     - Trusted-side access control, credential separation, review, rotation and audit.
     - These are trusted authorities. The deployment cannot defend against their authorized malicious decisions by attestation alone.
   * - Kill, pause, starve, partition, or never schedule a participant
     - Fail-closed authorization and federation checks avoid treating failure as success.
     - Host denial of service remains possible; strict participant requirements can stop the federation.

.. _coco_security_operations:

Operational security and lifecycle
==================================

Before deploying
----------------

* Complete the :ref:`attack-surface inventory <coco_security_attack_surface>`
  for the actual deployment, including local APIs, internal backends, and
  administrative routes as well as public endpoints.
* Establish who trusts each authority and who can change its configuration.
  Separate mutually distrustful administrative domains; the example is not a
  multi-tenant administrative isolation proof.
* Place the supplied AS public key and TLS roots with their intended operators.
  Use different credentials for TLS, token signing, image signing, registry
  publishing, and KBS administration.
* Review build sources, base images, dependencies, class allowlists, startup
  behavior and permitted outputs. CCManager blocks BYOC job submission in this
  flow; explicitly allowed component class paths are still trusted code, not
  a security sandbox or proof of safe computation.
* Keep the trusted-system evidence and private build/recovery material private.
  Handoffs contain only the material authorized for their recipient. Do not
  distribute publisher credentials, plaintext kits or release keys to CoCo IT.
* Record actual code/runtime/policy revisions, security configuration, time sources,
  and independent positive/negative acceptance evidence for each target.

Changes, revocation, and credential compromise
----------------------------------------------

Changing guest launch inputs, token-API configuration or the security baseline
requires reviewed references and a fresh capture as appropriate. Changing
image, command, security context, policy or guest trust configuration requires
a new authorized release. Editing host configuration does not update an already
running confidential guest.

Reference updates replace the authorized set; include every profile that should
remain accepted. SNP's multi-key updates are not transactional. Coordinate
writers and launches, maintain deny-on-failure behavior, verify RVPS readback
and persistence, and do not lower floors to recover from an error. Install
compatible reviewed AS policies before relying on new reference formats.

Removing an RVPS reference or KBS release rule affects future service decisions.
It **does not recall an image key already delivered**, erase plaintext already
loaded into a guest, or immediately invalidate every existing peer proof.
Local verifiers do not query policy on each verification, and a fresh outer
proof may carry a still-acceptable cached EAR. Stronger revocation requires a
separately designed short-lived application authorization/key protocol and
control of the relevant trusted endpoints; do not assume hostile IT will stop
an old guest on request.

AS signing-key rotation requires configuring the new verification
key at participants; do not learn it from incoming JWT headers. Retire
compromised FL identities and protect issuance of replacement kits. Rotation
of TLS certificates is not rotation of AS signing or KBS administrative keys.
Keep historical logs and backups access-controlled; deleting a local file
does not revoke credentials copied elsewhere.

Reference updates use the already configured appraisal policies. Workload
registration merges with a current complete KBS resource-policy export, keeping
unrelated rules. The helper maintains one canonical export and pending recovery
state around policy writes because the API cannot retrieve the current policy
bytes. Reconcile an uncertain pending write before registering another release.

Outages and diagnostics
-----------------------

New launches or fresh attestations depend on the relevant services and usable
vendor verification material. Caches may avoid some external calls; do not
assume every request reaches AMD KDS or NVIDIA NRAS. Caching requires lifecycle
management, especially after firmware changes or collateral expiry. Retries
must not accept stale/unverified evidence or relax authorization.

Use trusted secure-service diagnostics and authenticated application-level
evidence. Do not enable guest exec, dump AA responses, loosen policy, disable
TLS verification, or print private keys to make troubleshooting convenient.
The service administrator can see sensitive administrative material and some
attestation metadata; apply access control and retention rules there too.

.. _coco_security_acceptance:

Deployment acceptance and evidence limits
=========================================

Static checks and unit tests establish implementation behavior for tested
inputs. They do not establish hardware isolation or a successful real
deployment. Host diagnostics establish readiness observations; ``Running``
Pod status is controlled by the adversarial cluster. Trusted capture verifies
platform evidence, not an encrypted application's complete lifecycle.

For **each selected CPU/GPU target**, retain at least the following private
acceptance record, with relevant software, policy and configuration revisions:

.. list-table:: Test the enforcing boundary
   :header-rows: 1
   :widths: 24 43 33

   * - Boundary
     - Positive and negative checks
     - Evidence to retain
   * - Platform collection
     - Fresh verified capture; reject wrong challenge, tampered report/quote or event log, debug mode and unacceptable TCB.
     - Verifier results and bound launch/artifact records, not decoded claims alone.
   * - Reference policy
     - Accept each authorized profile; reject unauthorized or mixed TDX tuples, omitted RTMRs, wrong CPU type and insufficient SNP TCB.
     - Reviewed AS policy, RVPS readback, and live denial/acceptance results.
   * - Workload resource release
     - Release all three exact resources for the authorized launch; reject wrong paths, InitData or attested policy image/command claims, and missing/failed required GPU.
     - Trusted service and guest/application evidence correlated to the tested release.
   * - Effective image authorization
     - Reject unsigned/corrupt images; close the exact-digest binding limitation and test another accepted-key/same-repository signed image with unchanged InitData.
     - Evidence from the guest image-verification path. A test changing only attested policy claims does not establish rejection of the actual substituted image.
   * - Guest enforcement
     - Start the authorized process; reject prohibited exec/attach/debug and effective OCI requests outside guest rules. Also test permitted substitutions/omissions and equivalent-field changes.
     - Guest-policy behavior and the four launch outcomes; hostile-host launcher validation alone is insufficient.
   * - Federation identity
     - Register protected clients and, if selected, protected server; reject missing tokens, wrong site/audience, incomplete required sets and altered signatures.
     - Authenticated registration, periodic-round and real job results, including verification outside CoCo.
   * - Freshness and failure
     - Reject expired/replayed proofs within the stated cache scope; exercise bounded transient retry, terminal failures and required-participant loss.
     - Actual fail-closed behavior, timing and shutdown consequences; do not infer these from token generation alone.
   * - Lifecycle and output
     - Review guest disk integrity coverage, sensitive output, rebuild/configuration update, credential rotation and new-release denial after revocation.
     - Explicit limitations for issued keys, persistent state, external boot storage and host-visible metadata.

Service log lines are diagnostic observations, not necessarily cryptographic
correlation of all events to one attestation session. An empty log is not proof
that secret output is impossible. A standalone CoCoAuthorizer success is not
proof of encrypted image release, full registration, protected-server behavior,
or completion of a federated job. Record each result separately; absent tests
remain unverified. Never describe all four runtime variants as hardware-proven
merely because shared code or offline tests support them.

.. _coco_security_faq:

Security questions and direct answers
=====================================

Can CoCo IT SSH into a protected workload or use kubectl exec?
--------------------------------------------------------------

The intended image contains no SSH daemon/credentials, and the authorized guest
agent policy denies interactive exec/stream access. Host root does not grant
authorized guest-agent access. This does not rule out a vulnerability or an
interactive service deliberately included in the authorized application.

Can IT modify the Pod YAML or runtime?
--------------------------------------

**IT can edit the files, but that does not authorize a modified protected
workload to run.** Creating a Kubernetes Pod object or booting its confidential
VM is different from successfully starting the protected container. The entire
YAML is not authenticated byte-for-byte. The boundaries check the measured
guest, InitData, image signatures/integrity and effective guest requests, with
the coverage limits in :ref:`coco_security_pod_launch`.

.. list-table:: What happens after IT edits the launch inputs?
   :header-rows: 1
   :widths: 40 60

   * - IT's change
     - Expected enforcement under the documented configuration
   * - Change metadata outside the authorized guest policy, such as an unrelated label
     - The workload may still start. Not every YAML field is security-bound;
       do not assume every name, namespace or annotation is unconstrained.
   * - Change the command or another effective setting outside the permitted guest rules, keeping the original InitData
     - The guest agent rejects the unauthorized request. Platform attestation
       and KBS authorization can still pass for the original policy. Some
       substitutions, subsets or equivalent settings remain permitted.
   * - Change the image, keeping the original InitData
     - Guest signature/integrity checks still apply, but exact-digest
       authorization has an unresolved coverage limitation. Another signed
       image in the accepted repository is not universally shown to be rejected.
   * - Change the embedded policy or guest trust configuration to allow that modification
     - The InitData digest changes. KBS expects the configured
       digest and refuses to release the image decryption key.
   * - Replace measured guest components with an unauthorized version
     - Attestation appraisal fails and KBS withholds the key. This relies on
       demonstrated measurement/integrity coverage of the authorized boot path;
       a host runtime binary is not itself remotely authenticated.

For example, replacing the authorized command with ``/bin/sh`` does not grant a
shell. With the original InitData, the guest agent rejects that command. If IT
also edits the embedded policy to allow the shell, KBS rejects the changed
InitData digest. KBS checks the attested policy claims, not the host's live
container-creation request; the guest agent applies the permitted
effective-process rules to the actual request. Not every malicious YAML edit
must therefore fail at the attestation stage itself.

**The file-confidentiality requirement is that CoCo IT's cluster or host
privileges must not reveal the plaintext contents of files kept exclusively
inside the protected guest.** This includes decrypted application files,
startup credentials, and guest-local logs/job state. Image decryption and key
use stay inside the authorized confidential guest; reviewed agent policy
denies interactive exec and attach/stream access, including ``kubectl cp``
readback through those paths. The authorized application must not provide an
SSH/debug service that bypasses that boundary.

This is not an unconditional promise that any file mounted into any Pod is
secret. File data must remain in private guest memory or storage with verified
confidentiality/integrity protection. Plaintext hostPath/PVC storage,
host-visible logs, and files deliberately served by the application are outside
that boundary. Image-layer encryption alone does not protect arbitrary writable
disks, persistent volumes or checkpoints. Application vulnerabilities or
admitted code can also leak data; authorized storage, code and output channels
must be reviewed. The hardware, guest stack and independent secure services
remain trust assumptions.

IT can still launch unrelated Pods, falsify Kubernetes status or deny service.
None of those actions proves successful execution of the protected workload.
See :ref:`coco_security_workload_release` for the enforcement and storage limits.

Can another client request my image key if its attestation passes?
------------------------------------------------------------------

Platform appraisal alone is insufficient. Your release rule checks the exact
requested path and authorized workload/platform binding. A permissive shared
resource rule or compromised service administrator defeats that boundary.

Is the image digest the CPU measurement?
----------------------------------------

No. CPU launch/boot measurements identify authorized guest inputs. InitData binds
policy claims about the image and command, which KBS checks. Guest code checks
the actual image's signature/integrity and effective process request; the
exact-image-digest limitation above remains to be closed. Neither measurement
nor image authorization establishes application benevolence.

How are platform references produced?
--------------------------------------

The capture command verifies fresh hardware evidence and configured TCB/security
constraints, then exports the reference file directly. The existing Trustee
policies compare later appraisals with those references. CPU model or a matching
RuntimeClass name alone does not establish measurement equality.

Can a signed CPU+GPU token be changed into CPU-only?
----------------------------------------------------

Not without invalidating the AS signature. The authorizer also accepts a
legitimately issued CPU-only proof by default. GPU-required
KBS authorization is a separate constraint; an FL-wide or site-specific GPU
requirement needs an independently enforced participant policy.

Does the verifier need Trustee access or its own confidential VM?
-----------------------------------------------------------------

No. Peer verification uses local pinned AS trust and proof checks. Generation
requires the protected guest's AA API. Local verification therefore cannot
discover every service-policy change immediately.

Does a protected client mean the server and training data are safe?
-------------------------------------------------------------------

No. An ordinary server is still trusted with what it receives. A protected
server isolates authorized computation from its host, but does not automatically
provide differential privacy, prevent model inversion or membership inference,
detect poisoned updates, or prove aggregation correctness. Review recipients,
algorithms, data access and outputs separately.

Are logs, checkpoints, and container changes confidential forever?
------------------------------------------------------------------

No. Generated startup suppresses inherited console output, but admitted code
can emit information. Guest-local writable state is ephemeral; persistent
storage protection and rollback prevention are not supplied by image encryption.
Nor can reference revocation erase a key or plaintext already released.

Can the cluster owner still disrupt the federation?
---------------------------------------------------

Yes. The operator controls availability. Strict required-participant validation
can stop the federation after an outage or invalid proof; that is fail-closed
behavior, not a confidentiality mechanism that guarantees progress.

.. _coco_security_glossary:

Glossary and implementation references
======================================

.. glossary::

   CoCo
      Confidential Containers: the guest/runtime and attestation components
      used to protect container workloads in confidential VMs.

   TEE
      Trusted execution environment, here AMD SEV-SNP or Intel TDX, with
      NVIDIA confidential GPU protection for GPU profiles.

   TCB
      Trusted computing base. Also used in vendor evidence for the security
      version/status of platform components; acceptable versions need policy.

   AA / CDH
      Guest attestation agent / confidential data hub. They mediate
      attestation, resource retrieval, and guest data/image operations.

   AS / RVPS / KBS
      Attestation Service appraises evidence; Reference Value Provider Service
      supplies authorized values; Key Broker Service authorizes resource release.

   OPA / Rego
      Open Policy Agent and its policy language. The workflow uses Rego rules
      for appraisal/resource authorization and guest policy; these are distinct
      policies at distinct enforcement points, not one universal allow decision.

   EAR
      Entity Attestation Result: the AS-signed appraisal embedded in the
      NVFlare guest-key possession proof.

   InitData
      Guest configuration and policy whose digest is bound into CPU evidence.
      It is distinct from a CPU launch measurement and is not inherently secret.

   QGS / PCK / collateral
      TDX quote-generation service / platform certification key certification /
      signed certificate-status and identity/TCB material used by verification.

   KDS / VCEK
      AMD Key Distribution Service / Versioned Chip Endorsement Key. The
      VCEK-based SNP verification path uses a chip- and TCB-specific certificate.

   NRAS / RIM
      NVIDIA Remote Attestation Service / Reference Integrity Manifest. These
      support appraisal of GPU evidence; they do not authorize KBS resource paths.

Use the following guides for commands. They start from existing infrastructure
and complement this security model's acceptance requirements.

* :github_nvflare_link:`Workflow and prerequisites <examples/devops/coco/README.md>`
* :github_nvflare_link:`Capture measurements <examples/devops/coco/MEASUREMENTS.md>`
* :github_nvflare_link:`Configure Trustee and register releases <examples/devops/coco/TRUSTEE.md>`
* :github_nvflare_link:`Build NVFlare images and Pod YAML <examples/devops/coco/provision/README.md>`
* :github_nvflare_link:`Launch and trusted verification <examples/devops/coco/LAUNCH.md>`
* :ref:`Trustee-backed peer-proof protocol <coco_security_peer_attestation>`

Implementation sources for reviewers:

* :github_nvflare_link:`CoCoAuthorizer <nvflare/app_opt/confidential_computing/coco_authorizer.py>` and
  :github_nvflare_link:`CCManager <nvflare/app_opt/confidential_computing/cc_manager.py>`
* :github_nvflare_link:`CoCo provisioning <nvflare/lighter/cc_provision/impl/coco.py>` and
  :github_nvflare_link:`packaging <nvflare/lighter/cc_provision/impl/coco_packager.py>`
* :github_nvflare_link:`Guest-policy validation <nvflare/lighter/cc_provision/workload_security.py>` and
  :github_nvflare_link:`target-aware release authorization <nvflare/lighter/cc_provision/workload_release.py>`
* :github_nvflare_link:`Policy and Pod generation <examples/devops/coco/admin/30-generate-pod-and-policies.sh>` and
  :github_nvflare_link:`pinned hardened guest rules <tests/unit_test/lighter/cc_provision/impl/fixtures/kata-3.29-hardened-rules.rego>`
* :github_nvflare_link:`AS CPU policy <examples/devops/coco/service/policies/default_cpu.rego>` and
  :github_nvflare_link:`KBS resource-policy template <examples/devops/coco/service/policies/workload-resource-policy.rego.template>`
* :github_nvflare_link:`Platform capture helper <examples/devops/coco/trusted_system/capture-platform.py>`

For upstream concepts, consult:

* `AMD VCEK and KDS specification <https://docs.amd.com/v/u/en-US/57230>`_
* `Intel TDX host setup <https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/05/host_os_setup/>`_
  and `attestation infrastructure <https://cc-enabling.trustedservices.intel.com/intel-tdx-enabling-guide/02/infrastructure_setup/>`_
* `CoCo InitData <https://confidentialcontainers.org/docs/features/initdata/>`_
  and `upstream trust model <https://confidentialcontainers.org/docs/architecture/trust-model/trust-model/>`_
* `NVIDIA Remote Attestation Service <https://docs.nvidia.com/attestation/cloud-services/latest/nras/nras_introduction.html>`_

Also review the configured runtime/verifier versions and generator rules.
Upstream capabilities and example policies do not, by themselves, establish
which controls this particular deployment has enabled and validated; do not
replace the reviewed release rules with a permissive upstream example.
