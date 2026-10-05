# Confidential workload deployment

A sanitized, role-separated deployment example for AMD SEV-SNP or Intel TDX,
with an optional NVIDIA confidential GPU. Scripts, policies, dependencies and configuration templates
are included; deployment credentials, certificates, measurements, private
evidence and generated handoffs are not.

For security guarantees, trust boundaries, threats, residual risks and the
attestation/key-release protocol, start with the comprehensive
[CoCo + NVFlare security architecture](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html).
The role guides below provide operational procedures, not separate security
models. Follow the architecture's implementation-scope and validation notes.

For a collaboration-level starting point, read
[one model owner and two data owners](FL-DEPLOYMENT.md): roles and approvals,
machine prerequisites, the guest/application distinction, reference rehearsal,
ordered handoffs, and the dataset integrations users must supply.

**A mixed TDX CPU-only and SNP+NVIDIA GPU functional run passed on
2026-10-04.** An ordinary trusted server authenticated both real Kata clients
with `coco_authorizer`, and a finite job received both nonce-bound results
(3 and 7, aggregate 10). The SNP client required CPU and GPU appraisal; the job
itself performed CPU arithmetic. This establishes the tested functional chain,
not full security qualification or TDX+GPU coverage. Renewal, confidentiality
and hardware denial acceptance remain incomplete. See
[support status and the exact test boundary](RUNTIME-VARIANTS.md#support-status-and-hardware-validation).

For that normal server/two-client configuration, start with the complete
[mixed TDX/SNP project](provision/mixed-tdx-snp/README.md). The ordinary server
verifies clients directly; an additional observation client is unnecessary for
the deployment workflow.

Choose the target using [runtime variants and deployment requirements](RUNTIME-VARIANTS.md).
The role scripts select SNP-only, SNP+GPU, TDX-only or TDX+GPU explicitly; they
do not infer trust from the cluster operator's runtime name. TDX host firmware,
kernel, SGX provisioning and pinned quote-generation services must already be
installed and reviewed. For the implementation/security contract, see
[runtime implementation details](RUNTIME-PORTING.md). TDX hardware validation
must be completed on the intended platform; offline tests are not that proof.

For a worked deployment with one model owner and two data owners, follow the
[two-site FL example](https://nvflare.readthedocs.io/en/2.9/user_guide/confidential_computing/coco_security_architecture.html#coco-security-two-site-example).
It explains responsibilities, machine prerequisites and setup order, including
the application-specific dataset and persistence integration you must supply.

Start with [CONFIGURATION.md](CONFIGURATION.md) for required inputs, configuration
commands, transfer boundaries and execution order. Read
[PUBLICATION.md](PUBLICATION.md) before publishing or redistributing a used copy.

## Roles and guides

For NVFlare clients and an optional protected server, see
[project.yaml-based provisioning](provision/README.md). The provisioning-node
packager creates a separate encrypted image and Pod handoff for each protected
participant. An ordinary server remains supported as a verifier-only participant.
The [CoCoAuthorizer guide](provision/CCMANAGER.md) covers protected-participant proof
generation, local verification, trusted AS public-key
distribution, and the [live cross-node test's scope](provision/CCMANAGER.md#verification-status).

- [Secure services](service/README.md): TLS registry, Trustee/KBS, AS, RVPS and resource authorization.
- [NVFlare provisioning node](admin/README.md): build, encrypt, sign, publish and generate workload handoffs.
- [Trusted platform system](trusted_system/README.md): verify hardware evidence and produce approved reference inputs.
- [Adversarial CoCo cluster](coco/README.md): install public pinned runtime inputs and launch unchanged workloads.

CoCo IT does not approve measurements or administer secure services. Workload
signatures, encrypted layers, measured init-data, CPU/GPU appraisal and exact
resource authorization enforce the trust boundary. The host can still deny
service and observe exposed metadata and workload output. These scripts do
not establish absolute protection against every hardware/software flaw.

## Local validation

For the CPU-only TDX acceptance harness, both federation topologies, the baked
validation application, and private evidence recording, see
[TDX acceptance](acceptance/README.md).

```bash
python3 validate-package.py
```

Run from this directory in the Git checkout with Python 3.11+, Git and Bash.
Validation is offline and checks tracked sources, syntax, document links and
shared dependencies; it does not run regression tests. For a full assembled
package, use `--assembled` (no Git required). Static validation is not a substitute
for hardware-backed positive and negative tests.
Software versions and public upstream digest pins are retained for
reproducibility, not a claim that these versions remain vulnerability-free.
Host-specific teardown scripts and the obsolete signed-bundle installer are
not included. No separate NVFlare clone is required by the cluster installer.

## Package capabilities, not live state

For protected NVFlare servers and clients, use the [token-API runtime profile](RUNTIME-PROFILE.md).
It requires a new trusted rehearsal and v4 admin contract; old measurements and
contracts are not silently reused.
The [approved security context](admin/APPROVED-LAUNCH-PROFILE.md#approved-application-security-context-v3)
pins application IDs, privileges, capabilities and rootfs mode. NVFlare's writable
rootfs requires explicit approval; the collector's diagnostic privileges are not
application permissions. The pinned runtime does not establish guest seccomp
filtering merely by requesting `RuntimeDefault` in Kubernetes YAML.

These files do not imply a running Pod, issued certificate, installed policy,
or approved measurement on any machine. Obtain fresh deployment inputs and
record execution results privately. Public version/digest pins live in the
role configuration templates. Workload and platform approvals come from the
trusted workflow, not a historical deployment. Hardware-backed validation is
required after configuration; offline package checks alone do not establish it.

## Assemble self-contained role kits

Shared configuration checks, Kubernetes bootstrap helpers, the Kubernetes
installer and its kubeadm template are maintained only in [shared/](shared/README.md).
In the source checkout, role entry points call these shared implementations.
Do not transfer an unassembled role directory by itself.

```bash
python3 role_kits.py /path/to/new-coco-role-kits
python3 /path/to/new-coco-role-kits/validate-package.py --assembled
```

Choose a new directory outside this source package. Assembly requires a clean
Git checkout: commit reviewed changes first. The assembler copies only tracked
files, excludes ignored/untracked files, and materializes shared dependencies.
It does not assemble from an unpacked source archive or an existing role kit.
Transfer the resulting `admin/`, `service/`, `coco/`, or `trusted_system/`
directory only to its intended operator. Its scripts run without sibling roles,
the shared source tree or an NVFlare checkout. Supply private configuration,
certificates and handoff material separately according to [CONFIGURATION.md](CONFIGURATION.md).
The full assembled package retains cross-role documentation links for reference.

## Design slides

[Design slides (Markdown)](docs/coco-security-design-3-slides.md)
