# Confidential workload deployment

A sanitized, role-separated runnable example for AMD SEV-SNP and NVIDIA
confidential GPUs. Scripts, policies, dependencies and configuration templates
are included; deployment credentials, certificates, measurements, private
evidence and generated handoffs are not.

Start with [CONFIGURATION.md](CONFIGURATION.md) for required inputs, configuration
commands, transfer boundaries and execution order. Read
[PUBLICATION.md](PUBLICATION.md) before publishing or redistributing a used copy.

## Roles and guides

For NVFlare clients, see [project.yaml-based provisioning](provision/README.md).
The provisioning-node packager protects clients; the server keeps its ordinary kit.
The [CoCoAuthorizer guide](provision/CCMANAGER.md) covers client-side proof
generation, local verification on an ordinary server, trusted AS public-key
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
