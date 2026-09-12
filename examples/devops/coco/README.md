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
python3 -m venv "$HOME/.venvs/coco-validation"
"$HOME/.venvs/coco-validation/bin/python" -m pip install PyYAML==6.0.3
"$HOME/.venvs/coco-validation/bin/python" validate-package.py
sha256sum --check --strict PACKAGE-SHA256SUMS
```

Requires Python 3.11+, PyYAML and Bash. Dependency installation needs network
access; validation itself is offline and non-deploying. Keep the validation
environment outside this public package. Validation is not a substitute for
hardware-backed positive and negative tests.
Software versions and public upstream digest pins are retained for
reproducibility, not a claim that these versions remain vulnerability-free.
Host-specific teardown scripts and the obsolete signed-bundle installer are
not included. No separate NVFlare clone is required by the cluster installer.

## Design slides

[PDF](docs/coco-security-design-3-slides.pdf) · [PowerPoint](docs/coco-security-design-3-slides.pptx) · [HTML](docs/coco-security-design-3-slides.html) · [Markdown](docs/coco-security-design-3-slides.md)
