# NVIDIA FLARE Security Policy

NVIDIA is dedicated to the security and trust of its software products and
services, including its open-source repositories.

## Reporting a Vulnerability

Do not report a suspected vulnerability through a public GitHub issue, pull
request, or Discussion. Report it privately through one of these NVIDIA PSIRT
channels:

- [NVIDIA Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail)
- Email [`psirt@nvidia.com`](mailto:psirt@nvidia.com). For sensitive reports,
  use the [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key/).

Include the affected NVFlare version, branch, or commit; the vulnerability
type; reproduction instructions; proof-of-concept material when available; the
deployment mode and platform; and the potential impact. Keep the issue
confidential until NVIDIA PSIRT completes coordinated disclosure. See the
[NVIDIA PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/)
for the vulnerability-management and disclosure process.

For non-sensitive configuration and hardening questions, use
[NVFlare Discussions](https://github.com/NVIDIA/NVFlare/discussions). Do not
include secrets, private data, exploit details, or unpublished vulnerabilities.

## Supported Versions

Security fixes are delivered through maintained NVFlare releases and branches
as determined by the project maintainers and NVIDIA PSIRT. Use the latest
applicable release and review the
[NVIDIA security bulletins](https://www.nvidia.com/en-us/security/). When
reporting an issue, identify the exact installed version or commit because
behavior on `main` may differ from a released package.

## Architecture and Trust Boundaries

NVFlare is an extensible Python SDK and runtime for federated workflows. A
typical deployment contains an FL server, participating FL clients, and one or
more administrators. Provisioned startup kits establish participant identity
and contain connection material. Administrators submit jobs containing Python
code, configuration, and serialized components. The server coordinates work;
clients execute approved job code against site-local data and return model
updates, metrics, or statistics.

The principal trust boundaries are:

- **Administrator to server:** job submission and lifecycle operations cross
  an authorization boundary and can cause code to run at participating sites.
- **Server to client:** commands, tasks, model parameters, metrics, and
  statistics cross authenticated network connections.
- **Job package to site runtime:** custom components and dependencies execute
  with the permissions of the local NVFlare process.
- **Site data to federated result:** raw data should remain local, but model
  updates, metrics, logs, and aggregate statistics can still disclose
  information if the workflow and privacy controls are unsuitable.
- **Provisioning and dashboard to deployment:** identities, certificates,
  authorization policies, project metadata, and administrative credentials are
  security-sensitive inputs.
- **Agent skill to host environment:** files under `skills/` are instruction
  bundles for coding agents. A skill can guide file edits, local simulation, or
  job submission, but it does not grant permissions or override host approval,
  site policy, or authentication.

Sensitive assets include site data; model parameters and updates; aggregate
statistics and metrics; startup-kit private keys, certificates, and tokens;
job packages; workspaces and logs; and dashboard or provisioning state.

## Threat Model

NVFlare deployments should account for malicious or compromised job authors,
administrators, participants, dependencies, artifacts, and network peers, as
well as local users who can access runtime files. Important threats and the
expected control points include:

| Threat | Security impact | Primary controls and responsibilities |
| --- | --- | --- |
| Unauthorized job or administrative action | Remote code execution, data access, or deployment changes | Use certificate-based identity, least-privilege authorization policies, site policy, and audit logging. Review job submitters and targets before approval. |
| Malicious job code or custom component | Code execution at clients or the server, credential theft, data exfiltration | Treat job packages as executable code. Review source and dependencies, restrict allowed components, use unsafe-component detection, and isolate runtimes with OS/container controls. |
| Unsafe deserialization or model loading | Code execution or object-confusion attacks | Use NVFlare's supported serializers and registered decomposers, preserve type allowlists, prefer data-only formats such as `safetensors` where applicable, and do not load untrusted pickle artifacts. |
| Network interception or participant impersonation | Disclosure or tampering of tasks, updates, and credentials | Provision unique identities, validate certificates, use TLS or mutual TLS for production connections, protect private keys, and rotate compromised credentials. |
| Over-broad authorization or disabled site policy | A legitimate identity performs an unsafe operation | Define and test federated authorization policies, keep site-local approval in force, and review policy changes before deployment. |
| Privacy leakage from updates, metrics, statistics, or logs | Reconstruction, membership inference, or disclosure of sensitive site information | Minimize outputs, validate aggregation semantics, configure privacy filters or appropriate DP/HE/PSI controls, restrict logs, and assess the complete workload's privacy model. Federated execution alone is not a privacy guarantee. |
| Path traversal or malicious archives | Overwrite or disclosure outside an intended workspace | Use the maintained package/archive handling paths, preserve destination-boundary and symlink checks, and never extract untrusted content with ad hoc archive code. |
| Denial of service or resource exhaustion | Unavailable server, client, dashboard, or training capacity | Apply deployment-level resource limits and monitoring, validate job resource requests, bound message/artifact sizes, and use NVFlare fault-tolerance controls. |
| Dependency or artifact supply-chain compromise | Compromised code executes during install or runtime | Pin and review dependencies, verify artifact provenance, avoid credential-bearing package URLs, scan releases and containers, and update vulnerable components promptly. |
| Agent prompt injection or tampered skill content | Unsafe edits, commands, downloads, or submissions | Treat repository and dataset text as untrusted input, retain host approvals, install each skill only from the expected signed catalog artifact, and verify the skill signature after content changes. |

## Critical Security Assumptions

NVFlare's security controls depend on the following deployment assumptions:

- Operators secure the host OS, Python environment, containers, storage,
  network, and orchestrator independently of NVFlare.
- Startup kits and private keys are delivered only to their intended
  participants, stored with restrictive permissions, and replaced after
  suspected compromise.
- Production systems use properly validated secure connections. Development
  POC or clear-text configurations are not promoted to production unchanged.
- Administrators and sites review job code, component configuration, declared
  dependencies, and requested resources before authorizing execution.
- Site authorization and privacy policies are configured for the collaboration
  and are not bypassed merely to make a job run.
- A federated algorithm, secure transport, or aggregate output does not by
  itself establish regulatory compliance or prevent privacy leakage. Workload
  owners perform the required privacy, legal, and data-governance review.
- Logs, workspaces, checkpoints, metrics, and downloaded job artifacts are
  handled as potentially sensitive data and are retained only as required.
- Participants run supported software versions and remediate relevant NVIDIA
  and third-party security advisories.

## Deployment Hardening

Before production use:

1. Follow the [NVFlare security guidance](docs/user_guide/admin_guide/nvflare_security.rst)
   and [production-readiness checklist](docs/production_readiness.rst).
2. Provision distinct participant identities and enable secure transport,
   least-privilege authorization, site policy, and audit logging.
3. Run custom job code in a dedicated, minimally privileged environment with
   explicit filesystem, network, device, and resource limits.
4. Review and test serialization, data filters, aggregation, privacy controls,
   and failure behavior using non-sensitive data before admitting a workload.
5. Protect and monitor administrative endpoints, dashboard state, startup kits,
   workspaces, exported jobs, logs, and model artifacts.
6. Keep NVFlare, Python, base images, and third-party dependencies current and
   subscribe to relevant security bulletins.

Additional design detail is available in the
[security overview](docs/system_architecture/security_overview.rst) and the
[security FAQ](docs/security_faq.rst).
