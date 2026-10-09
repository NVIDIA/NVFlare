# Unified Confidential Computing Provisioning

Status: implemented on branch `cc_ui_redesign`

Created: 2026-09-28  
Updated: 2026-10-05  
Baseline: `upstream/2.9` at `6690fd4e3564d360bc86eaae0916d7b8c0cb3a6e`

## Purpose

NVFlare 2.9 has three confidential-computing deployment modes:

- `bare_metal_cvm`: a complete Intel TDX or AMD SEV-SNP VM produced by CVM
  Builder, with an application-specific encrypted vault;
- `coco`: a confidential Kata container on Kubernetes, using Trustee for
  attestation and key release; and
- `azure_cc`: an Azure confidential VM or confidential Azure Container
  Instance, using Microsoft Azure Attestation (MAA).

They currently use different provisioning entry points, configuration names,
packaging hooks, and documentation. This design gives them one provisioning
contract while preserving their different trust boundaries and deliverables.

## Decisions

1. A participant opts in to confidential computing only through `cc_config`.
2. Every `cc_config` requires:

   ```yaml
   cc_deployment_mode: bare_metal_cvm  # bare_metal_cvm | coco | azure_cc
   ```

3. `CCBuilder` validates each participant and dispatches it through one
   `CCDeployment` interface.
4. One class implements each mode: `BareMetalCVMDeployment`,
   `CoCoDeployment`, and `AzureCCDeployment`.
5. The same concept has the same name and type in every mode. Values unique to
   a mode remain in a mode-specific block.
6. Shared project settings live in a separate `cc_project.yml`, referenced
   once from `project.yml`.
7. Bare-metal CVM and CoCo both resolve `attestation.service: trustee` to the
   same `cc_project.yml` object and consume the same Trustee/KBS schema.
8. The participant role comes from `participant.type`; `cc_config` does not
   repeat it.
9. One common packager handles all post-signing work and produces a common
   result manifest.
10. One user guide contains the common workflow once and then the three
    mode-specific sections.

## Current 2.9 state

| Concern | Bare-metal CVM | CoCo | Azure CC |
|---|---|---|---|
| Selection | Top-level `cvm_vault.participants` | Participant `cc_config` | Participant `cc_config` |
| Discriminator | Implied by `cvm_vault` | `compute_env: confidential_containers` | `compute_env: azure_cvm` or `azure_confidential_container` |
| Entry point | Special `VaultAdapter` branch in `provision()` | `CCBuilder` and exclusive `CoCoPackager` | `CCBuilder` |
| CPU TEE | CVM image contract and optional `platforms` | `cc_cpu_mechanism` | `cc_cpu_mechanism` |
| GPU | `requires_gpu` | `cc_gpu` | Not in current examples |
| Role | Derived | Repeated in `role` | Repeated in `role` |
| Trustee | Separate `cvm_project.yml` | Repeated in participant/operator files | Not applicable |
| Public result | CVM OCI delivery | Pod YAML | Signed startup kit |

At the baseline revision, CoCo supports AMD SEV-SNP and Intel TDX, with or
without an NVIDIA confidential GPU. Bare-metal CVM supports TDX and SEV-SNP;
the approved CVM image contract determines GPU support. Azure has two targets
under one mode: confidential VM and confidential container.

## Unified files

### `project.yml`

All modes use the same participant property. The project-wide CC file is
referenced once.

```yaml
api_version: 3
name: unified_cc_project

cc_project_config: cc_project.yml

participants:
  - name: server.example.com
    type: server
    org: example
    fed_learn_port: 8002
    admin_port: 8003
    cc_config: cc_server.yml

  - name: site-coco
    type: client
    org: example
    cc_config: cc_site_coco.yml

  - name: site-azure
    type: client
    org: example
    cc_config: cc_site_azure.yml

  - name: admin@example.com
    type: admin
    org: example
    role: project_admin

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
  - path: nvflare.lighter.impl.cert.CertBuilder
  - path: nvflare.lighter.cc_provision.impl.cc.CCBuilder
  - path: nvflare.lighter.impl.signature.SignatureBuilder

packager:
  path: nvflare.lighter.cc_provision.impl.cc_packager.CCPackager
```

The presence of `cc_config` means the participant must be confidential.
Missing or invalid input is fatal; provisioning must never fall back to a
plaintext startup kit. `cc_project_config` is required when any participant
has `cc_config`.

Relative paths resolve from the file that declares them: `cc_config` from
`project.yml`, participant values from that participant file, and project
values from `cc_project.yml`.

### `cc_project.yml`

This operator-side file owns project-wide endpoints, authenticated public
keys, administrative credentials, approval keys, and trusted build tools.

```yaml
schema_version: 1

attestation_services:
  trustee:
    type: trustee

    # Trustee/KBS administration and resource release.
    kbs_endpoint: https://trustee.example.org:8443
    ca_cert_file: ./credentials/kbs_ca.pem
    admin_token_file: ./credentials/kbs_resource_token.jwt

    # CoCo guest-local Attestation Agent token API. Loopback is inside the
    # confidential guest, not the provisioning machine.
    attestation_token_endpoint: http://127.0.0.1:8006/aa/token
    attestation_signing_public_key_file: ./credentials/trustee_as_public.pem

    token_expiration_seconds: 300
    check_frequency_seconds: 120
    registration_token_timeout_seconds: 300
    refresh_token_timeout_seconds: 30
    get_token_request_timeout_seconds: 45

    # Optional shared verifier policy. If present, the site map is complete.
    proof_iat_leeway_seconds: 180
    workload_constraints:
      server:
        cpu_tee: tdx
        tdx_mr_td: <approved-96-character-lowercase-hex-value>
      site-1:
        cpu_tee: snp
        init_data: <approved-64-character-lowercase-hex-value>

    retry:
      max_attempts: 10
      initial_delay_seconds: 1.0
      max_delay_seconds: 15.0
      backoff_multiplier: 2.0
      jitter_ratio: 0.5

  azure_maa:
    type: azure_maa
    endpoint: https://sharedeus2.eus2.attest.azure.net
    token_expiration_seconds: 100
    check_frequency_seconds: 60

container_registries:
  coco_workloads:
    endpoint: secure-services.example.com:5000
    ca_cert_file: ./credentials/registry_ca.pem
    publisher_username_file: ./credentials/registry_username
    publisher_password_file: ./credentials/registry_password

approval:
  public_key_files:
    - ./credentials/acceptance_signing.pub

build_tools:
  bare_metal_cvm:
    cvm_builder_dir: ../../../nvflare/lighter/cc/image_builder
    output_root: /srv/nvflare/cvm_builds
  coco:
    build_command: ./admin/build_coco_image.sh
    build_timeout_seconds: 3600
```

This `trustee` object is the only Trustee/KBS configuration schema.
`BareMetalCVMDeployment` and `CoCoDeployment` both resolve
`attestation.service: trustee` to this object. The bare-metal class uses the
KBS administration and approval fields. The CoCo class also uses the
Attestation Service key, token endpoint, timing, and retry fields. Shared
fields are not renamed by mode.

`token_expiration_seconds` is the maximum accepted EAR age. Bare-metal CVM
provisioning carries it into the measured guest's proof-renewal schedule and
uses the earlier of signed `exp` and `iat + token_expiration_seconds`. The value
must leave room for one complete appraisal, the 15-second expiry margin, and a
separate 15-second allowance for issuance, retrieval, and publication: at least
90 seconds for CPU-only CVMs and 270 seconds for GPU CVMs.

`proof_iat_leeway_seconds` and `workload_constraints` preserve the existing
fail-closed Trustee verifier policy at project scope. When constraints are
present, their keys must be exactly the protected participants that select the
service. The root server uses logical key `server`; clients use participant
names. Both deployment modes receive the same validated policy. The public
mapping must not contain `gpu_required`; provisioning derives that internal
boolean from each participant's `gpu_tee`, making `nvidia_cc` require CPU plus
GPU evidence and `none` require CPU-only evidence.

`attestation_token_endpoint` is the token API exposed by the CoCo
Attestation Agent inside each confidential guest. The loopback address is
therefore intentional and is validated as guest-local; it does not refer to
the provisioning host or to a remote Trustee service.

`container_registries.coco_workloads` supplies the registry host, TLS trust,
and publisher credential references. A participant selects this named
registry and supplies only a repository path. The resulting image reference
is `<endpoint>/<repository>@sha256:<digest>`. Registry credentials remain in
operator-only state.

`admin_token_file` and registry credentials stay operator-only. Public trust
material is copied only where required. Secret contents and absolute operator
paths must not enter startup kits, image build contexts, Pod YAML, or public
manifests.

The CoCo `build_command` and a participant's `workload.source` have
different responsibilities. `workload.source` declares the application
context and Dockerfile. `CCPackager` copies those inputs into private
staging, adds the signed NVFlare kit and generated Dockerfile, and emits a
validated build-request file. `build_command` is the trusted project-wide
executable that consumes that request and runs the reviewed build, encryption,
signing, publication, policy-generation, and handoff pipeline. It cannot
replace or override the participant workload source; there is no precedence
between the two. The participant cannot select an executable.

### Common participant schema

```yaml
schema_version: 1
cc_deployment_mode: bare_metal_cvm  # bare_metal_cvm | coco | azure_cc
cpu_tee: intel_tdx                  # intel_tdx | amd_sev_snp
gpu_tee: none                       # none | nvidia_cc

attestation:
  service: trustee

class_allow_list: []

workload:
  source:
    type: docker_archive
    path: /srv/nvflare/images/application.tar
```

| Field | Meaning |
|---|---|
| `schema_version` | Unified participant schema version |
| `cc_deployment_mode` | Required selector: `bare_metal_cvm`, `coco`, or `azure_cc` |
| `cpu_tee` | One target: `intel_tdx` or `amd_sev_snp` |
| `gpu_tee` | `none` or `nvidia_cc` |
| `attestation.service` | Name in `cc_project.yml.attestation_services` |
| `class_allow_list` | Additional reviewed, fully qualified application class paths |
| `workload.source` | Typed participant workload input |

`participant.type` supplies the role. A participant file must not contain
`role`. `cpu_tee` is a scalar so it has the same meaning in all modes. One
participant configuration produces one deployment target.

For a client, `gpu_tee` also controls the default FL resource capacity.
`gpu_tee: nvidia_cc` defaults an omitted `capacity.num_of_gpus` in
`project.yml` to `1`. A positive explicit value is preserved for a reviewed
multi-GPU profile. With `gpu_tee: none`, `capacity.num_of_gpus` must be zero
or omitted. `capacity.mem_per_gpu_in_GiB` remains an optional scheduling
minimum in `project.yml`.

`class_allow_list` is optional and normalizes to `[]`. An empty list means
“add no application classes to the framework's existing safe defaults.” It
never means unrestricted, and it does not disable the built-in classes needed
to run NVFlare. Every non-empty entry must be a reviewed, fully qualified
class name; wildcards and module-prefix grants are rejected.

`workload.source.type` is a closed enum whose allowed value also depends on
the deployment mode:

| Source type | Required fields | Meaning | Allowed mode |
|---|---|---|---|
| `docker_archive` | `path` | Existing Linux AMD64 Docker-save archive; its image identity is derived from archive metadata | `bare_metal_cvm` |
| `docker_build` | `context`, `dockerfile` | Application source copied to private staging and passed to the trusted CoCo release pipeline | `coco` |
| `external` | No additional fields | NVFlare does not build or copy an image; it provisions the signed kit and attestation components for an image/resource managed by the Azure workflow | `azure_cc` |

An `external` source does not skip CC validation or startup-kit signing. It
only declares that image construction, Azure image selection, and Azure
resource creation occur outside `CCPackager`. Unknown source fields and
mode/source combinations fail validation.

### Bare-metal CVM example

```yaml
schema_version: 1
cc_deployment_mode: bare_metal_cvm  # bare_metal_cvm | coco | azure_cc
cpu_tee: intel_tdx
gpu_tee: none
attestation:
  service: trustee

workload:
  source:
    type: docker_archive
    path: /srv/nvflare/images/application.tar

bare_metal_cvm:
  cvm_image: oci://registry.example.org/cvm/tdx_cpu@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
  storage:
    vault_size_gib: 16
    applog_size_gib: 1
    user_config_size_gib: 1
    user_data_size_gib: 1
  network:
    allowed_in_ports: []
    allowed_out_ports: [8443]
    allowed_in_cidrs: []
    allowed_out_cidrs: []
```

`cvm_image` remains either an immutable registry reference or a pulled
directory. Its approved contract must match `cpu_tee` and `gpu_tee`. The
application image identity comes from the Docker archive.

The network lists contain operator-requested additions, not the complete
generated policy. `BareMetalCVMDeployment` derives the federation, admin,
relay, and listener ports from the finalized signed startup kit, adds required
Trustee bootstrap egress from the approved CVM contract, and unions those
ports with this block. In this server example, ports 8002 and 8003 become
inbound rules automatically; clients receive outbound rules for the server
endpoint. Operators must not duplicate derived FL ports in participant YAML.

### CoCo example

```yaml
schema_version: 1
cc_deployment_mode: coco  # bare_metal_cvm | coco | azure_cc
cpu_tee: amd_sev_snp
gpu_tee: nvidia_cc
attestation:
  service: trustee

workload:
  source:
    type: docker_build
    context: ./site_coco
    dockerfile: Dockerfile

coco:
  release_name: site-coco-v1
  registry: coco_workloads
  registry_repository: workloads/site-coco
  platform_config_file: ../admin/platform.env
```

`registry` selects the named project-level registry. `registry_repository`
is only the path within that registry. `platform_config_file` contains the
authority-approved Kata/runtime profile and related platform pins; the unified
schema removes registry endpoint and publisher credentials from that file so
they have one owner in `cc_project.yml`.

The implementation derives the Kata runtime instead of exposing another
runtime selector:

| `cpu_tee` | `gpu_tee` | Kata runtime class |
|---|---|---|
| `amd_sev_snp` | `none` | `kata-qemu-snp` |
| `amd_sev_snp` | `nvidia_cc` | `kata-qemu-nvidia-gpu-snp` |
| `intel_tdx` | `none` | `kata-qemu-tdx` |
| `intel_tdx` | `nvidia_cc` | `kata-qemu-nvidia-gpu-tdx` |

### Azure CC example

```yaml
schema_version: 1
cc_deployment_mode: azure_cc  # bare_metal_cvm | coco | azure_cc
cpu_tee: amd_sev_snp
gpu_tee: none
attestation:
  service: azure_maa

workload:
  source:
    type: external

azure_cc:
  deployment_target: confidential_vm  # confidential_vm | confidential_container
```

`cc_deployment_mode` selects Azure; `deployment_target` chooses the Azure
execution unit. Azure settings must not be represented as Trustee settings
because MAA and Trustee have different trust and key-release semantics.

The first unified implementation deliberately matches the NVFlare 2.9 Azure
integration rather than every workload that the Azure platform can host. It
accepts only `cpu_tee: amd_sev_snp` and `gpu_tee: none` for `azure_cc`.
`AzureCCDeployment` rejects Intel TDX and NVIDIA confidential GPU requests
before generating kits. Supporting either later requires the Azure
authorizer, claim policy, deployment workflow, and end-to-end tests to be
extended together. Cloud product availability alone does not enable a
combination in this provisioning schema.

### Server configuration and cross-mode peers

The bare-metal example above can be used as `cc_server.yml` for
`server.example.com`; its role is derived as `server`. A server's
`cc_deployment_mode` describes only where that server runs and how its own
deliverable is packaged. It does not limit clients to the same mode.

`CCBuilder` computes one project-wide peer-verification matrix from all
protected participants. Each participant receives:

- issuers for its own deployment mode;
- verifiers for every protected peer mode it must accept; and
- a `required_site_verifier_ids` mapping from each peer identity to that
  peer's required verifier.

For example, a bare-metal server can accept a CoCo client and an Azure client
when its startup kit contains the CoCo and Azure verifier components. This
design also requires `BareMetalCVMDeployment` to install a Trustee-backed
NVFlare peer authorizer that turns a fresh, authenticated CVM appraisal into
an identity-bound peer proof; vault-key release by itself is not a peer proof.
CoCo and Azure participants then receive its verifier. Provisioning fails
before signing when any requested mode pair lacks an issuer, verifier,
authenticated site binding, or compatible timing policy. KBS resource release
and NVFlare peer verification remain separate decisions.

## Standard names

| Current name or location | Unified name | Notes |
|---|---|---|
| Top-level `cvm_vault` or `compute_env` | `cc_deployment_mode` | Required in each `cc_config` |
| `cvm_vault.participants` | Participant `cc_config` | One selection mechanism |
| `role` in `cc_config` | Removed | Derived from `participant.type` |
| `cc_cpu_mechanism`, `platforms` | `cpu_tee` | One scalar |
| `requires_gpu`, `cc_gpu` | `gpu_tee` | `none` or `nvidia_cc` |
| `cc_issuers` and Python class path | `attestation.service` | Resolves a typed service |
| `trustee.url` | `kbs_endpoint` | Trustee KBS endpoint |
| `trustee.ca` | `ca_cert_file` | KBS TLS CA |
| `trustee_public_key_file` | `attestation_signing_public_key_file` | Trustee AS signing key |
| `token_url` | `attestation_token_endpoint` | Trustee AS token endpoint |
| `token_expiration` | `token_expiration_seconds` | Explicit units |
| `cc_attestation.check_frequency` | `check_frequency_seconds` | Project service setting |
| Timeout fields without units | Names ending in `_seconds` | Project service settings |
| `cc_issuers[].args.proof_iat_leeway_seconds` | `proof_iat_leeway_seconds` | Project Trustee verifier policy |
| `cc_issuers[].args.workload_constraints` | `workload_constraints` | Complete project Trustee claim pins |
| `docker_archive` | `workload.source.type: docker_archive` | Common workload envelope |
| `image_build` | `workload.source.type: docker_build` | Common workload envelope |
| Existing Azure-managed image/resource | `workload.source.type: external` | Azure builds and resources remain outside `CCPackager` |
| `vault_drive_size` | `bare_metal_cvm.storage.vault_size_gib` | Mode-specific |
| `platform_config` | `coco.platform_config_file` | Mode-specific |
| Registry host in CoCo `platform.env` | `container_registries.<name>.endpoint` | Project-level service setting |
| Azure `compute_env` variants | `azure_cc.deployment_target` | Azure sub-target |

Distinct concepts remain distinct. A bare-metal `cvm_image`, a CoCo
encrypted workload image, and an Azure VM image occupy different layers and
must not share a misleading name.

## Provisioning interface

### Proposed package layout

```text
nvflare/lighter/cc_provision/
  config.py                         # common schema and service resolution
  deployment.py                     # interface, plan, and result types
  impl/
    cc.py                           # CCBuilder and mode registry
    cc_packager.py                  # common post-signing packager
    bare_metal_cvm.py               # BareMetalCVMDeployment
    coco.py                         # CoCoDeployment
    coco_release.py                 # private adapter for the trusted release worker
    azure_cc.py                     # AzureCCDeployment
```

CVM image and vault construction remain in
`nvflare/lighter/cc/image_builder`. The bare-metal class adapts that code to
the interface.

### Interface

```python
@dataclass(frozen=True)
class CCDeploymentPlan:
    participant_name: str
    participant_type: str
    config_path: Path
    mode: CCDeploymentMode
    cpu_tee: CPUTEE
    gpu_tee: GPUTEE
    attestation_service: ResolvedAttestationService
    class_allow_list: tuple[str, ...]
    workload_source: WorkloadSource
    mode_config: Mapping[str, Any]
    internal: Mapping[str, Any] = field(default_factory=dict, repr=False)


class CCDeployment(ABC):
    mode: CCDeploymentMode

    @abstractmethod
    def validate_project_config(self, project_config: dict) -> None:
        pass

    @abstractmethod
    def create_plan(
        self,
        participant: Participant,
        cc_config: dict,
        project_config: dict,
    ) -> CCDeploymentPlan:
        pass

    def configure_startup_kit(
        self,
        plan: CCDeploymentPlan,
        project: Project,
        ctx: ProvisionContext,
    ) -> None:
        return None

    @abstractmethod
    def package(
        self,
        plan: CCDeploymentPlan,
        private_kit: Path,
        public_output: Path,
        ctx: ProvisionContext,
    ) -> CCDeploymentResult:
        pass
```

`CCDeploymentPlan` contains normalized values, resolved private input paths,
participant identity and role, and validated mode-specific data. Deployment
methods do not receive unvalidated YAML. `CCDeploymentResult` contains public
artifact records.

`configure_startup_kit()` has an explicit no-op default. A mode overrides it
when it must change mode-specific kit content before signing. Shared authorizer
and `CCManager` assembly stays in `CCBuilder` and must not be duplicated in
mode classes. The current CoCo implementation overrides this hook to suppress
host-visible startup output; bare-metal CVM and Azure use the no-op default.

The registry is explicit:

```python
DEPLOYMENTS = {
    CCDeploymentMode.BARE_METAL_CVM: BareMetalCVMDeployment,
    CCDeploymentMode.COCO: CoCoDeployment,
    CCDeploymentMode.AZURE_CC: AzureCCDeployment,
}
```

Missing or unknown `cc_deployment_mode` fails before kit generation.
Configuration cannot supply arbitrary deployment class paths.

### Lifecycle

`CCBuilder`:

1. loads `cc_project.yml`, validates its exact common schema and every named
   service once;
2. loads every participant `cc_config`;
3. requires `schema_version` and `cc_deployment_mode`;
4. validates common fields and resolves `attestation.service`;
5. determines the set of modes used by participants;
6. calls `validate_project_config()` for every used mode and for every
   mode-specific project block that is present, even when unused;
7. selects the class and creates an immutable plan;
8. builds the project-wide peer-verification matrix;
9. marks the participant CC-enabled so the complete kit is signed;
10. installs common `CCManager` settings where the mode uses peer
    attestation; and
11. calls `configure_startup_kit()`.

An omitted block for an unused mode is valid. A used mode must have all of its
required project settings. Any supplied mode-specific block is validated even
when no participant currently selects that mode, preventing dormant invalid
configuration from being accepted.

`CCPackager`:

1. verifies every selected signed kit;
2. moves all protected plaintext kits into private staging before starting an
   external build;
3. calls the selected class's `package()` for each participant;
4. retains recovery input, logs, credentials, and receipts in private state;
5. publishes only approved artifacts and a common manifest; and
6. succeeds only when every selected participant validates.

This removes the special top-level `cvm_vault` branch, the separate
`VaultAdapter` activation path, and the exclusive CoCo packager. Different
modes can coexist in one project because the common packager owns the private
and public handoff for each participant.

### Class responsibilities

| Class | Pre-signing work | Packaging work | Public artifact |
|---|---|---|---|
| `BareMetalCVMDeployment` | Install the Trustee-backed peer authorizer/manager and derive runtime/network settings | Validate CVM contract, build vault, install KBS resource, validate OCI | Complete CVM OCI artifact |
| `CoCoDeployment` | Install authorizer/manager; redirect startup-process stdout and stderr to `/dev/null` before signing so an untrusted host cannot collect sensitive console output, while retaining guest-local NVFlare file logs | Build/encrypt image, create Trustee handoff, validate measured Pod | Pod YAML and image metadata |
| `AzureCCDeployment` | Install Azure authorizer and manager; reject non-SNP or GPU targets in the initial 2.9 implementation | Validate/publish signed kit and sanitized deployment references | Startup kit and Azure metadata |

The interface standardizes provisioning; it does not make the artifacts or
attestation services interchangeable.

## Common result manifest

```json
{
  "schema": "nvflare-cc-delivery/v1",
  "participant": "site-coco",
  "cc_deployment_mode": "coco",
  "cpu_tee": "amd_sev_snp",
  "gpu_tee": "nvidia_cc",
  "attestation_service": "trustee",
  "artifacts": [
    {
      "type": "coco_pod",
      "path": "site-coco-v1-pod.yaml",
      "sha256": "<sha256>"
    }
  ]
}
```

Bare-metal adds public OCI/CVM identities, CoCo adds the encrypted-image and
launch-profile identities, and Azure adds the startup-kit checksum and target.
Secrets, private receipts, and operator paths are forbidden.

## Documentation

Create one canonical guide, for example
`docs/user_guide/confidential_computing/deployment.rst`, containing:

1. concepts, trust boundaries, and mode selection;
2. common `project.yml`, `cc_project.yml`, and `cc_config` usage;
3. the common field reference;
4. Trustee/KBS setup once for bare-metal and CoCo;
5. bare-metal settings, artifacts, and launch;
6. CoCo settings, artifacts, and Kubernetes launch;
7. Azure settings, artifacts, and launch; and
8. common results, troubleshooting, and migration.

Mode sections contain only their differences. Example runbooks may retain
detailed commands, but they link to the canonical Trustee/KBS section instead
of duplicating endpoints, certificates, tokens, retry policy, or timeouts.

## Migration plan

1. Add common schema types, exact validation, service resolution, the interface,
   and plan/result types with unit tests.
2. Put Azure behavior behind `AzureCCDeployment`.
3. Put CoCo builder/packager behavior behind `CoCoDeployment`, preserving all
   private staging and fail-closed checks.
4. Adapt `VaultAdapter` into `BareMetalCVMDeployment`, then remove the
   `cvm_vault` branch from `provision()`.
5. Add `CCPackager`, common signing, mixed-mode integration tests, and common
   result manifests.
6. Convert all examples to participant `cc_config` plus one
   `cc_project.yml`.
7. Consolidate user documentation and replace duplicate common sections with
   links.
8. Remove old field names after all in-repository callers and tests use the
   unified schema.

The unified schema is intentionally a breaking CC configuration change.
Runtime backward compatibility is out of scope: `cvm_vault`, `compute_env`,
and the old field names are rejected with an error that points to this
migration table. All in-repository examples and tests move in the same change.
A standalone conversion tool can be proposed separately, but the
implementation does not depend on one and deployment classes contain no
legacy aliases.

## Validation

- Reject missing, unknown, or non-string `cc_deployment_mode`.
- Reject `cc_config` on participant types other than server or client.
- Reject `role`, raw Python authorizer paths, and fields from another mode.
- Validate paths relative to their declaring YAML file.
- Verify that `class_allow_list: []` adds no custom classes and never enables
  unrestricted loading.
- Reject unknown workload sources and invalid source/mode combinations.
- Verify that `external` produces a signed Azure kit without invoking an
  image builder.
- Verify that CoCo's trusted `build_command` receives the staged
  `docker_build` source and cannot replace it.
- Require a named CoCo registry, validate its endpoint and CA, and keep its
  publisher credentials out of public output.
- Run the same common-schema tests for all three implementations.
- Run one Trustee fixture through bare-metal and CoCo validation.
- Validate every supplied mode-specific project block, including unused
  blocks, and require settings for every used mode.
- Prevent participant overrides of project-wide Trustee policy.
- Reject `azure_cc` with `intel_tdx` or `nvidia_cc` in the initial
  implementation.
- Test a protected server with clients in different deployment modes and fail
  when any required verifier pair is unavailable.
- Verify signed kits before packaging and retain recovery evidence on failure.
- Prove bare-metal and CoCo plaintext kits never enter public output.
- Validate checksums and immutable image or OCI digests.
- Test a project with one participant in each mode.

The end-to-end matrix must cover TDX CPU-only and SNP GPU cases for both
bare-metal and CoCo, plus SNP CPU-only Azure confidential VM and confidential
container cases. Negative tests cover the intentionally unsupported Azure TDX
and GPU combinations.

## Acceptance criteria

- Every confidential participant is selected with `cc_config`.
- Every participant file requires one of the three `cc_deployment_mode`
  values.
- One interface and three registered classes implement provisioning.
- Common concepts use the standardized names and types.
- Project-wide services and trusted tools live in `cc_project.yml`.
- Bare-metal and CoCo consume the same Trustee/KBS object.
- CoCo workload input, trusted build command, and named registry have
  independent, unambiguous ownership.
- Empty class allow-lists and all workload-source types have fail-closed
  semantics.
- The server's deployment mode does not restrict client modes; the generated
  peer-verification matrix covers every protected participant.
- Mixed-mode provisioning does not expose plaintext protected kits.
- Every mode produces the common result manifest.
- One guide covers all modes and defines common configuration only once.
