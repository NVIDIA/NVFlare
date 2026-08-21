# NVFlare stages 40 through 60 for Confidential Containers

This guide is for host IT administrators and Kubernetes cluster administrators
who have completed shared stages 00 through 30 in `USER_GUIDE.md` and want to
run one NVFlare server and one GPU-enabled NVFlare client in that AMD SEV-SNP
or Intel TDX cluster. It covers only the alternative NVFlare workload stages:

- `40-nvflare-build-sign-images.sh` builds NVFlare into separate server and
  client images, pushes them to the suite's private registry, signs their
  immutable digests, and installs a deny-by-default verification policy.
- `50-nvflare-deploy.sh` provisions the NVFlare identities and deploys the
  server and client as digest-pinned Kata Confidential Container Pods.
- `60-nvflare-submit-hello-numpy.sh` submits the standard one-client
  `hello-numpy` job and requires it to finish without an NVFlare execution
  error.

From the Stage-30 fork point, run these exact filenames in order:

```bash
./40-nvflare-build-sign-images.sh
./50-nvflare-deploy.sh
./60-nvflare-submit-hello-numpy.sh
```

Run those exact filenames in order. The NVFlare stages 40 through 60 replace the
generic `40-sign-image.sh` and `50-launch-and-attest.sh` workload stages; do not
run both variants or use a `*-*.sh` glob. The original `run-all.sh` continues
to run only the generic attestation sample.

## Security model and deployment layout

The server and client are long-running NVFlare parent processes. Each parent
runs in its own Kata confidential VM:

```text
Kubernetes node (SNP or TDX)
  |
  +-- nvflare-server Deployment
  |     runtimeClass: kata-qemu-{snp,tdx}
  |     no GPU request or limit
  |     /dev/{sev-guest,tdx_guest} created inside the Kata VM
  |     CPU evidence issuer; CPU/GPU peer-token verifiers
  |     signed server image at an immutable sha256 digest
  |     provisioned server startup kit
  |
  +-- site-1 Deployment
        runtimeClass: kata-qemu-nvidia-gpu-{snp,tdx}
        /dev/{sev-guest,tdx_guest} created inside the Kata VM
        signed client-parent image at an immutable sha256 digest
        one passthrough nvidia.com/pgpu device
        CPU and GPU evidence issuers; CPU/GPU peer-token verifiers
        provisioned client startup kit
```

The server is a CPU-only confidential VM. It uses the platform's non-GPU CoCo
RuntimeClass and its Pod specification contains no extended GPU resource. The
client uses the GPU-enabled CoCo RuntimeClass and is the only Pod that requests
`nvidia.com/pgpu`. Consequently, on a one-GPU bare-metal host, the sole GPU is
assigned exclusively to the client. The server and client can run
simultaneously because the server does not reserve or attach that device.

The generated NVFlare configuration uses the process job launcher. Server and
client job processes therefore run inside their existing confidential VM and
inside the already verified parent image. This is deliberate. A conventional
NVFlare Kubernetes job launcher creates new job Pods dynamically; those Pods
would also need the Kata RuntimeClass, measured init-data, digest pinning, and
the image-signature policy. Do not switch this example to the Kubernetes job
launcher until those controls are added to every generated job Pod.

The two parent images have different purposes:

| Image | Dockerfile | Contents and intended use |
|---|---|---|
| Server | `docker/Dockerfile.coco` | NVIDIA Distroless Python runtime with NVFlare, NumPy, the standard `hello-numpy` trainer and TensorBoard receiver, `snpguest`, Intel Trust Authority CLI, and the native NVAT `nvattest` CLI. It deliberately excludes the Kubernetes Python package and has no shell or package manager. |
| Client | `docker/Dockerfile.coco` with a client-role OCI label | The same signed `hello-numpy` trainer, NumPy/TensorBoard, and pinned attestation tool set in a separate repository and at a separately signed digest. It deliberately excludes the Kubernetes Python package. The Pod requests the passthrough GPU, but this parent image intentionally omits the large CUDA/PyTorch training stack. |

Both images are built from the enclosing NVFlare checkout. `COPY . .` places
the build context in the builder, and each Dockerfile installs that source with
pip. The image therefore contains the NVFlare code from the checkout being
built; it does not depend on a separately published NVFlare wheel.

## Starting point and prerequisites

First follow `USER_GUIDE.md` through the successful completion of
`30-deploy-security-services.sh`. This guide begins at the choice immediately
after that stage; it does not repeat host preparation, Kubernetes installation,
CoCo/GPU installation, or security-service deployment.

Before Stage 40, confirm that the shared workflow reported `Registry and Trustee
KBS are ready`. Its completed state must include the platform-specific Kata
RuntimeClasses, `nvidia.com/pgpu` capacity, VFIO-bound GPU, released
`block-encrypted` confidential-`emptyDir` support, registry and Trustee
Deployments, registry CA, and Cosign key pair. See `USER_GUIDE.md` if any shared
prerequisite is absent.

The shared registry is digest-pinned and binds its host port only to
`REGISTRY_BIND_ADDRESS` (normally `NODE_IP`), but this demo registry has no
client authentication and permits push/delete. Firewall it to trusted
administrators. Runtime pulls remain protected by immutable digests, Cosign,
and the measured deny-by-default image policy.

Stages 30 and 50 treat `NVFLARE_NAMESPACE` as a dedicated security boundary.
When first needed, they create it with
`nvflare.nvidia.com/coco-managed=true`. If the namespace already exists without
that label, they fail before writing hardware-authorizer or participant
credentials and before changing Pod Security Admission. Never add the ownership
label to a shared namespace. For a dedicated namespace created by an older
version of this workflow, inspect its workloads, RBAC, Secrets, and ConfigMaps
before explicitly labeling it for migration.

Both participant containers use `privileged: true` inside their isolated Kata
VM so they can materialize `/dev/sev-guest` or `/dev/tdx_guest` from guest
sysfs. Restricted or baseline Pod Security Admission would reject them. After
the ownership check, Stage 50 performs the equivalent of the following before
applying the Deployments:

```bash
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf \
  label namespace "$NVFLARE_NAMESPACE" \
  pod-security.kubernetes.io/enforce=privileged --overwrite
```

Restrict who can create or modify Pods and who can read Secrets in this
namespace. The admission label permits privileged admission for every workload
there; Kata isolation is established separately by each Pod's RuntimeClass.

On TDX, Stage 30 also requires an Intel Trust Authority `config.json` through
`NVFLARE_TDX_ATTESTATION_CONFIG_FILE`. It must be a JSON object containing a
nonempty `trustauthority_url`, an HTTPS `trustauthority_api_url`, and a nonempty
`trustauthority_api_key`. That key is distinct from an Intel PCS subscription
key. If Stage 30 was run before setting this file, set it and rerun Stage 30;
the script validates it without printing its contents and creates
`nvflare-tdx-attestation` in the NVFlare namespace.

Stage 50 defaults to self-contained local NVAT verification through
`NVFLARE_GPU_VERIFIER=local`; that mode does not use a service key. To select
remote NRAS verification, set `NVFLARE_GPU_VERIFIER=remote` and, when required,
set
`NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE` to a root-readable, one-line service
key file before Stage 30. Stage 30 stores it in `nvflare-gpu-attestation`, and
the Pod template exposes it to NVAT under its official
`NV_ATTESTATION_SERVICE_KEY` environment variable. `NVFLARE_GPU_NRAS_URL`
selects the HTTPS remote service. When constructing `GPUAuthorizer`
directly, a non-`None` `service_key` argument takes precedence; the environment
variable is consulted only when that argument is `None`.

The deployment stage needs the `nvflare` provisioning CLI on the host. Create a
virtual environment from the same checkout used as the image build context:

```bash
cd ../../..
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[K8S]' tensorboard
cd examples/devops/CoCo
```

This repository's `main` branch is development for the next NVFlare release.
Use the checkout installation above when that version is not yet published on
PyPI. `50-nvflare-deploy.sh` and `60-nvflare-submit-hello-numpy.sh` discover
`nvflare` on `PATH`, then `NVFlare/.venv/bin/nvflare`; `NVFLARE_CLI` can select
another executable. Stage 60 uses the interpreter beside that CLI, then
`python3`; set `NVFLARE_PYTHON` when neither is the intended environment.

The image build needs outbound access to the base-image registries and Python
package sources. The script installs Podman if it is absent. Docker is not
required. Registry, signing, and Kubernetes commands require passwordless
non-interactive `sudo` when the scripts are not run as root.

## Configure the NVFlare deployment

Review these values in `config.env`:

| Variable | Default | Meaning |
|---|---|---|
| `NVFLARE_NAMESPACE` | `nvflare-coco` | Dedicated Kubernetes namespace for the server and client. Stages 30/50 create it with `nvflare.nvidia.com/coco-managed=true` and refuse an existing namespace without that ownership label. |
| `NVFLARE_PROJECT_NAME` | `nvflare_coco` | NVFlare provisioned project name. |
| `NVFLARE_SERVER_NAME` | `nvflare-server` | NVFlare server identity, Service, and Deployment name. |
| `NVFLARE_CLIENT_NAME` | `site-1` | NVFlare client identity and Deployment name. |
| `NVFLARE_ADMIN_NAME` | `admin@nvidia.com` | Provisioned project-administrator identity. |
| `NVFLARE_ORG` | `nvidia` | Organization recorded in all three identities. |
| `NVFLARE_SERVER_IMAGE_NAME` | `coco/nvflare-server` | Server repository below the private registry. |
| `NVFLARE_CLIENT_IMAGE_NAME` | `coco/nvflare-client` | Client repository below the private registry. |
| `NVFLARE_SERVER_IMAGE_TAG` | `dev` | Mutable server distribution tag; deployment does not use it after digest resolution. |
| `NVFLARE_CLIENT_IMAGE_TAG` | `dev` | Mutable client distribution tag; deployment does not use it after digest resolution. |
| `NVFLARE_SERVER_MEMORY` | `8Gi` | Server Pod memory request and limit. |
| `NVFLARE_CLIENT_MEMORY` | `32Gi` | Client Pod memory request and limit. |
| `NVFLARE_SERVER_RUNTIME_CLASS` | `auto` | Selects `kata-qemu-snp` or `kata-qemu-tdx`, the CPU-only confidential runtime used by the server. |
| `NVFLARE_CLIENT_RUNTIME_CLASS` | `auto` | Selects `kata-qemu-nvidia-gpu-snp` or `kata-qemu-nvidia-gpu-tdx`, the GPU-enabled confidential runtime used by the client. |
| `NVFLARE_CLIENT_GPU_COUNT` | `1` | Number of passthrough GPU resources requested and limited by the client only. |
| `NVFLARE_WORKSPACE_SIZE` | `16Gi` | Ephemeral encrypted workspace limit for each participant. |
| `NVFLARE_FS_GROUP` | `65532` | Pod `fsGroup` applied to each writable encrypted workspace. The NVFlare process starts as root only inside its isolated Kata VM because the CPU evidence clients require guest-device/configfs access. |
| `NVFLARE_CLI` | empty | Optional absolute path to the host provisioning CLI. |
| `NVFLARE_PYTHON` | empty | Optional Python interpreter used to export `hello-numpy`; it must contain NVFlare, NumPy, and the configured TensorBoard version. |
| `NVFLARE_ADMIN_LOCAL_PORT` | `18003` | Loopback port used by Stage 60's temporary host-to-Kata administration tunnel to server port 8003. |
| `NVFLARE_CLI_CONNECT_TIMEOUT` | `30` | Seconds allowed for each Stage-60 administrator CLI connection. |
| `NVFLARE_JOB_WAIT_TIMEOUT` | `900` | Maximum seconds Stage 60 waits for the job to reach a terminal state. |
| `NVFLARE_JOB_WAIT_INTERVAL` | `2` | Seconds between Stage-60 job-status polls. |
| `TENSORBOARD_VERSION` | `2.20.0` | TensorBoard version installed and runtime-checked in both signed parent images for the standard `hello-numpy` receiver. |
| `NVFLARE_COCO_PYTHON_BUILD_BASE` / `NVFLARE_COCO_PYTHON_RUNTIME_BASE` | digest-pinned Python 3.12 images | ABI-matched builder and NVIDIA Distroless runtime used only by Stage 40. This is the validated image combination; NVAT itself is a native CLI and has no Python ABI dependency. |
| `NVFLARE_COCO_PYTHON_MINOR` | `3.12` | Python ABI directory used when copying the built NVFlare installation into the runtime image; it must match both base images. |
| `SNPGUEST_VERSION` | `0.10.0` | Pinned `snpguest` release installed in both images. |
| `TRUSTAUTHORITY_CLI_VERSION` | `v1.10.1` | Pinned Intel Trust Authority CLI installed in both images. |
| `NVAT_VERSION` | `1.2.2` | Pinned NVIDIA NVAT CLI and native library installed in both images. |
| `NVAT_BUILD_BASE` | digest-pinned Ubuntu 24.04 image | Resolves NVAT's supported native dependency closure; Stage 40 relocates it with private RPATHs into the final distroless image. |
| `AMD_KDS_PRODUCT` | `Genoa` | AMD product generation passed to `snpguest certificates`; set it to the generation used by the SNP host. |
| `NVFLARE_TDX_ATTESTATION_CONFIG_FILE` | empty | Required on TDX before Stage 30; path to a secret Trust Authority `config.json`. |
| `NVFLARE_GPU_ATTESTATION_SERVICE_KEY_FILE` | empty | Optional one-line NVIDIA attestation service key file staged by Stage 30 and injected as a Secret-backed environment variable. |
| `NVFLARE_GPU_VERIFIER` | `local` | NVAT verifier mode used by both participants. Set `remote` only with approved NRAS access. |
| `NVFLARE_GPU_NRAS_URL` | `https://nras.attestation.nvidia.com` | HTTPS endpoint used only when `NVFLARE_GPU_VERIFIER=remote`. |

The server and client names must be lowercase Kubernetes RFC 1123 names. The
server and client image repositories must be different. `GPU_RESOURCE` remains
the common suite setting and normally has the value `nvidia.com/pgpu`.
The direct-download URLs and SHA-256 pins remain in `config.env`; their audited
defaults and exact artifact meanings are recorded in `CHECKSUMS.md`. Treat a
change to any of those values as a supply-chain review, not routine deployment
configuration.

Leave both RuntimeClass settings at `auto` unless the cluster administrator has
reviewed an equivalent custom Kata installation. `RUNTIME_CLASS` remains the
GPU RuntimeClass used by the generic stage-50 attestation sample and is also the
default source for `NVFLARE_CLIENT_RUNTIME_CLASS`; it is not used by the NVFlare
server. Stage 50 fails before provisioning if either selected RuntimeClass is
absent. It also verifies after deployment that the server has no GPU request
and that both the client's GPU request and limit equal
`NVFLARE_CLIENT_GPU_COUNT`.

`REGISTRY_HOST` is derived from `NODE_IP` and `REGISTRY_PORT` and must match the
registry certificate and services created by the completed shared workflow. If
that address must change, return to `USER_GUIDE.md` before running Stage 40.

## Stage 40: build, push, sign, and authorize the images

Run the NVFlare image stage after stage 30 succeeds:

```bash
./40-nvflare-build-sign-images.sh
```

The script fails if the stage-30 signing material, registry deployment, or KBS
deployment is absent. It then performs the following operations:

1. Builds `docker/Dockerfile.coco` as both compact parent-process images,
   with pinned `snpguest`, Intel Trust Authority CLI, and
   NVIDIA NVAT installation. The direct artifacts are SHA-256 checked,
   and Stage 40 starts each completed runtime image to check both CLI versions,
   the NVAT CLI, all three NVFlare authorizer imports, and the absence of the
   Kubernetes Python package before applying
   distinct server/client OCI role labels and pushing the images to separate
   repositories. The NGC job image is intentionally not used as the long-running
   client root filesystem: its expanded size can exceed confidential-guest image
   storage. The script uses rootful Podman storage
   below `state/`, not the Docker daemon or a global Podman image store. The
   build uses the host network because some TDX host firewall configurations
   do not provide package-repository egress to Podman's private build network.
2. Pushes both tagged images to the TLS registry and validates the registry
   with `state/registry-tls/ca.crt`.
3. Reads each digest back from the registry with Skopeo. From this point on,
   the authoritative references have the form
   `REGISTRY/repository@sha256:...`; tags are not trusted deployment inputs.
4. Signs each immutable digest with `state/security/cosign.key` and immediately
   verifies it with `state/security/cosign.pub`.
5. Writes a policy whose default action is `reject` and whose only allowed
   repositories require a `sigstoreSigned` signature from that public key.
6. Updates Trustee's public verification-resource Secret with that policy, the
   public key, and the private registry CA.

The key outputs are:

| File | Purpose |
|---|---|
| `state/nvflare-images.env` | Exact digest references consumed by stage 50. |
| `state/nvflare-image-security-policy.json` | Deny-by-default guest image policy for the two NVFlare repositories. |
| `state/nvflare-server-cosign-verification.json` | Host-side server signature-verification result. |
| `state/nvflare-client-cosign-verification.json` | Host-side client signature-verification result. |
| `state/nvflare-image-signing-report.txt` | Source revision, Dockerfiles, signed references, and overall result. |

For each recorded digest, the script's signing operation is equivalent to:

```bash
COSIGN_PASSWORD="${COSIGN_PASSWORD:-}" cosign sign --yes \
  --tlog-upload=false --allow-insecure-registry \
  --key state/security/cosign.key \
  'REGISTRY/repository@sha256:APPROVED_DIGEST'
```

The actual reference is resolved from the registry and is never constructed
from a tag alone. Rekor mode changes `--tlog-upload` to `true`. Let the script
perform this operation so the signed references, policy, verification results,
and deployment input stay synchronized.

Inspect the result without printing the private key:

```bash
sed -n '1,120p' state/nvflare-image-signing-report.txt
. state/nvflare-images.env
cosign verify --allow-insecure-registry --insecure-ignore-tlog \
  --key state/security/cosign.pub "$NVFLARE_SERVER_IMAGE"
cosign verify --allow-insecure-registry --insecure-ignore-tlog \
  --key state/security/cosign.pub "$NVFLARE_CLIENT_IMAGE"
jq . state/nvflare-image-security-policy.json
```

Those verification commands match the default
`COSIGN_TLOG_MODE=disabled`. In that mode the cryptographic signature is
verified, but no public transparency-log inclusion is claimed. Set
`COSIGN_TLOG_MODE=rekor` before stage 40 to upload and require public Rekor
evidence; this requires network access and publishes signature metadata.

Treat `state/security/cosign.key` as a secret. Never commit it, place it in an
image, or copy it into the workload namespace. A production deployment should
use a protected KMS/HSM or an approved keyless identity and an auditable key
rotation process.

### Source and base-image provenance

The Cosign signature protects the exact bytes represented by the final registry
digest. It does not prove that the source checkout was reviewed, that the build
host was trustworthy, or that a mutable base-image tag had a particular value.
`Dockerfile.coco` pins its bases by digest. If you separately build
`Dockerfile.job` for GPU training, its NGC PyTorch base currently uses a mutable
tag. For a production build:

- build from a reviewed, clean commit and record that commit independently;
- pin every base image by digest;
- build in a controlled CI builder;
- produce and retain an SBOM and signed provenance attestation; and
- promote only an independently approved final digest.

Re-signing an image authorizes it under the configured repository policy. Limit
access to the signing key accordingly.

### Include NVFlare authorizer dependencies in the signed images

If the deployment enables the authorizers under
`nvflare/app_opt/confidential_computing`, every Python module, executable,
configuration file, trust root, and native library used by the selected
authorizers must be present in the image that runs them. Installing a tool on
the bare-metal host does not make it available inside a Kata confidential VM.
Do not download these dependencies after launch: add pinned, reviewed artifacts
at build time, validate them in the Dockerfile, and let Stage 40 sign the
resulting immutable digest.

The authorizer-specific requirements visible in the current NVFlare source are:

| Authorizer | Content required in the image | Runtime requirements outside the image |
|---|---|---|
| `SNPAuthorizer` (`snp_authorizer.py`) | Pinned `snpguest` at `/opt/attestation/bin/snpguest`; Stage 50 puts its certificate cache under the encrypted workspace and selects `AMD_KDS_PRODUCT`. | `/dev/sev-guest` created from the guest's own sysfs device identity and egress to AMD KDS. |
| `TDXAuthorizer` (`tdx_authorizer.py`) | Pinned Intel Trust Authority CLI at `/opt/attestation/bin/trustauthority-cli`, its runtime libraries/trust roots, and a read-only secret `config.json`. | `/dev/tdx_guest`, configfs TSM, QGS, and egress to the configured Intel Trust Authority endpoints. |
| `GPUAuthorizer` (`gpu_authorizer.py`) | Pinned native NVAT `nvattest` CLI at `/opt/attestation/bin/nvattest`, `libnvat`, its runtime libraries, CA certificates, and the default Rego relying-party policy embedded in the authorizer. | A GPU assigned to the issuing VM and the required guest GPU device/driver interfaces. This workflow defaults to NVAT local verification; remote mode uses NVIDIA NRAS and verifier egress. For authenticated NRAS, inject the key as `NV_ATTESTATION_SERVICE_KEY` from a Kubernetes Secret or pass the `service_key` constructor argument. A non-`None` constructor argument takes precedence over the environment. Never bake the key into the image or startup kit. A CPU-only peer verifies the client's transported evidence from a temporary file and does not need a GPU. |

Both role images contain the verifier dependencies because every participant
must verify its peers. Only evidence issuance follows the hardware assignment:

| Image | AMD SEV-SNP host | Intel TDX host | GPU authorizer |
|---|---|---|---|
| Server | Issues SNP evidence on an SNP host and verifies peer SNP evidence. | Issues TDX evidence on a TDX host and verifies peer TDX evidence. | Contains NVAT and verifies client GPU evidence, but does not issue GPU evidence and receives no GPU. |
| Client | Issues and verifies SNP evidence on an SNP host. | Issues and verifies TDX evidence on a TDX host. | Issues and verifies nonce-bound GPU evidence; it is the only Pod receiving a GPU. |

Stage 40 uses `docker/Dockerfile.coco` for both builds. The supplied Dockerfile
always installs the attestation tools, base NVFlare, and TensorBoard without
the `K8S` extra, and verifies that the Kubernetes Python package is absent. Stage 40
validates the pinned downloads and performs runtime checks
equivalent to:

```bash
python -c 'from nvflare.app_opt.confidential_computing.snp_authorizer import SNPAuthorizer'
python -c 'from nvflare.app_opt.confidential_computing.tdx_authorizer import TDXAuthorizer'
python -c 'from nvflare.app_opt.confidential_computing.gpu_authorizer import GPUAuthorizer'
python -c 'import numpy, tensorboard; assert tensorboard.__version__ == "2.20.0"'
python -c 'import importlib.util; assert importlib.util.find_spec("kubernetes") is None'
/opt/attestation/bin/snpguest --version
/opt/attestation/bin/trustauthority-cli version
/opt/attestation/bin/nvattest version
```

These checks run for both images. They do not replace the Stage-50 in-guest
device check and NVFlare peer authorization handshake.

`TDXAuthorizer` invokes the pinned Trust Authority CLI directly, captures
output without shared files, checks its exit status, requires
exactly one compact JWT in stdout, bounds token size, and enforces a command
timeout. It retains SHA-256 fingerprints of successfully verified JWTs and
rejects replay. The class's default `use_sudo=None` chooses non-interactive
sudo only for a non-root host process when sudo is available; Stage 50 sets
`use_sudo: false` because the container starts as root inside the isolated Kata
VM and contains no sudo. The default token command omits a TEE selector because
the pinned v1.10.1 CLI selects TDX in that case, preserving compatibility with
older CLIs that do not recognize `--tdx`. `token_options` permits reviewed,
version-specific token flags without parsing informational output as a token.
Hardware testing also found a narrow clock-boundary condition: Intel Trust
Authority can issue a token whose `nbf` (not-before) time is about one second
ahead of an otherwise NTP-synchronized TDX guest. Immediate CLI verification
then reports `token is not valid yet`. `TDXAuthorizer` handles only that exact
diagnostic from either stdout or stderr with a bounded five-attempt window,
waiting two seconds between attempts. It does not retry
signature, collateral, configuration, timeout, or other verification failures,
and it never writes verifier stderr (which can contain the bearer token) to the
NVFlare log. A not-before failure after that eight-second window remains
fail-closed and should be
investigated as a host, guest, or service clock-synchronization problem.

`GPUAuthorizer` uses NVIDIA's supported native NVAT client and has no
`nv_attestation_sdk` dependency. The issuer runs `nvattest collect-evidence`
with a fresh 32-byte nonce and transports the resulting JSON evidence. The peer
runs `nvattest attest` against that file, requires successful NVAT appraisal and
the configured Rego policy, checks the verified claims nonce, and only then
updates its replay cache. NVAT 1.2.2 specifies a distinct failure when
`nv_match` is false; for the built-in policy the authorizer additionally checks
every secure-boot, debug, signature, certificate/OCSP, RIM, and mismatch claim
itself. Thus a zero CLI/result code with weak claims still fails closed. NVAT
failures, timeouts, malformed evidence, policy mismatches, nonce mismatches,
and replays all fail closed.

GPU generation and verification are not serialized: every invocation uses
isolated temporary files, while the replay cache is internally synchronized.
Stage 50 provisions `get_token_request_timeout: 200`, above NVAT's 180-second
command limit, so the CC peer request does not expire first and shut down the
federation during a slow appraisal.

`SNPAuthorizer` retries only report generation and AMD CA/VCEK network fetches.
Local `snpguest verify attestation` runs once; deterministic signature failure
does not enter the exponential network retry loop.

If overriding `policy_file`, supply an NVAT Rego policy that defines the
boolean rule `nv_match`. The JSON policy format accepted by the retired Python
SDK is not compatible with NVAT and is rejected during authorizer
initialization. The legacy `verifier_url` constructor argument remains an alias
for `nras_url` to ease configuration migration.

Dependencies alone do not activate authorization. Stage 50 now generates a
platform-specific CPU authorizer configuration for both participants, adds the
GPU issuer only to the client, and provisions both through `CCBuilder`. It
fails if `CCBuilder` rejects a configuration or omits any expected manager or
authorizer resource.

### Generate authorizer-configured startup kits with `CCBuilder`

NVFlare can add confidential-computing authorization configuration to startup
kits during `nvflare provision`. The
[Azure confidential-VM deployment guide](https://nvflare.readthedocs.io/en/main/user_guide/confidential_computing/azure/azure_confidential_virtual_machine_deployment.html#azure-confidential-virtual-machine-deployment-guide)
demonstrates the provisioning pattern: assign a `cc_config` YAML file to every
confidential participant and place
`nvflare.lighter.cc_provision.impl.cc.CCBuilder` after `StaticFileBuilder` and
before `SignatureBuilder`.

The supplied `templates/nvflare-project.yml.in` uses this pattern:

```yaml
participants:
  - name: nvflare-server
    type: server
    org: nvidia
    fed_learn_port: 8002
    admin_port: 8003
    cc_config: "@@SERVER_CC_CONFIG@@"
  - name: site-1
    type: client
    org: nvidia
    cc_config: "@@CLIENT_CC_CONFIG@@"

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
      scheme: grpc
  - path: nvflare.lighter.impl.cert.CertBuilder
  - path: nvflare.lighter.cc_provision.impl.cc.CCBuilder
  - path: nvflare.lighter.impl.signature.SignatureBuilder
```

The paths must be readable by the host-side `nvflare provision` process used in
Stage 50. Use a separate configuration for each participant. Select one CPU
issuer matching the host platform:

```yaml
# AMD SEV-SNP participant
compute_env: onprem_cvm
cc_cpu_mechanism: amd_sev_snp
role: server  # use client in cc_site-1.yml
cc_issuers:
  - id: snp_authorizer
    path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
    token_expiration: 100
    args:
      snpguest_binary: /opt/attestation/bin/snpguest
      amd_certs_dir: /var/tmp/nvflare/workspace/attestation/snp-certs
      cpu_model: genoa  # use the generation that matches the deployed AMD host
cc_attestation:
  check_frequency: 120
```

```yaml
# Intel TDX participant
compute_env: onprem_cvm
cc_cpu_mechanism: intel_tdx
role: server  # use client in cc_site-1.yml
cc_issuers:
  - id: tdx_authorizer
    path: nvflare.app_opt.confidential_computing.tdx_authorizer.TDXAuthorizer
    token_expiration: 100
    args:
      tdx_cli_command: /opt/attestation/bin/trustauthority-cli
      config_dir: /etc/nvflare/tdx-attestation
      use_sudo: false
cc_attestation:
  check_frequency: 120
```

The TDX `config_dir` receives the reviewed `config.json` read-only from the
`nvflare-tdx-attestation` Secret created by Stage 30. The authorizer does not
write token, verification, or error files there.

For the GPU client, add the GPU mechanism and append a second issuer alongside
the selected CPU issuer. This complete SNP-plus-GPU client fragment illustrates
the required single `cc_issuers` list:

```yaml
compute_env: onprem_cvm
cc_cpu_mechanism: amd_sev_snp
cc_gpu_mechanism: nvidia_cc
role: client
cc_issuers:
  - id: snp_authorizer
    path: nvflare.app_opt.confidential_computing.snp_authorizer.SNPAuthorizer
    token_expiration: 100
    args:
      snpguest_binary: /opt/attestation/bin/snpguest
      amd_certs_dir: /var/tmp/nvflare/workspace/attestation/snp-certs
      cpu_model: genoa  # use the generation that matches the deployed AMD host
  - id: gpu_authorizer
    path: nvflare.app_opt.confidential_computing.gpu_authorizer.GPUAuthorizer
    token_expiration: 100
    args:
      nvat_command: /opt/attestation/bin/nvattest
      verifier: local
cc_attestation:
  check_frequency: 120
```

For a TDX-plus-GPU client, replace the SNP mechanism and issuer in that fragment
with the complete TDX mechanism and issuer shown above; retain the GPU issuer in
the same list. Do not declare a second `cc_issuers` key, because YAML would
replace the first list.

Do not configure `GPUAuthorizer` as an issuer for the CPU-only server. It is
still installed there as a verifier so the server can validate the client's GPU
token. `CCBuilder` adds `CCManager` and authorizer resources to the startup
kits; Stage 40 supplies their binaries and Python packages. Quotes are generated
at runtime, not at provision time.

Treat provisioning diagnostics as a security gate. `CCBuilder.initialize()`
reports a bad or missing `cc_config` as `CC is not enabled for ...`; Stage 50
rejects that diagnostic and requires the expected `cc_manager__p_resources.json`
and authorizer resource files before creating Kubernetes objects.

After changing any authorizer dependency or Dockerfile input, rerun Stage 40 so
both new image digests are signed and authorized, then rerun Stage 50.

## Stage 50: provision and deploy the participants

Run the NVFlare deployment stage:

```bash
./50-nvflare-deploy.sh
```

The script first re-verifies both signatures and checks that both references are
immutable digests. It then:

1. Generates SNP-or-TDX participant `cc_config` files, renders
   `templates/nvflare-project.yml.in`, and runs `nvflare provision`.
   `CCBuilder` adds the CPU issuer to both participants, the GPU issuer to the
   client, and CPU/GPU peer verifiers to both startup kits. Stage 50 rejects
   provisioning diagnostics or missing component resources.
2. Requires a suite-owned dedicated namespace before storing each server/client
   `startup/` directory in a Kubernetes Secret and each `local/` directory in a
   ConfigMap. The administrator kit remains only under suite state and is not
   copied to the cluster.
   Server job and snapshot storage is relocated from its provisioned `/tmp`
   default to the encrypted participant workspace.
3. Constructs CoCo init-data containing the deny-by-default image policy,
   Cosign public key, private registry CA, and guest-agent policy. The compressed
   init-data is placed in the Kata `cc_init_data` Pod annotation, so the TEE
   launch measurement binds these verification inputs. The demo agent policy
   denies `SetPolicyRequest`, preventing runtime replacement of the measured
   policy. It deliberately permits copy, exec, and stream requests because
   Stage 50 uses `kubectl exec` for in-guest device/tool checks. Consequently,
   this demo does not isolate the workload from a Kubernetes host administrator;
   production deployments should remove agent operations they do not require.
4. Creates the dedicated namespace with
   `nvflare.nvidia.com/coco-managed=true` if absent, refuses an existing
   namespace without that ownership marker, labels the owned namespace
   `pod-security.kubernetes.io/enforce=privileged`, and renders
   `templates/nvflare-coco.yaml.in` with the exact signed digests, the
   platform-specific non-GPU Kata RuntimeClass for the server, and the
   GPU-enabled Kata RuntimeClass for the client. Only the client manifest
   contains the `nvidia.com/pgpu` request and limit. Each container is
   privileged only inside its Kata VM, creates `/dev/sev-guest` or
   `/dev/tdx_guest` from guest sysfs before starting NVFlare, and mounts TDX
   configuration read-only when applicable.
5. Creates the server Service and server/client Deployments, waits for both Pods
   to be Ready, and checks each expected runtime class, the exclusive client GPU
   assignment, init-data annotation, Pod image references, and runtime-reported
   image IDs, opens the guest CPU device, checks the three in-image tool
   versions, and waits for each peer to log a successful NVFlare CC validation.
   Stage 40 checked all three authorizer imports; starting the configured
   participants also exercises the selected authorizers. The report treats
   successful container creation under the selected CoCo runtimes as indirect
   evidence that guest image-rs accepted both images; it does not run a
   separate unsigned-image rejection test.
6. After every preceding check passes, atomically records the exact active
   namespace, participant names, signed image digests, and administrator startup
   kit path in `state/nvflare-deployment/current.env`. Stage 50 removes an older
   pointer before reprovisioning, so a failed rerun cannot direct Stage 60 to a
   stale kit.

The server Service is cluster-internal and exposes:

| Port | Purpose |
|---|---|
| `8002` | NVFlare federation traffic from clients. |
| `8003` | NVFlare administration traffic. |
| `8102` | NVFlare parent/child process communication. |

Check the deployed objects and integrity report:

```bash
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf \
  -n nvflare-coco get service,deployments,pods -o wide
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf \
  -n nvflare-coco get pods \
  -o custom-columns=NAME:.metadata.name,RUNTIME:.spec.runtimeClassName,IMAGE:.spec.containers[0].image,IMAGE_ID:.status.containerStatuses[0].imageID
sed -n '1,160p' state/nvflare-deployment-integrity-report.txt
```

To inspect the GPU allocation directly, the server value must be `none` and the
client value must be `1` with the defaults:

```bash
for app in nvflare-server site-1; do
  sudo kubectl --kubeconfig /etc/kubernetes/admin.conf -n nvflare-coco \
    get pod -l "app=$app" -o json |
    jq -r --arg app "$app" --arg gpu nvidia.com/pgpu \
      '[$app, (.items[0].spec.runtimeClassName),
       (.items[0].spec.containers[0].resources.requests[$gpu] // "none")] | @tsv'
done
```

Participant logs are available through Kubernetes:

```bash
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf \
  -n nvflare-coco logs deployment/nvflare-server
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf \
  -n nvflare-coco logs deployment/site-1
```

The administrator startup kit is reported at the end of stage 50 under a fresh,
randomized provisioning workspace, for example:

```text
state/nvflare-deployment/workspace.A1b2C3/nvflare_coco/prod_00/admin@nvidia.com/startup
```

Keep it private. The generated server certificate contains the in-cluster
server names. If an administrator connects from outside the cluster, use an
approved network path to the ClusterIP Service or another tunnel to port 8003,
and preserve the provisioned server hostname for TLS verification. Standard
`kubectl port-forward` cannot reach a listener inside the Kata VM on every CRI
implementation; Stage 60 therefore uses a loopback-only host proxy to the
routable server Pod IP. Do not expose
the administration port publicly without authentication, authorization, and
network policy controls appropriate to the deployment.

This reference workflow uses ordinary Kubernetes Secrets for participant
startup kits, the TDX service configuration, and an optional NVIDIA service
key. A Kubernetes or etcd administrator can retrieve those values;
Confidential Containers cannot make an already host-visible Secret
confidential retroactively. For a production threat model that excludes the
host or cluster administrator, keep participant keys and service credentials
outside Kubernetes and release them from Trustee or another key broker only
after successful remote attestation. Mount the released material directly
inside the guest and remove the ordinary Secrets from the manifest.

## Stage 60: submit and verify hello-numpy

After Stage 50 succeeds, run:

```bash
./60-nvflare-submit-hello-numpy.sh
```

Stage 60 is an end-to-end functional check of the already attested NVFlare
parents. It does not create another Kubernetes workload or assign another GPU.
The script performs these operations:

1. Loads `state/nvflare-deployment/current.env`, requires it to match the
   configured namespace and participant names, waits for both Deployments to be
   available, and confirms their live image references still equal the exact
   Stage-50 signed digests. A missing state file means Stage 50 did not complete.
2. Uses the repository's `NumpyFedAvgRecipe` and
   `examples/hello-world/hello-numpy/client.py` to export the standard
   one-client, three-round job with full NumPy parameter transfer and
   TensorBoard experiment tracking. One client is intentional because this
   deployment has exactly one `site-1` participant. Because CCManager rejects
   BYOC, Stage 60 removes the exported custom copy and changes the client
   configuration to the narrowly scoped `CoCoHelloNumpyExecutor`. That executor
   accepts no job-controlled entry point or arguments and always runs the
   identical trainer baked into the signed parent image at
   `/opt/nvflare/examples/hello-numpy/client.py`. Stage 50 allow-lists that exact
   class, while the generic `ClientAPIExecutor` remains blocked. Stage 60
   refuses to submit if any custom directory remains. The export is retained
   below the run directory for inspection.
3. Copies the active administrator startup kit into a mode-0700 temporary
   directory. Only that temporary copy changes: its connection host and port
   are set to `127.0.0.1:$NVFLARE_ADMIN_LOCAL_PORT`. The provisioned
   `server_identity` remains `nvflare-server`, so mTLS still authenticates the
   expected server certificate rather than weakening hostname identity checks.
   Stage 50 includes `127.0.0.1` as an IP subject-alternative name in that
   server certificate specifically for this local tunnel; hostname verification
   remains enabled.
4. Resolves the running server Pod IP and starts a small Python TCP proxy bound
   only to host loopback, forwarding the local port to the server's
   cluster-internal administration port 8003. This is used instead of
   `kubectl port-forward`, whose host-network-namespace loopback does not reach
   the listener inside a Kata VM with every CRI implementation. The private
   administrator kit is never placed in a Kubernetes Secret or admin Pod. The
   tunnel and temporary kit are removed by the exit trap on success, failure,
   or interruption.
5. Runs `nvflare --format json job submit`, extracts the returned job ID, then
   runs `nvflare --format json job wait` with the configured timeout and poll
   interval. NVFlare returns a nonzero exit for failed, aborted, abandoned,
   timed-out, or execution-exception terminal states. Stage 60 additionally
   requires the returned status to start with `FINISHED:COMPLETED`.
6. Prints up to the last 120 server/client job-log lines and writes a PASS report.
   Log retrieval is diagnostic and can be unavailable when client log streaming
   is disabled; the authenticated `job wait` terminal status is the authoritative
   completion check.

Successful output ends with text equivalent to:

```text
hello-numpy finished without error
Job ID: <server-assigned-id>
Status: FINISHED:COMPLETED
Report: .../report.txt
Run artifacts: ...
```

Each invocation creates
`state/nvflare-job-runs/<UTC-timestamp>.<random-suffix>/` containing the exported job,
submission JSON, wait JSON, bounded job logs, CLI diagnostics, tunnel log,
and `report.txt`. `state/latest-nvflare-job-run` points to the latest successful
run. These files do not contain the copied administrator private key; that copy
exists only in the temporary directory removed at exit. Treat all suite state
as administrator-controlled data nonetheless.

Stage 60 uses `ProcessJobLauncher`: the server and client child processes run
inside the existing digest-pinned Kata VMs and inherit the signed parent
images' Python environment. Consequently, Stage 40 must be rerun after adding
or changing a job dependency. The supplied `Dockerfile.coco` includes NumPy
through base NVFlare, the `hello-numpy` trainer as integrity-protected image
content, and TensorBoard because the standard recipe enables that receiver. It
still excludes the Kubernetes Python package. A submitted custom trainer would
be BYOC and is intentionally rejected when confidential-computing validation is
enabled. The dedicated `CoCoHelloNumpyExecutor` has a fixed constructor so its
allow-list entry cannot be repurposed to execute another path from job config.

## How container-image integrity is enforced

The complete container-image path is:

```text
reviewed NVFlare source
  -> OCI build
  -> registry sha256 digest
  -> Cosign signature over that digest
  -> deny-by-default measured guest policy
  -> digest-pinned Kubernetes Pod
  -> the selected CoCo runtime invokes image-rs before container creation
  -> readiness, runtime image ID, and Pod controls checked by stage 50
```

The mutable tags are used only to push and locate a newly built object. Stage 50
reads the recorded `@sha256:` references, and the default CoCo runtimes pass the
measured image policy to guest image-rs. Stage 50 verifies the signatures on the
host and confirms that both containers become Ready with matching runtime image
IDs. Readiness is indirect enforcement evidence, not an independent negative
test proving that an unsigned image is rejected. Because the policy, public key,
and registry CA are in init-data, modifying those inputs changes SNP `HOST_DATA`
or TDX `MRCONFIGID`.

The CPU-only server VM and GPU-enabled client VM do not have interchangeable VM
launch reference values: their Kata guest images and launch configurations can
differ. A production verifier must maintain and appraise the approved reference
set for each selected RuntimeClass, in addition to checking the shared measured
init-data binding.

Container signing alone is not complete VM-image verification. The verifier
that releases production credentials or data must also appraise the signed TEE
report or quote and compare the VM launch measurements with independently
trusted references:

- SNP: verify the VCEK chain, report signature, debug policy, fresh
  `REPORT_DATA`, the full launch `MEASUREMENT`, and the init-data binding in
  `HOST_DATA`. A signed SNP ID block can enforce an approved launch identity.
- TDX: verify Intel PCS collateral and quote signature, debug state, fresh
  `REPORTDATA`, MRTD and all relevant RTMR values, and the init-data binding in
  `MRCONFIGID`.

For complete VM-image verification, the launch measurement must be compared
against a trusted reference—typically through Trustee/RVPS policy—or enforced
with a signed SNP ID block. CoCo's
[reference-value architecture](https://confidentialcontainers.org/docs/attestation/reference-values/)
is specifically designed for that comparison.

`50-nvflare-deploy.sh` now requires the NVFlare peers to generate and
cryptographically verify their hardware tokens before it reports success. That
peer handshake does not independently compare the Kata VM launch measurement
and init-data binding against an operator-approved reference set. The report
keeps that distinction explicit. Before releasing production data or
credentials, use Trustee/RVPS or another verifier outside the workload host's
failure domain with approved SNP/TDX references and this deployment's expected
init-data hash.

## Workspace encryption and persistence

Both Deployments mount an `emptyDir` at `/var/tmp/nvflare/workspace` and a
256 MiB `emptyDir` at `/applog`, which is required by the provisioned NVFlare
file-logging configuration. Stage 20
requires Kata's released `block-encrypted` mode for the GPU RuntimeClass used by
the client. The pinned chart's default non-GPU SNP and TDX RuntimeClasses use the
same mode for the server, but stage 50 checks only that the selected server
RuntimeClass exists. Verify `emptydir_mode = "block-encrypted"` independently
before using a custom server RuntimeClass. In that mode Kata creates a
host-backed sparse block device and applies LUKS2/dm-crypt plus dm-integrity with
an ephemeral key inside the confidential guest. The startup Secret and local
ConfigMap are nested read-only mounts under that workspace.

This protects transient workspace blocks from direct host inspection and
detects block modification. It is not persistent storage: deleting the Pod
deletes the backing file and loses its ephemeral key. It also does not prevent
whole-volume rollback. Production persistent datasets need a separately
designed encrypted-volume and key-release policy bound to successful remote
attestation.

## Submit other jobs safely

The deployed parent processes use NVFlare's process launcher. A submitted job's
Python code and dependencies must therefore be compatible with the signed
parent image in which that participant runs:

- server-side job code runs in the minimal server image;
- the deployed client parent has NVFlare but not CUDA/PyTorch training
  dependencies; and
- packages installed after measurement or downloaded without independent
  verification weaken the image-based software identity.

For an in-process GPU training demonstration, derive a compact client-parent
Dockerfile with only the required CUDA/Python libraries, keep its expanded size
within the confidential guest's image-pull capacity, then rerun NVFlare stages
40 and 50. For production, prefer a separately digest-pinned and signed job
image under a launcher that injects all CoCo controls listed below. Avoid `pip
install` from the network inside a running participant. The signature policy
verifies container images, not arbitrary job archives or datasets; retain
NVFlare's job-signature and authorization controls and govern who may submit
jobs.

If separate per-job Kubernetes Pods are required, extend the NVFlare Kubernetes
launcher or an admission controller so every job Pod receives all of the
following before enabling it:

- the SNP/TDX GPU Kata RuntimeClass;
- the measured `cc_init_data` annotation;
- a digest-pinned, signed job image allowed by the measured policy;
- the intended passthrough GPU resource and resource limits; and
- remote-attestation-gated data and credential release.

## Update or rotate the deployment

After changing NVFlare code, dependencies, or a Dockerfile, rebuild and deploy:

```bash
./40-nvflare-build-sign-images.sh
./50-nvflare-deploy.sh
./60-nvflare-submit-hello-numpy.sh
```

Stage 40 records new digests. Stage 50 reprovisions the identities, updates the
Secrets and ConfigMaps, applies those digests, restarts both Deployments, and
waits for their health checks. Stage 60 verifies that the rebuilt parents can
run the standard functional job.

Security-material rotation belongs to the shared workflow in `USER_GUIDE.md`.
After that procedure completes, rerun NVFlare Stages 40 and 50 so the images
and participants use the new public key and registry CA. Previously signed
images cease to be authorized unless the policy explicitly retains the old
public key.

## Troubleshooting

### Build cannot pull a base image

Confirm DNS, proxy, and firewall access from the host to Docker Hub, NGC, and
the Python package sources used by the Dockerfiles. Podman storage for this
workflow is below `state/podman-storage`; it is safe to discard only when no
build from this suite is in progress.

### Registry TLS or push fails

Confirm `NODE_IP` and inspect the certificate SAN:

```bash
openssl x509 -in state/registry-tls/registry.crt -noout -issuer -subject -ext subjectAltName
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf \
  -n coco-system get pods,service
```

If `NODE_IP` or the registry certificate must change, return to the shared
security-service procedure in `USER_GUIDE.md`. After it succeeds, rerun
NVFlare Stages 40 and 50 to publish and deploy references using the corrected
registry address.

### Deployment reports an unsigned image

Do not change the manifest to use a tag. Rerun stage 40, inspect both
Cosign verification JSON files, and verify that
`state/nvflare-image-security-policy.json` contains both repository keys. Then
rerun stage 50.

### Client remains Pending

Check the GPU resource and VFIO state:

```bash
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf get node \
  -o custom-columns=NAME:.metadata.name,PGPU:.status.allocatable.nvidia\.com/pgpu
sudo kubectl --kubeconfig /etc/kubernetes/admin.conf \
  -n nvflare-coco describe pod -l app=site-1
lspci -Dnnk -d 10de:
```

Only one passthrough-GPU Pod can consume a single `nvidia.com/pgpu` device at a
time. The NVFlare server is not a GPU consumer. Delete or stop the generic
stage-50 validation Pod if it was deliberately kept with
`KEEP_ATTESTATION_POD=1`, because it otherwise competes with the NVFlare client
for the sole device.

### Parent starts but jobs fail

Inspect participant logs and the provisioned `local/resources.json`. Confirm
that it contains the process launcher and that all job imports exist in the
corresponding signed parent image. For Stage 60, inspect
`state/latest-nvflare-job-run/wait.json`, `job-logs.txt`, and the participant
logs. Confirm the host export environment and both signed images contain
TensorBoard. Rebuild instead of installing dependencies interactively in the
running VM.

For shared-stage administration, recovery guidance, and the generic attestation
workflow, return to `USER_GUIDE.md`.
