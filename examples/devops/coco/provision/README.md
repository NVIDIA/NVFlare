# Provision NVFlare participants as confidential containers

Run `nvflare provision -p project.yaml` on the trusted **NVFlare provisioning
node**. This follows the CC CVM example's `cc_config`, builder, and packager
structure. Protect clients and, optionally, the server with a separate encrypted
image for each participant. Without a server `cc_config`, the server receives
an ordinary kit that verifies protected clients. FL console admins receive
ordinary startup kits. CoCo IT is not an FL project admin.

Relative `cc_config` paths resolve beside the source project YAML, not the
process working directory. POC preparation preserves absolute CC-config paths
when saving its project in the workspace. Python API callers must pass
`prepare_project(project_dict, project_file="/path/to/project.yaml")` when
using relative CC-config paths, or provide absolute paths themselves.

## Prerequisites

1. Use an NVFlare installation from the checkout containing `CoCoPackager`.
   An older installed package does not contain this implementation. For source
   development, create a project `.venv` and install with
   `python -m pip install -e .` from the repository root.
2. Prepare the [admin kit](../admin/README.md): tools, authenticated public
   certificates, registry publisher credentials, trusted `platform.env`, and
   the authority-pinned approved workload launch profile. Never take approval
   values from CoCo IT.
3. Prepare an application Dockerfile containing the same NVFlare version used
   for provisioning, Python 3.11+, bash, standard coreutils, and all reviewed
   application code and dependencies (including required NVIDIA userspace
   libraries). The supplied [client Dockerfile](site-1/Dockerfile) and
   [server Dockerfile](server/Dockerfile) deliberately have non-runnable
   placeholder bases. Replace each selected base with your reviewed application
   image pinned by digest, or replace its Dockerfile entirely. Include the
   server's aggregation/controller code in its image when protecting the server.
4. Obtain the authenticated AS signing public key and rehearse the guest token
   API described in [CCManager integration](CCMANAGER.md). Do this before image
   building. Every protected participant needs that API and a valid signed
   appraisal. Registration fails when a required client or server proof is missing
   or fails verification.

For example, the application base can be extended with `COPY code/ /local/custom/`.
Do not put credentials in that public source directory. Do not use CVM-specific
`/user_config` mounts, Docker `VOLUME`, privileged setup, root-only startup, or
runtime dependency downloads. Each protected server or client runs as UID/GID 65532.

## Configuration and command

[project.yaml](project.yaml) uses Workspace, StaticFile, Cert, CC, and Signature
builders, then `CoCoPackager`. Keep that order. Replace the server/admin names.
The default protects `site-1` using [cc_site-1.yml](cc_site-1.yml):

```yaml
compute_env: confidential_containers
cc_cpu_mechanism: amd_sev_snp
cc_gpu: nvidia
role: client
cc_issuers:
  - id: coco_authorizer
    path: nvflare.app_opt.confidential_computing.coco_authorizer.CoCoAuthorizer
    token_expiration: 300
    args:
      trustee_public_key_file: ./trustee-as-public.pem
      token_url: http://127.0.0.1:8006/aa/token
cc_attestation:
  check_frequency: 120
image_build:
  context: ./site-1
  dockerfile: Dockerfile
release_name: site-1-v1
registry_repository: workloads/site-1
platform_config: ../admin/platform.env
```

To protect the server too, uncomment its `cc_config: cc_server.yml` in
`project.yaml` and prepare [cc_server.yml](cc_server.yml) and the
[server build context](server/Dockerfile):

```yaml
- name: server.example.com
  type: server
  org: example
  fed_learn_port: 8002
  cc_config: cc_server.yml
```

The server CC YAML uses `role: server`, `image_build.context: ./server`,
`release_name: server-v1`, and `registry_repository: workloads/server`.
The CC role must match the participant type. Both roles currently require the
same AMD SEV-SNP plus NVIDIA confidential-GPU profile, including successful GPU
appraisal even if the server's controller does not use a GPU. This does not
provide a CPU-only server profile. Admin participants cannot use CoCo packaging.
The client name `server` is reserved for NVFlare's logical root-server identity.
The current profile reserves one `nvidia.com/pgpu` per Pod, including the server.
Running one protected server and one protected client concurrently requires two
allocatable confidential GPUs, which can be on separate nodes or clusters.
A node with only one allocatable GPU cannot schedule both Pods concurrently.

`cc_config` is relative to `project.yaml`. Context, platform configuration, and
`build_image_cmd` paths are relative to the participant's CC YAML. The Dockerfile
is relative to the build context. Absolute paths also work. Use a different
release name for each participant and each image/key/policy revision. Participants can
use different Dockerfiles. `platform_config` must name the prepared admin kit's
`platform.env`; the pipeline uses scripts from that same kit.

`CoCoPackager.build_image_cmd` is required: the Python wheel does not ship the
example image builder. Supply the reviewed executable explicitly. This is trusted
operator configuration, not workload-supplied input; relative `../admin/...` and
absolute paths intentionally support separate role-kit directories. Authenticate
and review that executable and its parent directory. The packager validates its
returned YAML, fixed security context, complete InitData policy and approved
pause-container profile before releasing a handoff. Unsupported output remains
in private recovery state.

The deployment host requires system Python 3.11+ for InitData/TOML validation.
An NVFlare virtualenv on Python 3.10 uses `/usr/bin/python3` for this parsing
step; it fails closed if that system interpreter is too old. No third-party
TOML backport is needed, and importing provisioning modules on 3.10 remains safe.

`class_allow_list`, when supplied, must contain fully qualified reviewed class
names such as `my_application.executor.ReviewedExecutor` or
`my_application.controller.ReviewedController`, with their reviewed code baked
into the corresponding image. Wildcards, module-prefix grants and non-string
entries are rejected. Omit it or use `[]` if unnecessary.

`cc_issuers` configures the implemented Trustee CoCo authorizer. Each protected
participant issues proofs and verifies the other protected participants through
CCManager. A protected server attests with logical site name `server`, regardless
of its project DNS name; its TLS certificate still identifies `server.example.com`.
An ordinary server verifies without attesting itself. When the server is
protected, ordinary clients also receive verifier-only configuration to check
its proof. The pinned public-key path is relative to the CC YAML. Protected
participants in one project must share that key, token expiration, check frequency,
and manager timeouts. `cc_gpu: nvidia` declares the required
profile; signed CPU and GPU appraisals supply the evidence. Invalid configuration
aborts provisioning. Mixing other CC compute environments into this project is
not supported yet. See [the checks and limitations](CCMANAGER.md).

The ordinary server's `CoCoAuthorizer.verify_for_site(token, authenticated_site)`
needs no CoCo runtime or Trustee connection: it verifies locally using the pinned
AS public key and binds the proof to the authenticated peer. Obtain
`authenticated_site` from authenticated FL/mTLS identity, not the submitted
token or its envelope. The compatible `verify(token)` method checks proof
validity only, without expected-peer binding. The current API has no
`expected_workloads` constructor argument. See the
[direct client/server API](CCMANAGER.md#direct-clientserver-api) and
[authorization boundaries](CCMANAGER.md#what-verification-does-not-authorize)
before integrating it outside the generated kits.

Activate the updated NVFlare environment on the provisioning node, then run:

```bash
cd /path/to/coco/provision
nvflare provision -p project.yaml
```

## What the packager does

After startup-kit signing, it moves all protected participants' plaintext kits
out of `prod_NN` before invoking any image builder. For each protected participant it:

1. Copies the application's context to a private build directory. Symlinks and
   special files are rejected. The context cannot contain the provisioning
   workspace. `.nvflare-kit` and `Dockerfile.coco*` are reserved names.
2. Creates `Dockerfile.coco`, appending only that participant's signed kit to the final
   application stage, UID/GID 65532, and the exact command
   `/opt/nvflare/startup/sub_start.sh --once --verify`. A server image contains
   its own `server.key`; a client image contains its own `client.key`.
   The generated startup script selects the correct NVFlare server/client process.
   Other participants' kits and admin credentials are not included.
3. Generates `workload.env` and invokes
   [build_coco_image.sh](../admin/build_coco_image.sh), which runs stages
   10, 20, 25, 30, and 40. After stage 10, inspect the image records and type
   the release name to approve signing/encryption/publication. There is no
   unattended approval switch in the wrapper.
4. Checks the final Pod and copies only its YAML into the participant's public folder.

The CC builder configures proof issuance and verification before signing using
the shared pinned Trustee key, project-specific audience, and required protected
participant identities. The server contributes logical identity `server` to
that required set only when its own `cc_config` enables protection.

The runner receives one JSON file with schema `nvflare-coco-build-request/v1`
and absolute `workload_env`, `admin_dir`, `result_file` paths. Success writes a
`nvflare-coco-build-result/v1` receipt with `release_name`, `pod_yaml`, and
`trusted_service` paths. A custom runner is trusted code and must preserve the
same checks; a generated YAML alone is not proof of encryption.

## Writable runtime state

The standalone demo stays read-only by default. NVFlare servers and clients
explicitly set `APP_READ_ONLY_ROOT_FILESYSTEM=false`: logs, received jobs, and checkpoints need
writable space. Genpolicy records that choice, and the InitData hash binds it
to KBS resource release. The guest-pulled image's writable container layer is
used, without adding hostPath, PVCs, shared directories, or Kubernetes volumes.
The server's `/opt/nvflare` workspace and `/tmp/nvflare` job/snapshot paths remain
inside that guest-local writable storage. This is ephemeral state, not persistent
encrypted storage. The workload can
modify its own runtime files; the cluster owner still must not substitute the
agent policy or inject exec requests. CoCo IT can stop the Pod and observe its
exposed logs/output. Never log secrets to stdout/stderr. Rehearse the actual
NVFlare workload against the approved runtime/profile before deployment.

## Handoffs

```text
workspace/coco_project/
  prod_NN/
    server.example.com/server-v1-pod.yaml  Pod-only handoff, if server protected
    admin@example.com/              ordinary FL console admin kit
    site-1/site-1-v1-pod.yaml        Pod-only handoff for this client
  state/coco-private/prod_NN/        PRIVATE; keep on provisioning node
    server.example.com/             present only when server protection is enabled
      startup-kit/                  includes this server's private server.key
      build-context/
      workload.env
      build-request.json
      result.json
    site-1/                         same private layout, with its own client.key
```

With the default ordinary server, `prod_NN/server.example.com/` instead contains
its ordinary signed kit with a CCManager verifier. Deliver it only to its trusted
server owner. Ordinary client kits and FL console admin kits likewise go only
to their trusted owners.

The admin workflow creates `WORK_ROOT/releases/<release_name>/handoff/` for
each protected participant, including `server-v1` when selected. Each contains
its own `trusted-service/` and `coco-it/` handoffs; its private receipt records
their locations. Authenticate and confidentially deliver every `trusted-service/`
to the secure-services administrator for review and
[stage 12](../service/TRUSTED-HANDOFF-RUNBOOK.md).
The administrator installs each release's own image key and authorization
fragment; a server release does not reuse a client's key or overwrite its policy.
After all required releases are authorized, deliver only each Pod YAML and independently
authenticated hash to CoCo IT for [launch](../coco/COCO-IT-RUNBOOK.md).
Provisioning does not install service policies, transfer secrets, or launch Pods.

Never distribute the whole workspace, `state/`, private build context, or a
protected participant's plaintext kit. A protected server's kit remains private
on the provisioning node and enters its encrypted image through the private
build context; it must not also be handed to the cluster owner as a plain kit.

## Protected server network endpoint

Before provisioning, choose a stable server DNS name reachable by every client
and FL console admin, and use it as the server participant name. For this example,
`server.example.com:8002` must route to TCP port 8002 on the protected server Pod.
The TLS certificate remains bound to that project name. Provisioning does not
create DNS records, Kubernetes Services, load balancers, or firewall rules.
The default FL admin endpoint shares `fed_learn_port`; if the project explicitly
sets a different `admin_port`, provide a second TCP Service port and corresponding
external routing for it.

After the secure-services owner authorizes all protected releases, CoCo IT
launches the unchanged Pod handoffs and creates a separate network Service.
For the sample `server-v1` release, on a cluster with a configured load-balancer
implementation:

```bash
kubectl apply -f - <<'YAML'
apiVersion: v1
kind: Service
metadata:
  name: nvflare-server
  namespace: default
spec:
  type: LoadBalancer
  selector:
    app: server-v1
  ports:
    - name: nvflare
      protocol: TCP
      port: 8002
      targetPort: 8002
YAML
kubectl -n default get service nvflare-server
kubectl -n default get endpointslices -l kubernetes.io/service-name=nvflare-server
```

The selector must match the protected server's actual `release_name` and the
Service must be in the Pod's namespace. Point `server.example.com` at the
Service's stable external address and allow the configured FL TCP port through
the required network boundaries. A bare-metal cluster needs a load-balancer
implementation or a separately reviewed external TCP forwarding arrangement;
`type: LoadBalancer` alone does not allocate a reachable external address there.
Forward TCP without terminating NVFlare TLS outside the guest. Clients and admins
must authenticate the server using their provisioned trust and DNS identity.

Keep the approved Pod unchanged: no `hostNetwork`, `hostPort`, container `ports`,
volumes, sidecars, or alternate startup command are needed for a Service to
target TCP 8002. Do not expose the guest-local AA port 8006. Update routing to
the new release selector during an approved replacement, and verify DNS,
connectivity, mutual authentication, and attestation before submitting jobs.
The cluster controls routing and availability; the Service does not grant
attestation or image-key authorization.

## Recovery

Treat a failed provisioning run's entire `prod_NN` as private until inspected.
Configuration or private-state validation can fail before the kits are moved;
that directory can still contain plaintext server/client keys at that point.
The guarantee that all selected kits are private applies before the first
external image build starts, not to every earlier provisioning failure.

If packaging fails, no Pod-only result is published for the failing participant;
private recovery inputs remain. Earlier participants may already have published
encrypted images and completed handoffs. There is no cross-participant registry
rollback. Review the retained records, choose new unique release names, then
rerun provision. Do not reuse keys/releases or delete existing registry objects
to force a retry.

If documented cleanup of generated `prod_NN` directories causes a later
`nvflare provision --force` run to reuse a stage number, the packager preserves
the previous private stage under a uniquely created, mode-0700 directory:

```text
state/coco-private/prod_NN.superseded-<unique-id>/prod_NN/
```

It then creates a fresh `state/coco-private/prod_NN/` for the new kits and build
inputs. This also applies when the previous build failed partway through.
Earlier archives are never overwritten or deleted, even across repeated retries.
Keep all these trees private. Retaining a directory does not rewrite absolute
paths in its historical receipts or automatically resume an external build.
Do not delete `state/coco-private` to work around a reused stage number, and do
not run concurrent provisioning commands against the same workspace.

Each image-runner invocation has a timeout of 3600 seconds, including time spent
waiting for interactive approval. For a longer reviewed build, configure the
packager in `project.yaml`, for example:

```yaml
packager:
  path: nvflare.lighter.cc_provision.impl.coco_packager.CoCoPackager
  args:
    build_image_cmd: ../admin/build_coco_image.sh
    build_timeout: 7200
```

`build_timeout` must be a positive integer in seconds. On timeout, packaging
fails without publishing a Pod handoff for that participant and retains its private
recovery inputs. The direct runner process is stopped; this is not a rollback
of published images or a guarantee that its descendants or external services
have stopped. Inspect and stop any remaining build activity before retrying.

Offline tests create/sign real startup kits with a mocked image runner. They
do not certify actual Docker builds, GPU operation, or hardware attestation.
The [separate live cross-node test](CCMANAGER.md#verification-status) verified
that a client inside CoCo could generate a proof accepted by an ordinary-host
verifier. It did not test full NVFlare registration, a protected server, or
protected image-key release. Server support requires a separate hardware-backed
rehearsal of the complete federation and each release's authorization path.
