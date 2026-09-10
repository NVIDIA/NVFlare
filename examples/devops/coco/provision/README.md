# Provision NVFlare clients as confidential containers

Run `nvflare provision -p project.yaml` on the trusted **NVFlare provisioning
node**. This follows the CC CVM example's `cc_config`, builder, and packager
structure. Only clients are protected; servers and FL console admins receive
ordinary startup kits. CoCo IT is not an FL project admin.

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
   libraries). The supplied [Dockerfile](site-1/Dockerfile) deliberately has a
   non-runnable placeholder base. Replace it with your reviewed application
   image pinned by digest, or replace the Dockerfile entirely.
4. Obtain the authenticated AS signing public key and rehearse the guest token
   API described in [CCManager integration](CCMANAGER.md). Do this before image
   building. The generated client refuses registration when that API is missing
   or its signed appraisal does not satisfy the configured checks.

For example, the application base can be extended with `COPY code/ /local/custom/`.
Do not put credentials in that public source directory. Do not use CVM-specific
`/user_config` mounts, Docker `VOLUME`, privileged setup, root-only startup, or
runtime dependency downloads. The final client runs as UID/GID 65532.

## Configuration and command

[project.yaml](project.yaml) uses Workspace, StaticFile, Cert, CC, and Signature
builders, then `CoCoPackager`. Keep that order. Replace the server/admin names.
Each protected client references its own [cc_site-1.yml](cc_site-1.yml):

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

`cc_config` is relative to `project.yaml`. Context, platform configuration, and
`build_image_cmd` paths are relative to the client's CC YAML. The Dockerfile
is relative to the build context. Absolute paths also work. Use a different
release name for each client and each image/key/policy revision. Clients can
use different Dockerfiles. `platform_config` must name the prepared admin kit's
`platform.env`; the pipeline uses scripts from that same kit.

`cc_issuers` configures the implemented Trustee CoCo authorizer. The confidential
client issues proofs; the ordinary FL server verifies them at registration and
periodically through CCManager without attesting itself. The pinned public-key
path is relative to the CC YAML. Clients in one project must share that key,
token expiration, and check frequency. `cc_gpu: nvidia` declares the required
profile; signed CPU and GPU appraisals supply the evidence. Invalid configuration
aborts provisioning. Mixing other CC compute environments into this project is
not supported yet. See [the checks and limitations](CCMANAGER.md).

Activate the updated NVFlare environment on the provisioning node, then run:

```bash
cd /path/to/coco/provision
nvflare provision -p project.yaml
```

## What the packager does

After startup-kit signing, for each protected client it:

1. Moves the plaintext signed kit out of `prod_NN` into private state.
2. Copies the application's context to a private build directory. Symlinks and
   special files are rejected. The context cannot contain the provisioning
   workspace. `.nvflare-kit` and `Dockerfile.coco*` are reserved names.
3. Creates `Dockerfile.coco`, appending the signed client kit to the final
   application stage, UID/GID 65532, and the exact command
   `/opt/nvflare/startup/sub_start.sh --once --verify`. Server/admin kits are
   never copied into client images.
4. Generates `workload.env` and invokes
   [build_coco_image.sh](../admin/build_coco_image.sh), which runs stages
   10, 20, 25, 30, and 40. After stage 10, inspect the image records and type
   the release name to approve signing/encryption/publication. There is no
   unattended approval switch in the wrapper.
5. Checks the final Pod and copies only its YAML into the client's public folder.
6. Installs final per-client InitData hash, encrypted image reference, and command
   pins in the ordinary server's verifier configuration. These final pins cannot
   be embedded in the client image itself: that would create a hash dependency
   cycle. Deliver the completed server kit, not a pre-packaging intermediate.

The runner receives one JSON file with schema `nvflare-coco-build-request/v1`
and absolute `workload_env`, `admin_dir`, `result_file` paths. Success writes a
`nvflare-coco-build-result/v1` receipt with `release_name`, `pod_yaml`, and
`trusted_service` paths. A custom runner is trusted code and must preserve the
same checks; a generated YAML alone is not proof of encryption.

## Writable runtime state

The standalone demo stays read-only by default. NVFlare clients explicitly set
`APP_READ_ONLY_ROOT_FILESYSTEM=false`: logs, received jobs, and checkpoints need
writable space. Genpolicy records that choice, and the InitData hash binds it
to KBS resource release. The guest-pulled image's writable container layer is
used, without adding hostPath, PVCs, shared directories, or Kubernetes volumes.
This is ephemeral state, not persistent encrypted storage. The workload can
modify its own runtime files; the cluster owner still must not substitute the
agent policy or inject exec requests. CoCo IT can stop the Pod and observe its
exposed logs/output. Never log secrets to stdout/stderr. Rehearse the actual
NVFlare workload against the approved runtime/profile before deployment.

## Handoffs and recovery

```text
workspace/coco_project/
  prod_NN/
    server.example.com/             ordinary server kit with CCManager verifier
    admin@example.com/              ordinary FL console admin kit
    site-1/site-1-v1-pod.yaml        ONLY this file goes to CoCo IT
  state/coco-private/prod_NN/site-1/ PRIVATE; keep on provisioning node
    startup-kit/
    build-context/
    workload.env
    build-request.json
    result.json
```

The admin workflow's `WORK_ROOT/releases/site-1-v1/handoff/` contains the
`trusted-service/` and `coco-it/` handoffs; the receipt records their locations.
Authenticate and confidentially deliver `trusted-service/` to the secure-services
administrator for review and [stage 12](../service/TRUSTED-HANDOFF-RUNBOOK.md).
After authorization is confirmed, deliver only the Pod YAML and independently
authenticated hash to CoCo IT for [launch](../coco/COCO-IT-RUNBOOK.md).
Provisioning does not install service policies, transfer secrets, or launch Pods.

Never distribute the whole workspace, `state/`, private build context, or a
CoCo client's plaintext kit. Server/admin kits go only to their trusted owners.

If packaging fails, no Pod-only result is published for the failing client;
private recovery inputs remain. Earlier clients may already have published
encrypted images and completed handoffs. There is no cross-client registry
rollback. Review the retained records, choose new unique release names, then
rerun provision. Do not reuse keys/releases or delete existing registry objects
to force a retry. Retained `state/coco-private/prod_NN` directories cannot be
overwritten; use a fresh workspace if intentionally resetting stage numbers.

Offline tests create/sign real startup kits with a mocked image runner. They
do not certify actual Docker builds, GPU operation, or hardware attestation.
