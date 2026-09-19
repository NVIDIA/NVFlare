.. _cvm_builder:

Provisioning with CVM Builder
========================================

The optional ``cvm_vault`` section in ``project.yml`` connects NVFlare provisioning
to CVM Builder. Build and approve the generic CVM once per platform and profile
version, then reuse it when provisioning a fresh encrypted application vault for
each selected participant and provisioning run. Stage 2 copies the existing
CVM into each delivery; it does not rebuild it or register new boot measurements.
Projects without ``cvm_vault`` retain their usual provisioning behavior.

CVM Builder is included in this repository at
``nvflare/lighter/cc/image_builder``. Point ``cvm_builder_dir`` at that directory
in your NVFlare source checkout. NVFlare invokes its ``cvmctl vault`` interface
as a separate process. Do not combine ``cvm_vault`` with a ``packager`` or edge
provisioning.

Prepare the worker
------------------

Run provisioning and vault construction on a trusted Linux worker configured for
CVM Builder (the tested builder environment is Ubuntu 26.04). Follow the
builder's ``nvflare/lighter/cc/image_builder/BUILD_GUIDE.md`` and
``TRUSTEE_GUIDE.md`` for disk tools, disabled swap,
core-dump policy, locked memory, approved bundles, and Trustee administration configuration.
Provisioning does not change those host settings or launch a CVM.

Install NVFlare and the builder's requirements in their respective environments.
For a source checkout at ``/opt/NVFlare``, prepare the builder environment with:

.. code-block:: bash

   cd /opt/NVFlare/nvflare/lighter/cc/image_builder
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt

``cvmctl vault`` selects ``CVM_BUILDER_PYTHON``, its own ``.venv/bin/python``, or
``python3``, in that order. If provisioning is unprivileged, the adapter uses
``sudo -n`` for construction and, when necessary, public output metadata collection
using the provisioning Python interpreter. Configure that worker boundary in
advance, including its interpreter/environment policy. No interactive sudo prompt
or remote worker submission is implemented by this adapter. Root-owned build
outputs retain their private permissions.

Set ``cvm_image`` to an approved generic CVM artifact in an OCI registry or a
previously pulled directory. A pulled directory must contain ``profile_set.json``
and all referenced platform bundle files, including ``cvm_manifest.json``,
``approval.json``, and ``resource_policy.rego``. For registry images, install ORAS
and configure its registry authentication for the account running provisioning.
The builder retrieves registry images through ORAS under the build worker's
identity and validates approval for every bundle in the resulting profile set.
The included builder supports shared project configuration and generates the
deployment ID for each vault build.
Trustee reference values, reusable policies, and enabled CVM build IDs must already
be installed. The existing CoCo Trustee accepts native vault key uploads using a scoped
bearer token. CVM Builder installs no backend services. Native uploads can
overwrite keys, and deletions require operator-controlled retry and restore
procedures to remain effective.

Prepare a Linux amd64 image containing NVFlare, Bash, and the workload dependencies:

.. code-block:: bash

   image=my-nvflare-application:1.0
   docker pull --platform linux/amd64 "$image"
   docker save "$image" -o /srv/nvflare/images/application.tar

NVFlare derives the Docker image ID from the configuration bytes in
``docker_archive``. Save exactly one Linux amd64 image per archive; multiple tags
for that same image are allowed. Classic and containerd-backed ``docker save``
archives, including gzip-compressed archives, are supported. Use ``docker save``,
not ``docker export``. No Docker daemon is needed to inspect the archive.

Create a shared ``cvm_project.yml`` beside ``project.yml``:

.. code-block:: yaml

   trustee:
     url: https://trustee.example.org:8443
     ca: ./credentials/kbs-ca.pem
     admin_token_file: ./credentials/kbs-resource-token.jwt

Credential paths resolve against this file. NVFlare finds the nearest
``cvm_project.yml`` starting beside ``project.yml`` and walking upward, then
passes its absolute path through ``--project-config`` on every build. To select
a different file, set ``cvm_vault.project_config``; relative paths resolve against
``project.yml``. Invalid nearest or explicit configuration fails without fallback.
The shared settings and credentials are not copied into application vaults or
public result metadata. There is no per-participant Trustee credential override.

Configure provisioning
----------------------

Add the following section to a project that defines ``site-1`` and its server.
Paths below illustrate a configured worker; the files must exist.

.. code-block:: yaml

   cvm_vault:
     cvm_builder_dir: /opt/NVFlare/nvflare/lighter/cc/image_builder
     cvm_image: /srv/cvm/images/cpu-2026.09
     docker_archive: /srv/nvflare/images/application.tar
     participants: [site-1]
     requires_gpu: false
     vault_drive_size: 16
     allowed_out_ports: [8443]

To pull the CVM directly from a registry, replace ``cvm_image`` with an immutable
reference. Use the actual manifest digest from the CVM publisher:

.. code-block:: yaml

   cvm_vault:
     # ...the other settings above...
     cvm_image: oci://registry.example.org/cvm/cpu-intel-tdx@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

``https://registry/repository@sha256:...`` and
``registry/repository@sha256:...`` are also accepted. These identify OCI registry
artifacts, not arbitrary HTTP downloads. Mutable tags and embedded credentials
are rejected. ``cvm_image`` points to the reusable generic CVM, not a completed
participant vault delivery. For a local image, supply the containing directory
rather than the ``profile_set.json`` file itself.

Registry references are passed intact to ``cvmctl vault``. The builder retrieves
and verifies the generic CVM, includes it in the completed delivery, and removes
its temporary download afterward. To reuse an offline copy, materialize it with
``cvmctl pull`` and set ``cvm_image`` to the resulting folder. For registry
sources, include all profile bootstrap ports in ``allowed_out_ports``; local
images also allow NVFlare to read and add those ports before construction.

Omit ``platforms`` to use the platforms supplied by ``cvm_image``. For an image
containing multiple platforms, an optional ``platforms`` list selects a subset;
every requested platform must be present in that image. A single-platform TDX or
AMD image therefore needs no separate platform setting. To combine separately
published platform bundles, use ``cvmctl pull`` with ``--merge`` and point
``cvm_image`` at the combined directory. Participant overrides can also select
different registry images or pulled directories.

Relative input paths resolve against ``project.yml``. Generated builder input
paths are absolute. ``output_root`` is optional: by default, each selected
participant's delivery goes into ``workspace/<project>/prod_NN/<participant>/``,
the same participant folder used by the previous packager. Its original signed
startup kit is retained in private staging under
``workspace/<project>/.cvm-vault-builds/<run-id>-inputs/startup-kit/``.

An explicit ``output_root`` places each build in a fresh subdirectory there and
leaves the original ``prod_NN`` startup kits in place. It must be outside the
provisioning workspace and private to the operator (mode ``0700``); it is created
if absent. Each invocation gets fresh staging and output paths. No ``release_id``
is needed. The builder generates the deployment ID, which NVFlare reads from
``vault_set.json`` after success.

Only explicitly selected servers and clients are supported. Unknown, duplicate,
admin, relay, and overseer selections are rejected.

``WorkspaceBuilder`` must be the first builder. Keep the normal static-file and
certificate builders in the project. The adapter adds a final signing step for
selected participants, after other builders finalize their files and before
``WorkspaceBuilder`` moves the kits into the new ``prod_NN``. Provisioning errors
or failure to produce that new directory prevent vault construction.

Participant-specific settings go under ``participant_overrides``. Overrides can
replace any application setting, including ``cvm_image``, Docker archive, platforms,
GPU requirement, ports, sizes, host mappings, public input directories,
TEE-device access, and workspace ownership. Override values replace the
corresponding shared values; mappings are not merged recursively.

.. code-block:: yaml

   cvm_vault:
     # ...shared settings above...
     participants: [site-1, server.example.com]
     participant_overrides:
       server.example.com:
         platforms: [amd_sev_snp, intel_tdx]
         allowed_ports: [8080]
         user_data: ./public-server-data
       site-1:
         tee_device: true

``requires_gpu: true`` requires a profile with ``contract.gpu: nvidia_cc``;
CPU-only profiles use ``requires_gpu: false``. GPU passthrough and the guest GPU
gate remain the builder's responsibility. ``tee_device`` defaults to false; enable
it only when the application's own attestation requires a CPU TEE device.
The runtime chooses the device for the selected platform. Existing NVFlare
``CCBuilder``/authorizer configuration remains separate from the CVM boot gate.

Network and filesystem behavior
-------------------------------

Outbound TCP ports include HTTPS (443), a local profile's ``bootstrap_egress``,
provisioned server ports, signed server endpoint ports, explicit connection/relay ports, and
``allowed_out_ports``. Add ports for any other application services explicitly.
Server ports and fixed participant listening ports are published through Docker
and included in the guest inbound allowance, together with ``allowed_ports``.
The adapter uses the configured ports rather than assuming 8002/8003; the current
default admin port is the federated-learning port. Clients must have a reachable
server address in their signed configuration; loopback addresses are rejected.
``hosts_entries`` can supply literal address mappings. Existing overseer settings
are obsolete in current NVFlare provisioning.

The selected participant's finalized workspace, including its private key,
``local/`` configuration and ``transfer/`` content, goes into encrypted
``application_files``. Other participants and provisioning ``state/`` are excluded.
Signed file contents are copied unchanged and verified before construction.
Keep any application-specific ``cc_params.yml`` within this workspace; the generic
builder does not parse it.

The container starts the signed ``startup/sub_start.sh`` through Bash with
``--verify --foreground``. Its source kit is at
``/vault/application/workspace``. ``NVFL_WORKSPACE=/vault/application/runtime``
keeps writable state separate and refreshes a verified working copy on each
container start. The supervisor remains in the foreground, handles TERM/INT,
honors NVFlare restart/shutdown markers, and exits nonzero on repeated startup
failure. It does not use the backgrounding ``start.sh`` wrapper.

Staging preserves file modes and ownership. For an image with a non-root runtime
UID, set ``workspace_uid`` and ``workspace_gid`` to that UID/GID using a worker
permitted to set ownership. Both the source workspace and runtime directory use
that ownership. The image must allow the runtime user to traverse ``/vault``.

``user_config`` and ``user_data`` are optional clear, read-only input directories.
Private-key files/PEM content and symlinks are rejected. Never place a startup kit
there. ``/applog`` is clear public output; confidential logs and state belong under
``/vault``. The four drive sizes are positive GiB integers: ``vault_drive_size``
defaults to 8 and the three sidecar sizes default to 1. Allow space for both the
Docker archive and imported image/layers, runtime state, and storage overhead.

Outputs and failure recovery
----------------------------

Run ``nvflare provision -p project.yml -w /srv/nvflare/provisioning`` on the worker.
The adapter calls
``cvmctl vault <absolute-config> --project-config <absolute-project-config> --output <fresh-output>``
once per participant, without ``--candidate`` or ``--dev``. Generated
``vault_build.yml`` contains ``cvm_image`` and the automatically derived
``image_id``. It never contains ``deployment_id``, ``cvm_profile`` or
``trustee``. Omitted ``platforms`` remains omitted so the builder selects all
available platforms; an explicit subset is preserved.

With the default output location:

.. code-block:: text

   workspace/project1/
     .cvm-vault-builds/<run-id>-inputs/
       startup-kit/               # original signed kit, retained
       application/workspace/     # signed kit copied into the encrypted vault
       application/runtime/       # initially empty runtime directory
       vault_build.yml            # generated build inputs
       build.log                  # retained on failure
       result.json                # successful public delivery metadata
     prod_NN/site-1/
       vault_set.json
       oci_artifacts.json
       vault_<generated-id>_intel_tdx.oci.tar
       intel_tdx/                 # CVM bundle, vault, sidecars, recovery records

With an explicit ``output_root``, the corresponding paths are
``output_root/<run-id>-inputs/`` and ``output_root/<run-id>/``. The caller's staging
ID is separate from the deployment ID generated by CVM Builder.

The CLI's JSON result adds ``cvm_vaults`` with participant/deployment IDs, output
directories, and each platform's complete OCI archive path, archive SHA-256,
OCI manifest digest, CVM build ID, and vault resource identity. The adapter reads
JSON metadata, discovers filenames from the inventory, verifies archive checksums
and OCI identities, and checks requested platforms and local-profile build IDs. It never parses console
messages to locate deliveries or includes credentials in result metadata.

Distribute the platform ``.oci.tar`` and its checksum through an authenticated
channel. Keep input staging and administrative records private. Use the builder's
``cvmctl publish`` for registry publication or ``cvmctl pull`` to
materialize a local archive or immutable registry digest. Then run the delivered
``launch_cvm.sh``/``shutdown_cvm.sh`` on the appropriate runtime host, following
``USER_GUIDE.md``. Provisioning does not publish or launch the delivery.

A failed build exits nonzero and retains inputs, logs, outputs and any successful
platform deliveries. Inspect ``build_failure.json`` when present and each
platform's ``provisioning.json``: ``uploading`` can mean the key was accepted but
the acknowledgement was lost; ``active`` confirms acknowledgement. Packaging can
fail after activation without a ``build_failure.json``. Resolve partial or
uncertain activation with the Trustee administrator before deliberately
starting a fresh build. The adapter never retries construction or removes
recovery records automatically.

Never attach the same writable vault file to two CVMs. Copy only stopped,
detached images; copies retain the same identity and revocation scope. Different
participants require separate vault builds and startup kits.
