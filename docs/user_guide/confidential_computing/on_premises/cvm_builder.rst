.. _cvm_builder:

Provisioning with CVM Builder
=============================

CVM Builder creates reusable, approved generic CVMs and combines them with an
application-specific encrypted vault during NVFlare provisioning. Configure the
participant through the common :ref:`cc_deployment` interface, select
``cc_deployment_mode: bare_metal_cvm``, and keep Trustee/KBS, approval keys, and
the builder location in the project's single ``cc_project.yml``. This page
covers the mode-specific operator workflow; it does not define a second
provisioning schema.

Prepare the worker
------------------

Run provisioning and vault construction on a trusted Linux worker configured
for CVM Builder. The tested builder environment is Ubuntu 26.04. Follow the
:github_nvflare_link:`build guide
<nvflare/lighter/cc/image_builder/BUILD_GUIDE.md>` and
:github_nvflare_link:`Trustee guide
<nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md>` for disk tools, disabled
swap, core-dump policy, locked memory, approved bundles, and Trustee
administration. Provisioning does not change these host settings or launch a
CVM.

For a source checkout at ``/opt/NVFlare``, prepare the builder environment with:

.. code-block:: bash

   cd /opt/NVFlare/nvflare/lighter/cc/image_builder
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt

``cvmctl vault`` selects ``CVM_BUILDER_PYTHON``, its own ``.venv/bin/python``,
or ``python3``. An unprivileged provisioner uses ``sudo -n`` where construction
or output inspection requires it. Configure that boundary in advance; no
interactive sudo prompt or remote worker submission is implemented.

Prepare the generic CVM and application
---------------------------------------

Set ``bare_metal_cvm.cvm_image`` to an immutable OCI registry reference or a
previously pulled directory. A pulled directory contains ``profile_set.json``
and every referenced platform bundle. For registry input, install ORAS and
configure authentication for the provisioning account. CVM Builder validates
the approval of every selected bundle.

Build a Linux amd64 application image containing NVFlare, Bash, application
code, and dependencies, then save it:

.. code-block:: bash

   image=my-nvflare-application:1.0
   docker pull --platform linux/amd64 "$image"
   docker save "$image" -o /srv/nvflare/images/application.tar

Set ``workload.source.type: docker_archive`` and point ``path`` at that file.
NVFlare derives the image identity from the archive. The archive must contain
one distinct image; multiple tags for that image are allowed. ``docker export``
output is not supported.

The generic CVM contract must match the participant's ``cpu_tee`` and
``gpu_tee``. CPU-only and GPU-enabled bundles have different approval and
measurement contracts. The configuration has no independent platform list:
the requested TEE pair selects the matching profile from ``cvm_image``.

Provision and distribute
------------------------

Use the common project, service, participant, builder, and packager structure
in :ref:`cc_deployment`, then run:

.. code-block:: bash

   nvflare provision -p project.yml -w /srv/nvflare/provisioning

The common packager verifies every selected startup kit before moving the
plaintext kits into private state. It then invokes ``cvmctl vault`` once per
participant. ``build_tools.bare_metal_cvm.output_root`` is optional. Without it,
deliveries use the production directory layout; with it, each build uses a
fresh operator-private output directory.

The public CC manifest records each OCI archive checksum, OCI manifest digest,
CVM build ID, platform, and vault resource identity. It contains no credentials
or private staging paths. Distribute the complete ``.oci.tar`` and checksum
through an authenticated channel, or publish it with ``cvmctl publish``. A
recipient can use ``cvmctl pull`` to verify and materialize an archive or
immutable registry reference.

On a matching TEE host, enter the materialized participant directory and run
``sudo ./launch_cvm.sh``. The verified guest attests, unlocks the application
vault, and starts the NVFlare participant. Stop it with
``sudo ./shutdown_cvm.sh``. See the :github_nvflare_link:`CVM Builder user guide
<nvflare/lighter/cc/image_builder/USER_GUIDE.md>` for delivery and launch
commands.

Network, data, and recovery behavior
------------------------------------

Federation, admin, relay, and listener ports are derived from the signed kit.
``bare_metal_cvm.network`` adds only application-specific ports and CIDRs.
``user_config`` and ``user_data`` are optional clear, read-only directories;
private keys, PEM private-key content, symlinks, and startup kits are rejected.
``/applog`` is clear writable output. Confidential logs and runtime state belong
under the encrypted vault.

A failed build keeps its private input, logs, output, and recovery records.
Resolve an uncertain Trustee upload with the Trustee administrator before
starting a fresh build. Never attach one writable vault file to two CVMs or copy
it while running; copies retain the same identity and revocation scope.
