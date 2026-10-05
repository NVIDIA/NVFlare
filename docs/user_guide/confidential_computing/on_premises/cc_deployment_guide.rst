.. _cc_deployment_guide:

################################################
FLARE Confidential Federated AI Deployment Guide
################################################

Use the unified :ref:`cc_deployment` workflow for bare-metal CVMs, CoCo, and
Azure CC. For an on-premises CVM deployment, select ``bare_metal_cvm`` for each
protected server or client. The common guide defines the project and
participant files, shared Trustee service, builder order, and result manifest.

The on-premises workflow has two stages:

1. Build, validate, and approve a reusable generic CVM for each CPU/GPU TEE
   profile.
2. Provision NVFlare participants and seal each signed startup kit and Docker
   application into a fresh encrypted vault.

Prepare the hosts and key service
=================================

Configure a trusted Linux build worker with the tools and Python requirements
from ``nvflare/lighter/cc/image_builder``. Hardware finalization and launch need
a host supporting the selected TEE. Follow its ``BUILD_GUIDE.md``,
``TRUSTEE_GUIDE.md``, ``USER_GUIDE.md``, and the relevant TDX or SEV-SNP host
guide. Build and approve generic images before provisioning participants.

Prepare the application and project
===================================

Save one Linux amd64 application image with ``docker save``. Use
``examples/advanced/cvm_builder`` as the complete configuration example. Point
each participant at its own Docker archive and an approved CVM image whose
contract matches ``cpu_tee`` and ``gpu_tee``. Put Trustee/KBS and approval-key
settings once in ``cc_project.yml``.

Provision and distribute
========================

Run provisioning on the prepared worker:

.. code-block:: bash

   nvflare provision -p project.yml -w /srv/nvflare/provisioning

Read the generated ``cc_manifests`` entries for archive paths and digests.
Transfer each complete OCI artifact through an authenticated channel or publish
it with ``cvmctl publish``. Keep private state, build logs, credentials, and
plaintext startup kits on the trusted worker.

Launch and run a job
====================

Materialize each delivery on the matching TEE host and run
``sudo ./launch_cvm.sh``. Start the ordinary NVFlare administrator console,
verify that the expected server and clients are connected and attested, and
submit a job whose code and dependencies are already in the application image.
Stop each delivery with ``sudo ./shutdown_cvm.sh``.

Successful candidate testing is not production approval. Production approval
must cover the exact generic CVM bundle, application archive, Trustee policy,
TEE profile, and resulting delivery.
