.. _cc_deployment_guide:

################################################
FLARE Confidential Federated AI Deployment Guide
################################################

NVFlare uses CVM Builder to deploy a server and clients inside confidential VMs
on Intel TDX or AMD SEV-SNP hosts. GPU clients additionally require a supported
NVIDIA confidential-computing GPU and a GPU-enabled CVM profile.

CVM Builder is included in the NVFlare source tree at
``nvflare/lighter/cc/image_builder``. The deployment has two stages:

1. Build, validate and approve a reusable generic CVM for each platform and profile.
2. Provision NVFlare participants and seal each participant's signed startup kit
   and Docker application into a fresh encrypted vault using those approved CVMs.

Each participant receives a complete OCI delivery containing the generic CVM,
application vault, sidecars and launch scripts. Application updates reuse the
approved generic image and create a new vault.

Prepare the hosts and key service
=================================

Configure a trusted Linux build worker with the builder's host tools and Python
requirements. Hardware finalization and launch require a host supporting the
selected TEE. The tested builder environment is Ubuntu 26.04; package, firmware,
QEMU, driver and attester pins belong to each reviewed profile.

Use these guides in ``nvflare/lighter/cc/image_builder``:

- ``BUILD_GUIDE.md``: build-worker prerequisites, generic CVM construction,
  hardware finalization, approval and application-vault construction.
- ``TRUSTEE_GUIDE.md``: the pinned Trustee deployment, immutable appraisal
  policies, mutual TLS, administrator access and vault-key lifecycle.
- ``USER_GUIDE.md``: delivery verification, host preparation, launch and shutdown.
- ``VALIDATION.md``: recorded test results and remaining production acceptance.

Build and approve the generic images before provisioning participants. CPU-only
and GPU-enabled images have distinct profile contracts; select the image whose
capabilities match the participant. Each approved platform bundle has its own
measurements and key-resource namespace.

Prepare the application and project
===================================

Build a Linux amd64 Docker image containing NVFlare, Bash and the application's
code and dependencies. Save it with ``docker save``. NVFlare derives the image ID
from the archive; the archive must contain exactly one distinct image.

Use the example in ``examples/advanced/cvm_builder``. It configures
an Intel TDX server and an AMD SEV-SNP GPU client. Update:

- ``project.yml``: participant names, the builder directory, each participant's
  approved ``cvm_image``, Docker archive, GPU requirements and network ports.
- ``cvm_project.yml``: the shared key-service endpoint and existing builder
  credentials. Relative credential paths resolve against this file.

``cvm_image`` accepts a pulled generic-image folder or an immutable OCI registry
reference. The selected platforms default to those in the image. See
:ref:`cvm_builder` for the complete configuration and validation rules.

Provision and distribute
========================

Run provisioning on the prepared Linux worker:

.. code-block:: bash

   cd examples/advanced/cvm_builder
   nvflare provision -p project.yml -w ./workspace

The ``cvm_vault`` adapter invokes ``cvmctl vault`` after startup-kit generation
and signing finish. Deliveries default to
``workspace/cvm_project/prod_NN/<participant>/``. The administrator's ordinary
startup kit remains in the same production directory. Private build inputs,
logs and recovery records are retained separately.

Read the returned OCI archive paths and digests. Transfer the complete archives
offline or publish them using the included ``cvmctl publish``. Recipients
use ``cvmctl pull`` to verify and materialize a delivery from its archive or
immutable registry reference. Follow ``USER_GUIDE.md`` for the exact commands.

Launch and run a job
====================

On each configured TEE host, enter the materialized participant directory and
run ``sudo ./launch_cvm.sh``. The launcher discovers the included CVM bundle.
The verified guest attests, unlocks the vault and starts the NVFlare participant;
GPU profiles also require successful GPU appraisal before workload execution.

Start the ordinary NVFlare administrator console from its generated kit, verify
that the server and clients are connected, and submit a job whose code and
dependencies are present in the application image. Inspect the job results and
participant logs. ``/applog`` is clear writable output; ``user_config`` and
``user_data`` are clear, read-only inputs. Put confidential data in the vault.

Stop each delivery with ``sudo ./shutdown_cvm.sh``. For failed builds, retain the
logs and recovery records and resolve uncertain key uploads before deliberately
starting another build. Production approval requires the exact bundle's complete
acceptance evidence; a successful candidate-mode test is not an approval.
