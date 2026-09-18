.. _base_image_build:

##################################
CVM Base Image and Binary Building
##################################

CVM Builder is included at ``nvflare/lighter/cc/image_builder`` in the NVFlare
source tree. Its ``cvmctl build`` command creates a reusable generic CVM;
``cvmctl vault`` seals an application for that CVM. NVFlare provisioning uses the
vault command through the configuration described in :doc:`cvm_builder`.

Prepare the generic image
=========================

Follow ``nvflare/lighter/cc/image_builder/BUILD_GUIDE.md`` for the tested Ubuntu
26.04 construction environment, exact package pins and input preparation. The
profile in ``config/cvm_profile.yml`` selects the cloud image, platform firmware,
upstream CoCo KBS client, attestation trust roots and approved TCB references.
For GPU profiles, also follow ``GPU_BUILD.md`` for driver packages, authenticated
package repositories and the NVIDIA attestation library.

Run from the builder directory after preparing those inputs:

.. code-block:: bash

   sudo ./cvmctl build

The command detects the target platform, builds and measures the generic image,
and invokes the configured site acceptance runner. Use
``./cvmctl build --help`` for explicit platform selection, deferred measurement
and other build options. Production use requires approval of the exact bundle.

Configure the existing CoCo Trustee using ``TRUSTEE_GUIDE.md``. Guest KBS clients
and Trustee must use the documented matching upstream versions. Build a generic
bundle once per platform and profile, then reuse it for subsequent application
vaults. The older Ansible image workflow and its Ubuntu 24.04 guest inputs do not
apply to this builder.
