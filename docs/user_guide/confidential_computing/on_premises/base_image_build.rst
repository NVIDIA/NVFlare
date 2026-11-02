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

Follow the :github_nvflare_link:`CVM Builder build guide
<nvflare/lighter/cc/image_builder/BUILD_GUIDE.md>` for the tested Ubuntu 26.04
construction environment, exact package pins and input preparation. The
profile in ``config/cvm_profile.yml`` selects the cloud image, platform firmware,
upstream CoCo KBS client, attestation trust roots and approved TCB references.
For GPU profiles, also follow :github_nvflare_link:`GPU_BUILD.md
<nvflare/lighter/cc/image_builder/GPU_BUILD.md>` for driver packages,
authenticated package repositories and the NVIDIA attestation library.

Run from the builder directory after preparing those inputs:

.. code-block:: bash

   sudo --preserve-env=CVM_BUILDER_PYTHON ./cvmctl build config/cvm_profile.yml

The command screens the target platform, builds and measures the generic image,
and emits an unapproved candidate. Continue with the build guide's exact-manifest
acceptance flow: qualify the isolated Trustee's administration, install that
candidate, seal its test vault, run the shipped acceptance runner, sign approval,
then install and publish. Production use requires approval of the exact bundle.

For site automation, an authorized operator may add ``--acceptance-runner``,
``--approval-key``, ``--install-config``, and signed-publication options. This
requires a **site-supplied adapter** implementing ``RUNNER BUNDLE REPORT``;
``/usr/local/sbin/site_acceptance_runner`` is an example name, not a shipped
program. The shipped lower-level runner cannot create its own candidate vault
and must not be passed directly to the combined build command. See the build
guide for that adapter's prerequisites and role boundaries. Use
``./cvmctl build --help`` for explicit platform selection, a TDX quote probe,
deferred measurement, and other options.

Configure upstream Trustee using the :github_nvflare_link:`Trustee guide
<nvflare/lighter/cc/image_builder/TRUSTEE_GUIDE.md>`. Use its standalone path or
merge the same CVM policy and references into a CoCo-managed instance. Guest KBS
clients and Trustee must use the documented matching upstream versions. Build a
generic bundle once per platform and profile, then reuse it for subsequent
application vaults.
