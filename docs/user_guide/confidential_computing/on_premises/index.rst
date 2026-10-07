*************************
On-Premises IP Protection
*************************

The following documents describe FLARE's Confidential Federated AI deployment
and runtime attestation support:

- :ref:`cc_architecture` - Trust model, trust boundaries, chain of trust, and attestation-bound key release
- :ref:`cvm_builder` - Provision application vaults using reusable, approved CVM Builder bundles
- :ref:`cc_deployment_guide` - Deployment guide for Intel TDX and AMD SEV-SNP CVMs, with optional NVIDIA GPU support
- :ref:`base_image_build` - Build instructions for Ubuntu base image, firmware, and required binaries
- :ref:`confidential_computing_attestation` - Attestation mechanisms and trust establishment

.. toctree::
   :maxdepth: 2

   security_architecture
   cvm_builder
   cc_deployment_guide
   base_image_build
   attestation
