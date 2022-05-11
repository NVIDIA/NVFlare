.. _example_apps:

#############################
Example Apps for NVIDIA FLARE
#############################
NVIDIA FLARE has several examples to help you get started with federated learning and to explore certain features in
`the examples directory <https://github.com/NVIDIA/NVFlare/tree/main/examples>`_.

The following quickstart guides walk you through some of these examples:

.. toctree::
   :maxdepth: 1

   examples/hello_pt
   examples/hello_pt_tb
   examples/hello_numpy
   examples/hello_tf2
   examples/hello_cross_val
   Federated Learning with CIFAR-10 <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>

For the complete collection of example applications, see https://github.com/NVIDIA/NVFlare/tree/main/examples.

Custom Code in Example Apps
===========================
There are several ways to make :ref:`custom code <custom_code>` available to clients when using NVIDIA FLARE. Most
hello-* examples use a custom folder within the FL application. Note that using a custom folder in the app needs to be
:ref:`allowed <troubleshooting_byoc>` when using secure provisioning. By default, this option is disabled in the secure
mode. POC mode, however, will work with custom code by default.

In contrast, the `CIFAR-10 <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_,
`prostate segmentation <https://github.com/NVIDIA/NVFlare/tree/main/examples/prostate>`_,
and `BraTS18 segmentation <https://github.com/NVIDIA/NVFlare/tree/main/examples/brats18>`_ examples assume that the
learner code is already installed on the client's system and
available in the PYTHONPATH. Hence, the app folders do not include the custom code there. The PYTHONPATH is
set in the ``run_poc.sh`` or ``run_secure.sh`` scripts of the example. Running these scripts as described in the README
will make the learner code available to the clients.
