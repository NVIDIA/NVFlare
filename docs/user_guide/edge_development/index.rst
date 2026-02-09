:orphan:

.. _edge_mobile_overview:

################
Edge & Mobile
################

FLARE extends federated learning beyond data centers to edge devices and mobile platforms.

.. toctree::
   :maxdepth: 1
   :hidden:

   edge_devices
   mobile_training

Edge Device Training
====================

Train on GPU-capable edge devices (e.g., NVIDIA Jetson) using standard PyTorch with the
``EdgeFedBuffRecipe``. Supports hierarchical architecture for scaling to thousands of devices
with flexible synchronous-to-asynchronous aggregation.

- :doc:`Edge Device Training (Jetson / GPU) <edge_devices>` -- Architecture, EDIP protocol, simulation, and EdgeFedBuffRecipe

Mobile Federated Training
=========================

Train on iOS and Android devices using `ExecuTorch <https://github.com/pytorch/executorch>`_.
**No device-side programming needed** -- develop your model in standard PyTorch and use the
``ETFedBuffRecipe`` to handle export, deployment, and federated orchestration.

- :doc:`Mobile Federated Training (iOS / Android) <mobile_training>` -- DeviceModel, ETFedBuffRecipe, and mobile SDK guides
