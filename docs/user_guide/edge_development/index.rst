:orphan:

.. _edge_mobile_overview:

################
Edge & Mobile
################

FLARE extends federated learning beyond data centers to edge devices and mobile platforms.

.. toctree::
   :maxdepth: 1
   :hidden:

   mobile_training

Mobile Federated Training
=========================

Train on iOS and Android devices using `ExecuTorch <https://github.com/pytorch/executorch>`_.
**No device-side programming needed** -- develop your model in standard PyTorch and use the
``ETFedBuffRecipe`` to handle export, deployment, and federated orchestration.

- :doc:`Mobile Federated Training (iOS / Android) <mobile_training>` -- DeviceModel, ETFedBuffRecipe, and mobile SDK guides

Hierarchical FLARE
==================

For scaling to thousands of devices with hierarchical aggregation and relay-based communication,
see :ref:`Hierarchical FLARE <flare_hierarchical_architecture>` and
:ref:`Hierarchical Communication <hierarchical_communication>`.
